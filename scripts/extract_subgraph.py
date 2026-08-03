import pickle
import re
from pathlib import Path
from typing import List, Set, Tuple

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-m3"
GRAPH_PATH = Path("artifacts/graphs/financial_graph.pkl")

# Топ-K узлов для первоначального поиска
TOP_K_NODES = 15
# Радиус BFS для расширения подграфа
BFS_RADIUS = 1
MAX_SUBGRAPH_NODES = 30


class SubgraphRetriever:
    def __init__(self, graph_path: Path = GRAPH_PATH):
        # Загружаем граф
        with open(graph_path, "rb") as f:
            self.G: nx.DiGraph = pickle.load(f)
        
        # Загружаем модель эмбеддингов (один раз)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Готовим индекс: для каждого узла — его эмбеддинг
        self._build_index()
    
    def _build_index(self):
        """Строит индекс узлов для поиска."""
        self.node_ids = []
        self.node_texts = []
        
        for node, data in self.G.nodes(data=True):
            # Формируем текстовое представление узла для эмбеддинга
            text = self._node_to_text(node, data)
            self.node_ids.append(node)
            self.node_texts.append(text)
        
        # Считаем эмбеддинги один раз
        self.node_embeddings = self.embedder.encode(
            self.node_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    
    def _node_to_text(self, node_id: str, data: dict) -> str:
        """Превращает узел в текст для эмбеддинга с богатым контекстом."""
        node_type = data.get("type", "")
        
        if node_type == "table":
            desc = data.get("description", "")
            name = data.get("name", "")
            # Добавляем список колонок (через HAS_COLUMN рёбра)
            cols = []
            for neighbor in self.G.successors(node_id):
                if self.G.nodes[neighbor].get("type") == "column":
                    cols.append(self.G.nodes[neighbor].get("name", ""))
            cols_str = ", ".join(cols[:10])
            return f"TABLE {name}: {desc}. Columns: {cols_str}"
        
        elif node_type == "column":
            full_name = data.get("full_name", "")
            dtype = data.get("data_type", "")
            desc = data.get("description", "")
            # Добавляем примеры значений (через VALUE_IN рёбра)
            values = []
            for neighbor in self.G.predecessors(node_id):
                if self.G.nodes[neighbor].get("type") == "value":
                    val = self.G.nodes[neighbor].get("value", "")
                    if val:
                        values.append(val)
            values_str = ", ".join(f"'{v}'" for v in values[:5])
            text = f"COLUMN {full_name} ({dtype}): {desc}"
            if values_str:
                text += f" Values: {values_str}"
            # Для дат — диапазон
            if "date" in dtype or "timestamp" in dtype:
                min_d = data.get("min_date")
                max_d = data.get("max_date")
                if min_d and max_d:
                    text += f" [range: {min_d} to {max_d}]"
            return text
        
        elif node_type == "value":
            value = data.get("value", "")
            col = data.get("column", "")
            if value == "":
                value_repr = "<empty string>"
            else:
                value_repr = value
            # Добавляем описание колонки
            col_node = f"COL:{col}"
            col_desc = ""
            if self.G.has_node(col_node):
                col_desc = self.G.nodes[col_node].get("description", "")
            return f"VALUE '{value_repr}' in {col}. {col_desc}"
        
        return f"{node_id}"
    
    def retrieve(self, question: str) -> nx.DiGraph:
        """
        Извлекает подграф, релевантный вопросу.
        
        Алгоритм:
        1. Эмбеддим вопрос
        2. Находим TOP_K самых похожих узлов
        3. Расширяем через BFS на радиус BFS_RADIUS
        4. Возвращаем подграф
        """
        # 1. Эмбеддинг вопроса
        q_emb = self.embedder.encode([question], normalize_embeddings=True)[0]
        
        # 2. Поиск ближайших узлов (cosine similarity)
        similarities = self.node_embeddings @ q_emb
        top_indices = np.argsort(similarities)[::-1][:TOP_K_NODES]
        
        seed_nodes: Set[str] = set()
        for idx in top_indices:
            node_id = self.node_ids[idx]
            sim = similarities[idx]
            # Берём только узлы с разумной похожести
            if sim >= 0.3:
                seed_nodes.add(node_id)
        
        # 3. BFS-расширение: добавляем связанные узлы
        expanded_nodes: Set[str] = set(seed_nodes)
        
        for seed in seed_nodes:
            # BFS в обе стороны (граф направленный, но связи важны в обе)
            for neighbor in list(nx.bfs_tree(self.G, seed, depth_limit=BFS_RADIUS)):
                expanded_nodes.add(neighbor)
            for neighbor in list(nx.bfs_tree(self.G.reverse(), seed, depth_limit=BFS_RADIUS)):
                expanded_nodes.add(neighbor)
        
        # 4. Формируем подграф
        if len(expanded_nodes) > MAX_SUBGRAPH_NODES:
            # Сортируем по релевантности (seed-узлы имеют приоритет)
            node_scores = {}
            for node in expanded_nodes:
                if node in seed_nodes:
                    node_scores[node] = 1.0  # Максимальный приоритет
                else:
                    # Считаем расстояние до ближайшего seed-узла
                    min_dist = min(
                        nx.shortest_path_length(self.G, seed, node) 
                        for seed in seed_nodes 
                        if nx.has_path(self.G, seed, node)
                    ) if any(nx.has_path(self.G, seed, node) for seed in seed_nodes) else 999
                    node_scores[node] = 1.0 / (1.0 + min_dist)
            
            # Берём топ-N узлов
            expanded_nodes = set(sorted(node_scores.keys(), key=lambda x: -node_scores[x])[:MAX_SUBGRAPH_NODES])

        subgraph = self.G.subgraph(expanded_nodes).copy()
        
        return subgraph
    
    def subgraph_to_context(self, subgraph: nx.DiGraph) -> str:
        """
        Превращает подграф в текстовый контекст для target-LLM.
        Формат — компактный M-Schema-подобный.
        """
        # Группируем узлы по типу
        tables = {}
        for node, data in subgraph.nodes(data=True):
            node_type = data.get("type")
            if node_type == "table":
                tables[data["name"]] = {"desc": data.get("description", ""), "columns": []}
        
        # Собираем колонки по таблицам
        for node, data in subgraph.nodes(data=True):
            if data.get("type") == "column":
                full_name = data.get("full_name", "")
                if "." in full_name:
                    table_name = full_name.split(".")[0]
                    if table_name in tables:
                        col_info = {
                            "name": data.get("name"),
                            "type": data.get("data_type"),
                            "desc": data.get("description", ""),
                            "is_pk": data.get("is_pk", False),
                            "values": [],
                        }
                        # Для дат — диапазон
                        if "date" in data.get("data_type", ""):
                            min_d = data.get("min_date")
                            max_d = data.get("max_date")
                            if min_d and max_d:
                                col_info["date_range"] = f"{min_d} to {max_d}"
                        tables[table_name]["columns"].append(col_info)
        
        # Собираем значения по колонкам
        for node, data in subgraph.nodes(data=True):
            if data.get("type") == "value":
                col_full = data.get("column", "")
                if "." in col_full:
                    table_name, col_name = col_full.split(".", 1)
                    if table_name in tables:
                        for col in tables[table_name]["columns"]:
                            if col["name"] == col_name:
                                col["values"].append(data.get("value"))
        
        # Собираем FK
        fk_lines = []
        for u, v, data in subgraph.edges(data=True):
            if data.get("relation") == "FK_REFERENCES":
                u_short = u.replace("COL:", "")
                v_short = v.replace("COL:", "")
                fk_lines.append(f"  {u_short} -> {v_short}")
        
        # Формируем итоговый текст
        lines = ["[Schema (relevant subset)]"]
        for table_name, table_data in tables.items():
            lines.append(f"\n# Table: {table_name}")
            if table_data["desc"]:
                lines.append(f"# {table_data['desc']}")
            lines.append("[")
            for col in table_data["columns"]:
                pk = ", PK" if col["is_pk"] else ""
                vals = ""
                if col["values"]:
                    # Показываем до 5 значений
                    sample = col["values"][:5]
                    vals = f", Examples: {sample}"
                date_range = ""
                if "date_range" in col:
                    date_range = f", Range: {col['date_range']}"
                lines.append(f"  ({col['name']}: {col['type']}{pk}{date_range}{vals})")
            lines.append("]")
        
        if fk_lines:
            lines.append("\n[Foreign Keys]")
            lines.extend(fk_lines)
        
        return "\n".join(lines)


# ---- Пример использования ----
if __name__ == "__main__":
    retriever = SubgraphRetriever()
    
    questions = [
        "Покажи все транзакции клиентов из Праги за 1997 год",
        "Какие клиенты имеют золотую карту?",
        "Сумма всех кредитов со статусом A",
        "Средний баланс по счетам с периодичностью POPLATEK MESICNE",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        subgraph = retriever.retrieve(q)
        context = retriever.subgraph_to_context(subgraph)
        
        print(f"\nПодграф: {subgraph.number_of_nodes()} узлов, {subgraph.number_of_edges()} рёбер")
        print(f"\nКонтекст для LLM ({len(context)} символов):")
        print(context)