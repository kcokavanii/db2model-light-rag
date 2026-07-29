"""
preview_graph.py — «взгляд архитектора» на собранные знания о БД.

Читает db_knowledge/*.json и показывает, как эти данные
лягут в граф LightRAG на Неделе 3:
  • узлы — таблицы, колонки, значения-справочники
  • рёбра — внешние ключи (FK)
"""
import json
from pathlib import Path

KB_DIR = Path(__file__).parent.parent / "db_knowledge"
LOW_CARDINALITY_THRESHOLD = 20 


def parse_n_distinct(val) -> float | None:
    """
    В pg_stats n_distinct бывает:
      > 0  — точное число уникальных значений
      < 0  — доля от размера таблицы (например -0.5 = 50% уникальных)
      None — ANALYZE не запускался
    Для справочников нам нужны только положительные маленькие значения.
    """
    if val is None:
        return None
    try:
        v = float(val)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def analyze_db(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tables = data.get("tables", {})
    n_tables = len(tables)
    n_edges = 0            # FK = рёбра графа
    n_dict_nodes = 0       # потенциальные узлы-справочники
    n_regular_nodes = 0    # обычные колонки
    dict_examples = []     # для красивой печати

    for table_name, table in tables.items():
        # Рёбра = внешние ключи
        n_edges += len(table.get("foreign_keys", []))

        # Узлы = колонки + возможные значения-справочники
        for col in table.get("columns", []):
            col_name = col["column_name"]
            stats = table.get("column_statistics", {}).get(col_name, {})
            n_distinct = parse_n_distinct(stats.get("n_distinct"))

            if n_distinct is not None and n_distinct < LOW_CARDINALITY_THRESHOLD:
                n_dict_nodes += 1
                if len(dict_examples) < 5:  # топ-5 примеров для отчёта
                    dict_examples.append(
                        f"  • {table_name}.{col_name} "
                        f"(≈{int(n_distinct)} значений)"
                    )
            else:
                n_regular_nodes += 1

    return {
        "db": path.stem.replace("_knowledge", ""),
        "tables": n_tables,
        "edges": n_edges,
        "regular_nodes": n_regular_nodes,
        "dict_nodes": n_dict_nodes,
        "dict_examples": dict_examples,
    }


def main():
    if not KB_DIR.exists():
        print(f"Папка {KB_DIR} не найдена. Сначала запусти explore_database.py")
        return

    files = sorted(KB_DIR.glob("*_knowledge.json"))
    if not files:
        print(f"В {KB_DIR} нет JSON-файлов.")
        return

    print(f"\n🔎 Анализируем {len(files)} БД из {KB_DIR}\n")
    print(f"{'Database':<25} {'Tables':>7} {'Edges(FK)':>10} "
          f"{'Dict nodes':>11} {'Regular':>8}")
    print("-" * 68)

    total_tables = total_edges = total_dict = 0
    all_examples = []

    for f in files:
        r = analyze_db(f)
        print(f"{r['db']:<25} {r['tables']:>7} {r['edges']:>10} "
              f"{r['dict_nodes']:>11} {r['regular_nodes']:>8}")
        total_tables += r["tables"]
        total_edges += r["edges"]
        total_dict += r["dict_nodes"]
        all_examples.extend(r["dict_examples"])

    print("-" * 68)
    print(f"{'ИТОГО':<25} {total_tables:>7} {total_edges:>10} {total_dict:>11}")

    print(f"\n💡 Для LightRAG это означает:")
    print(f"   • ~{total_tables + total_dict} узлов графа "
          f"(таблицы + справочники)")
    print(f"   • ~{total_edges} рёбер (FK-связи)")
    print(f"   • {total_dict} справочников можно развернуть "
          f"в отдельные сущности графа")

    if all_examples:
        print(f"\nПримеры справочников (top-5):")
        for ex in all_examples[:5]:
            print(ex)
    print()


if __name__ == "__main__":
    main()