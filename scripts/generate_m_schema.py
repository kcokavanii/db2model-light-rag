import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Динамическое определение корня проекта для корректного импорта промптов
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("generate_m_schema.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

from src.adv_text2sql.mcp_servers.text2sql_tool.src.prompts import (
    M_SCHEMA_TEACHER_PROMPT
)

def format_value(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return str(val)


def generate_xiyan_m_schema(db_data: Dict[str, Any], db_name: str) -> str:
    lines = [f"[DB_ID] {db_name}", "[Schema]"]
    
    for table in db_data.get("tables", []):
        table_name = table["table_name"]
        lines.append(f"# Table: {table_name}")
        lines.append("[")
        
        col_lines = []
        columns = table.get("columns", [])
        col_stats = table.get("column_statistics", {})
        primary_keys = table.get("primary_keys", [])
        
        for col in columns:
            col_name = col["column_name"]
            col_type = col["data_type"].upper()

            is_pk = col_name in primary_keys
            pk_marker = ", Primary Key" if is_pk else ""
            
            stats = col_stats.get(col_name, {})
            examples = stats.get("most_common_vals", [])
            
            if examples:
                examples_str = ", ".join(format_value(v) for v in examples)
                col_lines.append(f"  ({col_name}:{col_type}{pk_marker}, Examples: [{examples_str}])")
            else:
                col_lines.append(f"  ({col_name}:{col_type}{pk_marker})")
        
        lines.append(",\n".join(col_lines))
        lines.append("]\n")

        fks = table.get("foreign_keys", [])
        if fks:
            lines.append("[Foreign Keys]")
            for fk in fks:
                fk_str = f"{table_name}.{fk['column_name']}={fk['foreign_table']}.{fk['foreign_column']}"
                lines.append(fk_str)
        
        lines.append("")
    
    return "\n".join(lines)


def build_semantic_prompt(table: Dict[str, Any]) -> str:
    table_name = table["table_name"]
    lines = [f"Таблица: {table_name}"]
    
    if table.get("table_comment"):
        lines.append(f"Комментарий к таблице: {table['table_comment']}")
    
    lines.append("\nКолонки:")
    columns = table.get("columns", [])
    col_stats = table.get("column_statistics", {})
    primary_keys = table.get("primary_keys", []) 
    
    for col in columns:
        col_name = col["column_name"]
        col_type = col["data_type"]
        nullable = "YES" if col.get("is_nullable") else "NO"
        is_pk = col_name in primary_keys
        pk_text = " (Primary Key)" if is_pk else ""
        
        stats = col_stats.get(col_name, {})
        examples = stats.get("most_common_vals", [])        
        parts = [f"  - {col_name}{pk_text} (тип: {col_type}, nullable: {nullable})"]
        if examples:
            parts.append(f"примеры: {examples}")
        
        lines.append(", ".join(parts))
    
    fks = table.get("foreign_keys", [])
    if fks:
        lines.append("\nВнешние ключи:")
        for fk in fks:
            lines.append(f"  - {fk['column_name']} -> {fk['foreign_table']}({fk['foreign_column']})")
    
    samples = table.get("sample_rows", [])
    if samples:
        lines.append(f"\nПримеры строк (первые {min(3, len(samples))}):")
        for row in samples[:3]:
            lines.append(f"  {row}")
    
    lines.append("\nСгенерируй M-Schema описание в указанном формате.")
    return "\n".join(lines)


def call_teacher(client: OpenAI, model: str, user_prompt: str, max_retries: int = 3) -> tuple[str, int, int]:
    """Вызов teacher-LLM с retry и увеличенным таймаутом для локальной модели."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": M_SCHEMA_TEACHER_PROMPT}, {"role": "user", "content": user_prompt}],
                temperature=0.2, timeout=300
            )
            usage = response.usage or type('obj', (object,), {'prompt_tokens': 0, 'completion_tokens': 0})
            return response.choices[0].message.content or "", usage.prompt_tokens, usage.completion_tokens
        except Exception as e:
            log.warning(f"  attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries: time.sleep(2 ** attempt)
    
    return "", 0, 0



def generate_semantic_m_schema(db_data: Dict[str, Any], db_name: str, client: OpenAI, model: str) -> tuple[str, Dict]:
    """Генерирует семантическое M-Schema через teacher-LLM."""
    tables = db_data.get("tables", [])
    log.info(f"=== DB: {db_name} | {len(tables)} tables ===")
    
    sections = []
    total_p, total_c, total_time = 0, 0, 0.0
    failed = []
    
    for table in tables:
        table_name = table["table_name"]
        log.info(f"  -> {table_name}")
        t0 = time.time()
        user_prompt = build_semantic_prompt(table)
        
        try:
            content, p_tok, c_tok = call_teacher(client, model, user_prompt)
            dt = time.time() - t0
            total_p += p_tok
            total_c += c_tok
            total_time += dt
            
            log.info(f"     in={p_tok} out={c_tok} | {dt:.1f}s")
            
            if "[TABLE]" not in content or "[COLUMN]" not in content:
                log.warning(f"error: bad format, skipping table")
                failed.append(table_name)
                continue
            
            sections.append(content)
        except Exception as e:
            log.error(f"     ✗ failed: {e}")
            failed.append(table_name)
    
    header = (
        f"# Semantic M-Schema: {db_name}\n"
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Model: {model}\n"
        f"# Tables: {len(tables)} | Failed: {len(failed)}\n"
        f"# Tokens: in={total_p} out={total_c}\n"
    )
    if failed:
        header += f"# Failed tables: {', '.join(failed)}\n"
    
    stats = {
        "db_name": db_name,
        "tables": len(tables),
        "failed": failed,
        "prompt_tokens": total_p,
        "completion_tokens": total_c,
        "time_s": total_time,
    }
    
    return header + "\n\n---\n\n".join(sections) + "\n", stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Путь к *_knowledge.json или директории")
    parser.add_argument("-o", "--output", required=True, help="Выходная директория для m_schema файлов")
    args = parser.parse_args()
    
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL_NAME")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    log.info(f"model={model} base_url={base_url}")
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(in_path.glob("*_knowledge.json")) if in_path.is_dir() else [in_path]
    if not json_files:
        raise FileNotFoundError(f"Нет *_knowledge.json в {in_path}")
    
    all_stats = []    
    for jf in json_files:
        # Извлекаем db_name из имени файла (например, financial_knowledge.json -> financial)
        db_name = jf.stem.replace("_knowledge", "")
        log.info(f"\nProcessing: {jf} (db_name={db_name})")
        
        with jf.open("r", encoding="utf-8") as f:
            db_data = json.load(f)
        
        # M_schema (XiYan-SQL формат) (без LLM, мгновенно)
        xiyan_schema = generate_xiyan_m_schema(db_data, db_name)
        xiyan_file = out_path / f"{db_name}_m_schema.txt"
        xiyan_file.write_text(xiyan_schema, encoding="utf-8")
        log.info(f"M-Schema saved - {xiyan_file}")
        
        # семантическое описание (через teacher-LLM)
        semantic_schema, stats = generate_semantic_m_schema(db_data, db_name, client, model)
        semantic_file = out_path / f"{db_name}_semantic.txt"
        semantic_file.write_text(semantic_schema, encoding="utf-8")
        log.info(f"Semantic M-Schema saved - {semantic_file}")
        
        all_stats.append(stats)
    
    log.info("\n" + "=" * 60 + "\nSUMMARY")
    for s in all_stats:
        log.info(f"  {s['db_name']}: {s['tables']}t, "
                 f"{s['prompt_tokens']+s['completion_tokens']}tok, "
                )
        


if __name__ == "__main__":
    main()