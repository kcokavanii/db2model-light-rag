import psycopg
import argparse
from pathlib import Path
from dotenv import load_dotenv
import json
from typing import Dict, List, Any
import os


load_dotenv(override=True)

db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
db_host = os.getenv("DB_HOST", "localhost")
db_port = int(os.getenv("DB_PORT", "5444"))

# Определяем путь относительно текущего скрипта
script_dir = Path(__file__).parent 
project_root = script_dir.parent   
db_knowledge_dir = project_root / 'artifacts' / 'db_knowledge'
db_knowledge_dir.mkdir(exist_ok=True)

class DatabaseExplorer:
    def __init__(self, host: str, port: int, dbname: str, user: str, password: str):
        """Подключение к БД через туннель"""
        self.conn = psycopg.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        self.cursor = self.conn.cursor()
    
    def get_tables(self) -> List[Dict[str, Any]]:
        """Получить список таблиц с метаданными"""
        query = """
        SELECT 
            table_name,
            table_type
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
        self.cursor.execute(query)
        return [
            {'table_name': row[0], 'table_type': row[1]}
            for row in self.cursor.fetchall()
        ]
    
    def get_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Получить колонки таблицы"""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        ORDER BY ordinal_position;
        """
        self.cursor.execute(query, (table_name,))
        return [
            {
                'column_name': row[0],
                'data_type': row[1],
                'is_nullable': row[2] == 'YES',
                'default': row[3],
                'max_length': row[4]
            }
            for row in self.cursor.fetchall()
        ]
    
    def get_primary_keys(self, table_name: str) -> List[str]:
        """Получить первичные ключи через pg_catalog (более надёжно)"""
        query = """
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = %s::regclass
          AND i.indisprimary
        ORDER BY a.attnum;
        """
        try:
            self.cursor.execute(query, (f'public.{table_name}',))
            result = [row[0] for row in self.cursor.fetchall()]
            return result
        except Exception as e:
            print(f"  Ошибка при получении PK для {table_name}: {e}")
            return []
    
    
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """Получить внешние ключи через pg_catalog (более надёжно)"""
        query = """
        SELECT
            a.attname AS column_name,
            c2.relname AS foreign_table,
            a2.attname AS foreign_column
        FROM pg_constraint con
        JOIN pg_class c ON con.conrelid = c.oid
        JOIN pg_class c2 ON con.confrelid = c2.oid
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(con.conkey)
        JOIN pg_attribute a2 ON a2.attrelid = c2.oid AND a2.attnum = ANY(con.confkey)
        WHERE c.relname = %s
          AND con.contype = 'f'
        ORDER BY con.conname, a.attnum;
        """
        try:
            self.cursor.execute(query, (table_name,))
            result = [
                {
                    'column_name': row[0],
                    'foreign_table': row[1],
                    'foreign_column': row[2]
                }
                for row in self.cursor.fetchall()
            ]
            return result
        except Exception as e:
            print(f"  Ошибка при получении FK для {table_name}: {e}")
            return []

    def get_table_comment(self, table_name: str) -> str:
        """Получить комментарий к таблице (COMMENT ON TABLE ...)"""
        query = """
        SELECT obj_description(
            (quote_ident('public') || '.' || quote_ident(%s))::regclass,
            'pg_class'
        );
        """
        try:
            self.cursor.execute(query, (table_name,))
            row = self.cursor.fetchone()
            return row[0] if row and row[0] else ''
        except Exception:
            self.conn.rollback()
            return ''
    
    def get_statistics(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Получить статистики из pg_stats"""
        query = """
        SELECT
            n_distinct,
            null_frac,
            avg_width,
            most_common_vals,
            most_common_freqs,
            histogram_bounds
        FROM pg_stats
        WHERE schemaname = 'public'
          AND tablename = %s
          AND attname = %s;
        """
        self.cursor.execute(query, (table_name, column_name))
        row = self.cursor.fetchone()
        if not row:
            return {}

        n_distinct = row[0]
        most_common_vals = []
        if row[3]:
            vals = row[3].strip('{}').split(',')
            if n_distinct < 0:
                most_common_vals = vals[:10]
            elif n_distinct <= 20:
                most_common_vals = vals
            elif n_distinct <= 200:
                most_common_vals = vals[:20]
            else:
                most_common_vals = vals[:10]

        most_common_freqs = []
        if row[4] and most_common_vals:
            most_common_freqs = row[4][:len(most_common_vals)]
        
        return {
            'n_distinct': n_distinct,
            'null_frac': row[1],
            'avg_width': row[2],
            'most_common_vals': most_common_vals,
            'most_common_freqs': most_common_freqs,
            'histogram_bounds': row[5]
        }
    
    def get_sample_rows(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Получить пример строк из таблицы"""
        query = f'SELECT * FROM "{table_name}" LIMIT %s;'
        self.cursor.execute(query, (limit,))
        columns = [desc[0] for desc in self.cursor.description]
        return [
            dict(zip(columns, row))
            for row in self.cursor.fetchall()
        ]
    
    def explore_database(self) -> Dict[str, Any]:
        """Полное исследование БД"""
        result = {
            'tables': []
        }
        
        tables = self.get_tables()
        for table_info in tables:
            table_name = table_info['table_name']
            print(f"Exploring table: {table_name}")

            primary_keys = self.get_primary_keys(table_name)
            columns = self.get_columns(table_name)
            foreign_keys = self.get_foreign_keys(table_name)
            sample_rows = self.get_sample_rows(table_name, limit=5)
            comment = self.get_table_comment(table_name)
                       
            column_stats = {}
            for col in columns:
                stats = self.get_statistics(table_name, col['column_name'])
                column_stats[col['column_name']] = stats
            
            result['tables'].append({
                'table_name': table_name,
                'table_type': table_info['table_type'],
                'primary_keys': primary_keys,
                'columns': columns,
                'foreign_keys': foreign_keys,
                'sample_rows': sample_rows,
                'column_statistics': column_stats, 
                'table_comment': comment
            })
            
        
        return result
    
    def close(self):
        self.cursor.close()
        self.conn.close()

def process_single_database(db_name: str, output_dir: Path) -> None:
    """Обработать одну БД и сохранить результат"""
    print(f"\nExploring database: {db_name}")
    try:
        explorer = DatabaseExplorer(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_pass
        )

        knowledge = explorer.explore_database()

        output_path = output_dir / f'{db_name}_knowledge.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved in {output_path}")
        explorer.close()
    except psycopg.OperationalError as e:
        print(f"Ошибка подключения к {db_name}: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка при исследовании {db_name}: {e}")


    
def main():
    parser = argparse.ArgumentParser(description="Извлечение знаний о PostgreSQL БД в JSON-формате")    
    parser.add_argument(
        '--db', type=str,
        help="Имя одной БД для обработки (например: financial)"
    )    
    parser.add_argument(
        '--output-dir', type=Path, default=db_knowledge_dir,
        help="Папка для сохранения JSON-файлов"
    )    
    args = parser.parse_args()
    
    process_single_database(args.db, args.output_dir)    
    print(f"\nГотово! Все файлы в: {args.output_dir}")
    
if __name__ == '__main__':    
    main()