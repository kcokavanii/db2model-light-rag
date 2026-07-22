import psycopg
import argparse
from pathlib import Path
from dotenv import load_dotenv
import json
from typing import Dict, List, Any
import os

# Загружаем переменные из .env
load_dotenv(override=True)

db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
db_host = os.getenv("DB_HOST", "localhost")
db_port = int(os.getenv("DB_PORT", "5444"))

# Определяем путь относительно текущего скрипта
script_dir = Path(__file__).parent 
project_root = script_dir.parent   
db_knowledge_dir = project_root / 'db_knowledge'
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
        """Получить первичные ключи таблицы"""
        query = """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = 'public'
          AND tc.table_name = %s;
        """
        self.cursor.execute(query, (table_name,))
        return [row[0] for row in self.cursor.fetchall()]
    
    
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """Получить внешние ключи таблицы"""
        query = """
        SELECT
            tc.constraint_name,
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_name = %s;
        """
        self.cursor.execute(query, (table_name,))
        return [
            {
                'constraint_name': row[0],
                'column_name': row[1],
                'foreign_table': row[2],
                'foreign_column': row[3]
            }
            for row in self.cursor.fetchall()
        ]

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
        
        most_common_3 = []
        if row[3]: 
            vals = row[3].strip('{}').split(',')
            most_common_3 = [v for v in vals[:3] if v]
        
        return {
            'n_distinct': row[0],
            'null_frac': row[1],
            'avg_width': row[2],
            'most_common_vals': most_common_3,
            'most_common_freqs': row[4],
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
            
            columns = self.get_columns(table_name)
            foreign_keys = self.get_foreign_keys(table_name)
            sample_rows = self.get_sample_rows(table_name, limit=5)
            
            # Статистики для каждой колонки            
            column_stats = {}
            for col in columns:
                stats = self.get_statistics(table_name, col['column_name'])
                column_stats[col['column_name']] = stats
            
            result['tables'].append({
                'table_name': table_name,
                'table_type': table_info['table_type'],
                'columns': columns,
                'foreign_keys': foreign_keys,
                'sample_rows': sample_rows,
                'column_statistics': column_stats
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



def load_db_list_from_file(config_path: Path) -> List[str]:
    """Загрузить список БД из текстового файла (по одной в строке)"""
    if not config_path.exists():
        return []
    with open(config_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
def main():
    parser = argparse.ArgumentParser(
        description="Извлечение знаний о PostgreSQL БД в JSON-формате"
    )
    
    # Группа 1: одна конкретная БД
    parser.add_argument(
        '--db', type=str,
        help="Имя одной БД для обработки (например: financial)"
    )
    
    # Группа 2: список БД из файла
    parser.add_argument(
        '--config', type=Path,
        default=project_root / 'databases.txt',
        help="Путь к файлу со списком БД (по одной в строке). По умолчанию: databases.txt"
    )
    
    # Группа 3: все БД из BIRD (предустановленный список)
    parser.add_argument(
        '--all-bird', action='store_true',
        help="Обработать все стандартные БД из BIRD dev"
    )
    
    # Опции вывода
    parser.add_argument(
        '--output-dir', type=Path, default=db_knowledge_dir,
        help="Папка для сохранения JSON-файлов"
    )
    
    args = parser.parse_args()

    # Определяем, какие БД обрабатывать
    databases: List[str] = []
    
    if args.db:
        databases = [args.db]
    elif args.all_bird:
        databases = [
           'toxicology', 'financial',
            'codebase_community'
        ]
    else:
        # По умолчанию — читаем из конфига
        databases = load_db_list_from_file(args.config)
        if not databases:
            print(f"Файл {args.config} не найден или пуст.")
            print("Используй один из вариантов:")
            print("  python extract_schema.py --db financial")
            print("  python extract_schema.py --all-bird")
            print("  python extract_schema.py --config my_databases.txt")
            return
    
    print(f"Будет обработано БД: {len(databases)}")
    print(f"Выходная папка: {args.output_dir}")
    
    for db_name in databases:
        process_single_database(db_name, args.output_dir)
    
    print(f"\nГотово! Все файлы в: {args.output_dir}")
    
if __name__ == '__main__':    
    main()