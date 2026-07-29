import argparse
import subprocess
import sys
from pathlib import Path

# Определяем корень проекта, чтобы пути работали из любой директории
project_root = Path(__file__).resolve().parent.parent

def run_step(command: list, description: str) -> None:
    """Запускает команду и проверяет код возврата."""
    print(f"\n{description}...")
    result = subprocess.run(command, cwd=project_root)
    if result.returncode != 0:
        print(f"\nОШИБКА: Не удалось выполнить шаг '{description}'")
        print(f"Код возврата: {result.returncode}")
        sys.exit(result.returncode)
    print(f"Успешно: {description}")


def main():
    parser = argparse.ArgumentParser(
        description="Неделя 2: Полный pipeline Exploration (БД -> JSON -> M-Schema) одной командой."
    )
    parser.add_argument("--db", type=str, required=True, help="Имя целевой БД (например, financial)")
    args = parser.parse_args()

    db_name = args.db
    json_path = project_root / "artifacts" / "db_knowledge" / f"{db_name}_knowledge.json"
    output_dir = project_root / "artifacts" / "m_schemas"

    print(f"Запуск pipeline для БД: '{db_name}'")

    # ---- Explore DB ----
    run_step(
        [sys.executable, "scripts/explore_database.py", "--db", db_name],
        f"Шаг 1/2: Подключение к БД и генерация {db_name}_knowledge.json"
    )
    if not json_path.exists():
        print(f"\nОшибка: Файл {json_path} не был создан на шаге 1.")
        sys.exit(1)

    # ---- Generate M_schema ----
    run_step(
        [sys.executable, "scripts/generate_m_schema.py", "-i", str(json_path), "-o", str(output_dir)],
        f"Шаг 2/2: Генерация M-Schema (XiYan + Semantic) через Teacher-LLM"
    )

    print("\n" + "=" * 60)
    print(f"'{db_name}' успешно обработана!")
    print(f"Артефакты сохранены в:")
    print(f"   - {json_path}")
    print(f"   - {output_dir / f'{db_name}_m_schema.txt'}")
    print(f"   - {output_dir / f'{db_name}_semantic.txt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()