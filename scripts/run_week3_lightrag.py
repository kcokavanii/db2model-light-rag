"""
Неделя 3: Оркестратор pipeline Light-RAG.

Запуск:
    python scripts/run_week3_lightrag.py --db financial
    python scripts/run_week3_lightrag.py --db financial --no-llm
    python scripts/run_week3_lightrag.py --db financial --inspect
"""
import argparse
import subprocess
import sys
from pathlib import Path


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
        description="Неделя 3: Light-RAG pipeline для одной БД."
    )
    parser.add_argument(
        "--db", type=str, required=True,
        help="Имя целевой БД (например, financial)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Не использовать Teacher-LLM при построении графа (экономия токенов)"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Показать примеры узлов графа после построения"
    )
    parser.add_argument(
        "--skip-graph", action="store_true",
        help="Пропустить построение графа (использовать уже готовый из artifacts/graphs/)"
    )
    args = parser.parse_args()

    db_name = args.db
    graph_file = project_root / "artifacts" / "graphs" / f"{db_name}_graph.pkl"

    print(f"\n🚀 ЗАПУСК LIGHT-RAG PIPELINE ДЛЯ БД: '{db_name}'")
    print("=" * 60)

    # ===== ШАГ 1: Построение графа знаний =====
    if not args.skip_graph:
        cmd = [
            sys.executable,
            "scripts/build_db_graph.py",
            "--db", db_name,
        ]
        if args.no_llm:
            cmd.append("--no-llm")
        if args.inspect:
            cmd.append("--inspect")

        run_step(cmd, f"Шаг 1/?: Построение графа знаний для '{db_name}'")

        if not graph_file.exists():
            print(f"\nКритическая ошибка: граф не был создан: {graph_file}")
            sys.exit(1)
    else:
        if not graph_file.exists():
            print(f"\nФлаг --skip-graph указан, но граф не найден: {graph_file}")
            sys.exit(1)
        print(f"\n⏭Пропускаем построение графа (используем {graph_file.name})")

    # ===== ШАГ 2: Загрузка графа в LightRAG (будет добавлена позже) =====
    # TODO: написать scripts/load_to_lightrag.py
    # run_step(
    #     [sys.executable, "scripts/load_to_lightrag.py", "--db", db_name],
    #     f"Шаг 2/?: Загрузка графа в LightRAG для '{db_name}'"
    # )

    # ===== ШАГ 3: Извлечение подграфа на тестовых запросах (будет добавлена позже) =====
    # TODO: написать scripts/query_subgraph.py
    # run_step(
    #     [sys.executable, "scripts/query_subgraph.py", "--db", db_name],
    #     f"Шаг 3/?: Извлечение релевантных подграфов для '{db_name}'"
    # )

    # ===== ШАГ 4: Eval на target-модели (будет добавлена позже) =====
    # TODO: интеграция с eval_harness.py
    # run_step(
    #     [sys.executable, "scripts/eval_harness.py", "--db", db_name, "--mode", "lightrag"],
    #     f"Шаг 4/?: Замер EX/VES/токенов на target-модели"
    # )

    print("\n" + "=" * 60)
    print(f"🎉 PIPELINE ДЛЯ '{db_name}' УСПЕШНО ЗАВЕРШЁН!")
    print(f"📁 Артефакты:")
    print(f"   - {graph_file}")
    print(f"   - {graph_file.with_suffix('.graphml')}")
    print(f"   - {graph_file.with_name(graph_file.stem + '_stats.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()