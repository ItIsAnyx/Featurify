import pandas as pd
import json

df = pd.read_csv("autotest_result.csv")

for i, row in df.iterrows():
    print("=" * 80)
    print(f"Тест #{i+1}")
    print(f"Запрос: {row['request']}")
    print(f"Датасет: {row.get('dataset', '—')}")
    print(f"Время обработки запроса: {row['work_time']} сек.")

    try:
        response = json.loads(row["response"])
        print(f"\nВходные токены: {response.get('prompt_tokens', '')}")
        print(f"Выходные токены: {response.get('completion_tokens', '')}")
        print(f"Всего токенов: {response.get('total_tokens', '')}\n")
        print(f"Ответ модели:\n{json.dumps(response, indent=4, ensure_ascii=False)}")

    except Exception:
        print("\nОшибка парсинга ответа")
        print(row["response"])

    print("\n")

print(f"Общее время обработки моделью всех запросов: {df['work_time'].sum():.4f} сек.")