import pandas as pd
import json

df = pd.read_csv("autotest_result.csv")
total_tokens = 0

for i, row in df.iterrows():
    print("=" * 80)
    print(f"Тест #{i+1}")
    print(f"Запрос: {row['request']}")
    print(f"Датасет: {row.get('dataset', '—')}")
    print(f"Время обработки запроса: {row['work_time']} сек.")
    print(f"Tool calls: {row.get('tool_calls_count', '')}")
    print(f"Valid JSON: {row.get('valid_json', '')}")

    try:
        response = json.loads(row["response"])
        print(f"\nВходные токены: {response.get('prompt_tokens', '')}")
        print(f"Выходные токены: {response.get('completion_tokens', '')}")
        total_request_tokens = response.get('total_tokens', '')
        total_tokens += total_request_tokens
        print(f"Всего токенов: {total_request_tokens}")
        print(f"Ответ модели:\n{json.dumps(response, indent=4, ensure_ascii=False)}")


    except Exception:
        print("\nОшибка парсинга ответа")
        print(row["response"])

    print("\n")

print(f"Общее время обработки моделью всех запросов: {df['work_time'].sum():.4f} сек.")
print("\n" + "="*80)
print("СРЕДНИЕ МЕТРИКИ:\n")

if "tool_calls_count" in df.columns:
    print(f"Количество вызовов инструментов: {df['tool_calls_count'].mean():.3f}")

if "valid_json" in df.columns:
    print(f"Валидный JSON: {df['valid_json'].mean():.3f}")

if "work_time" in df.columns:
    print(f"Среднее время работы: {df['work_time'].mean():.3f}")

print(f"Средняя стоимость в токенах: {(total_tokens / df.shape[0]):.3f}")