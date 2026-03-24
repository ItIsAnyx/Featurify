import pandas as pd
import json

df = pd.read_csv("autotest_result.csv")

for i, row in df.iterrows():
    print("=" * 80)
    print(f"Тест #{i+1}")
    print(f"Запрос: {row['request']}")
    print(f"Датасет: {row.get('dataset', '—')}")

    try:
        response = json.loads(row["response"])

        print("\nAnalysis:")
        print(response.get("analysis", ""))

        print("\nRemove features:", response.get("remove_features", []))
        print("Transform features:", response.get("transform_features", []))
        print("Create features:", response.get("create_features", []))
        print("Models:", response.get("recommended_models", []))

    except Exception:
        print("\nОшибка парсинга ответа")
        print(row["response"])

    print("\n")