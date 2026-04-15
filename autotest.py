import pandas as pd
import requests
import json
from config import settings
import os
import time

API_URL = "http://127.0.0.1:8080/api/response"
HEADERS = {"AI_BACKEND_KEY": settings.BACKEND_KEY}
DATASET_DIR = "autotest_data/datasets/"

def count_tool_calls(context):
    if not isinstance(context, list):
        return 0
    return sum(1 for m in context if m.get("role") == "tool")


def check_valid_json(response):
    required_fields = [
        "analysis",
        "remove_features",
        "transform_features",
        "create_features",
        "recommended_models"
    ]

    if not isinstance(response, dict):
        return 0

    return int(all(field in response for field in required_fields))

def run_tests(test_file):
    df = pd.read_csv(test_file)
    results = []
    for i, row in df.iterrows():
        message = row["message"]
        dataset_name = row["dataset_name"]

        payload ={
            "message": message,
            "context": [],
        }

        files = None
        file_obj = None
        if isinstance(dataset_name, str) and dataset_name.strip():
            file_path = os.path.join(DATASET_DIR, dataset_name)

            if os.path.exists(file_path):
                file_obj = open(file_path, "rb")
                files = {
                    "file": file_obj
                }
            else:
                print(f"Dataset not found: {file_path}")

        start_time = time.perf_counter()
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                data={"payload": json.dumps(payload)},
                files=files
            )
            result = response.json()

        except Exception as e:
            result = {"error": str(e)}

        finally:
            if file_obj:
                file_obj.close()

        work_time = time.perf_counter() - start_time
        try:
            parsed = result if isinstance(result, dict) else {}

            context = parsed.get("context", [])
            tool_calls_count = count_tool_calls(context)
            valid_json = check_valid_json(parsed)

        except Exception:
            tool_calls_count = 0
            valid_json = 0

        results.append({
            "request": message,
            "dataset": dataset_name,
            "response": json.dumps(result, ensure_ascii=False),
            "work_time": round(work_time, 4),
            "tool_calls_count": tool_calls_count,
            "valid_json": valid_json
        })
        print(f"=====Обработан запрос {i+1}=====")
        print(f"Время обработки: {work_time:.4f} сек.\n")

    result_df = pd.DataFrame(results)
    result_df.to_csv("autotest_result.csv", index=False)

    print("Автотест завершён, результаты сохранены в autotest_result.csv")

if __name__ == "__main__":
    run_tests("autotest_data/test_cases.csv")