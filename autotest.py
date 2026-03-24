import pandas as pd
import requests
import json
from config import settings
import os

API_URL = "http://127.0.0.1:8080/api/response"
HEADERS = {"AI_BACKEND_KEY": settings.BACKEND_KEY}
DATASET_DIR = "autotest_data/datasets/"

def run_tests(test_file):
    df = pd.read_csv(test_file)
    results = []
    for i, row in df.iterrows():
        message = row["message"]
        dataset_name = row["dataset_name"]

        payload ={
            "message": message,
            "context": []
        }

        files = None
        if isinstance(dataset_name, str) and dataset_name.strip():
            file_path = os.path.join(DATASET_DIR, dataset_name)

            if os.path.exists(file_path):
                files = {
                    "file": open(file_path, "rb")
                }
            else:
                print(f"Dataset not found: {file_path}")

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

        results.append({
            "request": message,
            "dataset": dataset_name,
            "response": json.dumps(result, ensure_ascii=False)
        })
        print(f"=====Обработан запрос {i+1}=====")

    result_df = pd.DataFrame(results)
    result_df.to_csv("autotest_result.csv", index=False)

    print("Автотест завершён, результаты сохранены в autotest_result.csv")

if __name__ == "__main__":
    run_tests("autotest_data/test_cases.csv")