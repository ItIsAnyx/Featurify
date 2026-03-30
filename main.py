from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from openai import OpenAI
from config import settings, validate_key
from pydantic import BaseModel
from typing import List, Dict
from dataset import read_dataset, info_to_str
import json

class BaseRequest(BaseModel):
    message: str
    context: List[Dict[str, str]] = []

class JSONOutput(BaseModel):
    analysis: str
    remove_features: list[str]
    transform_features: list[str]
    create_features: list[str]
    recommended_models: list[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class LoadFile(BaseModel):
    filename: str
    df_info: dict

app = FastAPI(title=settings.APP_NAME)
client = OpenAI(api_key=settings.AI_API_KEY, base_url="https://api.deepseek.com")

# Основная функция для вызова ответа модели
async def call_llm(context: list) -> JSONOutput:
    try:
        out = client.chat.completions.create(
            model="deepseek-chat",
            messages=context,
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        content = out.choices[0].message.content
        content = content.replace("```json", "").replace("```", "")
        parsed = json.loads(content)
        usage = getattr(out, "usage", None)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)

        parsed.update({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        })

        return JSONOutput(**parsed)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Получение ответа на вопрос пользователя с контекстом
@app.post("/api/response", response_model=JSONOutput)
async def get_response(payload: str = Form(...), file: UploadFile = File(None), backend_key: str = Header(..., alias="AI_BACKEND_KEY")) -> JSONOutput:
    validate_key(backend_key)
    payload_dict = json.loads(payload)
    payload = BaseRequest(**payload_dict)

    system_prompt = """You are an ML expert who comes up with feature engineering.
    You may receive Dataset Description (JSON) and User Request. Use BOTH for your recommendations.
    Return ONLY valid JSON in this format:
    {
      "analysis": "string",
      "remove_features": ["string"],
      "transform_features": ["string"],
      "create_features": ["string"],
      "recommended_models": ["string"]
    }
    In the analysis field answer in the user's language."""

    if payload.context:
        context = [{"role": "system", "content": system_prompt}] + payload.context + [{"role": "user", "content": payload.message}]
    else:
        context = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": payload.message}]

    if file:
        try:
            df = read_dataset(file.file)
            info = info_to_str(df)
            file_info = {
                "filename": file.filename,
                "df_info": info
            }
            context.insert(1, {"role": "system", "content": f"Dataset info:\n{json.dumps(file_info)}"})
            print(context)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        result = await call_llm(context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")