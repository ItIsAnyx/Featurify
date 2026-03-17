import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from openai import OpenAI
from config import settings, validate_key
from pydantic import BaseModel
from typing import List, Dict
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
        return JSONOutput(**parsed)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Простое получение ответа на вопрос пользователя с контекстом
@app.post("/api/response", response_model=JSONOutput)
async def get_response(payload: BaseRequest, backend_key: str = Header(..., alias="AI_BACKEND_KEY")) -> JSONOutput:
    validate_key(backend_key)
    system_prompt = """You are an ML expert who comes up with feature engineering. Return ONLY valid JSON in this format:
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

    try:
        result = await call_llm(context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")