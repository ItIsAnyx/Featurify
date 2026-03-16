import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from openai import OpenAI
from config import settings, validate_key
from pydantic import BaseModel

class BaseAnswer(BaseModel):
    message: str
    context: list

app = FastAPI(title=settings.APP_NAME)
client = OpenAI(api_key=settings.AI_API_KEY, base_url="https://api.deepseek.com")

def call_llm(context: list):
    try:
        out = client.chat.completions.create(model="deepseek-chat", messages=context)
        return out.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/response", response_model=BaseAnswer)
def get_response(payload: BaseAnswer, backend_key: str = Header(..., alias="AI_BACKEND_KEY")):
    validate_key(backend_key)
    if not(len(payload.context)):
        context = [
            {"role": "system", "content": "You are an ML expert who comes up with feature engineering"}
        ] + payload.context + [{"role": "user", "content": payload.message}]
    else:
        context = [
            {"role": "system", "content": "You are an ML expert who comes up with feature engineering"},
            {"role": "user", "content": payload.message}
        ]

    try:
        final_context = context + [{"role": "assistant", "content": call_llm(context)}]
        return {
            "message": final_context[-1]["content"],
            "context": final_context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")