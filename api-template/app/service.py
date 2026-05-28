"""
Базовый класс сервиса и демо-реализация.

Этот файл — ЕДИНСТВЕННОЕ место, которое нужно менять студенту.
Всё остальное (FastAPI, схемы, Docker) работает автоматически.

КАК ПОЛЬЗОВАТЬСЯ:
1. Прочитайте комментарии к ServiceBase — там описаны все методы.
2. Посмотрите на DemoService как на рабочий пример.
3. Создайте свой класс-наследник ServiceBase.
4. Обновите функцию get_service() в конце файла.
"""

from __future__ import annotations

import base64
import io
import os
import json
import requests
from abc import ABC, abstractmethod
from typing import Any

from app.schemas import (
    ContentPartImage,
    ContentPartText,
    InfoResponse,
    InputType,
    RunRequest,
    RunResponse,
    Schema,
)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  БАЗОВЫЙ КЛАСС — НЕ ИЗМЕНЯЙТЕ ЕГО                                         ║
# ║  Все helper-методы (get_text, get_image) уже реализованы ниже.            ║
# ╚════════════════════════════════════════════════════════════════════════════╝


class ServiceBase(ABC):
    """Абстрактный базовый класс для всех сервисов.

    Каждый студент должен унаследовать этот класс и реализовать два метода:
      - get_info() — возвращает тип входных данных сервиса.
      - run()      — основная логика обработки запроса.

    Также доступны вспомогательные методы (helpers):
      - get_text(request)  — извлекает текст из content.
      - get_image(request) — извлекает base64 картинку из content.
    """

    @abstractmethod
    def get_info(self) -> InfoResponse:
        """Вернуть метаданные сервиса.

        input_type определяет, какой контент будет отправлять нагрузочный тест:
          - InputType.TEXT           → content — строка с текстом
          - InputType.IMAGE          → content — список с одной картинкой
          - InputType.TEXT_AND_IMAGE → content — список с текстом и картинкой

        input_schema — JSON Schema для параметров extra_body.
        output_schema — JSON Schema для поля result в ответе.
        """
        ...

    @abstractmethod
    def run(self, request: RunRequest) -> RunResponse:
        """Выполнить основную логику сервиса.

        Аргумент request содержит:
          - request.content    : str | list[ContentPart]
              Строка = обычный текст.
              Список = типизированные части (текст + картинка).
          - request.extra_body : dict
              Дополнительные параметры (temperature, max_tokens, ...).

        Верните RunResponse(status="success", result={...}) или
        RunResponse(status="error", error="описание ошибки").
        """
        ...

    # ------------------------------------------------------------------
    # Helper methods — вспомогательные методы для извлечения данных
    # ------------------------------------------------------------------

    def get_text(self, request: RunRequest) -> str | None:
        """Извлечь текст из content.

        Если content — строка, возвращает её как есть.
        Если content — список, ищет первую часть с type="text".
        Если текст не найден, возвращает None.

        Пример использования:
            text = self.get_text(request)
            if text is None:
                return RunResponse(status="error", error="Текст не передан")

        Args:
            request: объект RunRequest с полем content.

        Returns:
            Строка с текстом или None.
        """
        if isinstance(request.content, str):
            return request.content

        for part in request.content:
            if isinstance(part, ContentPartText):
                return part.text

        return None

    def get_image(self, request: RunRequest) -> str | None:
        """Извлечь base64-кодированную картинку из content.

        Ищет первую часть с type="image" в списке content.
        Если content — строка (текст), возвращает None.

        Пример использования:
            image_b64 = self.get_image(request)
            if image_b64 is None:
                return RunResponse(status="error", error="Картинка не передана")
            image_bytes = base64.b64decode(image_b64)

        Args:
            request: объект RunRequest с полем content.

        Returns:
            Строка с base64-кодированной картинкой или None.
        """
        if isinstance(request.content, str):
            return None

        for part in request.content:
            if isinstance(part, ContentPartImage):
                return part.image

        return None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  СТУДЕНТУ: Ниже находится демо-сервис. Замените его на свою реализацию.   ║
# ║  1. Напишите свой класс-наследник ServiceBase.                            ║
# ║  2. Обновите функцию get_service() в конце файла.                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class FeaturifyService(ServiceBase):

    def get_info(self) -> InfoResponse:
        return InfoResponse(
            input_type=InputType.TEXT,
            input_schema=Schema.of(
                context=Schema.array(
                    "Conversation context"
                )
            ),
            output_schema=Schema.of(
                analysis=Schema.string("Feature engineering analysis"),
                remove_features=Schema.array(
                    "Features recommended for removal"
                ),
                transform_features=Schema.array(
                    "Features recommended for transformation"
                ),
                create_features=Schema.array(
                    "Features recommended for creation"
                ),
                recommended_models=Schema.array(
                    "Recommended ML models"
                ),
                prompt_tokens=Schema.integer("Prompt tokens used"),
                completion_tokens=Schema.integer("Completion tokens used"),
                total_tokens=Schema.integer("Total tokens used"),
                context=Schema.array("Updated conversation context")
            ),
        )

    def run(self, request: RunRequest) -> RunResponse:
        try:
            text = self.get_text(request)

            if text is None:
                return RunResponse(
                    status="error",
                    error="Text input is required"
                )

            payload = {
                "message": text,
                "context": request.extra_body.get("context", [])
            }

            response = requests.post(
                "http://backend:8000/api/response",
                headers={
                    "AI_BACKEND_KEY": os.getenv("BACKEND_KEY")
                },
                data={
                    "payload": json.dumps(payload)
                },
                timeout=120
            )

            if response.status_code != 200:
                return RunResponse(
                    status="error",
                    error=f"Backend error: {response.text}"
                )

            return RunResponse(
                status="success",
                result=response.json()
            )

        except Exception as exc:
            return RunResponse(
                status="error",
                error=str(exc)
            )


_service_instance: ServiceBase | None = None


def get_service() -> ServiceBase:
    global _service_instance

    if _service_instance is None:
        _service_instance = FeaturifyService()

    return _service_instance

