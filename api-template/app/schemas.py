"""
API schemas — unified request/response models.

The /run endpoint uses an OpenAI-style content format:
  - content: str | list[ContentPart]
    - str = plain text
    - list = typed content parts (text + image)
  - extra_body: dict = optional service parameters
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field

_UNSET = object()


class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TEXT_AND_IMAGE = "text_and_image"


class ContentPartText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentPartImage(BaseModel):
    type: Literal["image"] = "image"
    image: str = Field(description="Base64-encoded image data (PNG, JPEG, etc.)")


class Schema:
    """Builder for JSON Schema dicts used in input_schema / output_schema.

    Example::

        input_schema = Schema.of(
            temperature=Schema.number("Temperature", default=0.7),
            max_tokens=Schema.integer("Max tokens", default=512),
        )

    Produces::

        {
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "description": "Temperature", "default": 0.7},
                "max_tokens":  {"type": "integer", "description": "Max tokens", "default": 512},
            },
        }
    """

    @staticmethod
    def of(**fields: dict[str, Any]) -> dict[str, Any]:
        """Build an object schema from named fields.

        Each keyword argument is ``name=Schema.<type>(...)``.
        """
        return {
            "type": "object",
            "properties": {name: field for name, field in fields.items()},
        }

    @staticmethod
    def string(
        description: str = "",
        default: Any = _UNSET,
        enum: list[str] | None = None,
    ) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "string"}
        if description:
            d["description"] = description
        if default is not _UNSET:
            d["default"] = default
        if enum is not None:
            d["enum"] = enum
        return d

    @staticmethod
    def number(
        description: str = "",
        default: Any = _UNSET,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "number"}
        if description:
            d["description"] = description
        if default is not _UNSET:
            d["default"] = default
        if minimum is not None:
            d["minimum"] = minimum
        if maximum is not None:
            d["maximum"] = maximum
        return d

    @staticmethod
    def integer(
        description: str = "",
        default: Any = _UNSET,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "integer"}
        if description:
            d["description"] = description
        if default is not _UNSET:
            d["default"] = default
        if minimum is not None:
            d["minimum"] = minimum
        if maximum is not None:
            d["maximum"] = maximum
        return d

    @staticmethod
    def boolean(
        description: str = "",
        default: Any = _UNSET,
    ) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "boolean"}
        if description:
            d["description"] = description
        if default is not _UNSET:
            d["default"] = default
        return d

    @staticmethod
    def array(
        items: dict[str, Any],
        description: str = "",
    ) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "array", "items": items}
        if description:
            d["description"] = description
        return d

    @staticmethod
    def object(
        description: str = "",
        **fields: dict[str, Any],
    ) -> dict[str, Any]:
        d = Schema.of(**fields)
        if description:
            d["description"] = description
        return d


ContentPart = Annotated[
    Union[ContentPartText, ContentPartImage],
    Field(discriminator="type"),
]


class RunRequest(BaseModel):
    """Unified request for /run endpoint.

    content:
      - str — plain text input. Example: "Hello world"
      - list[ContentPart] — typed parts. Example:
          [
            {"type": "text", "text": "Describe this image"},
            {"type": "image", "image": "<base64>"}
          ]

    extra_body:
      - Optional dict with service-specific parameters.
      - Example: {"temperature": 0.7, "max_tokens": 512}
    """

    content: str | list[ContentPart] = Field(
        description=(
            "Main input: plain text string or list of typed content parts. "
            "Use a string for text-only input. "
            "Use a list for image or multimodal input."
        ),
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameters (temperature, max_tokens, etc.)",
    )


class RunResponse(BaseModel):
    """Unified response from /run endpoint."""

    status: Literal["success", "error"]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class InfoResponse(BaseModel):
    """Service metadata returned by /info endpoint.

    input_type tells the load tester what content to generate:
      - "text"            -> send content as a string
      - "image"           -> send content as [{"type": "image", "image": "<base64>"}]
      - "text_and_image"  -> send content as [{"type": "text", ...}, {"type": "image", ...}]

    input_schema: JSON Schema describing extra_body parameters.
    output_schema: JSON Schema describing the result dict.
    """

    input_type: InputType
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
