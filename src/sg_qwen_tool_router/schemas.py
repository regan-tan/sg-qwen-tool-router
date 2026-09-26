from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ToolName = Literal[
    "get_weather",
    "convert_currency",
    "convert_units",
    "calculate",
    "get_time",
    "no_tool",
]


class ToolCall(BaseModel):
    tool: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_arguments(self):
        required_arguments = {
            "get_weather": {"location"},
            "convert_currency": {
                "amount",
                "from_currency",
                "to_currency",
            },
            "convert_units": {
                "value",
                "from_unit",
                "to_unit",
            },
            "calculate": {"expression"},
            "get_time": {"location"},
            "no_tool": set(),
        }

        expected = required_arguments[self.tool]
        actual = set(self.arguments.keys())

        if actual != expected:
            raise ValueError(
                f"{self.tool} expects arguments {expected}, "
                f"but received {actual}"
            )

        return self