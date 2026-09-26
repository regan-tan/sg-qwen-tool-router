import pytest
from pydantic import ValidationError

from sg_qwen_tool_router.schemas import ToolCall


def test_valid_weather_call():
    call = ToolCall(
        tool="get_weather",
        arguments={
            "location": "Tampines, Singapore",
        },
    )

    assert call.tool == "get_weather"
    assert call.arguments["location"] == "Tampines, Singapore"


def test_valid_currency_call():
    call = ToolCall(
        tool="convert_currency",
        arguments={
            "amount": 100,
            "from_currency": "SGD",
            "to_currency": "MYR",
        },
    )

    assert call.tool == "convert_currency"


def test_valid_no_tool():
    call = ToolCall(
        tool="no_tool",
        arguments={},
    )

    assert call.arguments == {}


def test_missing_required_argument_fails():
    with pytest.raises(ValidationError):
        ToolCall(
            tool="convert_currency",
            arguments={
                "amount": 100,
                "from_currency": "SGD",
            },
        )


def test_unknown_tool_fails():
    with pytest.raises(ValidationError):
        ToolCall(
            tool="book_grab",
            arguments={},
        )


def test_extra_argument_fails():
    with pytest.raises(ValidationError):
        ToolCall(
            tool="get_weather",
            arguments={
                "location": "Orchard",
                "random_field": "hello",
            },
        )