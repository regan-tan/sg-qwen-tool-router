TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The location to get weather for.",
            }
        },
        "required": ["location"],
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "amount": {
                "type": "number",
                "description": "The monetary amount to convert.",
            },
            "from_currency": {
                "type": "string",
                "description": "The source currency code, for example SGD.",
            },
            "to_currency": {
                "type": "string",
                "description": "The target currency code, for example MYR.",
            },
        },
        "required": [
            "amount",
            "from_currency",
            "to_currency",
        ],
    },
    {
        "name": "convert_units",
        "description": "Convert a value from one measurement unit to another.",
        "parameters": {
            "value": {
                "type": "number",
                "description": "The value to convert.",
            },
            "from_unit": {
                "type": "string",
                "description": "The original measurement unit.",
            },
            "to_unit": {
                "type": "string",
                "description": "The target measurement unit.",
            },
        },
        "required": [
            "value",
            "from_unit",
            "to_unit",
        ],
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate.",
            }
        },
        "required": ["expression"],
    },
    {
        "name": "get_time",
        "description": "Get the current time in a specified location or timezone.",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The location or timezone to get the time for.",
            }
        },
        "required": ["location"],
    },
]