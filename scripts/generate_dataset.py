import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from sg_qwen_tool_router.prompts import build_system_prompt
from sg_qwen_tool_router.schemas import ToolCall

SEED = 42
EXAMPLES_PER_TOOL = 600
OUTPUT_PATH = Path("data/raw/examples.jsonl")

random.seed(SEED)

SYSTEM_PROMPT = build_system_prompt()


# ---------------------------------------------------------------------------
# Dataset balance
# ---------------------------------------------------------------------------

# 600 examples per tool:
# 30% standard English
# 35% natural Singapore English
# 25% colloquial Singlish
# 10% noisy / informal text
STYLE_TARGETS = {
    "standard": 180,
    "singapore_english": 210,
    "singlish": 150,
    "noisy": 60,
}


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

LOCATIONS = [
    "Tampines",
    "Orchard",
    "Jurong East",
    "Woodlands",
    "Bugis",
    "Punggol",
    "Bishan",
    "Toa Payoh",
    "Changi",
    "Marina Bay",
    "Serangoon",
    "Ang Mo Kio",
    "Clementi",
    "Pasir Ris",
    "Yishun",
    "Sengkang",
    "Hougang",
    "Novena",
    "HarbourFront",
    "Sentosa",
    "Raffles Place",
    "Dhoby Ghaut",
    "Bedok",
    "Katong",
    "Bukit Timah",
    "Queenstown",
    "Bukit Batok",
    "Choa Chu Kang",
    "Kallang",
    "Tiong Bahru",
]


CURRENCY_PAIRS = [
    ("SGD", "MYR"),
    ("SGD", "USD"),
    ("SGD", "JPY"),
    ("SGD", "EUR"),
    ("SGD", "GBP"),
    ("SGD", "AUD"),
    ("SGD", "KRW"),
    ("SGD", "THB"),
    ("SGD", "CNY"),
    ("USD", "SGD"),
    ("MYR", "SGD"),
    ("JPY", "SGD"),
    ("AUD", "SGD"),
    ("GBP", "SGD"),
]


CURRENCY_ALIASES = {
    "SGD": [
        "SGD",
        "Singapore dollars",
        "Sing dollars",
    ],
    "MYR": [
        "MYR",
        "ringgit",
        "Malaysian ringgit",
    ],
    "USD": [
        "USD",
        "US dollars",
        "American dollars",
    ],
    "JPY": [
        "JPY",
        "yen",
        "Japanese yen",
    ],
    "EUR": [
        "EUR",
        "euros",
    ],
    "GBP": [
        "GBP",
        "pounds",
        "British pounds",
    ],
    "AUD": [
        "AUD",
        "Australian dollars",
        "Aussie dollars",
    ],
    "KRW": [
        "KRW",
        "won",
        "Korean won",
    ],
    "THB": [
        "THB",
        "baht",
        "Thai baht",
    ],
    "CNY": [
        "CNY",
        "yuan",
        "RMB",
    ],
}


AMOUNTS = [
    5,
    10,
    20,
    35,
    50,
    75,
    80,
    100,
    150,
    200,
    300,
    500,
    750,
    1000,
]


UNIT_CONVERSIONS = [
    (170, "cm", "feet"),
    (180, "cm", "feet"),
    (160, "cm", "inches"),
    (5, "km", "miles"),
    (10, "km", "miles"),
    (42, "km", "miles"),
    (100, "kg", "lb"),
    (70, "kg", "lb"),
    (50, "kg", "lb"),
    (30, "celsius", "fahrenheit"),
    (25, "celsius", "fahrenheit"),
    (35, "celsius", "fahrenheit"),
    (2, "litres", "millilitres"),
    (1.5, "litres", "millilitres"),
    (500, "millilitres", "litres"),
    (12, "inches", "cm"),
    (24, "inches", "cm"),
    (3, "metres", "feet"),
    (1.8, "metres", "feet"),
    (500, "grams", "ounces"),
    (1000, "grams", "kg"),
    (2, "kg", "grams"),
    (60, "minutes", "hours"),
    (150, "minutes", "hours"),
    (2, "hours", "minutes"),
    (5, "feet", "cm"),
    (6, "feet", "cm"),
    (1, "mile", "km"),
    (10, "miles", "km"),
    (32, "fahrenheit", "celsius"),
]


TIME_LOCATIONS = [
    "Singapore",
    "Kuala Lumpur",
    "Jakarta",
    "Bangkok",
    "Manila",
    "Tokyo",
    "Seoul",
    "Hong Kong",
    "Taipei",
    "Beijing",
    "Shanghai",
    "Sydney",
    "Melbourne",
    "Mumbai",
    "Dubai",
    "London",
    "Paris",
    "Berlin",
    "Rome",
    "Amsterdam",
    "New York",
    "Boston",
    "Washington DC",
    "Toronto",
    "Vancouver",
    "San Francisco",
    "Los Angeles",
    "Chicago",
    "Honolulu",
    "Auckland",
]


CALCULATIONS = [
    ("17 * 43", "17 multiplied by 43"),
    ("250 * 0.09", "the GST amount on $250"),
    ("80 * 0.09", "the GST amount on $80"),
    ("85 * 1.09", "$85 after adding 9% GST"),
    ("120 * 1.09", "$120 after adding 9% GST"),
    ("500 * 1.09", "$500 including 9% GST"),
    ("120 / 4", "120 divided by 4"),
    ("45 + 78", "45 plus 78"),
    ("200 - 37", "200 minus 37"),
    ("15 * 12", "15 times 12"),
    ("900 / 3", "900 divided by 3"),
    ("75 * 4", "75 multiplied by 4"),
    ("250 - 89", "250 minus 89"),
    ("39 + 64", "39 plus 64"),
    ("144 / 12", "144 divided by 12"),
    ("22 * 18", "22 times 18"),
    ("1000 * 0.15", "15 percent of 1000"),
    ("80 * 0.8", "$80 after a 20% discount"),
    ("250 * 0.9", "$250 after a 10% discount"),
    ("120 * 0.75", "$120 after a 25% discount"),
    ("150 / 5", "150 split equally among 5 people"),
    ("240 / 8", "240 split equally among 8 people"),
    ("18 + 27 + 35", "18 plus 27 plus 35"),
    ("500 - 128", "500 minus 128"),
    ("64 * 7", "64 times 7"),
    ("980 / 14", "980 divided by 14"),
    ("75 * 1.09", "$75 including 9% GST"),
    ("320 * 0.09", "the GST amount on $320"),
    ("49 + 83", "49 plus 83"),
    ("420 / 6", "420 divided by 6"),
]


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

WEATHER_TEMPLATES = {
    "standard": [
        "What is the weather in {location}?",
        "Check the current weather in {location}.",
        "How is the weather in {location} right now?",
        "Tell me the current weather for {location}.",
        "What are the current weather conditions in {location}?",
        "Could you check the weather in {location}?",
    ],
    "singapore_english": [
        "Can help me check the weather in {location}?",
        "Can check {location} weather now?",
        "How is the weather at {location} now?",
        "Is {location} raining now?",
        "Can see whether {location} got rain now?",
        "Help me check if {location} is raining.",
        "{location} weather okay now or not?",
    ],
    "singlish": [
        "{location} weather now how?",
        "{location} raining anot?",
        "eh {location} weather how ah",
        "now {location} rain or not",
        "{location} got rain now?",
    ],
    "noisy": [
        "{location} weather rn",
        "{location} rain anot",
    ],
}


CURRENCY_TEMPLATES = {
    "standard": [
        "Convert {amount} {from_alias} to {to_alias}.",
        "How much is {amount} {from_alias} in {to_alias}?",
        "Convert {amount} from {from_alias} to {to_alias}.",
        "What is {amount} {from_alias} worth in {to_alias}?",
        "What would {amount} {from_alias} be in {to_alias}?",
        "Exchange {amount} {from_alias} into {to_alias}.",
    ],
    "singapore_english": [
        "Can help me convert {amount} {from_alias} to {to_alias}?",
        "How much is {amount} {from_alias} in {to_alias} now?",
        "Help me change {amount} {from_alias} to {to_alias}.",
        "{amount} {from_alias} convert to {to_alias} how much?",
        "Can check how much {amount} {from_alias} is in {to_alias}?",
        "If I change {amount} {from_alias}, how much {to_alias}?",
        "Help me see {amount} {from_alias} equals how much {to_alias}.",
    ],
    "singlish": [
        "{amount} {from_alias} change {to_alias} how much ah",
        "eh {amount} {from_alias} to {to_alias} how much",
        "{amount} {from_alias} change to {to_alias} can?",
        "{amount} {from_alias} become {to_alias} how much ah",
        "if change {amount} {from_alias} get how much {to_alias}",
    ],
    "noisy": [
        "{amount}{from_code} to {to_code}",
        "{amount} {from_code} -> {to_code}",
    ],
}


UNIT_TEMPLATES = {
    "standard": [
        "Convert {value} {from_unit} to {to_unit}.",
        "How many {to_unit} is {value} {from_unit}?",
        "What is {value} {from_unit} in {to_unit}?",
        "Convert {value} from {from_unit} into {to_unit}.",
        "Express {value} {from_unit} in {to_unit}.",
        "What does {value} {from_unit} convert to in {to_unit}?",
    ],
    "singapore_english": [
        "Can help convert {value} {from_unit} to {to_unit}?",
        "{value} {from_unit} is how many {to_unit}?",
        "Help me change {value} {from_unit} into {to_unit}.",
        "How much is {value} {from_unit} in {to_unit}?",
        "Can calculate {value} {from_unit} in {to_unit}?",
        "Help me see {value} {from_unit} equals how many {to_unit}.",
        "{value} {from_unit} convert to {to_unit} can?",
    ],
    "singlish": [
        "{value} {from_unit} how many {to_unit} ah",
        "eh {value} {from_unit} convert {to_unit} how",
        "{value} {from_unit} become {to_unit} how much",
        "help me change {value} {from_unit} to {to_unit} leh",
        "{value} {from_unit} in {to_unit} is how much sia",
    ],
    "noisy": [
        "{value}{from_unit} to {to_unit}",
        "{value} {from_unit} -> {to_unit}",
    ],
}


CALCULATE_TEMPLATES = {
    "standard": [
        "Calculate {description}.",
        "What is {description}?",
        "Work out {description}.",
        "Please calculate {description}.",
        "Find the result of {description}.",
        "Compute {description}.",
    ],
    "singapore_english": [
        "Can help me calculate {description}?",
        "Help me work out {description}.",
        "Can calculate {description} for me?",
        "What's {description}?",
        "Help me see how much is {description}.",
        "Can help work out {description}?",
        "Please help calculate {description}.",
    ],
    "singlish": [
        "help me calculate {description} leh",
        "{description} how much ah",
        "eh calculate {description} for me",
        "{description} total how much",
        "can calculate {description} anot",
    ],
    "noisy": [
        "calc {description}",
        "{description} pls",
    ],
}


TIME_TEMPLATES = {
    "standard": [
        "What time is it in {location}?",
        "Tell me the current time in {location}.",
        "What is the time in {location} right now?",
        "Check the current time in {location}.",
        "Could you tell me the local time in {location}?",
        "What is the local time in {location}?",
    ],
    "singapore_english": [
        "Can check what time it is in {location}?",
        "What time is {location} now?",
        "Can tell me the current time in {location}?",
        "{location} is what time now?",
        "Help me check {location} time now.",
        "Can see what time {location} is now?",
        "What's the time over at {location} now?",
    ],
    "singlish": [
        "{location} now what time ah",
        "eh {location} what time now",
        "{location} time now how",
        "what time {location} now sia",
        "{location} now几点 ah",
    ],
    "noisy": [
        "{location} time rn",
        "time {location} now",
    ],
}


# ---------------------------------------------------------------------------
# no_tool data
# ---------------------------------------------------------------------------

NO_TOOL_TOPICS = [
    "why Singapore is so humid",
    "how GST works",
    "why MRT trains are crowded during peak hours",
    "what the Malaysian ringgit is",
    "why Singapore uses SGD",
    "why Singapore gets frequent thunderstorms",
    "what an exchange rate means",
    "why Orchard Road gets crowded",
    "what humidity means",
    "how currency conversion works",
    "what Celsius means",
    "what Fahrenheit means",
    "why different countries use different currencies",
    "what a timezone is",
    "why Singapore is in GMT plus 8",
    "how weather forecasts are made",
    "why currencies change in value",
    "what the metric system is",
    "why kilometres are used in Singapore",
    "how GST is calculated generally",
    "what compound interest means",
    "why temperatures change during the day",
    "what foreign exchange markets are",
    "why Singapore weather changes so quickly",
    "what SGD stands for",
    "why Japan uses yen",
    "why Korea uses won",
    "what a unit conversion is",
    "how calculators work",
    "why timezones exist",
]


NO_TOOL_INFO_TEMPLATES = {
    "standard": [
        "Explain this: {topic}.",
        "Tell me more about this: {topic}.",
        "Can you explain this: {topic}?",
        "I want to understand this: {topic}.",
        "Could you explain the following: {topic}?",
        "Give me some background on this: {topic}.",
    ],
    "singapore_english": [
        "Can explain this: {topic}?",
        "Can tell me more about this: {topic}?",
        "Help me understand this: {topic}.",
        "Can explain this one to me: {topic}?",
        "I don't really understand this: {topic}.",
        "Can share more about this: {topic}?",
        "Can help explain this: {topic}?",
    ],
    "singlish": [
        "eh can explain this: {topic}",
        "this one I don't understand: {topic}",
        "can explain this one anot: {topic}",
        "what does this mean ah: {topic}",
        "help me understand this leh: {topic}",
    ],
    "noisy": [
        "explain pls: {topic}",
        "what is this: {topic}",
    ],
}


# Requests where the intent resembles a tool call but required information
# is missing or genuinely ambiguous.
AMBIGUOUS_CURRENCY_TEMPLATES = {
    "standard": [
        "Convert {amount} dollars to ringgit.",
        "How much is {amount} dollars in ringgit?",
        "Convert {amount} to MYR.",
        "Exchange {amount} dollars into yen.",
    ],
    "singapore_english": [
        "Can convert {amount} dollars to ringgit?",
        "{amount} dollars in ringgit how much?",
        "Help me change {amount} dollars to yen.",
        "Can check {amount} dollars to MYR?",
        "{amount} dollars convert to ringgit can?",
    ],
    "singlish": [
        "{amount} dollars to ringgit how much ah",
        "eh {amount} dollars change ringgit",
        "{amount} dollars become MYR how much",
        "change {amount} dollars to yen leh",
    ],
    "noisy": [
        "{amount} dollars -> myr",
        "{amount} to ringgit",
    ],
}


AMBIGUOUS_FIXED = {
    "standard": [
        "What is the weather right now?",
        "Check the weather for me.",
        "What time is it there?",
        "Tell me the local time.",
        "Convert this into kilometres.",
        "Convert SGD to MYR.",
        "How much is this in yen?",
    ],
    "singapore_english": [
        "Can check the weather now?",
        "Can tell me what time there now?",
        "Help me convert this to kilometres.",
        "Can convert SGD to ringgit?",
        "How much is this in MYR?",
        "Can check whether raining now?",
        "Help me calculate this amount.",
    ],
    "singlish": [
        "weather now how?",
        "now what time ah?",
        "convert this to km leh",
        "SGD to ringgit how much ah",
        "this one in yen how much",
        "raining anot?",
        "calculate this can?",
    ],
    "noisy": [
        "weather rn",
        "sgd to myr",
        "time now?",
        "convert to km",
    ],
}


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------


def make_record(
    user_text: str,
    tool: str,
    arguments: dict[str, Any],
    style: str,
    template_id: str,
    difficulty: str = "normal",
) -> dict[str, Any]:
    call = ToolCall(
        tool=tool,
        arguments=arguments,
    )

    assistant_output = json.dumps(
        call.model_dump(),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "assistant",
                "content": assistant_output,
            },
        ],
        "metadata": {
            "tool": tool,
            "style": style,
            "template_id": template_id,
            "difficulty": difficulty,
        },
    }


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    user_text = record["messages"][1]["content"]
    assistant_text = record["messages"][2]["content"]

    return user_text, assistant_text


def deduplicate(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}

    for candidate in candidates:
        unique[record_key(candidate)] = candidate

    return list(unique.values())


def sample_balanced(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = deduplicate(candidates)
    selected = []

    for style, target_count in STYLE_TARGETS.items():
        style_candidates = [
            record
            for record in candidates
            if record["metadata"]["style"] == style
        ]

        if len(style_candidates) < target_count:
            raise ValueError(
                f"Not enough unique '{style}' candidates: "
                f"need {target_count}, found {len(style_candidates)}"
            )

        selected.extend(
            random.sample(
                style_candidates,
                target_count,
            )
        )

    random.shuffle(selected)

    return selected


# ---------------------------------------------------------------------------
# Tool generators
# ---------------------------------------------------------------------------


def generate_weather() -> list[dict[str, Any]]:
    candidates = []

    for style, templates in WEATHER_TEMPLATES.items():
        for template_index, template in enumerate(templates):
            for location in LOCATIONS:
                candidates.append(
                    make_record(
                        user_text=template.format(
                            location=location,
                        ),
                        tool="get_weather",
                        arguments={
                            "location": location,
                        },
                        style=style,
                        template_id=(
                            f"weather_{style}_{template_index}"
                        ),
                    )
                )

    return sample_balanced(candidates)


def generate_currency() -> list[dict[str, Any]]:
    candidates = []

    for style, templates in CURRENCY_TEMPLATES.items():
        for template_index, template in enumerate(templates):
            for amount in AMOUNTS:
                for from_code, to_code in CURRENCY_PAIRS:
                    from_alias = random.choice(
                        CURRENCY_ALIASES[from_code]
                    )
                    to_alias = random.choice(
                        CURRENCY_ALIASES[to_code]
                    )

                    user_text = template.format(
                        amount=amount,
                        from_alias=from_alias,
                        to_alias=to_alias,
                        from_code=from_code,
                        to_code=to_code,
                    )

                    candidates.append(
                        make_record(
                            user_text=user_text,
                            tool="convert_currency",
                            arguments={
                                "amount": amount,
                                "from_currency": from_code,
                                "to_currency": to_code,
                            },
                            style=style,
                            template_id=(
                                f"currency_{style}_{template_index}"
                            ),
                        )
                    )

    return sample_balanced(candidates)


def generate_units() -> list[dict[str, Any]]:
    candidates = []

    for style, templates in UNIT_TEMPLATES.items():
        for template_index, template in enumerate(templates):
            for value, from_unit, to_unit in UNIT_CONVERSIONS:
                candidates.append(
                    make_record(
                        user_text=template.format(
                            value=value,
                            from_unit=from_unit,
                            to_unit=to_unit,
                        ),
                        tool="convert_units",
                        arguments={
                            "value": value,
                            "from_unit": from_unit,
                            "to_unit": to_unit,
                        },
                        style=style,
                        template_id=(
                            f"units_{style}_{template_index}"
                        ),
                    )
                )

    return sample_balanced(candidates)


def generate_calculations() -> list[dict[str, Any]]:
    candidates = []

    for style, templates in CALCULATE_TEMPLATES.items():
        for template_index, template in enumerate(templates):
            for expression, description in CALCULATIONS:
                candidates.append(
                    make_record(
                        user_text=template.format(
                            description=description,
                        ),
                        tool="calculate",
                        arguments={
                            "expression": expression,
                        },
                        style=style,
                        template_id=(
                            f"calculate_{style}_{template_index}"
                        ),
                    )
                )

    return sample_balanced(candidates)


def generate_time() -> list[dict[str, Any]]:
    candidates = []

    for style, templates in TIME_TEMPLATES.items():
        for template_index, template in enumerate(templates):
            for location in TIME_LOCATIONS:
                candidates.append(
                    make_record(
                        user_text=template.format(
                            location=location,
                        ),
                        tool="get_time",
                        arguments={
                            "location": location,
                        },
                        style=style,
                        template_id=(
                            f"time_{style}_{template_index}"
                        ),
                    )
                )

    return sample_balanced(candidates)


# ---------------------------------------------------------------------------
# no_tool generator
# ---------------------------------------------------------------------------


def generate_no_tool_information() -> list[dict[str, Any]]:
    candidates = []

    for style, templates in NO_TOOL_INFO_TEMPLATES.items():
        for template_index, template in enumerate(templates):
            for topic in NO_TOOL_TOPICS:
                candidates.append(
                    make_record(
                        user_text=template.format(
                            topic=topic,
                        ),
                        tool="no_tool",
                        arguments={},
                        style=style,
                        template_id=(
                            f"no_tool_info_{style}_{template_index}"
                        ),
                    )
                )

    return candidates


def generate_no_tool_ambiguous() -> list[dict[str, Any]]:
    candidates = []

    for style, templates in AMBIGUOUS_CURRENCY_TEMPLATES.items():
        for template_index, template in enumerate(templates):
            for amount in AMOUNTS:
                candidates.append(
                    make_record(
                        user_text=template.format(
                            amount=amount,
                        ),
                        tool="no_tool",
                        arguments={},
                        style=style,
                        template_id=(
                            f"no_tool_currency_"
                            f"{style}_{template_index}"
                        ),
                        difficulty="hard",
                    )
                )

    for style, examples in AMBIGUOUS_FIXED.items():
        for example_index, user_text in enumerate(examples):
            candidates.append(
                make_record(
                    user_text=user_text,
                    tool="no_tool",
                    arguments={},
                    style=style,
                    template_id=(
                        f"no_tool_ambiguous_"
                        f"{style}_{example_index}"
                    ),
                    difficulty="hard",
                )
            )

    return candidates


def generate_no_tool() -> list[dict[str, Any]]:
    normal_candidates = deduplicate(
        generate_no_tool_information()
    )

    hard_candidates = deduplicate(
        generate_no_tool_ambiguous()
    )

    selected = []

    # 20% of no_tool examples are deliberately difficult / ambiguous.
    for style, target_count in STYLE_TARGETS.items():
        hard_target = round(target_count * 0.20)
        normal_target = target_count - hard_target

        normal_pool = [
            record
            for record in normal_candidates
            if record["metadata"]["style"] == style
        ]

        hard_pool = [
            record
            for record in hard_candidates
            if record["metadata"]["style"] == style
        ]

        if len(normal_pool) < normal_target:
            raise ValueError(
                f"Not enough normal no_tool examples for {style}"
            )

        if len(hard_pool) < hard_target:
            raise ValueError(
                f"Not enough hard no_tool examples for {style}"
            )

        selected.extend(
            random.sample(
                normal_pool,
                normal_target,
            )
        )

        selected.extend(
            random.sample(
                hard_pool,
                hard_target,
            )
        )

    random.shuffle(selected)

    return selected


# ---------------------------------------------------------------------------
# Validation / reporting
# ---------------------------------------------------------------------------


def validate_dataset(
    records: list[dict[str, Any]],
) -> None:
    expected_total = EXAMPLES_PER_TOOL * 6

    if len(records) != expected_total:
        raise ValueError(
            f"Expected {expected_total} examples, "
            f"found {len(records)}"
        )

    keys = [
        record_key(record)
        for record in records
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate user/output pairs detected."
        )

    for record in records:
        assistant_text = record["messages"][2]["content"]
        parsed = json.loads(assistant_text)

        ToolCall.model_validate(parsed)


def print_summary(
    records: list[dict[str, Any]],
) -> None:
    tool_counts = Counter(
        record["metadata"]["tool"]
        for record in records
    )

    style_counts = Counter(
        record["metadata"]["style"]
        for record in records
    )

    difficulty_counts = Counter(
        record["metadata"]["difficulty"]
        for record in records
    )

    print()
    print("Dataset summary")
    print("---------------")
    print(f"Total examples: {len(records)}")
    print()

    print("By tool:")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")

    print()
    print("By style:")
    for style, count in sorted(style_counts.items()):
        print(f"  {style}: {count}")

    print()
    print("By difficulty:")
    for difficulty, count in sorted(
        difficulty_counts.items()
    ):
        print(f"  {difficulty}: {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    records = []

    records.extend(generate_weather())
    records.extend(generate_currency())
    records.extend(generate_units())
    records.extend(generate_calculations())
    records.extend(generate_time())
    records.extend(generate_no_tool())

    random.shuffle(records)

    validate_dataset(records)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Generated {len(records)} examples."
    )
    print(
        f"Saved to {OUTPUT_PATH}"
    )

    print_summary(records)


if __name__ == "__main__":
    main()