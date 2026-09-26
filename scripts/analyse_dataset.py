import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sg_qwen_tool_router.schemas import ToolCall

DATASET_PATH = Path("data/raw/examples.jsonl")

SEED = 42
SAMPLES_PER_STYLE = 3

random.seed(SEED)


def load_dataset() -> list[dict[str, Any]]:
    records = []

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            records.append(record)

    return records


def get_user_text(
    record: dict[str, Any],
) -> str:
    return record["messages"][1]["content"]


def get_assistant_text(
    record: dict[str, Any],
) -> str:
    return record["messages"][2]["content"]


def validate_structure(
    records: list[dict[str, Any]],
) -> None:
    errors = []

    for index, record in enumerate(records):
        try:
            messages = record["messages"]
            metadata = record["metadata"]

            if len(messages) != 3:
                raise ValueError(
                    "Expected exactly 3 messages."
                )

            expected_roles = [
                "system",
                "user",
                "assistant",
            ]

            actual_roles = [
                message["role"]
                for message in messages
            ]

            if actual_roles != expected_roles:
                raise ValueError(
                    f"Unexpected roles: {actual_roles}"
                )

            required_metadata = {
                "tool",
                "style",
                "template_id",
                "difficulty",
            }

            missing_metadata = (
                required_metadata
                - set(metadata.keys())
            )

            if missing_metadata:
                raise ValueError(
                    f"Missing metadata: {missing_metadata}"
                )

            assistant_output = json.loads(
                get_assistant_text(record)
            )

            ToolCall.model_validate(
                assistant_output
            )

        except (
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            errors.append(
                f"Record {index}: {error}"
            )

    if errors:
        print("\nSTRUCTURE ERRORS")
        print("----------------")

        for error in errors[:20]:
            print(error)

        raise ValueError(
            f"{len(errors)} invalid records found."
        )

    print("Structure validation: PASS")


def check_duplicates(
    records: list[dict[str, Any]],
) -> None:
    pair_counts = Counter(
        (
            get_user_text(record),
            get_assistant_text(record),
        )
        for record in records
    )

    duplicate_pairs = [
        pair
        for pair, count in pair_counts.items()
        if count > 1
    ]

    print(
        f"Exact duplicate input/output pairs: "
        f"{len(duplicate_pairs)}"
    )


def check_conflicting_labels(
    records: list[dict[str, Any]],
) -> None:
    outputs_by_input = defaultdict(set)

    for record in records:
        outputs_by_input[
            get_user_text(record)
        ].add(
            get_assistant_text(record)
        )

    conflicts = {
        user_text: outputs
        for user_text, outputs
        in outputs_by_input.items()
        if len(outputs) > 1
    }

    print(
        f"Inputs with conflicting outputs: "
        f"{len(conflicts)}"
    )

    if conflicts:
        print("\nCONFLICTING EXAMPLES")
        print("--------------------")

        for user_text, outputs in list(
            conflicts.items()
        )[:10]:
            print(f"\nInput: {user_text}")

            for output in outputs:
                print(f"  {output}")


def print_distribution(
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

    print("\nDATASET DISTRIBUTION")
    print("--------------------")

    print(f"Total examples: {len(records)}")

    print("\nBy tool:")
    for tool, count in sorted(
        tool_counts.items()
    ):
        percentage = (
            count / len(records)
        ) * 100

        print(
            f"  {tool:<20} "
            f"{count:>4} "
            f"({percentage:>5.1f}%)"
        )

    print("\nBy style:")
    for style, count in sorted(
        style_counts.items()
    ):
        percentage = (
            count / len(records)
        ) * 100

        print(
            f"  {style:<20} "
            f"{count:>4} "
            f"({percentage:>5.1f}%)"
        )

    print("\nBy difficulty:")
    for difficulty, count in sorted(
        difficulty_counts.items()
    ):
        percentage = (
            count / len(records)
        ) * 100

        print(
            f"  {difficulty:<20} "
            f"{count:>4} "
            f"({percentage:>5.1f}%)"
        )


def print_tool_style_matrix(
    records: list[dict[str, Any]],
) -> None:
    matrix = defaultdict(Counter)

    for record in records:
        tool = record["metadata"]["tool"]
        style = record["metadata"]["style"]

        matrix[tool][style] += 1

    styles = [
        "standard",
        "singapore_english",
        "singlish",
        "noisy",
    ]

    print("\nTOOL × STYLE")
    print("------------")

    header = (
        f"{'Tool':<20}"
        + "".join(
            f"{style:>20}"
            for style in styles
        )
    )

    print(header)

    for tool in sorted(matrix):
        row = f"{tool:<20}"

        for style in styles:
            row += (
                f"{matrix[tool][style]:>20}"
            )

        print(row)


def analyze_templates(
    records: list[dict[str, Any]],
) -> None:
    template_counts = Counter(
        record["metadata"]["template_id"]
        for record in records
    )

    print("\nTEMPLATES")
    print("---------")

    print(
        f"Unique templates: "
        f"{len(template_counts)}"
    )

    print("\n10 most common templates:")

    for template, count in (
        template_counts.most_common(10)
    ):
        print(
            f"  {template:<45} {count}"
        )


def analyze_text_lengths(
    records: list[dict[str, Any]],
) -> None:
    lengths = [
        len(
            get_user_text(record).split()
        )
        for record in records
    ]

    print("\nUSER TEXT LENGTH")
    print("----------------")

    print(
        f"Minimum words: {min(lengths)}"
    )

    print(
        f"Maximum words: {max(lengths)}"
    )

    print(
        "Average words: "
        f"{sum(lengths) / len(lengths):.1f}"
    )


def print_random_samples(
    records: list[dict[str, Any]],
) -> None:
    grouped = defaultdict(list)

    for record in records:
        style = record["metadata"]["style"]
        grouped[style].append(record)

    print("\nRANDOM SAMPLE REVIEW")
    print("--------------------")

    styles = [
        "standard",
        "singapore_english",
        "singlish",
        "noisy",
    ]

    for style in styles:
        print(
            f"\n=== {style.upper()} ==="
        )

        sample_size = min(
            SAMPLES_PER_STYLE,
            len(grouped[style]),
        )

        for record in random.sample(
            grouped[style],
            sample_size,
        ):
            print()

            print(
                "Tool:",
                record["metadata"]["tool"],
            )

            print(
                "Difficulty:",
                record["metadata"][
                    "difficulty"
                ],
            )

            print(
                "User:",
                get_user_text(record),
            )

            print(
                "Target:",
                get_assistant_text(record),
            )


def print_hard_examples(
    records: list[dict[str, Any]],
) -> None:
    hard_examples = [
        record
        for record in records
        if record["metadata"][
            "difficulty"
        ]
        == "hard"
    ]

    print("\nHARD EXAMPLES")
    print("-------------")

    print(
        f"Total hard examples: "
        f"{len(hard_examples)}"
    )

    sample_size = min(
        10,
        len(hard_examples),
    )

    for record in random.sample(
        hard_examples,
        sample_size,
    ):
        print()
        print(
            f"User: {get_user_text(record)}"
        )
        print(
            f"Target: "
            f"{get_assistant_text(record)}"
        )


def main() -> None:
    records = load_dataset()

    print(
        f"Loaded {len(records)} records "
        f"from {DATASET_PATH}"
    )
    print()

    validate_structure(records)
    check_duplicates(records)
    check_conflicting_labels(records)

    print_distribution(records)
    print_tool_style_matrix(records)
    analyze_templates(records)
    analyze_text_lengths(records)

    print_random_samples(records)
    print_hard_examples(records)


if __name__ == "__main__":
    main()