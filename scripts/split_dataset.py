import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEED = 42

INPUT_PATH = Path("data/raw/examples.jsonl")
OUTPUT_DIR = Path("data/processed")

SPLIT_RATIOS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}


def load_dataset() -> list[dict[str, Any]]:
    records = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def group_by_template(
    records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups = defaultdict(list)

    for record in records:
        template_id = record["metadata"]["template_id"]

        groups[template_id].append(record)

    return list(groups.values())


def build_targets(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    total = len(records)

    tools = Counter(
        record["metadata"]["tool"]
        for record in records
    )

    styles = Counter(
        record["metadata"]["style"]
        for record in records
    )

    targets = {}

    for split, ratio in SPLIT_RATIOS.items():
        targets[split] = {
            "total": total * ratio,
            "tools": {
                tool: count * ratio
                for tool, count in tools.items()
            },
            "styles": {
                style: count * ratio
                for style, count in styles.items()
            },
        }

    return targets


def group_properties(
    group: list[dict[str, Any]],
) -> tuple[int, str, str]:
    first = group[0]

    size = len(group)
    tool = first["metadata"]["tool"]
    style = first["metadata"]["style"]

    return size, tool, style


def assign_groups(
    groups: list[list[dict[str, Any]]],
    targets: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(SEED)

    # Shuffle first so similarly sized template groups
    # do not always get assigned in source-code order.
    rng.shuffle(groups)

    # Larger groups are allocated first because they
    # are harder to place later without overshooting.
    groups.sort(
        key=len,
        reverse=True,
    )

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    current_total = Counter()
    current_tools = defaultdict(Counter)
    current_styles = defaultdict(Counter)

    for group in groups:
        size, tool, style = group_properties(group)

        candidate_scores = []

        for split in SPLIT_RATIOS:
            target_total = targets[split]["total"]
            target_tool = targets[split]["tools"][tool]
            target_style = targets[split]["styles"][style]

            projected_total = (
                current_total[split] + size
            )

            projected_tool = (
                current_tools[split][tool] + size
            )

            projected_style = (
                current_styles[split][style] + size
            )

            total_ratio = (
                projected_total / target_total
            )

            tool_ratio = (
                projected_tool / target_tool
            )

            style_ratio = (
                projected_style / target_style
            )

            # Tool balance receives the largest weight
            # because tool-call accuracy is our primary
            # evaluation objective.
            score = (
                2.0 * total_ratio
                + 3.0 * tool_ratio
                + 1.5 * style_ratio
            )

            candidate_scores.append(
                (
                    score,
                    rng.random(),
                    split,
                )
            )

        _, _, chosen_split = min(
            candidate_scores
        )

        splits[chosen_split].extend(group)

        current_total[chosen_split] += size
        current_tools[chosen_split][tool] += size
        current_styles[chosen_split][style] += size

    return splits


def validate_template_isolation(
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    template_sets = {}

    for split, records in splits.items():
        template_sets[split] = {
            record["metadata"]["template_id"]
            for record in records
        }

    train_validation_overlap = (
        template_sets["train"]
        & template_sets["validation"]
    )

    train_test_overlap = (
        template_sets["train"]
        & template_sets["test"]
    )

    validation_test_overlap = (
        template_sets["validation"]
        & template_sets["test"]
    )

    if train_validation_overlap:
        raise ValueError(
            "Template leakage between train "
            "and validation."
        )

    if train_test_overlap:
        raise ValueError(
            "Template leakage between train and test."
        )

    if validation_test_overlap:
        raise ValueError(
            "Template leakage between validation "
            "and test."
        )

    print("Template isolation: PASS")


def validate_record_isolation(
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    user_sets = {}

    for split, records in splits.items():
        user_sets[split] = {
            record["messages"][1]["content"]
            for record in records
        }

    overlaps = {
        "train/validation": (
            user_sets["train"]
            & user_sets["validation"]
        ),
        "train/test": (
            user_sets["train"]
            & user_sets["test"]
        ),
        "validation/test": (
            user_sets["validation"]
            & user_sets["test"]
        ),
    }

    for name, overlap in overlaps.items():
        if overlap:
            raise ValueError(
                f"Input leakage in {name}: "
                f"{len(overlap)} examples"
            )

    print("Input isolation: PASS")


def print_split_summary(
    split: str,
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

    template_count = len(
        {
            record["metadata"]["template_id"]
            for record in records
        }
    )

    print()
    print(split.upper())
    print("-" * len(split))
    print(
        f"Examples: {len(records)}"
    )
    print(
        f"Templates: {template_count}"
    )

    print("\nTools:")

    for tool, count in sorted(
        tool_counts.items()
    ):
        print(
            f"  {tool:<20} {count}"
        )

    print("\nStyles:")

    for style, count in sorted(
        style_counts.items()
    ):
        print(
            f"  {style:<20} {count}"
        )


def save_split(
    name: str,
    records: list[dict[str, Any]],
) -> None:
    output_path = (
        OUTPUT_DIR / f"{name}.jsonl"
    )

    with output_path.open(
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


def main() -> None:
    records = load_dataset()

    print(
        f"Loaded {len(records)} examples."
    )

    groups = group_by_template(records)

    print(
        f"Found {len(groups)} template families."
    )

    targets = build_targets(records)

    splits = assign_groups(
        groups,
        targets,
    )

    validate_template_isolation(splits)
    validate_record_isolation(splits)

    rng = random.Random(SEED)

    for records_for_split in splits.values():
        rng.shuffle(records_for_split)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split, split_records in splits.items():
        save_split(
            split,
            split_records,
        )

        print_split_summary(
            split,
            split_records,
        )

    total_written = sum(
        len(records_for_split)
        for records_for_split
        in splits.values()
    )

    if total_written != len(records):
        raise ValueError(
            "Some records were lost during splitting."
        )

    print()
    print(
        f"Successfully wrote "
        f"{total_written} records."
    )


if __name__ == "__main__":
    main()