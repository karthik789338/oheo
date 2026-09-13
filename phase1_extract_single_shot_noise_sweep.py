from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


INPUT_DIR = Path("outputs/phase1f_final_audit")
OUTPUT_DIR = Path("outputs/phase1f_final_audit")

INPUT_FILE = INPUT_DIR / "mean_vector_reference.csv"

OUTPUT_CSV = OUTPUT_DIR / "single_shot_noise_sweep.csv"
OUTPUT_JSON = OUTPUT_DIR / "single_shot_noise_sweep.json"

EXPECTED_HISTORY_COUNTS = (2, 4, 6, 8)
EXPECTED_NOISE_LEVELS = (0.0, 0.1, 0.25, 0.5, 1.0)

EXPECTED_SUBSET_COUNTS = {
    2: 28,
    4: 70,
    6: 28,
    8: 1,
}


METRICS = (
    "mapping_accuracy",
    "exact_transformation_recovery",
    "relation_accuracy_full",
    "relation_balanced_accuracy_full",
    "relation_macro_f1_full",
    "full_accuracy_above_majority",
)


def load_single_shot_rows():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing Phase 1F input: {INPUT_FILE}"
        )

    rows = []

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:
            repetitions = int(
                row["repetitions"]
            )

            if repetitions != 1:
                continue

            parsed = {
                "history_count":
                    int(
                        row["history_count"]
                    ),

                "noise_fraction":
                    float(
                        row["noise_fraction"]
                    ),

                "history_subset_mask":
                    int(
                        row["history_subset_mask"]
                    ),
            }

            for metric in METRICS:
                parsed[metric] = float(
                    row[metric]
                )

            rows.append(parsed)

    return rows


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "minimum":
            float(
                np.min(values)
            ),

        "mean":
            float(
                np.mean(values)
            ),

        "maximum":
            float(
                np.max(values)
            ),
    }


def aggregate_rows(rows):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["history_count"],
            row["noise_fraction"],
        )

        grouped[key].append(row)

    output_rows = []
    output_json = {}

    for history_count in EXPECTED_HISTORY_COUNTS:

        for noise_fraction in EXPECTED_NOISE_LEVELS:

            key = (
                history_count,
                noise_fraction,
            )

            group = grouped.get(
                key,
                [],
            )

            expected_count = (
                EXPECTED_SUBSET_COUNTS[
                    history_count
                ]
            )

            if len(group) != expected_count:
                raise AssertionError(
                    f"Expected {expected_count} rows for "
                    f"h={history_count}, "
                    f"noise={noise_fraction}; "
                    f"found {len(group)}."
                )

            summary = {}

            for metric in METRICS:
                summary[metric] = summarize(
                    [
                        row[metric]
                        for row in group
                    ]
                )

            csv_row = {
                "history_count":
                    history_count,

                "noise_fraction":
                    noise_fraction,

                "subset_count":
                    len(group),
            }

            for metric in METRICS:
                csv_row[
                    f"{metric}_minimum"
                ] = summary[
                    metric
                ]["minimum"]

                csv_row[
                    f"{metric}_mean"
                ] = summary[
                    metric
                ]["mean"]

                csv_row[
                    f"{metric}_maximum"
                ] = summary[
                    metric
                ]["maximum"]

            output_rows.append(
                csv_row
            )

            label = (
                f"h{history_count}"
                f"_noise{noise_fraction}"
            )

            output_json[label] = {
                "history_count":
                    history_count,

                "noise_fraction":
                    noise_fraction,

                "repetitions":
                    1,

                "subset_count":
                    len(group),

                **summary,
            }

    return (
        output_rows,
        output_json,
    )


def run_sanity_checks(
    source_rows,
    output_rows,
):
    expected_source_rows = (
        sum(
            EXPECTED_SUBSET_COUNTS.values()
        )
        * len(
            EXPECTED_NOISE_LEVELS
        )
    )

    if len(source_rows) != expected_source_rows:
        raise AssertionError(
            "Unexpected number of single-shot "
            f"source rows: {len(source_rows)}"
        )

    expected_conditions = (
        len(
            EXPECTED_HISTORY_COUNTS
        )
        * len(
            EXPECTED_NOISE_LEVELS
        )
    )

    if len(output_rows) != expected_conditions:
        raise AssertionError(
            "Expected "
            f"{expected_conditions} aggregated conditions, "
            f"found {len(output_rows)}."
        )

    clean_full = [
        row
        for row in output_rows
        if (
            row["history_count"] == 8
            and row["noise_fraction"] == 0.0
        )
    ]

    if len(clean_full) != 1:
        raise AssertionError(
            "Missing clean h=8 condition."
        )

    clean_full = clean_full[0]

    required_clean_metrics = (
        "mapping_accuracy_mean",
        "exact_transformation_recovery_mean",
        "relation_accuracy_full_mean",
        "relation_balanced_accuracy_full_mean",
        "relation_macro_f1_full_mean",
    )

    for metric in required_clean_metrics:

        if not np.isclose(
            clean_full[metric],
            1.0,
        ):
            raise AssertionError(
                f"Expected {metric}=1.0 "
                "for clean full-history condition."
            )


def write_csv(rows):
    fieldnames = list(
        rows[0].keys()
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(summary):
    payload = {
        "experiment":
            "single-shot noise sweep",

        "status":
            "extracted from frozen Phase 1F results",

        "repetitions":
            1,

        "history_counts":
            list(
                EXPECTED_HISTORY_COUNTS
            ),

        "noise_levels":
            list(
                EXPECTED_NOISE_LEVELS
            ),

        "condition_count":
            (
                len(
                    EXPECTED_HISTORY_COUNTS
                )
                * len(
                    EXPECTED_NOISE_LEVELS
                )
            ),

        "important_note":
            (
                "This is an extraction of already generated "
                "Phase 1E/1F conditions. No benchmark data, "
                "noise realization, transformation, split, "
                "or decoder was regenerated."
            ),

        "conditions":
            summary,

        "sanity_checks":
            "passed",
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
        )

    return payload


def main():
    rows = load_single_shot_rows()

    (
        aggregated_rows,
        summary,
    ) = aggregate_rows(
        rows
    )

    run_sanity_checks(
        rows,
        aggregated_rows,
    )

    write_csv(
        aggregated_rows
    )

    payload = write_json(
        summary
    )

    print(
        "Single-shot noise sweep extracted successfully."
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print(
        f"CSV written to: {OUTPUT_CSV}"
    )

    print(
        f"JSON written to: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()
