from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from phase2e_evaluate_structural_recovery import (
    RELATIONS,
    CODE_TO_RELATION,
    equality_mask,
    relation_from_masks,
)


PHASE1D_DIR = Path(
    "outputs/phase1d_compositional_splits"
)

PHASE2E_DIR = Path(
    "outputs/phase2e_structural_recovery"
)

OUTPUT_DIR = Path(
    "outputs/phase2f_ood_structural_audit"
)


FROZEN_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

TUNING_SEED = 11

FROZEN_NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

ESTIMATORS = (
    "hard",
    "probe",
)

EXPECTED_TRANSFORMATION_COUNTS = {
    "train": 86,
    "val": 9,
    "test": 9,
}

EXPECTED_TRANSFORMATION_COUNT = 104
EXPECTED_ALL_PAIR_COUNT = 10712


def load_csv(path: Path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def write_csv(
    path: Path,
    rows,
):
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    value,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            value,
            handle,
            indent=2,
        )


def parse_bool(value: str) -> bool:
    value = value.strip().lower()

    if value == "true":
        return True

    if value == "false":
        return False

    raise ValueError(
        f"Cannot parse boolean value: {value}"
    )


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean":
            float(
                values.mean()
            ),

        "std":
            float(
                values.std(
                    ddof=1
                )
            )
            if len(values) > 1
            else 0.0,

        "minimum":
            float(
                values.min()
            ),

        "maximum":
            float(
                values.max()
            ),
    }


def load_transformation_splits():
    rows = load_csv(
        PHASE1D_DIR
        / "transformation_split.csv"
    )

    split_lookup = {
        row["element_id"]:
            row["split"]
        for row in rows
    }

    counts = Counter(
        split_lookup.values()
    )

    if dict(counts) != EXPECTED_TRANSFORMATION_COUNTS:
        raise AssertionError(
            "Frozen transformation split changed: "
            f"{dict(counts)}"
        )

    return split_lookup


def load_structural_rows():
    rows = load_csv(
        PHASE2E_DIR
        / "transformation_metrics.csv"
    )

    parsed = []

    for row in rows:
        parsed.append(
            {
                "seed":
                    int(
                        row["seed"]
                    ),

                "noise_fraction":
                    float(
                        row[
                            "noise_fraction"
                        ]
                    ),

                "estimator":
                    row[
                        "estimator"
                    ],

                "element_id":
                    row[
                        "element_id"
                    ],

                "true_rank":
                    int(
                        row[
                            "true_rank"
                        ]
                    ),

                "predicted_rank":
                    int(
                        row[
                            "predicted_rank"
                        ]
                    ),

                "mapping_accuracy":
                    float(
                        row[
                            "mapping_accuracy"
                        ]
                    ),

                "exact_mapping":
                    parse_bool(
                        row[
                            "exact_mapping"
                        ]
                    ),

                "partition_correct":
                    parse_bool(
                        row[
                            "partition_correct"
                        ]
                    ),

                "rank_correct":
                    parse_bool(
                        row[
                            "rank_correct"
                        ]
                    ),

                "true_mapping":
                    tuple(
                        int(value)
                        for value in json.loads(
                            row[
                                "true_mapping"
                            ]
                        )
                    ),

                "predicted_mapping":
                    tuple(
                        int(value)
                        for value in json.loads(
                            row[
                                "predicted_mapping"
                            ]
                        )
                    ),
            }
        )

    expected_row_count = (
        len(FROZEN_SEEDS)
        * len(FROZEN_NOISE_LEVELS)
        * len(ESTIMATORS)
        * EXPECTED_TRANSFORMATION_COUNT
    )

    if len(parsed) != expected_row_count:
        raise AssertionError(
            "Unexpected transformation-metric row count: "
            f"{len(parsed)}"
        )

    return parsed


def supported_relation_metrics(
    truth,
    prediction,
):
    truth = np.asarray(
        truth,
        dtype=np.int8,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.int8,
    )

    if truth.shape != prediction.shape:
        raise ValueError(
            "Truth and prediction shapes differ."
        )

    confusion = np.zeros(
        (
            len(RELATIONS),
            len(RELATIONS),
        ),
        dtype=np.int64,
    )

    for true_value, predicted_value in zip(
        truth,
        prediction,
    ):
        confusion[
            int(true_value),
            int(predicted_value),
        ] += 1

    total = int(
        confusion.sum()
    )

    accuracy = float(
        np.trace(
            confusion
        )
        / total
    )

    recalls = []
    f1_values = []
    class_metrics = {}

    for code, relation_name in enumerate(
        RELATIONS
    ):
        true_positive = int(
            confusion[
                code,
                code,
            ]
        )

        support = int(
            confusion[
                code,
                :
            ].sum()
        )

        predicted_count = int(
            confusion[
                :,
                code,
            ].sum()
        )

        precision = (
            true_positive
            / predicted_count
            if predicted_count > 0
            else 0.0
        )

        recall = (
            true_positive
            / support
            if support > 0
            else None
        )

        if (
            recall is not None
            and precision + recall > 0
        ):
            f1 = (
                2.0
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )
        elif recall is not None:
            f1 = 0.0
        else:
            f1 = None

        if recall is not None:
            recalls.append(
                recall
            )

            f1_values.append(
                f1
            )

        class_metrics[
            relation_name
        ] = {
            "support":
                support,

            "precision":
                float(
                    precision
                ),

            "recall":
                (
                    float(
                        recall
                    )
                    if recall is not None
                    else None
                ),

            "f1":
                (
                    float(
                        f1
                    )
                    if f1 is not None
                    else None
                ),
        }

    supports = confusion.sum(
        axis=1
    )

    majority_baseline = float(
        supports.max()
        / supports.sum()
    )

    return {
        "pair_count":
            total,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            float(
                np.mean(
                    recalls
                )
            ),

        "macro_f1":
            float(
                np.mean(
                    f1_values
                )
            ),

        "majority_baseline":
            majority_baseline,

        "accuracy_above_majority":
            float(
                accuracy
                - majority_baseline
            ),

        "class_metrics":
            class_metrics,

        "confusion_matrix":
            confusion.tolist(),
    }


def relation_scopes(
    split_a,
    split_b,
):
    scopes = [
        "all",
    ]

    if (
        split_a == "train"
        and split_b == "train"
    ):
        scopes.append(
            "train_train"
        )

    if (
        split_a == "val"
        or split_b == "val"
    ):
        scopes.append(
            "val_involved"
        )

    if (
        split_a == "test"
        or split_b == "test"
    ):
        scopes.append(
            "test_involved"
        )

    if split_a == "test":
        scopes.append(
            "test_as_first"
        )

    if split_b == "test":
        scopes.append(
            "test_as_second"
        )

    if (
        split_a == "test"
        and split_b == "test"
    ):
        scopes.append(
            "test_test"
        )

    if (
        {split_a, split_b}
        == {
            "train",
            "test",
        }
    ):
        scopes.append(
            "train_test_cross"
        )

    if (
        split_a != "train"
        or split_b != "train"
    ):
        scopes.append(
            "heldout_involved"
        )

    return scopes


def transformation_split_metrics(
    group,
    split_lookup,
):
    output = []

    grouped = defaultdict(
        list
    )

    for row in group:
        grouped[
            split_lookup[
                row[
                    "element_id"
                ]
            ]
        ].append(
            row
        )

    for split in (
        "train",
        "val",
        "test",
    ):
        rows = grouped[
            split
        ]

        expected = (
            EXPECTED_TRANSFORMATION_COUNTS[
                split
            ]
        )

        if len(rows) != expected:
            raise AssertionError(
                f"{split} contains "
                f"{len(rows)} transformations."
            )

        output.append(
            {
                "split":
                    split,

                "transformation_count":
                    len(rows),

                "mapping_accuracy":
                    float(
                        np.mean(
                            [
                                row[
                                    "mapping_accuracy"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "exact_transformation_rate":
                    float(
                        np.mean(
                            [
                                row[
                                    "exact_mapping"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "partition_accuracy":
                    float(
                        np.mean(
                            [
                                row[
                                    "partition_correct"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "rank_accuracy":
                    float(
                        np.mean(
                            [
                                row[
                                    "rank_correct"
                                ]
                                for row in rows
                            ]
                        )
                    ),
            }
        )

    return output


def relation_scope_metrics(
    group,
    split_lookup,
):
    rows_by_element = {
        row["element_id"]:
            row
        for row in group
    }

    element_ids = sorted(
        rows_by_element
    )

    if len(
        element_ids
    ) != EXPECTED_TRANSFORMATION_COUNT:
        raise AssertionError(
            "Transformation count changed."
        )

    truth_by_scope = defaultdict(
        list
    )

    prediction_by_scope = defaultdict(
        list
    )

    for first_id in element_ids:
        first = rows_by_element[
            first_id
        ]

        first_true_mask = equality_mask(
            first[
                "true_mapping"
            ]
        )

        first_prediction_mask = equality_mask(
            first[
                "predicted_mapping"
            ]
        )

        first_split = split_lookup[
            first_id
        ]

        for second_id in element_ids:
            if first_id == second_id:
                continue

            second = rows_by_element[
                second_id
            ]

            second_true_mask = equality_mask(
                second[
                    "true_mapping"
                ]
            )

            second_prediction_mask = equality_mask(
                second[
                    "predicted_mapping"
                ]
            )

            second_split = split_lookup[
                second_id
            ]

            true_relation = relation_from_masks(
                first_true_mask,
                second_true_mask,
            )

            predicted_relation = relation_from_masks(
                first_prediction_mask,
                second_prediction_mask,
            )

            for scope in relation_scopes(
                first_split,
                second_split,
            ):
                truth_by_scope[
                    scope
                ].append(
                    true_relation
                )

                prediction_by_scope[
                    scope
                ].append(
                    predicted_relation
                )

    if len(
        truth_by_scope[
            "all"
        ]
    ) != EXPECTED_ALL_PAIR_COUNT:
        raise AssertionError(
            "Full ordered-pair count changed."
        )

    output = []

    for scope in sorted(
        truth_by_scope
    ):
        metrics = supported_relation_metrics(
            truth_by_scope[
                scope
            ],
            prediction_by_scope[
                scope
            ],
        )

        output.append(
            {
                "scope":
                    scope,

                "pair_count":
                    metrics[
                        "pair_count"
                    ],

                "accuracy":
                    metrics[
                        "accuracy"
                    ],

                "balanced_accuracy":
                    metrics[
                        "balanced_accuracy"
                    ],

                "macro_f1":
                    metrics[
                        "macro_f1"
                    ],

                "majority_baseline":
                    metrics[
                        "majority_baseline"
                    ],

                "accuracy_above_majority":
                    metrics[
                        "accuracy_above_majority"
                    ],
            }
        )

    return output


def aggregate_rows(
    rows,
    seeds,
    key_fields,
    metric_fields,
):
    grouped = defaultdict(
        list
    )

    for row in rows:
        if row[
            "seed"
        ] not in seeds:
            continue

        key = tuple(
            row[field]
            for field in key_fields
        )

        grouped[
            key
        ].append(
            row
        )

    output = {}

    for key in sorted(
        grouped
    ):
        group = grouped[
            key
        ]

        if len(group) != len(
            seeds
        ):
            raise AssertionError(
                f"Group {key} has "
                f"{len(group)} seeds."
            )

        label = "|".join(
            str(value)
            for value in key
        )

        item = {
            field:
                value
            for field, value
            in zip(
                key_fields,
                key,
            )
        }

        item[
            "seed_count"
        ] = len(
            group
        )

        for metric in metric_fields:
            item[
                metric
            ] = summarize(
                [
                    row[
                        metric
                    ]
                    for row in group
                ]
            )

        output[
            label
        ] = item

    return output


def selected_summary(
    transformation_aggregate,
    relation_aggregate,
):
    selected = {}

    for estimator in ESTIMATORS:
        selected[
            estimator
        ] = {}

        for noise in FROZEN_NOISE_LEVELS:
            noise_key = str(
                noise
            )

            selected[
                estimator
            ][
                noise_key
            ] = {
                "test_transformations":
                    transformation_aggregate[
                        (
                            estimator
                            + "|"
                            + noise_key
                            + "|test"
                        )
                    ],

                "all_relations":
                    relation_aggregate[
                        (
                            estimator
                            + "|"
                            + noise_key
                            + "|all"
                        )
                    ],

                "test_involved_relations":
                    relation_aggregate[
                        (
                            estimator
                            + "|"
                            + noise_key
                            + "|test_involved"
                        )
                    ],

                "test_test_relations":
                    relation_aggregate[
                        (
                            estimator
                            + "|"
                            + noise_key
                            + "|test_test"
                        )
                    ],
            }

    return selected


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_lookup = (
        load_transformation_splits()
    )

    structural_rows = (
        load_structural_rows()
    )

    grouped = defaultdict(
        list
    )

    for row in structural_rows:
        key = (
            row["seed"],
            row["noise_fraction"],
            row["estimator"],
        )

        grouped[
            key
        ].append(
            row
        )

    expected_group_count = (
        len(FROZEN_SEEDS)
        * len(FROZEN_NOISE_LEVELS)
        * len(ESTIMATORS)
    )

    if len(grouped) != expected_group_count:
        raise AssertionError(
            "Unexpected run-estimator group count."
        )

    transformation_output = []
    relation_output = []

    for (
        seed,
        noise,
        estimator,
    ), group in sorted(
        grouped.items()
    ):
        if len(
            group
        ) != EXPECTED_TRANSFORMATION_COUNT:
            raise AssertionError(
                "A run does not contain "
                "104 transformations."
            )

        transformation_metrics = (
            transformation_split_metrics(
                group,
                split_lookup,
            )
        )

        for row in transformation_metrics:
            transformation_output.append(
                {
                    "seed":
                        seed,

                    "noise_fraction":
                        noise,

                    "estimator":
                        estimator,

                    **row,
                }
            )

        relation_metrics = (
            relation_scope_metrics(
                group,
                split_lookup,
            )
        )

        for row in relation_metrics:
            relation_output.append(
                {
                    "seed":
                        seed,

                    "noise_fraction":
                        noise,

                    "estimator":
                        estimator,

                    **row,
                }
            )

        print(
            f"seed={seed} | "
            f"noise={noise} | "
            f"estimator={estimator} | "
            "OOD structural audit complete"
        )

    write_csv(
        OUTPUT_DIR
        / "transformation_split_metrics.csv",
        transformation_output,
    )

    write_csv(
        OUTPUT_DIR
        / "relation_scope_metrics.csv",
        relation_output,
    )

    transformation_metric_fields = [
        "mapping_accuracy",
        "exact_transformation_rate",
        "partition_accuracy",
        "rank_accuracy",
    ]

    relation_metric_fields = [
        "pair_count",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "majority_baseline",
        "accuracy_above_majority",
    ]

    primary_transformation = aggregate_rows(
        rows=transformation_output,
        seeds=FROZEN_SEEDS,
        key_fields=[
            "estimator",
            "noise_fraction",
            "split",
        ],
        metric_fields=(
            transformation_metric_fields
        ),
    )

    primary_relation = aggregate_rows(
        rows=relation_output,
        seeds=FROZEN_SEEDS,
        key_fields=[
            "estimator",
            "noise_fraction",
            "scope",
        ],
        metric_fields=(
            relation_metric_fields
        ),
    )

    sensitivity_seeds = tuple(
        seed
        for seed in FROZEN_SEEDS
        if seed != TUNING_SEED
    )

    sensitivity_transformation = (
        aggregate_rows(
            rows=transformation_output,
            seeds=sensitivity_seeds,
            key_fields=[
                "estimator",
                "noise_fraction",
                "split",
            ],
            metric_fields=(
                transformation_metric_fields
            ),
        )
    )

    sensitivity_relation = (
        aggregate_rows(
            rows=relation_output,
            seeds=sensitivity_seeds,
            key_fields=[
                "estimator",
                "noise_fraction",
                "scope",
            ],
            metric_fields=(
                relation_metric_fields
            ),
        )
    )

    summary = {
        "phase":
            "2F OOD structural-generalization audit",

        "evaluated_run_estimator_groups":
            len(
                grouped
            ),

        "transformation_split_counts":
            EXPECTED_TRANSFORMATION_COUNTS,

        "primary_five_seed_results":
            selected_summary(
                primary_transformation,
                primary_relation,
            ),

        "four_seed_sensitivity_excluding_tuning_seed":
            selected_summary(
                sensitivity_transformation,
                sensitivity_relation,
            ),

        "training_performed":
            False,

        "checkpoint_selection_reopened":
            False,

        "checkpoints_modified":
            False,

        "phase2e_outputs_modified":
            False,

        "phase2f_status":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase2f_summary.json",
        summary,
    )

    print()
    print(
        "Phase 2F OOD structural audit completed."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"Outputs written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
