from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


PHASE1A_DIR = Path("outputs/phase1a_ground_truth")
PHASE1C_DIR = Path("outputs/phase1c_continuous_observations")
PHASE1E_DIR = Path("outputs/phase1e_identifiability_stress")

OUTPUT_DIR = Path("outputs/phase1f_final_audit")


STATE_COUNT = 8
TRANSFORMATION_COUNT = 104
INFORMATION_CLASS_COUNT = 9

HISTORY_COUNTS = (2, 4, 6, 8)
REPETITIONS = (1, 4, 16)

BASE_RANDOM_SEED = 8675309

RELATIONS = (
    "equivalent",
    "a_more_informative",
    "a_less_informative",
    "incomparable",
)

RELATION_TO_CODE = {
    name: index
    for index, name in enumerate(RELATIONS)
}

CODE_TO_RELATION = {
    index: name
    for name, index in RELATION_TO_CODE.items()
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def save_input_hashes() -> None:
    files = {
        "phase1a_semigroup":
            PHASE1A_DIR / "semigroup_elements.csv",

        "phase1a_relations":
            PHASE1A_DIR / "blackwell_relations.csv",

        "phase1a_summary":
            PHASE1A_DIR / "phase1a_summary.json",

        "phase1c_prototypes":
            PHASE1C_DIR / "state_prototypes.csv",

        "phase1c_noise":
            PHASE1C_DIR / "noise_conditions.json",

        "phase1e_history":
            PHASE1E_DIR
            / "history_subset_identifiability.csv",

        "phase1e_alphabet":
            PHASE1E_DIR
            / "alphabet_coverage.csv",

        "phase1e_noisy":
            PHASE1E_DIR
            / "noisy_oracle_recovery.csv",

        "phase1e_summary":
            PHASE1E_DIR / "phase1e_summary.json",
    }

    hashes = {}

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input: {path}"
            )

        hashes[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }

    with (
        OUTPUT_DIR
        / "input_hashes.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            hashes,
            handle,
            indent=2,
        )


def load_semigroup():
    path = (
        PHASE1A_DIR
        / "semigroup_elements.csv"
    )

    rows = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:
            row["mapping_tuple"] = tuple(
                int(value)
                for value in json.loads(
                    row["mapping"]
                )
            )

            rows.append(row)

    rows.sort(
        key=lambda row: row["element_id"]
    )

    return rows


def load_prototypes():
    path = (
        PHASE1C_DIR
        / "state_prototypes.csv"
    )

    rows = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:
            rows.append(row)

    rows.sort(
        key=lambda row: int(row["state"])
    )

    dimension_names = [
        name
        for name in rows[0]
        if name.startswith("z")
    ]

    dimension_names.sort(
        key=lambda name: int(name[1:])
    )

    return np.asarray(
        [
            [
                float(row[name])
                for name in dimension_names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def load_noise_conditions():
    rows = load_json(
        PHASE1C_DIR
        / "noise_conditions.json"
    )

    rows.sort(
        key=lambda row:
            int(row["noise_index"])
    )

    return rows


def load_csv(path: Path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def equality_mask(outputs) -> int:
    outputs = [
        int(value)
        for value in outputs
    ]

    mask = 0
    bit = 0

    for first in range(
        len(outputs)
    ):
        for second in range(
            first + 1,
            len(outputs),
        ):

            if (
                outputs[first]
                == outputs[second]
            ):
                mask |= (
                    1 << bit
                )

            bit += 1

    return mask


def relation_from_masks(
    first_mask: int,
    second_mask: int,
) -> int:

    if first_mask == second_mask:
        return RELATION_TO_CODE[
            "equivalent"
        ]

    if (
        first_mask
        & ~second_mask
    ) == 0:
        return RELATION_TO_CODE[
            "a_more_informative"
        ]

    if (
        second_mask
        & ~first_mask
    ) == 0:
        return RELATION_TO_CODE[
            "a_less_informative"
        ]

    return RELATION_TO_CODE[
        "incomparable"
    ]


def relation_vector(masks):
    relations = []

    for first in range(
        len(masks)
    ):
        for second in range(
            len(masks)
        ):

            if first == second:
                continue

            relations.append(
                relation_from_masks(
                    int(masks[first]),
                    int(masks[second]),
                )
            )

    return np.asarray(
        relations,
        dtype=np.int8,
    )


def build_ground_truth(
    semigroup_rows,
):
    mappings = [
        row["mapping_tuple"]
        for row in semigroup_rows
    ]

    masks = np.asarray(
        [
            equality_mask(mapping)
            for mapping in mappings
        ],
        dtype=np.int64,
    )

    relations = relation_vector(
        masks
    )

    return (
        mappings,
        relations,
    )


def relation_metrics(
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
            "Relation vectors have different shapes."
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

    correct = int(
        np.trace(
            confusion
        )
    )

    accuracy = (
        correct / total
    )

    recalls = []
    precisions = []
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

        recall = (
            true_positive / support
            if support > 0
            else 0.0
        )

        precision = (
            true_positive / predicted_count
            if predicted_count > 0
            else 0.0
        )

        if precision + recall > 0:
            f1 = (
                2.0
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )
        else:
            f1 = 0.0

        recalls.append(
            recall
        )

        precisions.append(
            precision
        )

        f1_values.append(
            f1
        )

        class_metrics[
            relation_name
        ] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    class_supports = (
        confusion.sum(
            axis=1
        )
    )

    majority_baseline = (
        int(
            np.max(
                class_supports
            )
        )
        / total
    )

    return {
        "accuracy":
            float(accuracy),

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

        "majority_baseline_accuracy":
            float(
                majority_baseline
            ),

        "accuracy_above_majority":
            float(
                accuracy
                - majority_baseline
            ),

        "class_metrics":
            class_metrics,

        "confusion_matrix":
            confusion,
    }


def history_subset_mask(
    states,
) -> int:

    mask = 0

    for state in states:
        mask |= (
            1 << int(state)
        )

    return mask


def make_condition_rng(
    subset_mask,
    noise_index,
):
    seed_sequence = np.random.SeedSequence(
        [
            BASE_RANDOM_SEED,
            int(subset_mask),
            int(noise_index),
        ]
    )

    return np.random.default_rng(
        seed_sequence
    )


def decode_nearest_prototype(
    observations,
    prototypes,
):
    differences = (
        observations[
            ...,
            None,
            :
        ]
        - prototypes[
            None,
            None,
            :,
            :
        ]
    )

    squared_distances = (
        differences
        * differences
    ).sum(
        axis=-1
    )

    return np.argmin(
        squared_distances,
        axis=-1,
    ).astype(
        np.int8
    )


def build_existing_reference_lookup():
    rows = load_csv(
        PHASE1E_DIR
        / "noisy_oracle_recovery.csv"
    )

    lookup = {}

    for row in rows:
        key = (
            int(
                row[
                    "history_subset_mask"
                ]
            ),
            int(
                row[
                    "noise_index"
                ]
            ),
            int(
                row[
                    "repetitions"
                ]
            ),
        )

        lookup[key] = row

    return lookup


def run_mean_vector_reference(
    mappings,
    prototypes,
    noise_conditions,
    full_relations,
):
    """
    Stronger repeated-measurement reference.

    For each transformation/history pair:
      1. generate the same repeated noisy measurements as Phase 1E,
      2. average the continuous vectors,
      3. decode the averaged vector using the known prototypes.

    This avoids information loss from hard decoding before
    aggregation.
    """

    existing_lookup = (
        build_existing_reference_lookup()
    )

    rows = []

    maximum_repetitions = max(
        REPETITIONS
    )

    for history_count in HISTORY_COUNTS:

        for subset in combinations(
            range(STATE_COUNT),
            history_count,
        ):

            subset = tuple(subset)

            subset_mask = (
                history_subset_mask(
                    subset
                )
            )

            true_outputs = np.asarray(
                [
                    [
                        mapping[state]
                        for state in subset
                    ]
                    for mapping in mappings
                ],
                dtype=np.int8,
            )

            restricted_masks = np.asarray(
                [
                    equality_mask(outputs)
                    for outputs in true_outputs
                ],
                dtype=np.int64,
            )

            restricted_relations = (
                relation_vector(
                    restricted_masks
                )
            )

            clean_vectors = prototypes[
                true_outputs
            ]

            for noise_condition in (
                noise_conditions
            ):

                noise_index = int(
                    noise_condition[
                        "noise_index"
                    ]
                )

                noise_fraction = float(
                    noise_condition[
                        "noise_fraction"
                    ]
                )

                sigma = float(
                    noise_condition[
                        "sigma"
                    ]
                )

                rng = make_condition_rng(
                    subset_mask,
                    noise_index,
                )

                repeated = np.repeat(
                    clean_vectors[
                        :,
                        :,
                        None,
                        :
                    ],
                    maximum_repetitions,
                    axis=2,
                )

                if sigma > 0.0:
                    repeated = (
                        repeated
                        + rng.normal(
                            loc=0.0,
                            scale=sigma,
                            size=repeated.shape,
                        )
                    )

                for repetitions in REPETITIONS:

                    mean_vectors = (
                        repeated[
                            :,
                            :,
                            :repetitions,
                            :
                        ].mean(
                            axis=2
                        )
                    )

                    estimated_outputs = (
                        decode_nearest_prototype(
                            mean_vectors,
                            prototypes,
                        )
                    )

                    mapping_accuracy = float(
                        np.mean(
                            estimated_outputs
                            == true_outputs
                        )
                    )

                    exact_recovery = float(
                        np.mean(
                            np.all(
                                estimated_outputs
                                == true_outputs,
                                axis=1,
                            )
                        )
                    )

                    estimated_masks = np.asarray(
                        [
                            equality_mask(outputs)
                            for outputs
                            in estimated_outputs
                        ],
                        dtype=np.int64,
                    )

                    estimated_relations = (
                        relation_vector(
                            estimated_masks
                        )
                    )

                    restricted_metrics = (
                        relation_metrics(
                            restricted_relations,
                            estimated_relations,
                        )
                    )

                    full_metrics = (
                        relation_metrics(
                            full_relations,
                            estimated_relations,
                        )
                    )

                    old_key = (
                        subset_mask,
                        noise_index,
                        repetitions,
                    )

                    old_row = (
                        existing_lookup[
                            old_key
                        ]
                    )

                    old_full_accuracy = float(
                        old_row[
                            "relation_accuracy_vs_full_ground_truth"
                        ]
                    )

                    rows.append(
                        {
                            "history_subset_mask":
                                subset_mask,

                            "history_count":
                                history_count,

                            "history_states":
                                json.dumps(
                                    list(subset),
                                    separators=(
                                        ",",
                                        ":",
                                    ),
                                ),

                            "noise_index":
                                noise_index,

                            "noise_fraction":
                                noise_fraction,

                            "sigma":
                                sigma,

                            "repetitions":
                                repetitions,

                            "mapping_accuracy":
                                mapping_accuracy,

                            "exact_transformation_recovery":
                                exact_recovery,

                            "relation_accuracy_restricted":
                                restricted_metrics[
                                    "accuracy"
                                ],

                            "relation_balanced_accuracy_restricted":
                                restricted_metrics[
                                    "balanced_accuracy"
                                ],

                            "relation_macro_f1_restricted":
                                restricted_metrics[
                                    "macro_f1"
                                ],

                            "relation_accuracy_full":
                                full_metrics[
                                    "accuracy"
                                ],

                            "relation_balanced_accuracy_full":
                                full_metrics[
                                    "balanced_accuracy"
                                ],

                            "relation_macro_f1_full":
                                full_metrics[
                                    "macro_f1"
                                ],

                            "full_majority_baseline":
                                full_metrics[
                                    "majority_baseline_accuracy"
                                ],

                            "full_accuracy_above_majority":
                                full_metrics[
                                    "accuracy_above_majority"
                                ],

                            "phase1e_majority_vote_full_accuracy":
                                old_full_accuracy,

                            "mean_vector_minus_majority_vote":
                                (
                                    full_metrics[
                                        "accuracy"
                                    ]
                                    - old_full_accuracy
                                ),
                        }
                    )

    return rows


def write_mean_reference(
    rows,
):
    path = (
        OUTPUT_DIR
        / "mean_vector_reference.csv"
    )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
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


def write_relation_baseline(
    full_relations,
):
    metrics = relation_metrics(
        full_relations,
        full_relations,
    )

    counts = Counter(
        int(value)
        for value in full_relations
    )

    result = {
        "ordered_distinct_pair_count":
            len(full_relations),

        "class_counts": {
            CODE_TO_RELATION[code]:
                counts[code]
            for code in range(
                len(RELATIONS)
            )
        },

        "majority_class":
            CODE_TO_RELATION[
                max(
                    counts,
                    key=counts.get,
                )
            ],

        "majority_baseline_accuracy":
            metrics[
                "majority_baseline_accuracy"
            ],
    }

    with (
        OUTPUT_DIR
        / "relation_class_baseline.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=2,
        )

    return result


def write_history_extremes():
    source_rows = load_csv(
        PHASE1E_DIR
        / "history_subset_identifiability.csv"
    )

    grouped = defaultdict(list)

    for row in source_rows:

        row["history_count"] = int(
            row["history_count"]
        )

        row[
            "distinct_transformation_signatures"
        ] = int(
            row[
                "distinct_transformation_signatures"
            ]
        )

        row[
            "full_relation_accuracy_from_restricted_view"
        ] = float(
            row[
                "full_relation_accuracy_from_restricted_view"
            ]
        )

        grouped[
            row["history_count"]
        ].append(row)

    output_rows = []

    for history_count in sorted(grouped):

        group = grouped[
            history_count
        ]

        worst = min(
            group,
            key=lambda row: (
                row[
                    "full_relation_accuracy_from_restricted_view"
                ],
                int(
                    row[
                        "history_subset_mask"
                    ]
                ),
            ),
        )

        best = max(
            group,
            key=lambda row: (
                row[
                    "full_relation_accuracy_from_restricted_view"
                ],
                -int(
                    row[
                        "history_subset_mask"
                    ]
                ),
            ),
        )

        for label, row in [
            ("worst", worst),
            ("best", best),
        ]:

            output_rows.append(
                {
                    "history_count":
                        history_count,

                    "extreme":
                        label,

                    "history_subset_mask":
                        row[
                            "history_subset_mask"
                        ],

                    "history_states":
                        row[
                            "history_states"
                        ],

                    "distinct_transformation_signatures":
                        row[
                            "distinct_transformation_signatures"
                        ],

                    "full_relation_accuracy":
                        row[
                            "full_relation_accuracy_from_restricted_view"
                        ],
                }
            )

    path = (
        OUTPUT_DIR
        / "history_subset_extremes.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    return output_rows


def audit_operation_alphabets():
    source_rows = load_csv(
        PHASE1E_DIR
        / "alphabet_coverage.csv"
    )

    rows = []

    for row in source_rows:

        parsed = {
            "alphabet_size":
                int(
                    row[
                        "alphabet_size"
                    ]
                ),

            "generator_subset":
                row[
                    "generator_subset"
                ],

            "reachable_transformation_count":
                int(
                    row[
                        "reachable_transformation_count"
                    ]
                ),

            "reachable_information_class_count":
                int(
                    row[
                        "reachable_information_class_count"
                    ]
                ),
        }

        rows.append(parsed)

    full_semigroup = [
        row
        for row in rows
        if (
            row[
                "reachable_transformation_count"
            ]
            == TRANSFORMATION_COUNT
        )
    ]

    all_information_classes = [
        row
        for row in rows
        if (
            row[
                "reachable_information_class_count"
            ]
            == INFORMATION_CLASS_COUNT
        )
    ]

    minimum_semigroup_size = min(
        row["alphabet_size"]
        for row in full_semigroup
    )

    minimum_information_size = min(
        row["alphabet_size"]
        for row in all_information_classes
    )

    minimal_semigroup_rows = [
        row
        for row in full_semigroup
        if (
            row["alphabet_size"]
            == minimum_semigroup_size
        )
    ]

    minimal_information_rows = [
        row
        for row in all_information_classes
        if (
            row["alphabet_size"]
            == minimum_information_size
        )
    ]

    output_rows = []

    for coverage_type, selected_rows in [
        (
            "full_semigroup",
            minimal_semigroup_rows,
        ),
        (
            "all_information_classes",
            minimal_information_rows,
        ),
    ]:

        for row in selected_rows:

            output_rows.append(
                {
                    "coverage_type":
                        coverage_type,

                    "alphabet_size":
                        row[
                            "alphabet_size"
                        ],

                    "generator_subset":
                        row[
                            "generator_subset"
                        ],

                    "reachable_transformation_count":
                        row[
                            "reachable_transformation_count"
                        ],

                    "reachable_information_class_count":
                        row[
                            "reachable_information_class_count"
                        ],
                }
            )

    path = (
        OUTPUT_DIR
        / "minimal_operation_alphabets.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    return {
        "minimum_full_semigroup_alphabet_size":
            minimum_semigroup_size,

        "minimal_full_semigroup_alphabets":
            [
                json.loads(
                    row[
                        "generator_subset"
                    ]
                )
                for row
                in minimal_semigroup_rows
            ],

        "minimum_all_information_classes_alphabet_size":
            minimum_information_size,

        "minimal_all_information_class_alphabets":
            [
                json.loads(
                    row[
                        "generator_subset"
                    ]
                )
                for row
                in minimal_information_rows
            ],
    }


def summarize_mean_reference(
    rows,
):
    grouped = defaultdict(list)

    comparison_counts = Counter()

    for row in rows:

        key = (
            row["history_count"],
            row["noise_fraction"],
            row["repetitions"],
        )

        grouped[key].append(
            row
        )

        difference = row[
            "mean_vector_minus_majority_vote"
        ]

        tolerance = 1e-12

        if difference > tolerance:
            comparison_counts[
                "mean_vector_better"
            ] += 1

        elif difference < -tolerance:
            comparison_counts[
                "majority_vote_better"
            ] += 1

        else:
            comparison_counts[
                "equal"
            ] += 1

    summary_rows = []

    for key in sorted(grouped):

        history_count, noise, repetitions = key

        group = grouped[key]

        summary_rows.append(
            {
                "history_count":
                    history_count,

                "noise_fraction":
                    noise,

                "repetitions":
                    repetitions,

                "subset_count":
                    len(group),

                "mean_mapping_accuracy":
                    float(
                        np.mean(
                            [
                                row[
                                    "mapping_accuracy"
                                ]
                                for row in group
                            ]
                        )
                    ),

                "mean_exact_transformation_recovery":
                    float(
                        np.mean(
                            [
                                row[
                                    "exact_transformation_recovery"
                                ]
                                for row in group
                            ]
                        )
                    ),

                "mean_relation_accuracy_full":
                    float(
                        np.mean(
                            [
                                row[
                                    "relation_accuracy_full"
                                ]
                                for row in group
                            ]
                        )
                    ),

                "mean_relation_balanced_accuracy_full":
                    float(
                        np.mean(
                            [
                                row[
                                    "relation_balanced_accuracy_full"
                                ]
                                for row in group
                            ]
                        )
                    ),

                "mean_relation_macro_f1_full":
                    float(
                        np.mean(
                            [
                                row[
                                    "relation_macro_f1_full"
                                ]
                                for row in group
                            ]
                        )
                    ),

                "mean_accuracy_above_majority":
                    float(
                        np.mean(
                            [
                                row[
                                    "full_accuracy_above_majority"
                                ]
                                for row in group
                            ]
                        )
                    ),

                "mean_improvement_over_phase1e_majority_vote":
                    float(
                        np.mean(
                            [
                                row[
                                    "mean_vector_minus_majority_vote"
                                ]
                                for row in group
                            ]
                        )
                    ),
            }
        )

    path = (
        OUTPUT_DIR
        / "mean_reference_summary.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    return (
        summary_rows,
        dict(comparison_counts),
    )


def run_sanity_checks(
    phase1e_summary,
    full_relations,
    relation_baseline,
    mean_rows,
):
    if (
        phase1e_summary[
            "sanity_checks"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 1E is not marked as passed."
        )

    if (
        len(full_relations)
        != 10712
    ):
        raise AssertionError(
            "Expected 10,712 ordered "
            "distinct relation pairs."
        )

    expected_counts = {
        "equivalent": 2008,
        "a_more_informative": 1728,
        "a_less_informative": 1728,
        "incomparable": 5248,
    }

    if (
        relation_baseline[
            "class_counts"
        ]
        != expected_counts
    ):
        raise AssertionError(
            "Frozen relation counts changed."
        )

    expected_rows = (
        127
        * 5
        * 3
    )

    if (
        len(mean_rows)
        != expected_rows
    ):
        raise AssertionError(
            f"Expected {expected_rows} "
            "reference conditions."
        )

    clean_full = [
        row
        for row in mean_rows
        if (
            row["history_count"] == 8
            and row["noise_fraction"] == 0.0
        )
    ]

    if len(clean_full) != 3:
        raise AssertionError(
            "Missing clean full-history conditions."
        )

    for row in clean_full:

        required = [
            "mapping_accuracy",
            "exact_transformation_recovery",
            "relation_accuracy_restricted",
            "relation_accuracy_full",
            "relation_balanced_accuracy_full",
            "relation_macro_f1_full",
        ]

        for metric in required:

            if not np.isclose(
                row[metric],
                1.0,
            ):
                raise AssertionError(
                    "Clean full-history reference "
                    f"must have {metric}=1."
                )


def write_final_summary(
    relation_baseline,
    alphabet_audit,
    history_extremes,
    mean_summary,
    comparison_counts,
):
    selected_conditions = {}

    wanted = [
        (2, 0.0, 1),
        (4, 0.0, 1),
        (6, 0.0, 1),
        (8, 0.0, 1),

        (4, 0.25, 16),
        (6, 0.25, 16),
        (8, 0.25, 16),

        (4, 0.5, 16),
        (6, 0.5, 16),
        (8, 0.5, 16),

        (8, 1.0, 16),
    ]

    lookup = {
        (
            row["history_count"],
            row["noise_fraction"],
            row["repetitions"],
        ): row

        for row in mean_summary
    }

    for key in wanted:

        if key not in lookup:
            continue

        label = (
            f"h{key[0]}"
            f"_noise{key[1]}"
            f"_r{key[2]}"
        )

        selected_conditions[
            label
        ] = lookup[key]

    summary = {
        "phase":
            "1F final benchmark audit",

        "relation_baseline":
            relation_baseline,

        "alphabet_audit":
            alphabet_audit,

        "history_subset_extremes":
            history_extremes,

        "selected_mean_vector_reference_conditions":
            selected_conditions,

        "mean_vector_vs_phase1e_majority_vote":
            comparison_counts,

        "important_interpretation": (
            "Repeated measurements can reduce measurement noise, "
            "but they cannot recover distinctions removed by missing "
            "history coverage. Raw relation accuracy must be interpreted "
            "relative to the imbalanced four-class relation baseline."
        ),

        "phase1e_correction": (
            "The Phase 1E majority-vote decoder should be described "
            "as a privileged reference decoder, not an oracle ceiling. "
            "Phase 1F adds a stronger mean-vector reference."
        ),

        "sanity_checks":
            "passed",
    }

    with (
        OUTPUT_DIR
        / "phase1f_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_input_hashes()

    phase1e_summary = load_json(
        PHASE1E_DIR
        / "phase1e_summary.json"
    )

    semigroup_rows = (
        load_semigroup()
    )

    prototypes = (
        load_prototypes()
    )

    noise_conditions = (
        load_noise_conditions()
    )

    (
        mappings,
        full_relations,
    ) = build_ground_truth(
        semigroup_rows
    )

    relation_baseline = (
        write_relation_baseline(
            full_relations
        )
    )

    history_extremes = (
        write_history_extremes()
    )

    alphabet_audit = (
        audit_operation_alphabets()
    )

    print(
        "Structural audit complete."
    )

    mean_rows = (
        run_mean_vector_reference(
            mappings=mappings,
            prototypes=prototypes,
            noise_conditions=noise_conditions,
            full_relations=full_relations,
        )
    )

    write_mean_reference(
        mean_rows
    )

    (
        mean_summary,
        comparison_counts,
    ) = summarize_mean_reference(
        mean_rows
    )

    run_sanity_checks(
        phase1e_summary=phase1e_summary,
        full_relations=full_relations,
        relation_baseline=relation_baseline,
        mean_rows=mean_rows,
    )

    summary = write_final_summary(
        relation_baseline=relation_baseline,
        alphabet_audit=alphabet_audit,
        history_extremes=history_extremes,
        mean_summary=mean_summary,
        comparison_counts=comparison_counts,
    )

    print(
        "Phase 1F final benchmark audit "
        "completed successfully."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"Outputs written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
