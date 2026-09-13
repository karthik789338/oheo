from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from phase2b_operation_channel_model import (
    OperationChannelModel,
)


PHASE1A_DIR = Path(
    "outputs/phase1a_ground_truth"
)

PHASE1C_DIR = Path(
    "outputs/phase1c_continuous_observations"
)

PHASE2A_DIR = Path(
    "outputs/phase2a_learning_protocol"
)

PHASE2D_DIR = Path(
    "outputs/phase2d_final_ocm"
)

OUTPUT_DIR = Path(
    "outputs/phase2e_structural_recovery"
)

RUN_DETAIL_DIR = (
    OUTPUT_DIR / "run_details"
)


STATE_COUNT = 8
OBSERVATION_DIM = 12
LATENT_STATE_COUNT = 8
EXPECTED_TRANSFORMATION_COUNT = 104
EXPECTED_RELATION_PAIR_COUNT = 10712

FROZEN_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

FROZEN_NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

TUNING_SEED = 11
EXPECTED_CONFIGURATION_ID = "c05"


RELATIONS = (
    "equivalent",
    "a_more_informative",
    "a_less_informative",
    "incomparable",
)

RELATION_TO_CODE = {
    name: index
    for index, name in enumerate(
        RELATIONS
    )
}

CODE_TO_RELATION = {
    index: name
    for name, index in (
        RELATION_TO_CODE.items()
    )
}

EXPECTED_RELATION_COUNTS = {
    "equivalent": 2008,
    "a_more_informative": 1728,
    "a_less_informative": 1728,
    "incomparable": 5248,
}


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
):
    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def load_csv(
    path: Path,
):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


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
        writer.writerows(
            rows
        )


def noise_slug(
    noise: float,
):
    return (
        f"{noise:g}"
        .replace(
            ".",
            "p",
        )
    )


def load_operation_vocabulary():
    return load_json(
        PHASE2A_DIR
        / "operation_vocabulary.json"
    )


def load_generators():
    rows = load_json(
        PHASE1A_DIR
        / "generators.json"
    )

    mappings = {}

    for row in rows:
        mappings[
            row["name"]
        ] = tuple(
            int(value)
            for value in row[
                "mapping"
            ]
        )

    return mappings


def parse_shortest_word(
    row,
):
    word_length = int(
        row["word_length"]
    )

    if word_length == 0:
        return tuple()

    text = row[
        "shortest_word"
    ].strip()

    if not text:
        raise AssertionError(
            "Nonempty transformation has "
            "an empty shortest word."
        )

    if text.startswith("["):
        operations = tuple(
            json.loads(
                text
            )
        )
    else:
        operations = tuple(
            item.strip()
            for item in text.split(
                "->"
            )
            if item.strip()
        )

    if len(
        operations
    ) != word_length:
        raise AssertionError(
            "Shortest-word length mismatch: "
            f"{text}"
        )

    return operations


def load_semigroup():
    rows = load_csv(
        PHASE1A_DIR
        / "semigroup_elements.csv"
    )

    output = []

    for row in rows:
        output.append(
            {
                "element_id":
                    row[
                        "element_id"
                    ],

                "mapping":
                    tuple(
                        int(value)
                        for value in json.loads(
                            row[
                                "mapping"
                            ]
                        )
                    ),

                "rank":
                    int(
                        row[
                            "rank"
                        ]
                    ),

                "word_length":
                    int(
                        row[
                            "word_length"
                        ]
                    ),

                "word":
                    parse_shortest_word(
                        row
                    ),
            }
        )

    output.sort(
        key=lambda row:
            row[
                "element_id"
            ]
    )

    if (
        len(output)
        != EXPECTED_TRANSFORMATION_COUNT
    ):
        raise AssertionError(
            "Semigroup size changed."
        )

    return output


def load_clean_prototypes():
    rows = load_csv(
        PHASE1C_DIR
        / "state_prototypes.csv"
    )

    rows.sort(
        key=lambda row:
            int(
                row["state"]
            )
    )

    dimension_names = [
        name
        for name in rows[0]
        if name.startswith("z")
    ]

    dimension_names.sort(
        key=lambda name:
            int(
                name[1:]
            )
    )

    prototypes = np.asarray(
        [
            [
                float(
                    row[name]
                )
                for name in dimension_names
            ]
            for row in rows
        ],
        dtype=np.float32,
    )

    if prototypes.shape != (
        STATE_COUNT,
        OBSERVATION_DIM,
    ):
        raise AssertionError(
            "Prototype shape changed."
        )

    return prototypes


def equality_mask(
    mapping,
):
    mapping = [
        int(value)
        for value in mapping
    ]

    mask = 0
    bit = 0

    for first in range(
        len(mapping)
    ):
        for second in range(
            first + 1,
            len(mapping),
        ):
            if (
                mapping[first]
                == mapping[second]
            ):
                mask |= (
                    1 << bit
                )

            bit += 1

    return mask


def partition_of(
    mapping,
):
    blocks = defaultdict(
        list
    )

    for source, destination in enumerate(
        mapping
    ):
        blocks[
            int(destination)
        ].append(
            int(source)
        )

    return tuple(
        sorted(
            tuple(block)
            for block in blocks.values()
        )
    )


def relation_from_masks(
    first_mask: int,
    second_mask: int,
):
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


def relation_vector(
    mappings,
):
    masks = [
        equality_mask(
            mapping
        )
        for mapping in mappings
    ]

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
                    masks[first],
                    masks[second],
                )
            )

    return np.asarray(
        relations,
        dtype=np.int8,
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
            "Relation vectors have "
            "different shapes."
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

    accuracy = float(
        np.trace(
            confusion
        )
        / confusion.sum()
    )

    class_metrics = {}
    recalls = []
    f1_values = []

    for code, name in enumerate(
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
            true_positive
            / support
            if support > 0
            else 0.0
        )

        precision = (
            true_positive
            / predicted_count
            if predicted_count > 0
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        recalls.append(
            recall
        )

        f1_values.append(
            f1
        )

        class_metrics[
            name
        ] = {
            "support":
                support,

            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "f1":
                float(
                    f1
                ),
        }

    class_counts = confusion.sum(
        axis=1
    )

    majority_baseline = float(
        class_counts.max()
        / class_counts.sum()
    )

    return {
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

        "majority_baseline_accuracy":
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


def best_latent_alignment(
    encoder_probabilities,
):
    """
    Find the one-to-one true-state to latent-state assignment
    that maximizes total encoder probability.

    This is evaluation-only label alignment.
    """

    probabilities = np.asarray(
        encoder_probabilities,
        dtype=np.float64,
    )

    if probabilities.shape != (
        STATE_COUNT,
        LATENT_STATE_COUNT,
    ):
        raise ValueError(
            "Encoder probability shape changed."
        )

    best_permutation = None
    best_score = -float(
        "inf"
    )

    for permutation in permutations(
        range(
            LATENT_STATE_COUNT
        )
    ):
        score = sum(
            probabilities[
                true_state,
                permutation[
                    true_state
                ],
            ]
            for true_state in range(
                STATE_COUNT
            )
        )

        if score > best_score:
            best_score = score
            best_permutation = (
                permutation
            )

    true_to_latent = np.asarray(
        best_permutation,
        dtype=np.int64,
    )

    latent_to_true = np.empty(
        LATENT_STATE_COUNT,
        dtype=np.int64,
    )

    for true_state, latent_state in enumerate(
        true_to_latent
    ):
        latent_to_true[
            latent_state
        ] = true_state

    assigned_probabilities = np.asarray(
        [
            probabilities[
                true_state,
                true_to_latent[
                    true_state
                ],
            ]
            for true_state in range(
                STATE_COUNT
            )
        ],
        dtype=np.float64,
    )

    top1_latent = np.argmax(
        probabilities,
        axis=1,
    )

    return {
        "true_to_latent":
            true_to_latent,

        "latent_to_true":
            latent_to_true,

        "mean_assigned_probability":
            float(
                assigned_probabilities.mean()
            ),

        "minimum_assigned_probability":
            float(
                assigned_probabilities.min()
            ),

        "unique_top1_latent_count":
            int(
                len(
                    set(
                        int(value)
                        for value
                        in top1_latent
                    )
                )
            ),
    }


def compose_hard_word(
    primitive_maps,
    word,
):
    current = np.arange(
        LATENT_STATE_COUNT,
        dtype=np.int64,
    )

    for operation in word:
        current = np.asarray(
            primitive_maps[
                operation
            ],
            dtype=np.int64,
        )[
            current
        ]

    return tuple(
        int(value)
        for value in current
    )


def align_latent_mapping(
    latent_mapping,
    true_to_latent,
    latent_to_true,
):
    aligned = []

    for true_source in range(
        STATE_COUNT
    ):
        latent_source = int(
            true_to_latent[
                true_source
            ]
        )

        latent_destination = int(
            latent_mapping[
                latent_source
            ]
        )

        true_destination = int(
            latent_to_true[
                latent_destination
            ]
        )

        aligned.append(
            true_destination
        )

    return tuple(
        aligned
    )


@torch.no_grad()
def probe_word(
    model,
    prototype_tensor,
    operation_vocabulary,
    word,
    temperature,
):
    """
    Apply the soft learned channels to encoded clean prototypes,
    decode, and classify by nearest clean prototype.
    """

    current = model.encode(
        prototype_tensor,
        temperature=temperature,
    )

    for operation in word:
        operation_id = (
            operation_vocabulary[
                operation
            ]
        )

        operation_ids = torch.full(
            (
                STATE_COUNT,
            ),
            fill_value=operation_id,
            dtype=torch.long,
            device=prototype_tensor.device,
        )

        current = model.apply_operation(
            latent_distribution=current,
            operation_ids=operation_ids,
            temperature=temperature,
        )

    decoded = model.decode(
        current
    )

    differences = (
        decoded[
            :,
            None,
            :
        ]
        - prototype_tensor[
            None,
            :,
            :
        ]
    )

    squared_distances = (
        differences
        * differences
    ).sum(
        dim=-1
    )

    prediction = torch.argmin(
        squared_distances,
        dim=1,
    )

    return tuple(
        int(value)
        for value in prediction
        .detach()
        .cpu()
        .tolist()
    )


def evaluate_mapping_family(
    true_rows,
    predicted_mappings,
    true_relations,
):
    transformation_rows = []

    correct_entries = 0
    total_entries = 0
    exact_count = 0
    partition_count = 0
    rank_count = 0

    for true_row, predicted in zip(
        true_rows,
        predicted_mappings,
    ):
        truth = true_row[
            "mapping"
        ]

        entry_correct = sum(
            int(
                predicted[index]
                == truth[index]
            )
            for index in range(
                STATE_COUNT
            )
        )

        mapping_accuracy = (
            entry_correct
            / STATE_COUNT
        )

        exact = bool(
            tuple(predicted)
            == tuple(truth)
        )

        partition_correct = bool(
            partition_of(
                predicted
            )
            == partition_of(
                truth
            )
        )

        predicted_rank = len(
            set(
                predicted
            )
        )

        rank_correct = bool(
            predicted_rank
            == true_row[
                "rank"
            ]
        )

        correct_entries += (
            entry_correct
        )

        total_entries += (
            STATE_COUNT
        )

        exact_count += int(
            exact
        )

        partition_count += int(
            partition_correct
        )

        rank_count += int(
            rank_correct
        )

        transformation_rows.append(
            {
                "element_id":
                    true_row[
                        "element_id"
                    ],

                "true_rank":
                    true_row[
                        "rank"
                    ],

                "predicted_rank":
                    predicted_rank,

                "mapping_accuracy":
                    mapping_accuracy,

                "exact_mapping":
                    exact,

                "partition_correct":
                    partition_correct,

                "rank_correct":
                    rank_correct,

                "true_mapping":
                    json.dumps(
                        list(
                            truth
                        ),
                        separators=(
                            ",",
                            ":",
                        ),
                    ),

                "predicted_mapping":
                    json.dumps(
                        list(
                            predicted
                        ),
                        separators=(
                            ",",
                            ":",
                        ),
                    ),
            }
        )

    predicted_relations = (
        relation_vector(
            predicted_mappings
        )
    )

    relation_result = (
        relation_metrics(
            true_relations,
            predicted_relations,
        )
    )

    return {
        "mapping_accuracy":
            float(
                correct_entries
                / total_entries
            ),

        "exact_transformation_rate":
            float(
                exact_count
                / len(
                    true_rows
                )
            ),

        "partition_accuracy":
            float(
                partition_count
                / len(
                    true_rows
                )
            ),

        "rank_accuracy":
            float(
                rank_count
                / len(
                    true_rows
                )
            ),

        "distinct_predicted_transformation_count":
            int(
                len(
                    set(
                        tuple(mapping)
                        for mapping
                        in predicted_mappings
                    )
                )
            ),

        "distinct_predicted_partition_count":
            int(
                len(
                    set(
                        partition_of(
                            mapping
                        )
                        for mapping
                        in predicted_mappings
                    )
                )
            ),

        "relation_metrics":
            relation_result,

        "transformation_rows":
            transformation_rows,
    }


def primitive_metrics(
    true_generators,
    predicted_generators,
):
    rows = []

    correct = 0
    total = 0
    exact = 0

    for operation in true_generators:
        truth = tuple(
            true_generators[
                operation
            ]
        )

        prediction = tuple(
            predicted_generators[
                operation
            ]
        )

        entry_correct = sum(
            int(
                truth[index]
                == prediction[index]
            )
            for index in range(
                STATE_COUNT
            )
        )

        mapping_accuracy = (
            entry_correct
            / STATE_COUNT
        )

        is_exact = bool(
            truth
            == prediction
        )

        correct += (
            entry_correct
        )

        total += (
            STATE_COUNT
        )

        exact += int(
            is_exact
        )

        rows.append(
            {
                "operation":
                    operation,

                "mapping_accuracy":
                    mapping_accuracy,

                "exact_mapping":
                    is_exact,

                "true_mapping":
                    json.dumps(
                        list(
                            truth
                        ),
                        separators=(
                            ",",
                            ":",
                        ),
                    ),

                "predicted_mapping":
                    json.dumps(
                        list(
                            prediction
                        ),
                        separators=(
                            ",",
                            ":",
                        ),
                    ),
            }
        )

    return {
        "mapping_accuracy":
            float(
                correct
                / total
            ),

        "exact_operation_rate":
            float(
                exact
                / len(
                    true_generators
                )
            ),

        "rows":
            rows,
    }


def summarize_values(
    values,
):
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


def aggregate_by_noise(
    run_rows,
    seeds,
):
    metrics = [
        "alignment_mean_probability",
        "alignment_unique_top1_count",
        "mean_channel_row_entropy",
        "mean_channel_row_max_probability",

        "hard_primitive_mapping_accuracy",
        "hard_primitive_exact_rate",
        "hard_transformation_mapping_accuracy",
        "hard_exact_transformation_rate",
        "hard_partition_accuracy",
        "hard_rank_accuracy",
        "hard_relation_accuracy",
        "hard_relation_balanced_accuracy",
        "hard_relation_macro_f1",

        "probe_primitive_mapping_accuracy",
        "probe_primitive_exact_rate",
        "probe_transformation_mapping_accuracy",
        "probe_exact_transformation_rate",
        "probe_partition_accuracy",
        "probe_rank_accuracy",
        "probe_relation_accuracy",
        "probe_relation_balanced_accuracy",
        "probe_relation_macro_f1",
    ]

    output = {}

    for noise in FROZEN_NOISE_LEVELS:
        group = [
            row
            for row in run_rows
            if (
                row[
                    "seed"
                ] in seeds
                and np.isclose(
                    row[
                        "noise_fraction"
                    ],
                    noise,
                )
            )
        ]

        if len(group) != len(
            seeds
        ):
            raise AssertionError(
                f"Noise {noise} has an "
                "incorrect seed count."
            )

        item = {
            "seed_count":
                len(
                    group
                )
        }

        for metric in metrics:
            item[
                metric
            ] = summarize_values(
                [
                    row[
                        metric
                    ]
                    for row in group
                ]
            )

        output[
            str(
                noise
            )
        ] = item

    return output


def main():
    arguments = parse_arguments()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_DETAIL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        arguments.device
    )

    operation_vocabulary = (
        load_operation_vocabulary()
    )

    true_generators = (
        load_generators()
    )

    semigroup_rows = (
        load_semigroup()
    )

    clean_prototypes = (
        load_clean_prototypes()
    )

    prototype_tensor = torch.tensor(
        clean_prototypes,
        dtype=torch.float32,
        device=device,
    )

    true_mappings = [
        row[
            "mapping"
        ]
        for row in semigroup_rows
    ]

    true_relations = (
        relation_vector(
            true_mappings
        )
    )

    if (
        len(true_relations)
        != EXPECTED_RELATION_PAIR_COUNT
    ):
        raise AssertionError(
            "Relation-pair count changed."
        )

    relation_counts = Counter(
        CODE_TO_RELATION[
            int(value)
        ]
        for value in true_relations
    )

    if dict(
        relation_counts
    ) != EXPECTED_RELATION_COUNTS:
        raise AssertionError(
            "Ground-truth relation counts changed: "
            f"{dict(relation_counts)}"
        )

    run_rows = []
    primitive_rows = []
    transformation_rows = []

    for seed in FROZEN_SEEDS:

        for noise in FROZEN_NOISE_LEVELS:

            run_name = (
                f"seed_{seed}"
                f"_noise_{noise_slug(noise)}"
            )

            run_dir = (
                PHASE2D_DIR
                / "runs"
                / run_name
            )

            result_path = (
                run_dir
                / "result.json"
            )

            checkpoint_path = (
                run_dir
                / "best_model.pt"
            )

            if not result_path.exists():
                raise FileNotFoundError(
                    result_path
                )

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    checkpoint_path
                )

            phase2d_result = (
                load_json(
                    result_path
                )
            )

            if (
                phase2d_result[
                    "configuration_id"
                ]
                != EXPECTED_CONFIGURATION_ID
            ):
                raise AssertionError(
                    f"{run_name} used the "
                    "wrong configuration."
                )

            if (
                phase2d_result[
                    "test_used_for_checkpoint_selection"
                ]
                is not False
            ):
                raise AssertionError(
                    f"Test leakage in {run_name}."
                )

            if (
                phase2d_result[
                    "numerically_valid"
                ]
                is not True
            ):
                raise AssertionError(
                    f"{run_name} is invalid."
                )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            model = OperationChannelModel(
                observation_dim=(
                    OBSERVATION_DIM
                ),
                latent_state_count=(
                    LATENT_STATE_COUNT
                ),
                operation_count=len(
                    operation_vocabulary
                ),
            ).to(
                device
            )

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            model.eval()

            if not all(
                torch.isfinite(
                    parameter
                ).all().item()
                for parameter
                in model.parameters()
            ):
                raise FloatingPointError(
                    f"Non-finite parameters in {run_name}."
                )

            temperature = float(
                checkpoint[
                    "temperature"
                ]
            )

            with torch.no_grad():

                encoder_probabilities = (
                    model.encode(
                        prototype_tensor,
                        temperature=temperature,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                channels = (
                    model.operation_channels(
                        temperature=temperature
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

            alignment = (
                best_latent_alignment(
                    encoder_probabilities
                )
            )

            true_to_latent = alignment[
                "true_to_latent"
            ]

            latent_to_true = alignment[
                "latent_to_true"
            ]

            hard_primitive_latent = {}

            for operation, operation_id in (
                operation_vocabulary.items()
            ):
                hard_primitive_latent[
                    operation
                ] = tuple(
                    int(value)
                    for value in np.argmax(
                        channels[
                            operation_id
                        ],
                        axis=1,
                    )
                )

            hard_primitive_true = {}

            for operation, mapping in (
                hard_primitive_latent.items()
            ):
                hard_primitive_true[
                    operation
                ] = align_latent_mapping(
                    latent_mapping=mapping,
                    true_to_latent=(
                        true_to_latent
                    ),
                    latent_to_true=(
                        latent_to_true
                    ),
                )

            probe_primitive_true = {
                operation:
                    probe_word(
                        model=model,
                        prototype_tensor=(
                            prototype_tensor
                        ),
                        operation_vocabulary=(
                            operation_vocabulary
                        ),
                        word=(
                            operation,
                        ),
                        temperature=temperature,
                    )
                for operation in true_generators
            }

            hard_all_mappings = []
            probe_all_mappings = []

            for semigroup_row in (
                semigroup_rows
            ):
                latent_mapping = (
                    compose_hard_word(
                        primitive_maps=(
                            hard_primitive_latent
                        ),
                        word=semigroup_row[
                            "word"
                        ],
                    )
                )

                hard_all_mappings.append(
                    align_latent_mapping(
                        latent_mapping=(
                            latent_mapping
                        ),
                        true_to_latent=(
                            true_to_latent
                        ),
                        latent_to_true=(
                            latent_to_true
                        ),
                    )
                )

                probe_all_mappings.append(
                    probe_word(
                        model=model,
                        prototype_tensor=(
                            prototype_tensor
                        ),
                        operation_vocabulary=(
                            operation_vocabulary
                        ),
                        word=semigroup_row[
                            "word"
                        ],
                        temperature=temperature,
                    )
                )

            hard_primitive_result = (
                primitive_metrics(
                    true_generators=(
                        true_generators
                    ),
                    predicted_generators=(
                        hard_primitive_true
                    ),
                )
            )

            probe_primitive_result = (
                primitive_metrics(
                    true_generators=(
                        true_generators
                    ),
                    predicted_generators=(
                        probe_primitive_true
                    ),
                )
            )

            hard_result = (
                evaluate_mapping_family(
                    true_rows=(
                        semigroup_rows
                    ),
                    predicted_mappings=(
                        hard_all_mappings
                    ),
                    true_relations=(
                        true_relations
                    ),
                )
            )

            probe_result = (
                evaluate_mapping_family(
                    true_rows=(
                        semigroup_rows
                    ),
                    predicted_mappings=(
                        probe_all_mappings
                    ),
                    true_relations=(
                        true_relations
                    ),
                )
            )

            safe_channels = np.clip(
                channels,
                1e-12,
                1.0,
            )

            row_entropy = -(
                safe_channels
                * np.log(
                    safe_channels
                )
            ).sum(
                axis=2
            )

            row_maximum = channels.max(
                axis=2
            )

            hard_relation = hard_result[
                "relation_metrics"
            ]

            probe_relation = probe_result[
                "relation_metrics"
            ]

            run_row = {
                "seed":
                    int(
                        seed
                    ),

                "noise_fraction":
                    float(
                        noise
                    ),

                "temperature":
                    temperature,

                "alignment_mean_probability":
                    alignment[
                        "mean_assigned_probability"
                    ],

                "alignment_minimum_probability":
                    alignment[
                        "minimum_assigned_probability"
                    ],

                "alignment_unique_top1_count":
                    alignment[
                        "unique_top1_latent_count"
                    ],

                "mean_channel_row_entropy":
                    float(
                        row_entropy.mean()
                    ),

                "mean_channel_row_max_probability":
                    float(
                        row_maximum.mean()
                    ),

                "hard_primitive_mapping_accuracy":
                    hard_primitive_result[
                        "mapping_accuracy"
                    ],

                "hard_primitive_exact_rate":
                    hard_primitive_result[
                        "exact_operation_rate"
                    ],

                "hard_transformation_mapping_accuracy":
                    hard_result[
                        "mapping_accuracy"
                    ],

                "hard_exact_transformation_rate":
                    hard_result[
                        "exact_transformation_rate"
                    ],

                "hard_partition_accuracy":
                    hard_result[
                        "partition_accuracy"
                    ],

                "hard_rank_accuracy":
                    hard_result[
                        "rank_accuracy"
                    ],

                "hard_relation_accuracy":
                    hard_relation[
                        "accuracy"
                    ],

                "hard_relation_balanced_accuracy":
                    hard_relation[
                        "balanced_accuracy"
                    ],

                "hard_relation_macro_f1":
                    hard_relation[
                        "macro_f1"
                    ],

                "hard_relation_accuracy_above_majority":
                    hard_relation[
                        "accuracy_above_majority"
                    ],

                "probe_primitive_mapping_accuracy":
                    probe_primitive_result[
                        "mapping_accuracy"
                    ],

                "probe_primitive_exact_rate":
                    probe_primitive_result[
                        "exact_operation_rate"
                    ],

                "probe_transformation_mapping_accuracy":
                    probe_result[
                        "mapping_accuracy"
                    ],

                "probe_exact_transformation_rate":
                    probe_result[
                        "exact_transformation_rate"
                    ],

                "probe_partition_accuracy":
                    probe_result[
                        "partition_accuracy"
                    ],

                "probe_rank_accuracy":
                    probe_result[
                        "rank_accuracy"
                    ],

                "probe_relation_accuracy":
                    probe_relation[
                        "accuracy"
                    ],

                "probe_relation_balanced_accuracy":
                    probe_relation[
                        "balanced_accuracy"
                    ],

                "probe_relation_macro_f1":
                    probe_relation[
                        "macro_f1"
                    ],

                "probe_relation_accuracy_above_majority":
                    probe_relation[
                        "accuracy_above_majority"
                    ],
            }

            for relation_name in RELATIONS:
                run_row[
                    "hard_recall_"
                    + relation_name
                ] = hard_relation[
                    "class_metrics"
                ][
                    relation_name
                ][
                    "recall"
                ]

                run_row[
                    "probe_recall_"
                    + relation_name
                ] = probe_relation[
                    "class_metrics"
                ][
                    relation_name
                ][
                    "recall"
                ]

            run_rows.append(
                run_row
            )

            for family, result in [
                (
                    "hard",
                    hard_primitive_result,
                ),
                (
                    "probe",
                    probe_primitive_result,
                ),
            ]:
                for row in result[
                    "rows"
                ]:
                    primitive_rows.append(
                        {
                            "seed":
                                seed,

                            "noise_fraction":
                                noise,

                            "estimator":
                                family,

                            **row,
                        }
                    )

            for family, result in [
                (
                    "hard",
                    hard_result,
                ),
                (
                    "probe",
                    probe_result,
                ),
            ]:
                for row in result[
                    "transformation_rows"
                ]:
                    transformation_rows.append(
                        {
                            "seed":
                                seed,

                            "noise_fraction":
                                noise,

                            "estimator":
                                family,

                            **row,
                        }
                    )

            detail = {
                "run_name":
                    run_name,

                "seed":
                    seed,

                "noise_fraction":
                    noise,

                "checkpoint_temperature":
                    temperature,

                "alignment": {
                    "true_to_latent":
                        [
                            int(value)
                            for value in true_to_latent
                        ],

                    "latent_to_true":
                        [
                            int(value)
                            for value in latent_to_true
                        ],

                    "mean_assigned_probability":
                        alignment[
                            "mean_assigned_probability"
                        ],

                    "minimum_assigned_probability":
                        alignment[
                            "minimum_assigned_probability"
                        ],

                    "unique_top1_latent_count":
                        alignment[
                            "unique_top1_latent_count"
                        ],
                },

                "hard_relation_metrics":
                    hard_relation,

                "probe_relation_metrics":
                    probe_relation,

                "training_performed":
                    False,

                "checkpoint_modified":
                    False,

                "structural_ground_truth_use":
                    "evaluation only",
            }

            write_json(
                RUN_DETAIL_DIR
                / f"{run_name}.json",
                detail,
            )

            print(
                f"{run_name} | "
                f"hard relation="
                f"{run_row['hard_relation_accuracy']:.4f} | "
                f"probe relation="
                f"{run_row['probe_relation_accuracy']:.4f}"
            )

    if len(
        run_rows
    ) != 25:
        raise AssertionError(
            "Expected exactly 25 structural evaluations."
        )

    write_csv(
        OUTPUT_DIR
        / "all_run_structural_metrics.csv",
        run_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "primitive_operation_metrics.csv",
        primitive_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "transformation_metrics.csv",
        transformation_rows,
    )

    primary_summary = (
        aggregate_by_noise(
            run_rows=run_rows,
            seeds=FROZEN_SEEDS,
        )
    )

    sensitivity_seeds = tuple(
        seed
        for seed in FROZEN_SEEDS
        if seed != TUNING_SEED
    )

    sensitivity_summary = (
        aggregate_by_noise(
            run_rows=run_rows,
            seeds=sensitivity_seeds,
        )
    )

    summary = {
        "phase":
            "2E blind structural recovery",

        "configuration_id":
            EXPECTED_CONFIGURATION_ID,

        "evaluated_checkpoint_count":
            len(
                run_rows
            ),

        "seeds":
            list(
                FROZEN_SEEDS
            ),

        "noise_levels":
            list(
                FROZEN_NOISE_LEVELS
            ),

        "ordered_distinct_relation_pair_count":
            EXPECTED_RELATION_PAIR_COUNT,

        "relation_class_counts":
            EXPECTED_RELATION_COUNTS,

        "majority_relation_baseline":
            (
                EXPECTED_RELATION_COUNTS[
                    "incomparable"
                ]
                / EXPECTED_RELATION_PAIR_COUNT
            ),

        "primary_estimator": (
            "Encoder-based one-to-one latent alignment, "
            "row-argmax hardened primitive channels, "
            "and exact hard composition."
        ),

        "secondary_estimator": (
            "Clean-prototype encoding, soft channel composition, "
            "continuous decoding, and nearest-prototype classification."
        ),

        "primary_five_seed_results":
            primary_summary,

        "four_seed_sensitivity_excluding_tuning_seed":
            sensitivity_summary,

        "training_performed":
            False,

        "checkpoint_selection_reopened":
            False,

        "checkpoint_modified":
            False,

        "structural_ground_truth_used_for":
            "evaluation only",

        "phase2e_status":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase2e_summary.json",
        summary,
    )

    print()
    print(
        "Phase 2E structural recovery "
        "evaluation completed."
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
