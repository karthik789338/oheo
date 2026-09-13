from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from phase3c_run_tier_b_smoke_tests import (
    find_channel_logits,
    load_operation_vocabulary,
)

from phase3f_evaluate_final_prediction import (
    ALL_MODELS,
    CHECKPOINT_MODELS,
    FROZEN_NOISE_LEVELS,
    FROZEN_SEEDS,
    load_checkpoint_model,
    predict_batch,
    resolve_device,
)


PHASE1A_DIR = Path(
    "outputs/phase1a_ground_truth"
)

PHASE1D_DIR = Path(
    "outputs/phase1d_compositional_splits"
)

PHASE3B_DATA_DIR = Path(
    "outputs/phase3b_tier_b_data"
)

PRIVILEGED_DIR = (
    PHASE3B_DATA_DIR / "privileged"
)

PHASE3F_DIR = Path(
    "outputs/phase3f_tier_b_prediction"
)

OUTPUT_DIR = Path(
    "outputs/phase3g_tier_b_structure"
)

RUN_OUTPUT_DIR = (
    OUTPUT_DIR / "runs"
)


STATE_COUNT = 8
OBSERVATION_DIMENSION = 32
SEMIGROUP_SIZE = 104

CARRIER_SPLITS = (
    "train",
    "val",
    "test",
)

RELATION_NAMES = (
    "equivalent",
    "more",
    "less",
    "incomparable",
)

EXPECTED_CARRIER_COUNTS = {
    "train": 96,
    "val": 32,
    "test": 32,
}

EXPECTED_EVALUATION_RUN_COUNTS = {
    "B0": 5,
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
    "OCM": 25,
}

EXPECTED_EVALUATION_RUN_COUNT = 135
EXPECTED_BEHAVIORAL_METRIC_ROWS = 405
EXPECTED_OCM_HARD_ROWS = 25


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

    parser.add_argument(
        "--run-id",
        default="all",
    )

    parser.add_argument(
        "--model-id",
        choices=[
            "all",
            "B0",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "OCM",
        ],
        default="all",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
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

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

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


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot summarize an empty collection."
        )

    return {
        "mean":
            float(values.mean()),

        "std":
            (
                float(
                    values.std(ddof=1)
                )
                if len(values) > 1
                else 0.0
            ),

        "minimum":
            float(values.min()),

        "maximum":
            float(values.max()),
    }


def parse_json_or_delimited_word(
    value,
    word_length,
):
    if word_length == 0:
        return tuple()

    if isinstance(value, list):
        return tuple(
            str(item)
            for item in value
        )

    text = str(value).strip()

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return tuple(
                str(item)
                for item in parsed
            )

    except json.JSONDecodeError:
        pass

    if "->" in text:
        return tuple(
            part.strip()
            for part in text.split("->")
            if part.strip()
        )

    return tuple(
        part.strip()
        for part in text.split(",")
        if part.strip()
    )


def parse_mapping(value):
    if isinstance(value, list):
        mapping = value
    else:
        mapping = json.loads(
            str(value)
        )

    mapping = tuple(
        int(item)
        for item in mapping
    )

    if len(mapping) != STATE_COUNT:
        raise AssertionError(
            "Mapping has the wrong state count."
        )

    return mapping


def validate_sources():
    phase3f_summary = load_json(
        PHASE3F_DIR
        / "phase3f_summary.json"
    )

    evaluation_registry = load_json(
        PHASE3F_DIR
        / "evaluation_registry.json"
    )

    if (
        phase3f_summary[
            "phase3f_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 3F has not passed."
        )

    if (
        phase3f_summary[
            "evaluation_run_count"
        ]
        != EXPECTED_EVALUATION_RUN_COUNT
    ):
        raise AssertionError(
            "Phase 3F evaluation-run count changed."
        )

    if (
        phase3f_summary[
            "checkpoint_selection_reopened"
        ]
        is not False
    ):
        raise AssertionError(
            "Checkpoint selection was reopened."
        )

    if (
        phase3f_summary[
            "test_metrics_used_for_selection"
        ]
        is not False
    ):
        raise AssertionError(
            "Test metrics affected model selection."
        )

    if (
        evaluation_registry[
            "registry_status"
        ]
        != "frozen"
    ):
        raise AssertionError(
            "Phase 3F registry is not frozen."
        )

    runs = evaluation_registry[
        "runs"
    ]

    counts = Counter(
        row["model_id"]
        for row in runs
    )

    if dict(counts) != EXPECTED_EVALUATION_RUN_COUNTS:
        raise AssertionError(
            "Evaluation model counts changed: "
            f"{dict(counts)}"
        )

    return phase3f_summary, runs


def load_clean_manifold():
    path = (
        PRIVILEGED_DIR
        / "carrier_state_observations.npz"
    )

    with np.load(path) as data:
        carrier_ids = np.asarray(
            data["carrier_ids"],
            dtype=np.int64,
        )

        carrier_splits = np.asarray(
            data["carrier_splits"]
        ).astype(str)

        observations = np.asarray(
            data["normalized_observations"],
            dtype=np.float32,
        )

    if observations.shape != (
        160,
        STATE_COUNT,
        OBSERVATION_DIMENSION,
    ):
        raise AssertionError(
            "Tier B manifold shape changed: "
            f"{observations.shape}"
        )

    if not np.isfinite(
        observations
    ).all():
        raise FloatingPointError(
            "Tier B manifold contains NaN or Inf."
        )

    split_counts = Counter(
        carrier_splits.tolist()
    )

    if dict(split_counts) != EXPECTED_CARRIER_COUNTS:
        raise AssertionError(
            "Carrier split counts changed: "
            f"{dict(split_counts)}"
        )

    if not np.array_equal(
        carrier_ids,
        np.arange(
            len(carrier_ids),
            dtype=np.int64,
        ),
    ):
        raise AssertionError(
            "Carrier IDs are no longer contiguous."
        )

    return {
        "carrier_ids":
            carrier_ids,

        "carrier_splits":
            carrier_splits,

        "observations":
            observations,
    }


def load_semigroup():
    rows = load_csv(
        PHASE1A_DIR
        / "semigroup_elements.csv"
    )

    semigroup = []

    for row in rows:
        word_length = int(
            row["word_length"]
        )

        word = parse_json_or_delimited_word(
            row["shortest_word"],
            word_length,
        )

        if len(word) != word_length:
            raise AssertionError(
                "Semigroup word length changed."
            )

        semigroup.append(
            {
                "element_id":
                    str(row["element_id"]),

                "mapping":
                    parse_mapping(
                        row["mapping"]
                    ),

                "word":
                    word,

                "word_length":
                    word_length,
            }
        )

    if len(semigroup) != SEMIGROUP_SIZE:
        raise AssertionError(
            "Semigroup size changed."
        )

    return semigroup


def load_generators():
    raw = load_json(
        PHASE1A_DIR
        / "generators.json"
    )

    generators = {}

    if isinstance(raw, dict):
        for name, value in raw.items():
            mapping_value = (
                value["mapping"]
                if isinstance(value, dict)
                else value
            )

            generators[
                str(name)
            ] = tuple(
                int(item)
                for item in mapping_value
            )

    elif isinstance(raw, list):
        for row in raw:
            generators[
                str(row["name"])
            ] = tuple(
                int(item)
                for item in row["mapping"]
            )

    else:
        raise TypeError(
            "Unsupported generator format."
        )

    if len(generators) != 6:
        raise AssertionError(
            "Primitive-operation count changed."
        )

    return generators


def load_transformation_splits(
    semigroup,
):
    rows = load_csv(
        PHASE1D_DIR
        / "transformation_split.csv"
    )

    split_lookup = {
        str(row["element_id"]):
            row["split"]
        for row in rows
    }

    indices = {
        "train": [],
        "val": [],
        "test": [],
    }

    for index, transformation in enumerate(
        semigroup
    ):
        element_id = transformation[
            "element_id"
        ]

        if element_id not in split_lookup:
            raise AssertionError(
                f"Missing split for {element_id}."
            )

        split = split_lookup[
            element_id
        ]

        indices[
            split
        ].append(index)

    observed_counts = {
        split:
            len(values)
        for split, values in indices.items()
    }

    expected_counts = {
        "train": 86,
        "val": 9,
        "test": 9,
    }

    if observed_counts != expected_counts:
        raise AssertionError(
            "Transformation split counts changed: "
            f"{observed_counts}"
        )

    return indices


def find_primitive_indices(
    semigroup,
    generators,
):
    mapping_to_indices = defaultdict(
        list
    )

    for index, transformation in enumerate(
        semigroup
    ):
        mapping_to_indices[
            transformation["mapping"]
        ].append(index)

    primitive_indices = {}

    for operation, mapping in (
        generators.items()
    ):
        mapping = tuple(mapping)

        candidates = mapping_to_indices[
            mapping
        ]

        if not candidates:
            raise AssertionError(
                f"Primitive {operation} is absent "
                "from the semigroup."
            )

        primitive_indices[
            operation
        ] = min(
            candidates,
            key=lambda index: (
                semigroup[index][
                    "word_length"
                ],
                index,
            ),
        )

    if len(
        set(
            primitive_indices.values()
        )
    ) != len(
        primitive_indices
    ):
        raise AssertionError(
            "Primitive generators do not map to "
            "six distinct semigroup elements."
        )

    return primitive_indices


def partition_masks(
    mappings,
):
    mappings = np.asarray(
        mappings,
        dtype=np.int64,
    )

    return (
        mappings[..., :, None]
        == mappings[..., None, :]
    )


def mapping_ranks(
    mappings,
):
    mappings = np.asarray(
        mappings,
        dtype=np.int64,
    )

    present = np.stack(
        [
            np.any(
                mappings == state,
                axis=-1,
            )
            for state in range(
                STATE_COUNT
            )
        ],
        axis=-1,
    )

    return present.sum(
        axis=-1
    )


def relation_pair_indices(
    transformation_count,
    test_indices,
):
    first = []
    second = []

    for first_index in range(
        transformation_count
    ):
        for second_index in range(
            transformation_count
        ):
            if first_index == second_index:
                continue

            first.append(first_index)
            second.append(second_index)

    first = np.asarray(
        first,
        dtype=np.int64,
    )

    second = np.asarray(
        second,
        dtype=np.int64,
    )

    test_mask = np.zeros(
        transformation_count,
        dtype=bool,
    )

    test_mask[
        np.asarray(
            test_indices,
            dtype=np.int64,
        )
    ] = True

    return {
        "all": (
            first,
            second,
        ),

        "test_involved": (
            first[
                np.logical_or(
                    test_mask[first],
                    test_mask[second],
                )
            ],
            second[
                np.logical_or(
                    test_mask[first],
                    test_mask[second],
                )
            ],
        ),

        "test_test": (
            first[
                np.logical_and(
                    test_mask[first],
                    test_mask[second],
                )
            ],
            second[
                np.logical_and(
                    test_mask[first],
                    test_mask[second],
                )
            ],
        ),
    }


def relation_labels(
    mappings,
    first_indices,
    second_indices,
):
    masks = partition_masks(
        mappings
    ).reshape(
        len(mappings),
        -1,
    )

    first = masks[
        first_indices
    ]

    second = masks[
        second_indices
    ]

    equivalent = np.all(
        first == second,
        axis=1,
    )

    first_more = np.all(
        np.logical_or(
            ~first,
            second,
        ),
        axis=1,
    )

    second_more = np.all(
        np.logical_or(
            ~second,
            first,
        ),
        axis=1,
    )

    labels = np.full(
        len(first_indices),
        3,
        dtype=np.int8,
    )

    labels[
        second_more
        & ~equivalent
    ] = 2

    labels[
        first_more
        & ~equivalent
    ] = 1

    labels[
        equivalent
    ] = 0

    return labels


def classification_metrics(
    true_labels,
    predicted_labels,
):
    true_labels = np.asarray(
        true_labels,
        dtype=np.int64,
    )

    predicted_labels = np.asarray(
        predicted_labels,
        dtype=np.int64,
    )

    accuracy = float(
        np.mean(
            true_labels
            == predicted_labels
        )
    )

    counts = np.bincount(
        true_labels,
        minlength=len(
            RELATION_NAMES
        ),
    )

    recalls = []
    f1_values = []

    for label in range(
        len(
            RELATION_NAMES
        )
    ):
        true_positive = int(
            np.sum(
                (true_labels == label)
                & (predicted_labels == label)
            )
        )

        false_positive = int(
            np.sum(
                (true_labels != label)
                & (predicted_labels == label)
            )
        )

        false_negative = int(
            np.sum(
                (true_labels == label)
                & (predicted_labels != label)
            )
        )

        support = int(
            counts[label]
        )

        if support > 0:
            recalls.append(
                true_positive
                / support
            )

        precision = (
            true_positive
            / (
                true_positive
                + false_positive
            )
            if (
                true_positive
                + false_positive
            ) > 0
            else 0.0
        )

        recall = (
            true_positive
            / (
                true_positive
                + false_negative
            )
            if (
                true_positive
                + false_negative
            ) > 0
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

        f1_values.append(f1)

    return {
        "accuracy":
            accuracy,

        "balanced_accuracy":
            float(
                np.mean(recalls)
            ),

        "macro_f1":
            float(
                np.mean(f1_values)
            ),

        "majority_baseline":
            float(
                counts.max()
                / len(true_labels)
            ),
    }


def consensus_mapping(
    carrier_mappings,
):
    carrier_mappings = np.asarray(
        carrier_mappings,
        dtype=np.int64,
    )

    counts = np.stack(
        [
            np.sum(
                carrier_mappings
                == state,
                axis=0,
            )
            for state in range(
                STATE_COUNT
            )
        ],
        axis=-1,
    )

    return np.argmax(
        counts,
        axis=-1,
    ).astype(
        np.int8
    )


def mapping_family_metrics(
    predicted_by_carrier,
    true_mappings,
    selected_indices,
):
    predicted_by_carrier = np.asarray(
        predicted_by_carrier,
        dtype=np.int64,
    )

    true_mappings = np.asarray(
        true_mappings,
        dtype=np.int64,
    )

    selected = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    predicted = predicted_by_carrier[
        :,
        selected,
        :,
    ]

    true = true_mappings[
        selected,
        :
    ]

    mapping_accuracy = float(
        np.mean(
            predicted
            == true[
                None,
                :,
                :,
            ]
        )
    )

    exact = np.all(
        predicted
        == true[
            None,
            :,
            :,
        ],
        axis=-1,
    )

    predicted_partitions = (
        partition_masks(
            predicted
        )
    )

    true_partitions = (
        partition_masks(
            true
        )
    )

    partition_exact = np.all(
        predicted_partitions
        == true_partitions[
            None,
            :,
            :,
            :,
        ],
        axis=(-1, -2),
    )

    predicted_rank = mapping_ranks(
        predicted
    )

    true_rank = mapping_ranks(
        true
    )

    rank_exact = (
        predicted_rank
        == true_rank[
            None,
            :,
        ]
    )

    consensus = consensus_mapping(
        predicted_by_carrier
    )

    consensus_selected = consensus[
        selected
    ]

    consensus_exact = np.all(
        consensus_selected
        == true,
        axis=-1,
    )

    consensus_partition_exact = np.all(
        partition_masks(
            consensus_selected
        )
        == true_partitions,
        axis=(-1, -2),
    )

    consensus_rank_exact = (
        mapping_ranks(
            consensus_selected
        )
        == true_rank
    )

    carrier_consistency = float(
        np.mean(
            predicted
            == consensus_selected[
                None,
                :,
                :,
            ]
        )
    )

    return {
        "mapping_accuracy":
            mapping_accuracy,

        "exact_transformation_rate":
            float(
                exact.mean()
            ),

        "partition_accuracy":
            float(
                partition_exact.mean()
            ),

        "rank_accuracy":
            float(
                rank_exact.mean()
            ),

        "consensus_mapping_accuracy":
            float(
                np.mean(
                    consensus_selected
                    == true
                )
            ),

        "consensus_exact_transformation_rate":
            float(
                consensus_exact.mean()
            ),

        "consensus_partition_accuracy":
            float(
                consensus_partition_exact.mean()
            ),

        "consensus_rank_accuracy":
            float(
                consensus_rank_exact.mean()
            ),

        "carrier_consistency":
            carrier_consistency,

        "consensus_mapping":
            consensus,
    }


def relation_scope_metrics(
    predicted_consensus,
    true_mappings,
    pair_scopes,
):
    output = {}

    for scope, (
        first_indices,
        second_indices,
    ) in pair_scopes.items():
        true_labels = relation_labels(
            mappings=true_mappings,
            first_indices=first_indices,
            second_indices=second_indices,
        )

        predicted_labels = relation_labels(
            mappings=predicted_consensus,
            first_indices=first_indices,
            second_indices=second_indices,
        )

        metrics = classification_metrics(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
        )

        output[scope] = {
            "pair_count":
                len(first_indices),

            **metrics,
        }

    return output


@torch.no_grad()
def evaluate_behavioral_mappings(
    run,
    model_bundle,
    semigroup,
    operation_vocabulary,
    clean_observations,
    device,
):
    model = model_bundle[
        "model"
    ]

    temperature = model_bundle[
        "temperature"
    ]

    clean_tensor = torch.from_numpy(
        clean_observations
    ).to(
        device=device,
        dtype=torch.float32,
    )

    carrier_count = clean_tensor.shape[0]

    initial = clean_tensor.reshape(
        carrier_count
        * STATE_COUNT,
        OBSERVATION_DIMENSION,
    )

    predicted_mappings = np.empty(
        (
            carrier_count,
            len(semigroup),
            STATE_COUNT,
        ),
        dtype=np.int8,
    )

    for transformation_index, transformation in enumerate(
        semigroup
    ):
        word = transformation[
            "word"
        ]

        if len(word) == 0:
            prediction = initial

        else:
            operation_ids = torch.tensor(
                [
                    operation_vocabulary[
                        operation
                    ]
                    for operation in word
                ],
                dtype=torch.long,
                device=device,
            ).unsqueeze(
                0
            ).repeat(
                len(initial),
                1,
            )

            step_mask = torch.ones(
                operation_ids.shape,
                dtype=torch.bool,
                device=device,
            )

            prediction = predict_batch(
                model_id=run[
                    "model_id"
                ],
                model=model,
                observations=initial.unsqueeze(
                    1
                ),
                operation_ids=operation_ids,
                step_mask=step_mask,
                temperature=temperature,
            )

        if not torch.isfinite(
            prediction
        ).all():
            raise FloatingPointError(
                f"{run['run_id']} produced "
                "non-finite structural predictions."
            )

        prediction = prediction.reshape(
            carrier_count,
            STATE_COUNT,
            OBSERVATION_DIMENSION,
        )

        distances = (
            prediction[
                :,
                :,
                None,
                :,
            ]
            - clean_tensor[
                :,
                None,
                :,
                :,
            ]
        ).pow(
            2
        ).sum(
            dim=-1
        )

        decoded = torch.argmin(
            distances,
            dim=-1,
        )

        predicted_mappings[
            :,
            transformation_index,
            :,
        ] = (
            decoded.detach()
            .cpu()
            .numpy()
            .astype(
                np.int8
            )
        )

    return predicted_mappings


def best_state_assignment(
    confusion,
):
    best_score = -1
    best_permutation = None

    for permutation in itertools.permutations(
        range(
            STATE_COUNT
        )
    ):
        score = sum(
            int(
                confusion[
                    true_state,
                    permutation[
                        true_state
                    ],
                ]
            )
            for true_state in range(
                STATE_COUNT
            )
        )

        if score > best_score:
            best_score = score
            best_permutation = permutation

    if best_permutation is None:
        raise RuntimeError(
            "Could not align OCM latent states."
        )

    return (
        np.asarray(
            best_permutation,
            dtype=np.int64,
        ),
        int(best_score),
    )


@torch.no_grad()
def align_ocm_states(
    model,
    temperature,
    clean_observations,
    carrier_splits,
    device,
):
    train_indices = np.where(
        carrier_splits == "train"
    )[0]

    train = torch.from_numpy(
        clean_observations[
            train_indices
        ]
    ).to(
        device=device,
        dtype=torch.float32,
    )

    encoded = model.encode(
        train.reshape(
            -1,
            OBSERVATION_DIMENSION,
        ),
        temperature=temperature,
    ).reshape(
        len(train_indices),
        STATE_COUNT,
        STATE_COUNT,
    )

    predicted_latent = torch.argmax(
        encoded,
        dim=-1,
    ).detach().cpu().numpy()

    confusion = np.zeros(
        (
            STATE_COUNT,
            STATE_COUNT,
        ),
        dtype=np.int64,
    )

    for true_state in range(
        STATE_COUNT
    ):
        counts = np.bincount(
            predicted_latent[
                :,
                true_state,
            ],
            minlength=STATE_COUNT,
        )

        confusion[
            true_state
        ] = counts

    true_to_latent, assignment_score = (
        best_state_assignment(
            confusion
        )
    )

    latent_to_true = np.empty(
        STATE_COUNT,
        dtype=np.int64,
    )

    for true_state, latent_state in enumerate(
        true_to_latent
    ):
        latent_to_true[
            latent_state
        ] = true_state

    split_accuracy = {}

    for split in CARRIER_SPLITS:
        indices = np.where(
            carrier_splits == split
        )[0]

        values = torch.from_numpy(
            clean_observations[
                indices
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        encoded = model.encode(
            values.reshape(
                -1,
                OBSERVATION_DIMENSION,
            ),
            temperature=temperature,
        ).reshape(
            len(indices),
            STATE_COUNT,
            STATE_COUNT,
        )

        latent_predictions = torch.argmax(
            encoded,
            dim=-1,
        ).detach().cpu().numpy()

        decoded_true = latent_to_true[
            latent_predictions
        ]

        true_labels = np.broadcast_to(
            np.arange(
                STATE_COUNT,
                dtype=np.int64,
            )[None, :],
            decoded_true.shape,
        )

        split_accuracy[
            split
        ] = float(
            np.mean(
                decoded_true
                == true_labels
            )
        )

    return {
        "true_to_latent":
            true_to_latent,

        "latent_to_true":
            latent_to_true,

        "confusion":
            confusion,

        "training_assignment_score":
            assignment_score,

        "training_assignment_accuracy":
            float(
                assignment_score
                / (
                    len(train_indices)
                    * STATE_COUNT
                )
            ),

        "split_alignment_accuracy":
            split_accuracy,
    }


def compose_operation_mappings(
    semigroup,
    primitive_mappings,
):
    output = []

    for transformation in semigroup:
        current = np.arange(
            STATE_COUNT,
            dtype=np.int64,
        )

        for operation in transformation[
            "word"
        ]:
            current = primitive_mappings[
                operation
            ][current]

        output.append(
            tuple(
                int(item)
                for item in current
            )
        )

    return np.asarray(
        output,
        dtype=np.int8,
    )


@torch.no_grad()
def evaluate_ocm_hard_structure(
    model,
    temperature,
    operation_count,
    operation_vocabulary,
    semigroup,
    clean_observations,
    carrier_splits,
    true_mappings,
    transformation_indices,
    primitive_indices,
    pair_scopes,
    device,
):
    alignment = align_ocm_states(
        model=model,
        temperature=temperature,
        clean_observations=clean_observations,
        carrier_splits=carrier_splits,
        device=device,
    )

    _, logits = find_channel_logits(
        model=model,
        operation_count=operation_count,
    )

    if logits is None:
        raise RuntimeError(
            "Could not locate OCM channel logits."
        )

    probabilities = torch.softmax(
        logits / temperature,
        dim=-1,
    )

    maximum_row_sum_error = float(
        (
            probabilities.sum(
                dim=-1
            )
            - 1.0
        ).abs().max().item()
    )

    if maximum_row_sum_error >= 1e-6:
        raise AssertionError(
            "OCM channel rows are not stochastic."
        )

    latent_operation_mappings = (
        torch.argmax(
            probabilities,
            dim=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )

    true_to_latent = alignment[
        "true_to_latent"
    ]

    latent_to_true = alignment[
        "latent_to_true"
    ]

    primitive_hard_mappings = {}

    for operation, operation_id in (
        operation_vocabulary.items()
    ):
        mapping = np.empty(
            STATE_COUNT,
            dtype=np.int64,
        )

        for true_state in range(
            STATE_COUNT
        ):
            current_latent = (
                true_to_latent[
                    true_state
                ]
            )

            next_latent = (
                latent_operation_mappings[
                    operation_id,
                    current_latent,
                ]
            )

            mapping[
                true_state
            ] = latent_to_true[
                next_latent
            ]

        primitive_hard_mappings[
            operation
        ] = mapping

    hard_mappings = compose_operation_mappings(
        semigroup=semigroup,
        primitive_mappings=(
            primitive_hard_mappings
        ),
    )

    as_single_carrier = hard_mappings[
        None,
        :,
        :,
    ]

    primitive_metrics = mapping_family_metrics(
        predicted_by_carrier=(
            as_single_carrier
        ),
        true_mappings=true_mappings,
        selected_indices=list(
            primitive_indices.values()
        ),
    )

    all_metrics = mapping_family_metrics(
        predicted_by_carrier=(
            as_single_carrier
        ),
        true_mappings=true_mappings,
        selected_indices=(
            transformation_indices[
                "train"
            ]
            + transformation_indices[
                "val"
            ]
            + transformation_indices[
                "test"
            ]
        ),
    )

    test_metrics = mapping_family_metrics(
        predicted_by_carrier=(
            as_single_carrier
        ),
        true_mappings=true_mappings,
        selected_indices=(
            transformation_indices[
                "test"
            ]
        ),
    )

    relations = relation_scope_metrics(
        predicted_consensus=hard_mappings,
        true_mappings=true_mappings,
        pair_scopes=pair_scopes,
    )

    return {
        "hard_mappings":
            hard_mappings,

        "primitive_metrics":
            primitive_metrics,

        "all_metrics":
            all_metrics,

        "test_metrics":
            test_metrics,

        "relation_metrics":
            relations,

        "alignment":
            alignment,

        "maximum_channel_row_sum_error":
            maximum_row_sum_error,
    }


def evaluate_run(
    run,
    semigroup,
    true_mappings,
    transformation_indices,
    primitive_indices,
    pair_scopes,
    clean_observations,
    carrier_splits,
    operation_vocabulary,
    device,
    run_dir,
):
    operation_count = len(
        operation_vocabulary
    )

    model_bundle = load_checkpoint_model(
        run=run,
        operation_count=operation_count,
        device=device,
    )

    predicted_mappings = (
        evaluate_behavioral_mappings(
            run=run,
            model_bundle=model_bundle,
            semigroup=semigroup,
            operation_vocabulary=(
                operation_vocabulary
            ),
            clean_observations=(
                clean_observations
            ),
            device=device,
        )
    )

    behavioral_rows = []
    consensus_arrays = {}

    for split in CARRIER_SPLITS:
        carrier_indices = np.where(
            carrier_splits == split
        )[0]

        split_mappings = predicted_mappings[
            carrier_indices
        ]

        primitive_metrics = (
            mapping_family_metrics(
                predicted_by_carrier=(
                    split_mappings
                ),
                true_mappings=true_mappings,
                selected_indices=list(
                    primitive_indices.values()
                ),
            )
        )

        all_metrics = mapping_family_metrics(
            predicted_by_carrier=(
                split_mappings
            ),
            true_mappings=true_mappings,
            selected_indices=(
                transformation_indices[
                    "train"
                ]
                + transformation_indices[
                    "val"
                ]
                + transformation_indices[
                    "test"
                ]
            ),
        )

        test_metrics = mapping_family_metrics(
            predicted_by_carrier=(
                split_mappings
            ),
            true_mappings=true_mappings,
            selected_indices=(
                transformation_indices[
                    "test"
                ]
            ),
        )

        predicted_consensus = (
            all_metrics[
                "consensus_mapping"
            ]
        )

        consensus_arrays[
            split
        ] = predicted_consensus

        relations = relation_scope_metrics(
            predicted_consensus=(
                predicted_consensus
            ),
            true_mappings=true_mappings,
            pair_scopes=pair_scopes,
        )

        behavioral_rows.append(
            {
                "run_id":
                    run["run_id"],

                "model_id":
                    run["model_id"],

                "configuration_id":
                    run[
                        "configuration_id"
                    ],

                "seed":
                    run["seed"],

                "training_noise_fraction":
                    float(
                        run[
                            "noise_fraction"
                        ]
                    ),

                "carrier_split":
                    split,

                "carrier_count":
                    len(
                        carrier_indices
                    ),

                "parameter_count":
                    model_bundle[
                        "parameter_count"
                    ],

                "primitive_mapping_accuracy":
                    primitive_metrics[
                        "mapping_accuracy"
                    ],

                "primitive_exact_transformation_rate":
                    primitive_metrics[
                        "exact_transformation_rate"
                    ],

                "all_mapping_accuracy":
                    all_metrics[
                        "mapping_accuracy"
                    ],

                "all_exact_transformation_rate":
                    all_metrics[
                        "exact_transformation_rate"
                    ],

                "all_partition_accuracy":
                    all_metrics[
                        "partition_accuracy"
                    ],

                "all_rank_accuracy":
                    all_metrics[
                        "rank_accuracy"
                    ],

                "all_consensus_exact_transformation_rate":
                    all_metrics[
                        "consensus_exact_transformation_rate"
                    ],

                "test_mapping_accuracy":
                    test_metrics[
                        "mapping_accuracy"
                    ],

                "test_exact_transformation_rate":
                    test_metrics[
                        "exact_transformation_rate"
                    ],

                "test_partition_accuracy":
                    test_metrics[
                        "partition_accuracy"
                    ],

                "test_rank_accuracy":
                    test_metrics[
                        "rank_accuracy"
                    ],

                "test_consensus_mapping_accuracy":
                    test_metrics[
                        "consensus_mapping_accuracy"
                    ],

                "test_consensus_exact_transformation_rate":
                    test_metrics[
                        "consensus_exact_transformation_rate"
                    ],

                "test_consensus_partition_accuracy":
                    test_metrics[
                        "consensus_partition_accuracy"
                    ],

                "test_consensus_rank_accuracy":
                    test_metrics[
                        "consensus_rank_accuracy"
                    ],

                "test_carrier_consistency":
                    test_metrics[
                        "carrier_consistency"
                    ],

                "all_relation_accuracy":
                    relations[
                        "all"
                    ][
                        "accuracy"
                    ],

                "all_relation_balanced_accuracy":
                    relations[
                        "all"
                    ][
                        "balanced_accuracy"
                    ],

                "test_involved_relation_accuracy":
                    relations[
                        "test_involved"
                    ][
                        "accuracy"
                    ],

                "test_involved_relation_balanced_accuracy":
                    relations[
                        "test_involved"
                    ][
                        "balanced_accuracy"
                    ],

                "test_involved_relation_macro_f1":
                    relations[
                        "test_involved"
                    ][
                        "macro_f1"
                    ],

                "test_involved_relation_majority_baseline":
                    relations[
                        "test_involved"
                    ][
                        "majority_baseline"
                    ],

                "test_test_relation_accuracy":
                    relations[
                        "test_test"
                    ][
                        "accuracy"
                    ],

                "test_test_relation_balanced_accuracy":
                    relations[
                        "test_test"
                    ][
                        "balanced_accuracy"
                    ],

                "test_test_relation_macro_f1":
                    relations[
                        "test_test"
                    ][
                        "macro_f1"
                    ],

                "test_test_relation_majority_baseline":
                    relations[
                        "test_test"
                    ][
                        "majority_baseline"
                    ],
            }
        )

    hard_result = None

    if run["model_id"] == "OCM":
        hard_result = (
            evaluate_ocm_hard_structure(
                model=model_bundle[
                    "model"
                ],
                temperature=model_bundle[
                    "temperature"
                ],
                operation_count=operation_count,
                operation_vocabulary=(
                    operation_vocabulary
                ),
                semigroup=semigroup,
                clean_observations=(
                    clean_observations
                ),
                carrier_splits=(
                    carrier_splits
                ),
                true_mappings=true_mappings,
                transformation_indices=(
                    transformation_indices
                ),
                primitive_indices=(
                    primitive_indices
                ),
                pair_scopes=pair_scopes,
                device=device,
            )
        )

    save_values = {
        "predicted_mappings":
            predicted_mappings,

        "train_consensus_mapping":
            consensus_arrays["train"],

        "val_consensus_mapping":
            consensus_arrays["val"],

        "test_consensus_mapping":
            consensus_arrays["test"],
    }

    if hard_result is not None:
        save_values[
            "ocm_hard_mappings"
        ] = hard_result[
            "hard_mappings"
        ]

        save_values[
            "ocm_true_to_latent"
        ] = hard_result[
            "alignment"
        ][
            "true_to_latent"
        ]

        save_values[
            "ocm_latent_to_true"
        ] = hard_result[
            "alignment"
        ][
            "latent_to_true"
        ]

        save_values[
            "ocm_alignment_confusion"
        ] = hard_result[
            "alignment"
        ][
            "confusion"
        ]

    np.savez_compressed(
        run_dir
        / "structural_predictions.npz",
        **save_values,
    )

    result = {
        "phase":
            "3G Tier B structural recovery",

        "run_id":
            run["run_id"],

        "model_id":
            run["model_id"],

        "configuration_id":
            run[
                "configuration_id"
            ],

        "seed":
            run["seed"],

        "training_noise_fraction":
            float(
                run[
                    "noise_fraction"
                ]
            ),

        "behavioral_carrier_split_metrics":
            behavioral_rows,

        "ocm_hard_result":
            (
                {
                    "primitive_metrics": {
                        key: value
                        for key, value in hard_result[
                            "primitive_metrics"
                        ].items()
                        if key
                        != "consensus_mapping"
                    },

                    "all_metrics": {
                        key: value
                        for key, value in hard_result[
                            "all_metrics"
                        ].items()
                        if key
                        != "consensus_mapping"
                    },

                    "test_metrics": {
                        key: value
                        for key, value in hard_result[
                            "test_metrics"
                        ].items()
                        if key
                        != "consensus_mapping"
                    },

                    "relation_metrics":
                        hard_result[
                            "relation_metrics"
                        ],

                    "training_assignment_accuracy":
                        hard_result[
                            "alignment"
                        ][
                            "training_assignment_accuracy"
                        ],

                    "carrier_split_alignment_accuracy":
                        hard_result[
                            "alignment"
                        ][
                            "split_alignment_accuracy"
                        ],

                    "maximum_channel_row_sum_error":
                        hard_result[
                            "maximum_channel_row_sum_error"
                        ],
                }
                if hard_result is not None
                else None
            ),

        "structural_predictions_path":
            str(
                run_dir
                / "structural_predictions.npz"
            ),

        "training_performed":
            False,

        "checkpoint_selection_reopened":
            False,

        "checkpoint_modified":
            False,

        "clean_manifold_read":
            True,

        "clean_manifold_use":
            (
                "evaluation-only behavioral probe "
                "and nearest-state decoding"
            ),

        "true_transformations_read":
            True,

        "structural_labels_used_for_evaluation_only":
            True,

        "test_metrics_used_for_selection":
            False,

        "status":
            "completed",
    }

    if model_bundle[
        "model"
    ] is not None:
        del model_bundle[
            "model"
        ]

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def flatten_completed_results(
    evaluation_runs,
):
    behavioral_rows = []
    hard_rows = []

    for run in evaluation_runs:
        path = (
            RUN_OUTPUT_DIR
            / run["run_id"]
            / "result.json"
        )

        if not path.exists():
            return None

        result = load_json(path)

        if result.get("status") != "completed":
            return None

        behavioral_rows.extend(
            result[
                "behavioral_carrier_split_metrics"
            ]
        )

        if (
            result[
                "ocm_hard_result"
            ]
            is not None
        ):
            hard = result[
                "ocm_hard_result"
            ]

            hard_rows.append(
                {
                    "run_id":
                        result["run_id"],

                    "configuration_id":
                        result[
                            "configuration_id"
                        ],

                    "seed":
                        result["seed"],

                    "training_noise_fraction":
                        result[
                            "training_noise_fraction"
                        ],

                    "parameter_count":
                        next(
                            row[
                                "parameter_count"
                            ]
                            for row in result[
                                "behavioral_carrier_split_metrics"
                            ]
                        ),

                    "training_assignment_accuracy":
                        hard[
                            "training_assignment_accuracy"
                        ],

                    "train_carrier_alignment_accuracy":
                        hard[
                            "carrier_split_alignment_accuracy"
                        ][
                            "train"
                        ],

                    "val_carrier_alignment_accuracy":
                        hard[
                            "carrier_split_alignment_accuracy"
                        ][
                            "val"
                        ],

                    "test_carrier_alignment_accuracy":
                        hard[
                            "carrier_split_alignment_accuracy"
                        ][
                            "test"
                        ],

                    "maximum_channel_row_sum_error":
                        hard[
                            "maximum_channel_row_sum_error"
                        ],

                    "hard_primitive_mapping_accuracy":
                        hard[
                            "primitive_metrics"
                        ][
                            "mapping_accuracy"
                        ],

                    "hard_primitive_exact_transformation_rate":
                        hard[
                            "primitive_metrics"
                        ][
                            "exact_transformation_rate"
                        ],

                    "hard_all_mapping_accuracy":
                        hard[
                            "all_metrics"
                        ][
                            "mapping_accuracy"
                        ],

                    "hard_all_exact_transformation_rate":
                        hard[
                            "all_metrics"
                        ][
                            "exact_transformation_rate"
                        ],

                    "hard_all_partition_accuracy":
                        hard[
                            "all_metrics"
                        ][
                            "partition_accuracy"
                        ],

                    "hard_all_rank_accuracy":
                        hard[
                            "all_metrics"
                        ][
                            "rank_accuracy"
                        ],

                    "hard_test_mapping_accuracy":
                        hard[
                            "test_metrics"
                        ][
                            "mapping_accuracy"
                        ],

                    "hard_test_exact_transformation_rate":
                        hard[
                            "test_metrics"
                        ][
                            "exact_transformation_rate"
                        ],

                    "hard_test_partition_accuracy":
                        hard[
                            "test_metrics"
                        ][
                            "partition_accuracy"
                        ],

                    "hard_test_rank_accuracy":
                        hard[
                            "test_metrics"
                        ][
                            "rank_accuracy"
                        ],

                    "hard_test_involved_relation_accuracy":
                        hard[
                            "relation_metrics"
                        ][
                            "test_involved"
                        ][
                            "accuracy"
                        ],

                    "hard_test_involved_relation_balanced_accuracy":
                        hard[
                            "relation_metrics"
                        ][
                            "test_involved"
                        ][
                            "balanced_accuracy"
                        ],

                    "hard_test_involved_relation_macro_f1":
                        hard[
                            "relation_metrics"
                        ][
                            "test_involved"
                        ][
                            "macro_f1"
                        ],

                    "hard_test_test_relation_accuracy":
                        hard[
                            "relation_metrics"
                        ][
                            "test_test"
                        ][
                            "accuracy"
                        ],

                    "hard_test_test_relation_balanced_accuracy":
                        hard[
                            "relation_metrics"
                        ][
                            "test_test"
                        ][
                            "balanced_accuracy"
                        ],
                }
            )

    if (
        len(behavioral_rows)
        != EXPECTED_BEHAVIORAL_METRIC_ROWS
    ):
        raise AssertionError(
            "Expected 405 behavioral metric rows."
        )

    if (
        len(hard_rows)
        != EXPECTED_OCM_HARD_ROWS
    ):
        raise AssertionError(
            "Expected 25 OCM hard-channel rows."
        )

    return behavioral_rows, hard_rows


def aggregate_behavioral_rows(
    behavioral_rows,
):
    grouped = defaultdict(list)

    for row in behavioral_rows:
        key = (
            row["model_id"],
            float(
                row[
                    "training_noise_fraction"
                ]
            ),
            row["carrier_split"],
        )

        grouped[key].append(row)

    metric_names = (
        "primitive_mapping_accuracy",
        "primitive_exact_transformation_rate",
        "test_mapping_accuracy",
        "test_exact_transformation_rate",
        "test_partition_accuracy",
        "test_rank_accuracy",
        "test_consensus_mapping_accuracy",
        "test_consensus_exact_transformation_rate",
        "test_consensus_partition_accuracy",
        "test_consensus_rank_accuracy",
        "test_carrier_consistency",
        "test_involved_relation_accuracy",
        "test_involved_relation_balanced_accuracy",
        "test_involved_relation_macro_f1",
        "test_test_relation_accuracy",
        "test_test_relation_balanced_accuracy",
        "test_test_relation_macro_f1",
    )

    output = []

    for (
        model_id,
        noise,
        carrier_split,
    ), rows in sorted(
        grouped.items()
    ):
        expected_count = (
            1
            if model_id in {
                "B0",
                "B1",
            }
            else 5
        )

        if len(rows) != expected_count:
            raise AssertionError(
                f"{model_id}/{noise}/{carrier_split} "
                "has an invalid fit count."
            )

        summary_row = {
            "model_id":
                model_id,

            "training_noise_fraction":
                noise,

            "carrier_split":
                carrier_split,

            "fit_count":
                len(rows),

            "parameter_count":
                int(
                    rows[0][
                        "parameter_count"
                    ]
                ),

            "test_involved_relation_majority_baseline":
                rows[0][
                    "test_involved_relation_majority_baseline"
                ],

            "test_test_relation_majority_baseline":
                rows[0][
                    "test_test_relation_majority_baseline"
                ],
        }

        for metric_name in metric_names:
            statistics = summarize(
                [
                    float(
                        row[
                            metric_name
                        ]
                    )
                    for row in rows
                ]
            )

            summary_row[
                metric_name
                + "_mean"
            ] = statistics[
                "mean"
            ]

            summary_row[
                metric_name
                + "_std"
            ] = statistics[
                "std"
            ]

        output.append(
            summary_row
        )

    if len(output) != 105:
        raise AssertionError(
            "Expected 105 behavioral summary rows."
        )

    return output


def aggregate_hard_rows(
    hard_rows,
):
    grouped = defaultdict(list)

    for row in hard_rows:
        grouped[
            float(
                row[
                    "training_noise_fraction"
                ]
            )
        ].append(row)

    metric_names = [
        key
        for key in hard_rows[0]
        if key not in {
            "run_id",
            "configuration_id",
            "seed",
            "training_noise_fraction",
            "parameter_count",
        }
    ]

    output = []

    for noise, rows in sorted(
        grouped.items()
    ):
        if len(rows) != 5:
            raise AssertionError(
                "Each OCM noise condition must "
                "contain five seeds."
            )

        summary_row = {
            "training_noise_fraction":
                noise,

            "fit_count":
                len(rows),

            "parameter_count":
                int(
                    rows[0][
                        "parameter_count"
                    ]
                ),
        }

        for metric_name in metric_names:
            statistics = summarize(
                [
                    float(
                        row[
                            metric_name
                        ]
                    )
                    for row in rows
                ]
            )

            summary_row[
                metric_name
                + "_mean"
            ] = statistics[
                "mean"
            ]

            summary_row[
                metric_name
                + "_std"
            ] = statistics[
                "std"
            ]

        output.append(
            summary_row
        )

    return output


def create_rankings(
    behavioral_summary,
):
    output = []

    for carrier_split in CARRIER_SPLITS:
        for noise in FROZEN_NOISE_LEVELS:
            candidates = [
                row
                for row in behavioral_summary
                if (
                    row["carrier_split"]
                    == carrier_split
                    and np.isclose(
                        row[
                            "training_noise_fraction"
                        ],
                        noise,
                    )
                )
            ]

            candidates.sort(
                key=lambda row: (
                    -row[
                        "test_involved_relation_balanced_accuracy_mean"
                    ],
                    -row[
                        "test_consensus_exact_transformation_rate_mean"
                    ],
                    -row[
                        "test_consensus_mapping_accuracy_mean"
                    ],
                    row["model_id"],
                )
            )

            for rank, row in enumerate(
                candidates,
                start=1,
            ):
                output.append(
                    {
                        "carrier_split":
                            carrier_split,

                        "training_noise_fraction":
                            noise,

                        "rank":
                            rank,

                        "model_id":
                            row[
                                "model_id"
                            ],

                        "test_relation_balanced_accuracy":
                            row[
                                "test_involved_relation_balanced_accuracy_mean"
                            ],

                        "test_consensus_exact_transformation_rate":
                            row[
                                "test_consensus_exact_transformation_rate_mean"
                            ],

                        "test_consensus_mapping_accuracy":
                            row[
                                "test_consensus_mapping_accuracy_mean"
                            ],

                        "ranking_rule":
                            (
                                "relation balanced accuracy, "
                                "then exact recovery, "
                                "then mapping accuracy"
                            ),
                    }
                )

    return output


def create_ocm_hard_behavioral_comparison(
    behavioral_rows,
    hard_rows,
):
    behavioral_lookup = {
        (
            row["run_id"],
            row["carrier_split"],
        ):
            row
        for row in behavioral_rows
        if row["model_id"] == "OCM"
    }

    output = []

    for hard in hard_rows:
        for carrier_split in CARRIER_SPLITS:
            behavioral = behavioral_lookup[
                (
                    hard["run_id"],
                    carrier_split,
                )
            ]

            output.append(
                {
                    "run_id":
                        hard["run_id"],

                    "seed":
                        hard["seed"],

                    "training_noise_fraction":
                        hard[
                            "training_noise_fraction"
                        ],

                    "carrier_split":
                        carrier_split,

                    "alignment_accuracy":
                        hard[
                            f"{carrier_split}_carrier_alignment_accuracy"
                        ],

                    "hard_test_exact_transformation_rate":
                        hard[
                            "hard_test_exact_transformation_rate"
                        ],

                    "behavioral_test_exact_transformation_rate":
                        behavioral[
                            "test_exact_transformation_rate"
                        ],

                    "behavioral_consensus_test_exact_transformation_rate":
                        behavioral[
                            "test_consensus_exact_transformation_rate"
                        ],

                    "hard_test_relation_accuracy":
                        hard[
                            "hard_test_involved_relation_accuracy"
                        ],

                    "hard_test_relation_balanced_accuracy":
                        hard[
                            "hard_test_involved_relation_balanced_accuracy"
                        ],

                    "behavioral_test_relation_accuracy":
                        behavioral[
                            "test_involved_relation_accuracy"
                        ],

                    "behavioral_test_relation_balanced_accuracy":
                        behavioral[
                            "test_involved_relation_balanced_accuracy"
                        ],
                }
            )

    return output


def finalize_phase(
    evaluation_runs,
):
    flattened = flatten_completed_results(
        evaluation_runs
    )

    if flattened is None:
        return None

    behavioral_rows, hard_rows = (
        flattened
    )

    behavioral_summary = (
        aggregate_behavioral_rows(
            behavioral_rows
        )
    )

    hard_summary = aggregate_hard_rows(
        hard_rows
    )

    rankings = create_rankings(
        behavioral_summary
    )

    comparison = (
        create_ocm_hard_behavioral_comparison(
            behavioral_rows=behavioral_rows,
            hard_rows=hard_rows,
        )
    )

    write_csv(
        OUTPUT_DIR
        / "all_behavioral_run_metrics.csv",
        behavioral_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "behavioral_model_noise_summary.csv",
        behavioral_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "ocm_internal_hard_run_metrics.csv",
        hard_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "ocm_internal_hard_summary.csv",
        hard_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "behavioral_structure_rankings.csv",
        rankings,
    )

    write_csv(
        OUTPUT_DIR
        / "ocm_hard_vs_behavioral.csv",
        comparison,
    )

    summary = {
        "phase":
            "3G Tier B structural recovery audit",

        "behavioral_evaluation_run_count":
            len(
                evaluation_runs
            ),

        "behavioral_run_carrier_split_metric_count":
            len(
                behavioral_rows
            ),

        "behavioral_model_noise_summary_count":
            len(
                behavioral_summary
            ),

        "ocm_internal_hard_run_count":
            len(
                hard_rows
            ),

        "ocm_internal_hard_summary_count":
            len(
                hard_summary
            ),

        "carrier_splits":
            list(
                CARRIER_SPLITS
            ),

        "transformation_count":
            SEMIGROUP_SIZE,

        "probe_input":
            (
                "Clean carrier-state observations "
                "for every carrier and state"
            ),

        "behavioral_decoder":
            (
                "Nearest clean state on the same "
                "carrier manifold"
            ),

        "ocm_alignment_data":
            "clean training carriers only",

        "ocm_alignment_applied_to":
            (
                "validation and test carriers "
                "without refitting"
            ),

        "training_performed":
            False,

        "checkpoint_selection_reopened":
            False,

        "checkpoints_modified":
            False,

        "structural_ground_truth_read":
            True,

        "structural_ground_truth_use":
            "final evaluation only",

        "test_metrics_used_for_selection":
            False,

        "phase2_outputs_modified":
            False,

        "phase3e_outputs_modified":
            False,

        "phase3f_outputs_modified":
            False,

        "all_evaluations_complete":
            True,

        "phase3g_status":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase3g_summary.json",
        summary,
    )

    return summary


def main():
    arguments = parse_arguments()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, evaluation_runs = validate_sources()

    requested_runs = [
        run
        for run in evaluation_runs
        if (
            (
                arguments.run_id == "all"
                or run["run_id"]
                == arguments.run_id
            )
            and (
                arguments.model_id == "all"
                or run["model_id"]
                == arguments.model_id
            )
        )
    ]

    if not requested_runs:
        raise ValueError(
            "No structural evaluation runs match "
            "the requested filters."
        )

    device = resolve_device(
        arguments.device
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision(
            "high"
        )

    operation_vocabulary = (
        load_operation_vocabulary()
    )

    manifold = load_clean_manifold()

    semigroup = load_semigroup()

    generators = load_generators()

    transformation_indices = (
        load_transformation_splits(
            semigroup
        )
    )

    primitive_indices = (
        find_primitive_indices(
            semigroup=semigroup,
            generators=generators,
        )
    )

    true_mappings = np.asarray(
        [
            transformation[
                "mapping"
            ]
            for transformation in semigroup
        ],
        dtype=np.int8,
    )

    pair_scopes = relation_pair_indices(
        transformation_count=(
            len(semigroup)
        ),
        test_indices=(
            transformation_indices[
                "test"
            ]
        ),
    )

    for index, run in enumerate(
        requested_runs,
        start=1,
    ):
        run_dir = (
            RUN_OUTPUT_DIR
            / run["run_id"]
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_path = (
            run_dir / "result.json"
        )

        if (
            result_path.exists()
            and not arguments.force
        ):
            existing = load_json(
                result_path
            )

            if existing.get(
                "status"
            ) == "completed":
                print(
                    f"[{index}/{len(requested_runs)}] "
                    f"{run['run_id']} already completed."
                )
                continue

        print()
        print(
            f"[{index}/{len(requested_runs)}] "
            f"Evaluating {run['run_id']}"
        )

        result = evaluate_run(
            run=run,
            semigroup=semigroup,
            true_mappings=true_mappings,
            transformation_indices=(
                transformation_indices
            ),
            primitive_indices=(
                primitive_indices
            ),
            pair_scopes=pair_scopes,
            clean_observations=(
                manifold[
                    "observations"
                ]
            ),
            carrier_splits=(
                manifold[
                    "carrier_splits"
                ]
            ),
            operation_vocabulary=(
                operation_vocabulary
            ),
            device=device,
            run_dir=run_dir,
        )

        write_json(
            result_path,
            result,
        )

        for row in result[
            "behavioral_carrier_split_metrics"
        ]:
            print(
                f"  {row['carrier_split']} | "
                f"test exact="
                f"{row['test_exact_transformation_rate']:.4f} | "
                f"consensus exact="
                f"{row['test_consensus_exact_transformation_rate']:.4f} | "
                f"relation balanced="
                f"{row['test_involved_relation_balanced_accuracy']:.4f}"
            )

    summary = finalize_phase(
        evaluation_runs
    )

    if summary is None:
        completed_count = sum(
            (
                RUN_OUTPUT_DIR
                / run["run_id"]
                / "result.json"
            ).exists()
            for run in evaluation_runs
        )

        print()
        print(
            "Phase 3G progress: "
            f"{completed_count}/"
            f"{EXPECTED_EVALUATION_RUN_COUNT} "
            "run results present."
        )

    else:
        print()
        print(
            "Phase 3G structural evaluation completed."
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
