from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn


PHASE4AR_DIR = Path(
    "outputs/phase4ar_tier_c_v2_protocol"
)

PHASE4BR_DIR = Path(
    "outputs/phase4br_tier_c_v2_development_data"
)

PHASE4CR_DIR = Path(
    "outputs/phase4cr_tier_c_v2_audit"
)

OUTPUT_DIR = Path(
    "outputs/phase4crd_tier_c_v2_failure_diagnosis"
)

PRIVILEGED_DIR = (
    PHASE4BR_DIR
    / "privileged"
)


PROTOCOL_VERSION = "tier_c_v2"

STATE_COUNT = 8
FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32

TRAIN_CARRIER_COUNT = 96
VALIDATION_CARRIER_COUNT = 32
DEVELOPMENT_CARRIER_COUNT = 128

EXPECTED_DIAGNOSTIC_SEEDS = (
    52011,
    52023,
    52037,
)

FACTOR_NAMES = (
    "coarsening_level",
    "anisotropy_level",
    "defect_recovery_level",
)

SEALED_TEST_CELLS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    return parser.parse_args()


def resolve_device(requested):
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def write_json(path, value):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(path, rows):
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with Path(path).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def safe_torch_load(
    path,
    map_location,
):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def validate_rejection_freeze():
    rejection_summary = load_json(
        PHASE4CR_DIR
        / "phase4cr_rejection_summary.json"
    )

    rejection_record = load_json(
        PHASE4CR_DIR
        / "phase4cr_rejection_record.json"
    )

    audit_summary = load_json(
        PHASE4CR_DIR
        / "phase4cr_summary.json"
    )

    decoder_summary = load_json(
        PHASE4CR_DIR
        / "state_decoder_summary.json"
    )

    if (
        rejection_summary[
            "protocol_version"
        ]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Tier C v2 protocol version changed."
        )

    if (
        rejection_summary[
            "tier_c_v2_rejected"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v2 rejection is not recorded."
        )

    if (
        rejection_summary[
            "phase4dr_predictive_training_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "Predictive training was incorrectly authorized."
        )

    if (
        rejection_summary[
            "sealed_test_set_remains_unopened"
        ]
        is not True
    ):
        raise AssertionError(
            "The sealed test set is not recorded as unopened."
        )

    if (
        rejection_summary[
            "failed_acceptance_gates"
        ]
        != [
            "three_seed_state_separability"
        ]
    ):
        raise AssertionError(
            "Unexpected Tier C v2 rejection reason."
        )

    summary_status = rejection_summary.get(
        "phase4cr_rejection_freeze_status"
    )

    record_status = rejection_record.get(
        "rejection_freeze_status"
    )

    if (
        summary_status != "passed"
        and record_status != "passed"
    ):
        raise AssertionError(
            "Tier C v2 rejection freeze did not pass."
        )

    if (
        rejection_record[
            "rejection_is_final"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v2 rejection is not final."
        )

    if (
        audit_summary[
            "phase4cr_status"
        ]
        != "failed_acceptance_gates"
    ):
        raise AssertionError(
            "Phase 4C-R audit status changed."
        )

    if (
        audit_summary[
            "tier_c_v2_development_data_accepted"
        ]
        is not False
    ):
        raise AssertionError(
            "Tier C v2 was unexpectedly accepted."
        )

    diagnostic_seeds = tuple(
        int(value)
        for value in decoder_summary[
            "diagnostic_seeds"
        ]
    )

    if (
        diagnostic_seeds
        != EXPECTED_DIAGNOSTIC_SEEDS
    ):
        raise AssertionError(
            "Frozen diagnostic seeds changed."
        )

    return {
        "rejection_summary":
            rejection_summary,

        "rejection_record":
            rejection_record,

        "audit_summary":
            audit_summary,

        "decoder_summary":
            decoder_summary,
    }


def find_sealed_test_files():
    found = []

    directories = (
        PHASE4BR_DIR / "visible",
        PHASE4BR_DIR / "privileged",
        PHASE4BR_DIR / "cell_manifests",
    )

    for directory in directories:
        if not directory.exists():
            continue

        for path in directory.iterdir():
            if any(
                cell_id in path.name
                for cell_id in SEALED_TEST_CELLS
            ):
                found.append(
                    str(path)
                )

    return found


def validate_test_sealing():
    sealed_files = find_sealed_test_files()

    if sealed_files:
        raise AssertionError(
            "Sealed test files unexpectedly exist: "
            + ", ".join(sealed_files)
        )

    carrier_rows = load_csv(
        PHASE4AR_DIR
        / "carrier_split_v2.csv"
    )

    frozen_test_ids = {
        int(row["carrier_id"])
        for row in carrier_rows
        if row["split"] == "test"
    }

    development_ids = set(
        int(value)
        for value in np.load(
            PRIVILEGED_DIR
            / "development_carrier_ids.npy"
        )
    )

    overlap = (
        frozen_test_ids
        & development_ids
    )

    if overlap:
        raise AssertionError(
            "Frozen test carrier IDs appear in "
            "development arrays."
        )

    parameter_rows = load_csv(
        PHASE4BR_DIR
        / "development_carrier_parameters.csv"
    )

    if any(
        row["split"] == "test"
        for row in parameter_rows
    ):
        raise AssertionError(
            "Test-carrier parameters are present."
        )

    return {
        "sealed_test_files_found":
            sealed_files,

        "frozen_test_carrier_count":
            len(frozen_test_ids),

        "development_test_carrier_overlap_count":
            len(overlap),

        "test_carrier_parameters_present":
            False,

        "test_fields_read":
            False,

        "test_metrics_computed":
            False,
    }


def load_state_factor_registry():
    rows = load_csv(
        PHASE4AR_DIR
        / "state_morphology_regimes_v2.csv"
    )

    rows.sort(
        key=lambda row:
            int(row["state_id"])
    )

    state_ids = [
        int(row["state_id"])
        for row in rows
    ]

    if state_ids != list(
        range(STATE_COUNT)
    ):
        raise AssertionError(
            "Tier C v2 state IDs changed."
        )

    factor_values = {
        factor:
            np.asarray(
                [
                    int(row[factor])
                    for row in rows
                ],
                dtype=np.int64,
            )
        for factor in FACTOR_NAMES
    }

    state_codes = [
        str(row["state_code"])
        for row in rows
    ]

    if len(set(state_codes)) != STATE_COUNT:
        raise AssertionError(
            "State morphology codes are not unique."
        )

    registry_rows = []

    for row in rows:
        registry_rows.append(
            {
                "state_id":
                    int(row["state_id"]),

                "state_code":
                    str(row["state_code"]),

                "coarsening_level":
                    int(
                        row[
                            "coarsening_level"
                        ]
                    ),

                "anisotropy_level":
                    int(
                        row[
                            "anisotropy_level"
                        ]
                    ),

                "defect_recovery_level":
                    int(
                        row[
                            "defect_recovery_level"
                        ]
                    ),
            }
        )

    write_csv(
        OUTPUT_DIR
        / "state_factor_registry.csv",
        registry_rows,
    )

    return {
        "rows":
            registry_rows,

        "state_codes":
            state_codes,

        "factor_values":
            factor_values,
    }


def load_development_dataset():
    carrier_ids = np.load(
        PRIVILEGED_DIR
        / "development_carrier_ids.npy"
    ).astype(
        np.int64
    )

    carrier_splits = np.load(
        PRIVILEGED_DIR
        / "development_carrier_splits.npy"
    ).astype(str)

    normalized_bank = np.load(
        PRIVILEGED_DIR
        / "development_carrier_state_fields_normalized.npy",
        mmap_mode="r",
    )

    expected_shape = (
        DEVELOPMENT_CARRIER_COUNT,
        STATE_COUNT,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    if normalized_bank.shape != expected_shape:
        raise AssertionError(
            "Development field-bank shape changed."
        )

    if normalized_bank.dtype != np.float32:
        raise AssertionError(
            "Development field-bank dtype changed."
        )

    train_indices = np.where(
        carrier_splits == "train"
    )[0]

    validation_indices = np.where(
        carrier_splits == "val"
    )[0]

    if len(train_indices) != TRAIN_CARRIER_COUNT:
        raise AssertionError(
            "Training carrier count changed."
        )

    if (
        len(validation_indices)
        != VALIDATION_CARRIER_COUNT
    ):
        raise AssertionError(
            "Validation carrier count changed."
        )

    labels = np.broadcast_to(
        np.arange(
            STATE_COUNT,
            dtype=np.int64,
        )[None, :],
        (
            DEVELOPMENT_CARRIER_COUNT,
            STATE_COUNT,
        ),
    )

    return {
        "carrier_ids":
            carrier_ids,

        "carrier_splits":
            carrier_splits,

        "train": {
            "carrier_ids":
                carrier_ids[
                    train_indices
                ].copy(),

            "fields":
                np.asarray(
                    normalized_bank[
                        train_indices
                    ],
                    dtype=np.float32,
                ).reshape(
                    -1,
                    FIELD_CHANNEL_COUNT,
                    GRID_HEIGHT,
                    GRID_WIDTH,
                ).copy(),

            "labels":
                labels[
                    train_indices
                ].reshape(-1).copy(),
        },

        "validation": {
            "carrier_ids":
                carrier_ids[
                    validation_indices
                ].copy(),

            "fields":
                np.asarray(
                    normalized_bank[
                        validation_indices
                    ],
                    dtype=np.float32,
                ).reshape(
                    -1,
                    FIELD_CHANNEL_COUNT,
                    GRID_HEIGHT,
                    GRID_WIDTH,
                ).copy(),

            "labels":
                labels[
                    validation_indices
                ].reshape(-1).copy(),
        },
    }


class ResidualBlock(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        stride=1,
    ):
        super().__init__()

        self.convolution_1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.normalization_1 = nn.GroupNorm(
            num_groups=8,
            num_channels=output_channels,
        )

        self.convolution_2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.normalization_2 = nn.GroupNorm(
            num_groups=8,
            num_channels=output_channels,
        )

        if (
            stride != 1
            or input_channels != output_channels
        ):
            self.skip = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    num_groups=8,
                    num_channels=output_channels,
                ),
            )

        else:
            self.skip = nn.Identity()

    def forward(self, inputs):
        residual = self.skip(inputs)

        values = self.convolution_1(
            inputs
        )

        values = self.normalization_1(
            values
        )

        values = nn.functional.gelu(
            values
        )

        values = self.convolution_2(
            values
        )

        values = self.normalization_2(
            values
        )

        return nn.functional.gelu(
            values + residual
        )


class FrozenDiagnosticStateDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                FIELD_CHANNEL_COUNT,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=32,
            ),
            nn.GELU(),
        )

        self.encoder = nn.Sequential(
            ResidualBlock(
                32,
                32,
            ),
            ResidualBlock(
                32,
                64,
                stride=2,
            ),
            ResidualBlock(
                64,
                64,
            ),
            ResidualBlock(
                64,
                128,
                stride=2,
            ),
            ResidualBlock(
                128,
                128,
            ),
            ResidualBlock(
                128,
                192,
                stride=2,
            ),
            ResidualBlock(
                192,
                192,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(
                p=0.10
            ),
            nn.Linear(
                192,
                STATE_COUNT,
            ),
        )

    def forward(self, fields):
        values = self.stem(fields)
        values = self.encoder(values)
        values = self.pool(values)

        return self.classifier(values)


@torch.no_grad()
def predict_dataset(
    model,
    fields,
    device,
    batch_size,
):
    model.eval()

    probability_batches = []

    for start in range(
        0,
        len(fields),
        batch_size,
    ):
        end = min(
            len(fields),
            start + batch_size,
        )

        batch = torch.from_numpy(
            fields[start:end].copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )

        logits = model(batch)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        probability_batches.append(
            probabilities.detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

    probabilities = np.concatenate(
        probability_batches,
        axis=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(
        np.int64
    )

    return predictions, probabilities


def confusion_matrix(
    labels,
    predictions,
    class_count,
):
    matrix = np.zeros(
        (
            class_count,
            class_count,
        ),
        dtype=np.int64,
    )

    for true_value, predicted_value in zip(
        labels,
        predictions,
    ):
        matrix[
            int(true_value),
            int(predicted_value),
        ] += 1

    return matrix


def balanced_accuracy_from_confusion(
    matrix,
):
    recalls = []

    for true_class in range(
        matrix.shape[0]
    ):
        denominator = matrix[
            true_class
        ].sum()

        if denominator == 0:
            continue

        recalls.append(
            matrix[
                true_class,
                true_class,
            ]
            / denominator
        )

    if not recalls:
        return float("nan")

    return float(
        np.mean(recalls)
    )


def state_accuracy_rows(
    seed,
    split,
    labels,
    predictions,
    state_registry,
):
    rows = []

    for state_id in range(
        STATE_COUNT
    ):
        selected = (
            labels == state_id
        )

        count = int(
            selected.sum()
        )

        correct = int(
            (
                predictions[selected]
                == labels[selected]
            ).sum()
        )

        registry = (
            state_registry["rows"][
                state_id
            ]
        )

        rows.append(
            {
                "seed":
                    seed,

                "split":
                    split,

                "state_id":
                    state_id,

                "state_code":
                    registry[
                        "state_code"
                    ],

                "coarsening_level":
                    registry[
                        "coarsening_level"
                    ],

                "anisotropy_level":
                    registry[
                        "anisotropy_level"
                    ],

                "defect_recovery_level":
                    registry[
                        "defect_recovery_level"
                    ],

                "sample_count":
                    count,

                "correct_count":
                    correct,

                "accuracy":
                    correct / count,
            }
        )

    return rows


def factor_metrics(
    seed,
    split,
    labels,
    predictions,
    state_registry,
):
    rows = []
    confusion_rows = []

    factor_values = (
        state_registry[
            "factor_values"
        ]
    )

    for factor_name in FACTOR_NAMES:
        lookup = factor_values[
            factor_name
        ]

        true_factor = lookup[
            labels
        ]

        predicted_factor = lookup[
            predictions
        ]

        matrix = confusion_matrix(
            labels=true_factor,
            predictions=predicted_factor,
            class_count=2,
        )

        accuracy = float(
            np.mean(
                true_factor
                == predicted_factor
            )
        )

        balanced_accuracy = (
            balanced_accuracy_from_confusion(
                matrix
            )
        )

        rows.append(
            {
                "seed":
                    seed,

                "split":
                    split,

                "factor":
                    factor_name,

                "accuracy":
                    accuracy,

                "balanced_accuracy":
                    balanced_accuracy,

                "error_rate":
                    1.0 - accuracy,
            }
        )

        for true_value in range(2):
            for predicted_value in range(2):
                confusion_rows.append(
                    {
                        "seed":
                            seed,

                        "split":
                            split,

                        "factor":
                            factor_name,

                        "true_value":
                            true_value,

                        "predicted_value":
                            predicted_value,

                        "count":
                            int(
                                matrix[
                                    true_value,
                                    predicted_value,
                                ]
                            ),
                    }
                )

    return rows, confusion_rows


def hamming_error_rows(
    seed,
    split,
    labels,
    predictions,
    state_registry,
):
    factor_values = (
        state_registry[
            "factor_values"
        ]
    )

    hamming_distances = np.zeros(
        len(labels),
        dtype=np.int64,
    )

    factor_difference_flags = {}

    for factor_name in FACTOR_NAMES:
        lookup = factor_values[
            factor_name
        ]

        differs = (
            lookup[labels]
            != lookup[predictions]
        )

        factor_difference_flags[
            factor_name
        ] = differs

        hamming_distances += (
            differs.astype(
                np.int64
            )
        )

    rows = []

    for distance in range(4):
        count = int(
            np.sum(
                hamming_distances
                == distance
            )
        )

        rows.append(
            {
                "seed":
                    seed,

                "split":
                    split,

                "factor_hamming_distance":
                    distance,

                "count":
                    count,

                "fraction":
                    count / len(labels),
            }
        )

    pattern_rows = []

    for coarsening_difference in (
        0,
        1,
    ):
        for anisotropy_difference in (
            0,
            1,
        ):
            for defect_difference in (
                0,
                1,
            ):
                selected = (
                    factor_difference_flags[
                        "coarsening_level"
                    ]
                    == bool(
                        coarsening_difference
                    )
                )

                selected &= (
                    factor_difference_flags[
                        "anisotropy_level"
                    ]
                    == bool(
                        anisotropy_difference
                    )
                )

                selected &= (
                    factor_difference_flags[
                        "defect_recovery_level"
                    ]
                    == bool(
                        defect_difference
                    )
                )

                count = int(
                    selected.sum()
                )

                pattern_rows.append(
                    {
                        "seed":
                            seed,

                        "split":
                            split,

                        "coarsening_differs":
                            coarsening_difference,

                        "anisotropy_differs":
                            anisotropy_difference,

                        "defect_recovery_differs":
                            defect_difference,

                        "factor_difference_count":
                            (
                                coarsening_difference
                                + anisotropy_difference
                                + defect_difference
                            ),

                        "sample_count":
                            count,

                        "fraction":
                            count
                            / len(labels),
                    }
                )

    return rows, pattern_rows


def state_confusion_rows(
    seed,
    split,
    matrix,
    state_registry,
):
    rows = []

    for true_state in range(
        STATE_COUNT
    ):
        for predicted_state in range(
            STATE_COUNT
        ):
            true_row = (
                state_registry["rows"][
                    true_state
                ]
            )

            predicted_row = (
                state_registry["rows"][
                    predicted_state
                ]
            )

            factor_difference_count = sum(
                int(
                    true_row[factor]
                    != predicted_row[factor]
                )
                for factor in FACTOR_NAMES
            )

            rows.append(
                {
                    "seed":
                        seed,

                    "split":
                        split,

                    "true_state":
                        true_state,

                    "true_state_code":
                        true_row[
                            "state_code"
                        ],

                    "predicted_state":
                        predicted_state,

                    "predicted_state_code":
                        predicted_row[
                            "state_code"
                        ],

                    "coarsening_differs":
                        int(
                            true_row[
                                "coarsening_level"
                            ]
                            != predicted_row[
                                "coarsening_level"
                            ]
                        ),

                    "anisotropy_differs":
                        int(
                            true_row[
                                "anisotropy_level"
                            ]
                            != predicted_row[
                                "anisotropy_level"
                            ]
                        ),

                    "defect_recovery_differs":
                        int(
                            true_row[
                                "defect_recovery_level"
                            ]
                            != predicted_row[
                                "defect_recovery_level"
                            ]
                        ),

                    "factor_difference_count":
                        factor_difference_count,

                    "count":
                        int(
                            matrix[
                                true_state,
                                predicted_state,
                            ]
                        ),
                }
            )

    return rows


def summarize_grouped_rows(
    rows,
    group_keys,
    value_key,
):
    grouped = {}

    for row in rows:
        key = tuple(
            row[group_key]
            for group_key in group_keys
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            float(
                row[value_key]
            )
        )

    output = []

    for key in sorted(grouped):
        values = np.asarray(
            grouped[key],
            dtype=np.float64,
        )

        result = {
            group_key:
                key[index]
            for index, group_key
            in enumerate(group_keys)
        }

        result.update(
            {
                "seed_count":
                    len(values),

                f"{value_key}_mean":
                    float(
                        values.mean()
                    ),

                f"{value_key}_std":
                    (
                        float(
                            values.std(
                                ddof=1
                            )
                        )
                        if len(values) > 1
                        else 0.0
                    ),

                f"{value_key}_minimum":
                    float(
                        values.min()
                    ),

                f"{value_key}_maximum":
                    float(
                        values.max()
                    ),
            }
        )

        output.append(result)

    return output


def evaluate_seed(
    seed,
    model,
    dataset,
    state_registry,
    device,
    batch_size,
):
    prediction_outputs = {}

    metric_rows = []
    state_rows = []
    factor_rows = []
    factor_confusion_rows = []
    state_confusion_output = []
    hamming_rows = []
    error_pattern_rows = []

    for split in (
        "train",
        "validation",
    ):
        fields = dataset[
            split
        ]["fields"]

        labels = dataset[
            split
        ]["labels"]

        (
            predictions,
            probabilities,
        ) = predict_dataset(
            model=model,
            fields=fields,
            device=device,
            batch_size=batch_size,
        )

        prediction_outputs[
            split
        ] = {
            "labels":
                labels,

            "predictions":
                predictions,

            "probabilities":
                probabilities,
        }

        state_matrix = confusion_matrix(
            labels=labels,
            predictions=predictions,
            class_count=STATE_COUNT,
        )

        accuracy = float(
            np.mean(
                labels == predictions
            )
        )

        balanced_accuracy = (
            balanced_accuracy_from_confusion(
                state_matrix
            )
        )

        metric_rows.append(
            {
                "seed":
                    seed,

                "split":
                    split,

                "sample_count":
                    len(labels),

                "state_accuracy":
                    accuracy,

                "state_balanced_accuracy":
                    balanced_accuracy,
            }
        )

        state_rows.extend(
            state_accuracy_rows(
                seed=seed,
                split=split,
                labels=labels,
                predictions=predictions,
                state_registry=state_registry,
            )
        )

        (
            split_factor_rows,
            split_factor_confusion,
        ) = factor_metrics(
            seed=seed,
            split=split,
            labels=labels,
            predictions=predictions,
            state_registry=state_registry,
        )

        factor_rows.extend(
            split_factor_rows
        )

        factor_confusion_rows.extend(
            split_factor_confusion
        )

        state_confusion_output.extend(
            state_confusion_rows(
                seed=seed,
                split=split,
                matrix=state_matrix,
                state_registry=state_registry,
            )
        )

        (
            split_hamming_rows,
            split_error_pattern_rows,
        ) = hamming_error_rows(
            seed=seed,
            split=split,
            labels=labels,
            predictions=predictions,
            state_registry=state_registry,
        )

        hamming_rows.extend(
            split_hamming_rows
        )

        error_pattern_rows.extend(
            split_error_pattern_rows
        )

    return {
        "prediction_outputs":
            prediction_outputs,

        "metric_rows":
            metric_rows,

        "state_rows":
            state_rows,

        "factor_rows":
            factor_rows,

        "factor_confusion_rows":
            factor_confusion_rows,

        "state_confusion_rows":
            state_confusion_output,

        "hamming_rows":
            hamming_rows,

        "error_pattern_rows":
            error_pattern_rows,
    }


def compute_ensemble_diagnostics(
    seed_outputs,
    dataset,
    state_registry,
):
    metric_rows = []
    factor_rows = []
    state_rows = []

    ensemble_predictions = {}

    for split in (
        "train",
        "validation",
    ):
        probability_stack = np.stack(
            [
                seed_outputs[seed][
                    "prediction_outputs"
                ][split][
                    "probabilities"
                ]
                for seed in EXPECTED_DIAGNOSTIC_SEEDS
            ],
            axis=0,
        )

        prediction_stack = np.stack(
            [
                seed_outputs[seed][
                    "prediction_outputs"
                ][split][
                    "predictions"
                ]
                for seed in EXPECTED_DIAGNOSTIC_SEEDS
            ],
            axis=0,
        )

        labels = dataset[
            split
        ]["labels"]

        mean_probabilities = (
            probability_stack.mean(
                axis=0
            )
        )

        predictions = np.argmax(
            mean_probabilities,
            axis=1,
        ).astype(
            np.int64
        )

        ensemble_predictions[
            split
        ] = predictions

        matrix = confusion_matrix(
            labels=labels,
            predictions=predictions,
            class_count=STATE_COUNT,
        )

        state_accuracy = float(
            np.mean(
                labels == predictions
            )
        )

        pairwise_agreements = []

        for first_index in range(
            len(EXPECTED_DIAGNOSTIC_SEEDS)
        ):
            for second_index in range(
                first_index + 1,
                len(
                    EXPECTED_DIAGNOSTIC_SEEDS
                ),
            ):
                pairwise_agreements.append(
                    float(
                        np.mean(
                            prediction_stack[
                                first_index
                            ]
                            == prediction_stack[
                                second_index
                            ]
                        )
                    )
                )

        unanimous_fraction = float(
            np.mean(
                np.all(
                    prediction_stack
                    == prediction_stack[
                        0:1
                    ],
                    axis=0,
                )
            )
        )

        any_seed_correct = float(
            np.mean(
                np.any(
                    prediction_stack
                    == labels[
                        None,
                        :,
                    ],
                    axis=0,
                )
            )
        )

        all_seed_correct = float(
            np.mean(
                np.all(
                    prediction_stack
                    == labels[
                        None,
                        :,
                    ],
                    axis=0,
                )
            )
        )

        metric_rows.append(
            {
                "split":
                    split,

                "sample_count":
                    len(labels),

                "probability_ensemble_state_accuracy":
                    state_accuracy,

                "probability_ensemble_state_balanced_accuracy":
                    balanced_accuracy_from_confusion(
                        matrix
                    ),

                "mean_pairwise_seed_prediction_agreement":
                    float(
                        np.mean(
                            pairwise_agreements
                        )
                    ),

                "unanimous_seed_prediction_fraction":
                    unanimous_fraction,

                "any_seed_correct_fraction":
                    any_seed_correct,

                "all_seeds_correct_fraction":
                    all_seed_correct,
            }
        )

        (
            ensemble_factor_rows,
            _,
        ) = factor_metrics(
            seed="ensemble",
            split=split,
            labels=labels,
            predictions=predictions,
            state_registry=state_registry,
        )

        factor_rows.extend(
            ensemble_factor_rows
        )

        state_rows.extend(
            state_accuracy_rows(
                seed="ensemble",
                split=split,
                labels=labels,
                predictions=predictions,
                state_registry=state_registry,
            )
        )

    write_csv(
        OUTPUT_DIR
        / "ensemble_metrics.csv",
        metric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "ensemble_factor_accuracy.csv",
        factor_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "ensemble_state_accuracy.csv",
        state_rows,
    )

    return {
        "metrics":
            metric_rows,

        "factor_rows":
            factor_rows,

        "state_rows":
            state_rows,

        "predictions":
            ensemble_predictions,
    }


def aggregate_validation_confusions(
    state_confusion_rows_all,
):
    aggregate = Counter()

    metadata = {}

    for row in state_confusion_rows_all:
        if row["split"] != "validation":
            continue

        key = (
            int(row["true_state"]),
            int(row["predicted_state"]),
        )

        aggregate[key] += int(
            row["count"]
        )

        metadata[key] = row

    rows = []

    for true_state in range(
        STATE_COUNT
    ):
        for predicted_state in range(
            STATE_COUNT
        ):
            key = (
                true_state,
                predicted_state,
            )

            source = metadata[key]

            rows.append(
                {
                    "true_state":
                        true_state,

                    "true_state_code":
                        source[
                            "true_state_code"
                        ],

                    "predicted_state":
                        predicted_state,

                    "predicted_state_code":
                        source[
                            "predicted_state_code"
                        ],

                    "coarsening_differs":
                        source[
                            "coarsening_differs"
                        ],

                    "anisotropy_differs":
                        source[
                            "anisotropy_differs"
                        ],

                    "defect_recovery_differs":
                        source[
                            "defect_recovery_differs"
                        ],

                    "factor_difference_count":
                        source[
                            "factor_difference_count"
                        ],

                    "count_across_seeds":
                        aggregate[key],
                }
            )

    write_csv(
        OUTPUT_DIR
        / "validation_aggregate_state_confusion.csv",
        rows,
    )

    off_diagonal = [
        row
        for row in rows
        if row["true_state"]
        != row["predicted_state"]
        and row[
            "count_across_seeds"
        ] > 0
    ]

    off_diagonal.sort(
        key=lambda row: (
            -row[
                "count_across_seeds"
            ],
            row["true_state"],
            row["predicted_state"],
        )
    )

    top_rows = off_diagonal[:20]

    write_csv(
        OUTPUT_DIR
        / "validation_top_state_confusions.csv",
        top_rows,
    )

    return rows, top_rows


def derive_diagnosis(
    factor_summary,
    state_summary,
    top_confusions,
    ensemble,
):
    validation_factor_rows = [
        row
        for row in factor_summary
        if row["split"] == "validation"
    ]

    validation_factor_rows.sort(
        key=lambda row:
            row["accuracy_mean"]
    )

    weakest_factor = (
        validation_factor_rows[0]
    )

    strongest_factor = (
        validation_factor_rows[-1]
    )

    validation_state_rows = [
        row
        for row in state_summary
        if row["split"] == "validation"
    ]

    validation_state_rows.sort(
        key=lambda row:
            row["accuracy_mean"]
    )

    weakest_states = (
        validation_state_rows[:3]
    )

    strongest_states = (
        list(
            reversed(
                validation_state_rows[-3:]
            )
        )
    )

    validation_ensemble = next(
        row
        for row in ensemble["metrics"]
        if row["split"] == "validation"
    )

    dominant_confusion = (
        top_confusions[0]
        if top_confusions
        else None
    )

    diagnosis = {
        "weakest_validation_factor":
            weakest_factor[
                "factor"
            ],

        "weakest_validation_factor_accuracy_mean":
            weakest_factor[
                "accuracy_mean"
            ],

        "weakest_validation_factor_accuracy_minimum":
            weakest_factor[
                "accuracy_minimum"
            ],

        "strongest_validation_factor":
            strongest_factor[
                "factor"
            ],

        "strongest_validation_factor_accuracy_mean":
            strongest_factor[
                "accuracy_mean"
            ],

        "weakest_validation_states":
            [
                {
                    "state_id":
                        row["state_id"],

                    "state_code":
                        row["state_code"],

                    "accuracy_mean":
                        row["accuracy_mean"],
                }
                for row in weakest_states
            ],

        "strongest_validation_states":
            [
                {
                    "state_id":
                        row["state_id"],

                    "state_code":
                        row["state_code"],

                    "accuracy_mean":
                        row["accuracy_mean"],
                }
                for row in strongest_states
            ],

        "dominant_validation_confusion":
            (
                {
                    "true_state":
                        dominant_confusion[
                            "true_state"
                        ],

                    "true_state_code":
                        dominant_confusion[
                            "true_state_code"
                        ],

                    "predicted_state":
                        dominant_confusion[
                            "predicted_state"
                        ],

                    "predicted_state_code":
                        dominant_confusion[
                            "predicted_state_code"
                        ],

                    "count_across_seeds":
                        dominant_confusion[
                            "count_across_seeds"
                        ],

                    "coarsening_differs":
                        dominant_confusion[
                            "coarsening_differs"
                        ],

                    "anisotropy_differs":
                        dominant_confusion[
                            "anisotropy_differs"
                        ],

                    "defect_recovery_differs":
                        dominant_confusion[
                            "defect_recovery_differs"
                        ],
                }
                if dominant_confusion
                else None
            ),

        "validation_probability_ensemble_accuracy":
            validation_ensemble[
                "probability_ensemble_state_accuracy"
            ],

        "validation_mean_pairwise_seed_agreement":
            validation_ensemble[
                "mean_pairwise_seed_prediction_agreement"
            ],

        "validation_any_seed_correct_fraction":
            validation_ensemble[
                "any_seed_correct_fraction"
            ],

        "validation_all_seeds_correct_fraction":
            validation_ensemble[
                "all_seeds_correct_fraction"
            ],

        "interpretation_rule": {
            "low_factor_accuracy":
                (
                    "The corresponding morphology control is "
                    "not reliably decoded across validation carriers."
                ),

            "high_any_seed_but_low_all_seed_accuracy":
                (
                    "The information is present but optimization "
                    "or representation extraction is seed-sensitive."
                ),

            "similar_any_seed_and_all_seed_accuracy":
                (
                    "Errors are consistent across seeds and more "
                    "likely reflect insufficient state separation."
                ),
        },

        "v3_design_authorized":
            False,

        "reason_v3_design_not_yet_authorized":
            (
                "This script freezes the diagnosis only. "
                "A separate Phase 4A-R2 protocol must use these "
                "results to define a fresh Tier C v3 dataset."
            ),

        "new_model_fitting_performed":
            False,

        "test_fields_used":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "diagnosis_findings.json",
        diagnosis,
    )

    return diagnosis


def write_input_hashes():
    paths = {
        "phase4cr_rejection_summary":
            PHASE4CR_DIR
            / "phase4cr_rejection_summary.json",

        "phase4cr_rejection_record":
            PHASE4CR_DIR
            / "phase4cr_rejection_record.json",

        "state_decoder_summary":
            PHASE4CR_DIR
            / "state_decoder_summary.json",

        "state_regimes":
            PHASE4AR_DIR
            / "state_morphology_regimes_v2.csv",

        "development_fields":
            PRIVILEGED_DIR
            / "development_carrier_state_fields_normalized.npy",
    }

    for seed in EXPECTED_DIAGNOSTIC_SEEDS:
        paths[
            f"state_decoder_seed_{seed}"
        ] = (
            PHASE4CR_DIR
            / f"state_decoder_seed_{seed}.pt"
        )

    hashes = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        hashes[name] = {
            "path":
                str(path),

            "sha256":
                sha256_file(path),
        }

    write_json(
        OUTPUT_DIR
        / "input_hashes.json",
        hashes,
    )


def main():
    arguments = parse_arguments()

    if arguments.batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = validate_rejection_freeze()

    sealing = validate_test_sealing()

    state_registry = (
        load_state_factor_registry()
    )

    dataset = load_development_dataset()

    device = resolve_device(
        arguments.device
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision(
            "high"
        )

    write_input_hashes()

    all_metric_rows = []
    all_state_rows = []
    all_factor_rows = []
    all_factor_confusion_rows = []
    all_state_confusion_rows = []
    all_hamming_rows = []
    all_error_pattern_rows = []

    seed_outputs = {}

    for index, seed in enumerate(
        EXPECTED_DIAGNOSTIC_SEEDS,
        start=1,
    ):
        print(
            f"[{index}/{len(EXPECTED_DIAGNOSTIC_SEEDS)}] "
            f"Diagnosing frozen decoder seed {seed}"
        )

        checkpoint_path = (
            PHASE4CR_DIR
            / f"state_decoder_seed_{seed}.pt"
        )

        checkpoint = safe_torch_load(
            checkpoint_path,
            map_location="cpu",
        )

        if int(
            checkpoint["seed"]
        ) != seed:
            raise AssertionError(
                "Checkpoint seed metadata changed."
            )

        if (
            checkpoint[
                "test_fields_used"
            ]
            is not False
        ):
            raise AssertionError(
                "A diagnostic checkpoint reports test use."
            )

        if (
            checkpoint[
                "eligible_for_predictive_use"
            ]
            is not False
        ):
            raise AssertionError(
                "Diagnostic checkpoint became "
                "eligible for predictive use."
            )

        model = (
            FrozenDiagnosticStateDecoder()
            .to(device)
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        result = evaluate_seed(
            seed=seed,
            model=model,
            dataset=dataset,
            state_registry=state_registry,
            device=device,
            batch_size=arguments.batch_size,
        )

        seed_outputs[seed] = result

        all_metric_rows.extend(
            result["metric_rows"]
        )

        all_state_rows.extend(
            result["state_rows"]
        )

        all_factor_rows.extend(
            result["factor_rows"]
        )

        all_factor_confusion_rows.extend(
            result[
                "factor_confusion_rows"
            ]
        )

        all_state_confusion_rows.extend(
            result[
                "state_confusion_rows"
            ]
        )

        all_hamming_rows.extend(
            result["hamming_rows"]
        )

        all_error_pattern_rows.extend(
            result[
                "error_pattern_rows"
            ]
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(
        OUTPUT_DIR
        / "state_metrics_by_seed.csv",
        all_metric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "state_accuracy_by_seed.csv",
        all_state_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "factor_accuracy_by_seed.csv",
        all_factor_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "factor_confusion_by_seed.csv",
        all_factor_confusion_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "state_confusion_by_seed.csv",
        all_state_confusion_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "factor_hamming_distance_by_seed.csv",
        all_hamming_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "factor_error_patterns_by_seed.csv",
        all_error_pattern_rows,
    )

    state_summary = summarize_grouped_rows(
        rows=all_state_rows,
        group_keys=(
            "split",
            "state_id",
            "state_code",
        ),
        value_key="accuracy",
    )

    factor_summary = summarize_grouped_rows(
        rows=all_factor_rows,
        group_keys=(
            "split",
            "factor",
        ),
        value_key="accuracy",
    )

    overall_summary = summarize_grouped_rows(
        rows=all_metric_rows,
        group_keys=(
            "split",
        ),
        value_key="state_accuracy",
    )

    write_csv(
        OUTPUT_DIR
        / "state_accuracy_summary.csv",
        state_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "factor_accuracy_summary.csv",
        factor_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "overall_accuracy_summary.csv",
        overall_summary,
    )

    ensemble = compute_ensemble_diagnostics(
        seed_outputs=seed_outputs,
        dataset=dataset,
        state_registry=state_registry,
    )

    (
        aggregate_confusion,
        top_confusions,
    ) = aggregate_validation_confusions(
        all_state_confusion_rows
    )

    diagnosis = derive_diagnosis(
        factor_summary=factor_summary,
        state_summary=state_summary,
        top_confusions=top_confusions,
        ensemble=ensemble,
    )

    observed_validation_rows = [
        row
        for row in all_metric_rows
        if row["split"] == "validation"
    ]

    observed_validation_accuracy = float(
        np.mean(
            [
                row["state_accuracy"]
                for row in observed_validation_rows
            ]
        )
    )

    frozen_validation_accuracy = float(
        sources[
            "decoder_summary"
        ][
            "mean_validation_accuracy"
        ]
    )

    reproduction_error = abs(
        observed_validation_accuracy
        - frozen_validation_accuracy
    )

    if reproduction_error > 1e-12:
        raise AssertionError(
            "Frozen validation accuracy was not "
            "reproduced exactly. "
            f"Observed={observed_validation_accuracy}, "
            f"frozen={frozen_validation_accuracy}."
        )

    summary = {
        "phase":
            (
                "4C-RD Tier C v2 development-only "
                "failure diagnosis"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_rejection_frozen":
            True,

        "tier_c_v2_rejected":
            True,

        "diagnostic_seed_count":
            len(
                EXPECTED_DIAGNOSTIC_SEEDS
            ),

        "diagnostic_seeds":
            list(
                EXPECTED_DIAGNOSTIC_SEEDS
            ),

        "training_sample_count":
            len(
                dataset[
                    "train"
                ][
                    "labels"
                ]
            ),

        "validation_sample_count":
            len(
                dataset[
                    "validation"
                ][
                    "labels"
                ]
            ),

        "frozen_mean_validation_state_accuracy":
            frozen_validation_accuracy,

        "reproduced_mean_validation_state_accuracy":
            observed_validation_accuracy,

        "validation_accuracy_reproduction_error":
            reproduction_error,

        "weakest_validation_factor":
            diagnosis[
                "weakest_validation_factor"
            ],

        "weakest_validation_factor_accuracy_mean":
            diagnosis[
                "weakest_validation_factor_accuracy_mean"
            ],

        "strongest_validation_factor":
            diagnosis[
                "strongest_validation_factor"
            ],

        "strongest_validation_factor_accuracy_mean":
            diagnosis[
                "strongest_validation_factor_accuracy_mean"
            ],

        "validation_probability_ensemble_accuracy":
            diagnosis[
                "validation_probability_ensemble_accuracy"
            ],

        "validation_mean_pairwise_seed_agreement":
            diagnosis[
                "validation_mean_pairwise_seed_agreement"
            ],

        "validation_any_seed_correct_fraction":
            diagnosis[
                "validation_any_seed_correct_fraction"
            ],

        "validation_all_seeds_correct_fraction":
            diagnosis[
                "validation_all_seeds_correct_fraction"
            ],

        "dominant_validation_confusion":
            diagnosis[
                "dominant_validation_confusion"
            ],

        "sealed_test_files_absent":
            not bool(
                sealing[
                    "sealed_test_files_found"
                ]
            ),

        "test_carrier_parameters_present":
            sealing[
                "test_carrier_parameters_present"
            ],

        "test_fields_read":
            False,

        "test_metrics_computed":
            False,

        "new_model_fitting_performed":
            False,

        "model_weights_modified":
            False,

        "predictive_training_performed":
            False,

        "predictive_checkpoint_selection_performed":
            False,

        "diagnostic_checkpoint_selection_reopened":
            False,

        "tier_c_v1_outputs_modified":
            False,

        "tier_c_v2_prior_outputs_modified":
            False,

        "v3_protocol_frozen":
            False,

        "phase4crd_status":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase4crd_summary.json",
        summary,
    )

    print()
    print(
        "Phase 4C-RD Tier C v2 failure "
        "diagnosis completed."
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
