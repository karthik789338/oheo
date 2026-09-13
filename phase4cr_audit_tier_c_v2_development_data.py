from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn


PHASE1A_DIR = Path(
    "outputs/phase1a_ground_truth"
)

PHASE1D_DIR = Path(
    "outputs/phase1d_compositional_splits"
)

PHASE4AR_DIR = Path(
    "outputs/phase4ar_tier_c_v2_protocol"
)

PHASE4BR_DIR = Path(
    "outputs/phase4br_tier_c_v2_development_data"
)

VISIBLE_DIR = PHASE4BR_DIR / "visible"
PRIVILEGED_DIR = PHASE4BR_DIR / "privileged"
MANIFEST_DIR = PHASE4BR_DIR / "cell_manifests"

OUTPUT_DIR = Path(
    "outputs/phase4cr_tier_c_v2_audit"
)


PROTOCOL_VERSION = "tier_c_v2"

STATE_COUNT = 8
OPERATION_COUNT = 6
SEMIGROUP_SIZE = 104

TRAIN_CARRIER_COUNT = 96
VALIDATION_CARRIER_COUNT = 32
DEVELOPMENT_CARRIER_COUNT = 128

FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32

FLATTENED_DIMENSION = (
    FIELD_CHANNEL_COUNT
    * GRID_HEIGHT
    * GRID_WIDTH
)

NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

DEVELOPMENT_CELLS = (
    "train_joint",
    "val_composition",
    "val_carrier",
    "val_joint",
)

SEALED_TEST_CELLS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)

CELL_CARRIER_SPLITS = {
    "train_joint": "train",
    "val_composition": "train",
    "val_carrier": "val",
    "val_joint": "val",
}

EXPECTED_TRANSFORMATION_SPLITS = {
    "train": 86,
    "val": 9,
    "test": 9,
}

DIAGNOSTIC_SEEDS = (
    52011,
    52023,
    52037,
)

STATE_DECODER_MAXIMUM_EPOCHS = 300
STATE_DECODER_PATIENCE = 50
STATE_DECODER_BATCH_SIZE = 64

GLOBAL_CLASSIFIER_MAXIMUM_EPOCHS = 1500
GLOBAL_CLASSIFIER_PATIENCE = 150

MINIMUM_MEAN_VALIDATION_STATE_ACCURACY = 0.80
MINIMUM_WORST_VALIDATION_STATE_ACCURACY = 0.75
MAXIMUM_GLOBAL_SHORTCUT_VALIDATION_ACCURACY = 0.95

MINIMUM_ENERGY_NONINCREASE_FRACTION = 0.99
MAXIMUM_PHASE_MEAN_DRIFT = 1e-5

PHASE_MINIMUM = -1.000001
PHASE_MAXIMUM = 1.000001

DEFECT_MINIMUM = -0.000001
DEFECT_MAXIMUM = 1.000001

AFFINE_PCA_DIMENSION = 128
AFFINE_RIDGE = 1e-4
AFFINE_MINIMUM_VALIDATION_MSE = 1e-8


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
        "--zero-noise-chunk-size",
        type=int,
        default=256,
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


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def parse_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


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


def parse_mapping(value):
    if isinstance(value, list):
        parsed = value
    else:
        parsed = json.loads(
            str(value)
        )

    mapping = tuple(
        int(item)
        for item in parsed
    )

    if len(mapping) != STATE_COUNT:
        raise AssertionError(
            "Transformation mapping size changed."
        )

    return mapping


def parse_word(
    value,
    word_length,
):
    if word_length == 0:
        return tuple()

    if isinstance(value, list):
        parsed = value

    else:
        text = str(value).strip()

        try:
            parsed = json.loads(text)

        except json.JSONDecodeError:
            if "->" in text:
                parsed = [
                    item.strip()
                    for item in text.split("->")
                    if item.strip()
                ]

            else:
                parsed = [
                    item.strip()
                    for item in text.split(",")
                    if item.strip()
                ]

    word = tuple(
        str(item)
        for item in parsed
    )

    if len(word) != word_length:
        raise AssertionError(
            "Transformation word length changed."
        )

    return word


def validate_sources():
    phase4ar = load_json(
        PHASE4AR_DIR
        / "phase4ar_summary.json"
    )

    field_bank = load_json(
        PHASE4BR_DIR
        / "phase4br_field_bank_summary.json"
    )

    dataset = load_json(
        PHASE4BR_DIR
        / "phase4br_summary.json"
    )

    acceptance_gates = load_json(
        PHASE4AR_DIR
        / "acceptance_gates_v2.json"
    )

    sealing_policy = load_json(
        PHASE4AR_DIR
        / "test_sealing_policy.json"
    )

    if (
        phase4ar["phase4ar_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A-R is not frozen."
        )

    if (
        phase4ar["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Protocol version changed."
        )

    if (
        field_bank[
            "field_bank_generation_status"
        ]
        != "completed"
    ):
        raise AssertionError(
            "Development field bank is incomplete."
        )

    if (
        dataset["phase4br_status"]
        != "completed"
    ):
        raise AssertionError(
            "Development dataset is incomplete."
        )

    if (
        field_bank[
            "test_carrier_count_generated"
        ]
        != 0
    ):
        raise AssertionError(
            "Test carriers were generated."
        )

    if (
        dataset["test_cell_count_generated"]
        != 0
    ):
        raise AssertionError(
            "Test cells were generated."
        )

    if (
        dataset["sealed_test_files_absent"]
        is not True
    ):
        raise AssertionError(
            "Sealed test files are not absent."
        )

    if (
        dataset["training_performed"]
        is not False
    ):
        raise AssertionError(
            "Predictive training occurred before Phase 4C-R."
        )

    if (
        dataset["test_metrics_computed"]
        is not False
    ):
        raise AssertionError(
            "Test metrics were computed before Phase 4C-R."
        )

    return {
        "phase4ar": phase4ar,
        "field_bank": field_bank,
        "dataset": dataset,
        "acceptance_gates": acceptance_gates,
        "sealing_policy": sealing_policy,
    }


def load_carrier_split():
    rows = load_csv(
        PHASE4AR_DIR
        / "carrier_split_v2.csv"
    )

    if len(rows) != 160:
        raise AssertionError(
            "Frozen carrier count changed."
        )

    split_counts = Counter(
        row["split"]
        for row in rows
    )

    if dict(split_counts) != {
        "train": 96,
        "val": 32,
        "test": 32,
    }:
        raise AssertionError(
            f"Frozen carrier counts changed: "
            f"{dict(split_counts)}"
        )

    split_ids = {
        split: {
            int(row["carrier_id"])
            for row in rows
            if row["split"] == split
        }
        for split in (
            "train",
            "val",
            "test",
        )
    }

    return rows, split_ids


def load_development_field_bank():
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

    raw_bank = np.load(
        PRIVILEGED_DIR
        / "development_carrier_state_fields_raw.npy",
        mmap_mode="r",
    )

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

    if raw_bank.shape != expected_shape:
        raise AssertionError(
            f"Raw field shape changed: {raw_bank.shape}"
        )

    if normalized_bank.shape != expected_shape:
        raise AssertionError(
            "Normalized field shape changed."
        )

    if raw_bank.dtype != np.float32:
        raise AssertionError(
            "Raw development fields are not float32."
        )

    if normalized_bank.dtype != np.float32:
        raise AssertionError(
            "Normalized development fields are not float32."
        )

    if len(set(carrier_ids.tolist())) != (
        DEVELOPMENT_CARRIER_COUNT
    ):
        raise AssertionError(
            "Development carrier IDs are not unique."
        )

    train_local_indices = np.where(
        carrier_splits == "train"
    )[0]

    validation_local_indices = np.where(
        carrier_splits == "val"
    )[0]

    if len(train_local_indices) != (
        TRAIN_CARRIER_COUNT
    ):
        raise AssertionError(
            "Training carrier count changed."
        )

    if len(validation_local_indices) != (
        VALIDATION_CARRIER_COUNT
    ):
        raise AssertionError(
            "Validation carrier count changed."
        )

    return {
        "carrier_ids":
            carrier_ids,

        "carrier_splits":
            carrier_splits,

        "raw_bank":
            raw_bank,

        "normalized_bank":
            normalized_bank,

        "train_local_indices":
            train_local_indices,

        "validation_local_indices":
            validation_local_indices,

        "carrier_id_to_local_index": {
            int(carrier_id):
                int(local_index)
            for local_index, carrier_id
            in enumerate(carrier_ids)
        },
    }


def audit_physics():
    rows = load_csv(
        PRIVILEGED_DIR
        / "development_physics_diagnostics.csv"
    )

    if len(rows) != (
        DEVELOPMENT_CARRIER_COUNT
        * STATE_COUNT
    ):
        raise AssertionError(
            "Physics diagnostic count changed."
        )

    all_values_finite = bool(
        all(
            parse_bool(
                row["all_values_finite"]
            )
            for row in rows
        )
    )

    energy_nonincrease_fraction = float(
        np.mean(
            [
                parse_bool(
                    row["energy_nonincrease"]
                )
                for row in rows
            ]
        )
    )

    maximum_phase_mean_drift = float(
        max(
            float(
                row["phase_mean_drift"]
            )
            for row in rows
        )
    )

    maximum_defect_mean_drift = float(
        max(
            float(
                row["defect_mean_drift"]
            )
            for row in rows
        )
    )

    phase_minimum = float(
        min(
            float(
                row["phase_minimum"]
            )
            for row in rows
        )
    )

    phase_maximum = float(
        max(
            float(
                row["phase_maximum"]
            )
            for row in rows
        )
    )

    defect_minimum = float(
        min(
            float(
                row["defect_minimum"]
            )
            for row in rows
        )
    )

    defect_maximum = float(
        max(
            float(
                row["defect_maximum"]
            )
            for row in rows
        )
    )

    total_rejected_steps = int(
        sum(
            int(
                row["rejected_step_count"]
            )
            for row in rows
        )
    )

    phase_range_passed = bool(
        phase_minimum >= PHASE_MINIMUM
        and phase_maximum <= PHASE_MAXIMUM
    )

    defect_range_passed = bool(
        defect_minimum >= DEFECT_MINIMUM
        and defect_maximum <= DEFECT_MAXIMUM
    )

    passed = bool(
        all_values_finite
        and energy_nonincrease_fraction
        >= MINIMUM_ENERGY_NONINCREASE_FRACTION
        and maximum_phase_mean_drift
        <= MAXIMUM_PHASE_MEAN_DRIFT
        and phase_range_passed
        and defect_range_passed
    )

    result = {
        "simulation_count":
            len(rows),

        "all_values_finite":
            all_values_finite,

        "energy_nonincrease_fraction":
            energy_nonincrease_fraction,

        "minimum_required_energy_nonincrease_fraction":
            MINIMUM_ENERGY_NONINCREASE_FRACTION,

        "maximum_phase_mean_drift":
            maximum_phase_mean_drift,

        "maximum_allowed_phase_mean_drift":
            MAXIMUM_PHASE_MEAN_DRIFT,

        "maximum_defect_mean_drift":
            maximum_defect_mean_drift,

        "phase_minimum":
            phase_minimum,

        "phase_maximum":
            phase_maximum,

        "phase_range_passed":
            phase_range_passed,

        "defect_minimum":
            defect_minimum,

        "defect_maximum":
            defect_maximum,

        "defect_range_passed":
            defect_range_passed,

        "total_rejected_step_proposals":
            total_rejected_steps,

        "unresolved_simulation_failure_count":
            0,

        "physics_gate_passed":
            passed,
    }

    write_json(
        OUTPUT_DIR
        / "physics_audit.json",
        result,
    )

    return result


def audit_basic_manifold(
    normalized_bank,
    train_local_indices,
):
    training_fields = np.asarray(
        normalized_bank[
            train_local_indices
        ],
        dtype=np.float32,
    )

    channel_variances = (
        training_fields.var(
            axis=(0, 1, 3, 4),
            dtype=np.float64,
        )
    )

    all_training_channels_nonconstant = bool(
        np.all(
            channel_variances > 1e-8
        )
    )

    state_variation_rows = []

    every_state_has_variation = True

    for state_id in range(
        STATE_COUNT
    ):
        fields = np.asarray(
            normalized_bank[
                :,
                state_id,
            ],
            dtype=np.float32,
        )

        mean_field = fields.mean(
            axis=0,
            dtype=np.float64,
        )

        mean_squared_variation = float(
            np.mean(
                (
                    fields
                    - mean_field[
                        None,
                        ...,
                    ]
                ) ** 2
            )
        )

        positive_variation = bool(
            mean_squared_variation > 1e-8
        )

        every_state_has_variation = bool(
            every_state_has_variation
            and positive_variation
        )

        state_variation_rows.append(
            {
                "state_id":
                    state_id,

                "mean_squared_carrier_variation":
                    mean_squared_variation,

                "positive_carrier_variation":
                    positive_variation,
            }
        )

    hashes = []

    flattened = normalized_bank.reshape(
        DEVELOPMENT_CARRIER_COUNT
        * STATE_COUNT,
        FLATTENED_DIMENSION,
    )

    for index in range(
        DEVELOPMENT_CARRIER_COUNT
        * STATE_COUNT
    ):
        values = np.asarray(
            flattened[index],
            dtype=np.float32,
        )

        hashes.append(
            hashlib.sha256(
                values.tobytes()
            ).hexdigest()
        )

    exact_duplicate_count = (
        len(hashes)
        - len(set(hashes))
    )

    no_exact_duplicates = bool(
        exact_duplicate_count == 0
    )

    result = {
        "field_count":
            DEVELOPMENT_CARRIER_COUNT
            * STATE_COUNT,

        "training_channel_variances":
            channel_variances.tolist(),

        "all_training_channels_nonconstant":
            all_training_channels_nonconstant,

        "every_state_has_positive_carrier_variation":
            every_state_has_variation,

        "exact_duplicate_field_count":
            exact_duplicate_count,

        "no_exact_duplicate_fields":
            no_exact_duplicates,

        "basic_manifold_gate_passed":
            bool(
                all_training_channels_nonconstant
                and every_state_has_variation
                and no_exact_duplicates
            ),
    }

    write_csv(
        OUTPUT_DIR
        / "state_carrier_variation.csv",
        state_variation_rows,
    )

    write_json(
        OUTPUT_DIR
        / "manifold_basic_audit.json",
        result,
    )

    return result


def build_state_dataset(
    normalized_bank,
    train_local_indices,
    validation_local_indices,
):
    fields = np.asarray(
        normalized_bank,
        dtype=np.float32,
    )

    state_labels = np.broadcast_to(
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
        "train": {
            "x":
                fields[
                    train_local_indices
                ].reshape(
                    -1,
                    FIELD_CHANNEL_COUNT,
                    GRID_HEIGHT,
                    GRID_WIDTH,
                ).copy(),

            "y":
                state_labels[
                    train_local_indices
                ].reshape(-1).copy(),
        },

        "val": {
            "x":
                fields[
                    validation_local_indices
                ].reshape(
                    -1,
                    FIELD_CHANNEL_COUNT,
                    GRID_HEIGHT,
                    GRID_WIDTH,
                ).copy(),

            "y":
                state_labels[
                    validation_local_indices
                ].reshape(-1).copy(),
        },
    }


def extract_global_features(fields):
    fields64 = fields.astype(
        np.float64,
        copy=False,
    )

    mean = fields64.mean(
        axis=(-2, -1)
    )

    centered = (
        fields64
        - mean[
            :,
            :,
            None,
            None,
        ]
    )

    variance = (
        centered**2
    ).mean(
        axis=(-2, -1)
    )

    standard_deviation = np.sqrt(
        np.maximum(
            variance,
            1e-12,
        )
    )

    skewness = (
        centered**3
    ).mean(
        axis=(-2, -1)
    ) / (
        standard_deviation**3
    )

    kurtosis = (
        centered**4
    ).mean(
        axis=(-2, -1)
    ) / (
        standard_deviation**4
    ) - 3.0

    quantile_10 = np.quantile(
        fields64,
        0.10,
        axis=(-2, -1),
    )

    median = np.quantile(
        fields64,
        0.50,
        axis=(-2, -1),
    )

    quantile_90 = np.quantile(
        fields64,
        0.90,
        axis=(-2, -1),
    )

    return np.concatenate(
        [
            mean,
            variance,
            skewness,
            kurtosis,
            quantile_10,
            median,
            quantile_90,
        ],
        axis=1,
    ).astype(
        np.float32
    )


class GlobalStatisticsClassifier(nn.Module):

    def __init__(self, feature_count):
        super().__init__()

        self.classifier = nn.Linear(
            feature_count,
            STATE_COUNT,
        )

    def forward(self, features):
        return self.classifier(features)


@torch.no_grad()
def evaluate_feature_classifier(
    model,
    features,
    labels,
    device,
):
    model.eval()

    x = torch.from_numpy(
        features
    ).to(
        device=device,
        dtype=torch.float32,
    )

    y = torch.from_numpy(
        labels
    ).to(
        device=device,
        dtype=torch.long,
    )

    logits = model(x)

    loss = nn.functional.cross_entropy(
        logits,
        y,
    )

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    return {
        "loss":
            float(
                loss.item()
            ),

        "accuracy":
            float(
                (
                    predictions == y
                ).float().mean().item()
            ),
    }


def train_global_statistics_classifier(
    dataset,
    device,
):
    seed = DIAGNOSTIC_SEEDS[0]

    set_random_seed(seed)

    train_features = extract_global_features(
        dataset["train"]["x"]
    )

    validation_features = (
        extract_global_features(
            dataset["val"]["x"]
        )
    )

    feature_mean = train_features.mean(
        axis=0,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    feature_standard_deviation = (
        train_features.std(
            axis=0,
            dtype=np.float64,
        ).astype(
            np.float32
        )
    )

    feature_standard_deviation = np.maximum(
        feature_standard_deviation,
        1e-6,
    )

    train_features = (
        train_features
        - feature_mean
    ) / feature_standard_deviation

    validation_features = (
        validation_features
        - feature_mean
    ) / feature_standard_deviation

    model = GlobalStatisticsClassifier(
        feature_count=train_features.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.03,
        weight_decay=1e-3,
    )

    train_x = torch.from_numpy(
        train_features
    ).to(
        device=device,
        dtype=torch.float32,
    )

    train_y = torch.from_numpy(
        dataset["train"]["y"]
    ).to(
        device=device,
        dtype=torch.long,
    )

    best_state = None
    best_validation_loss = math.inf
    best_epoch = None
    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        GLOBAL_CLASSIFIER_MAXIMUM_EPOCHS + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(train_x)

        loss = nn.functional.cross_entropy(
            logits,
            train_y,
        )

        loss.backward()
        optimizer.step()

        train_metrics = (
            evaluate_feature_classifier(
                model=model,
                features=train_features,
                labels=dataset["train"]["y"],
                device=device,
            )
        )

        validation_metrics = (
            evaluate_feature_classifier(
                model=model,
                features=validation_features,
                labels=dataset["val"]["y"],
                device=device,
            )
        )

        history.append(
            {
                "epoch":
                    epoch,

                "training_loss":
                    train_metrics["loss"],

                "training_accuracy":
                    train_metrics["accuracy"],

                "validation_loss":
                    validation_metrics["loss"],

                "validation_accuracy":
                    validation_metrics["accuracy"],
            }
        )

        improvement = (
            best_validation_loss
            - validation_metrics["loss"]
        )

        if improvement > 1e-7:
            best_validation_loss = (
                validation_metrics["loss"]
            )

            best_epoch = epoch

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= GLOBAL_CLASSIFIER_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "No global-statistics checkpoint selected."
        )

    model.load_state_dict(
        best_state
    )

    train_metrics = (
        evaluate_feature_classifier(
            model=model,
            features=train_features,
            labels=dataset["train"]["y"],
            device=device,
        )
    )

    validation_metrics = (
        evaluate_feature_classifier(
            model=model,
            features=validation_features,
            labels=dataset["val"]["y"],
            device=device,
        )
    )

    write_csv(
        OUTPUT_DIR
        / "global_statistics_classifier_history.csv",
        history,
    )

    torch.save(
        {
            "phase":
                "4C-R global-statistics shortcut diagnostic",

            "protocol_version":
                PROTOCOL_VERSION,

            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "feature_mean":
                feature_mean,

            "feature_standard_deviation":
                feature_standard_deviation,

            "model_state_dict":
                best_state,

            "eligible_for_predictive_use":
                False,
        },
        OUTPUT_DIR
        / "global_statistics_classifier.pt",
    )

    shortcut_absent = bool(
        validation_metrics["accuracy"]
        <= MAXIMUM_GLOBAL_SHORTCUT_VALIDATION_ACCURACY
    )

    result = {
        "feature_count":
            int(
                train_features.shape[1]
            ),

        "features_per_channel": [
            "mean",
            "variance",
            "skewness",
            "excess_kurtosis",
            "10th_percentile",
            "median",
            "90th_percentile",
        ],

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "epochs_executed":
            len(history),

        "training_accuracy":
            train_metrics["accuracy"],

        "validation_accuracy":
            validation_metrics["accuracy"],

        "maximum_allowed_validation_accuracy":
            MAXIMUM_GLOBAL_SHORTCUT_VALIDATION_ACCURACY,

        "global_statistics_shortcut_absent":
            shortcut_absent,

        "test_fields_used":
            False,

        "eligible_for_predictive_use":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "global_statistics_classifier_metrics.json",
        result,
    )

    return result


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


def periodic_translation_augmentation(
    fields,
):
    shift_y = int(
        torch.randint(
            low=0,
            high=GRID_HEIGHT,
            size=(1,),
            device=fields.device,
        ).item()
    )

    shift_x = int(
        torch.randint(
            low=0,
            high=GRID_WIDTH,
            size=(1,),
            device=fields.device,
        ).item()
    )

    return torch.roll(
        fields,
        shifts=(
            shift_y,
            shift_x,
        ),
        dims=(-2, -1),
    )


@torch.no_grad()
def evaluate_state_decoder(
    model,
    fields,
    labels,
    device,
    batch_size=128,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    count = 0

    for start in range(
        0,
        len(fields),
        batch_size,
    ):
        end = min(
            len(fields),
            start + batch_size,
        )

        x = torch.from_numpy(
            fields[start:end].copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )

        y = torch.from_numpy(
            labels[start:end].copy()
        ).to(
            device=device,
            dtype=torch.long,
        )

        logits = model(x)

        loss = nn.functional.cross_entropy(
            logits,
            y,
            reduction="sum",
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        total_loss += float(
            loss.item()
        )

        correct += int(
            (
                predictions == y
            ).sum().item()
        )

        count += len(x)

    return {
        "loss":
            total_loss / count,

        "accuracy":
            correct / count,
    }


def train_one_state_decoder(
    dataset,
    seed,
    device,
):
    set_random_seed(seed)

    model = FrozenDiagnosticStateDecoder().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    train_x = dataset["train"]["x"]
    train_y = dataset["train"]["y"]

    generator = np.random.default_rng(
        seed
    )

    best_state = None
    best_validation_loss = math.inf
    best_epoch = None
    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        STATE_DECODER_MAXIMUM_EPOCHS + 1,
    ):
        model.train()

        permutation = generator.permutation(
            len(train_x)
        )

        total_loss = 0.0
        total_count = 0
        correct = 0

        for start in range(
            0,
            len(train_x),
            STATE_DECODER_BATCH_SIZE,
        ):
            indices = permutation[
                start:
                start + STATE_DECODER_BATCH_SIZE
            ]

            x = torch.from_numpy(
                train_x[indices].copy()
            ).to(
                device=device,
                dtype=torch.float32,
            )

            y = torch.from_numpy(
                train_y[indices].copy()
            ).to(
                device=device,
                dtype=torch.long,
            )

            x = periodic_translation_augmentation(
                x
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_count = len(x)

            total_loss += float(
                loss.item()
            ) * batch_count

            correct += int(
                (
                    torch.argmax(
                        logits,
                        dim=1,
                    )
                    == y
                ).sum().item()
            )

            total_count += batch_count

        validation_metrics = (
            evaluate_state_decoder(
                model=model,
                fields=dataset["val"]["x"],
                labels=dataset["val"]["y"],
                device=device,
            )
        )

        history.append(
            {
                "seed":
                    seed,

                "epoch":
                    epoch,

                "training_loss":
                    total_loss / total_count,

                "training_accuracy":
                    correct / total_count,

                "validation_loss":
                    validation_metrics["loss"],

                "validation_accuracy":
                    validation_metrics["accuracy"],
            }
        )

        improvement = (
            best_validation_loss
            - validation_metrics["loss"]
        )

        if improvement > 1e-6:
            best_validation_loss = (
                validation_metrics["loss"]
            )

            best_epoch = epoch

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= STATE_DECODER_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            f"No decoder checkpoint selected "
            f"for seed {seed}."
        )

    model.load_state_dict(
        best_state
    )

    training_metrics = evaluate_state_decoder(
        model=model,
        fields=dataset["train"]["x"],
        labels=dataset["train"]["y"],
        device=device,
    )

    validation_metrics = (
        evaluate_state_decoder(
            model=model,
            fields=dataset["val"]["x"],
            labels=dataset["val"]["y"],
            device=device,
        )
    )

    write_csv(
        OUTPUT_DIR
        / f"state_decoder_seed_{seed}_history.csv",
        history,
    )

    torch.save(
        {
            "phase":
                "4C-R frozen diagnostic state decoder",

            "protocol_version":
                PROTOCOL_VERSION,

            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "model_state_dict":
                best_state,

            "trained_on":
                "clean training-carrier fields only",

            "selected_on":
                "clean validation-carrier fields only",

            "test_fields_used":
                False,

            "eligible_for_predictive_use":
                False,
        },
        OUTPUT_DIR
        / f"state_decoder_seed_{seed}.pt",
    )

    return {
        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "epochs_executed":
            len(history),

        "training_loss":
            training_metrics["loss"],

        "training_accuracy":
            training_metrics["accuracy"],

        "validation_loss":
            validation_metrics["loss"],

        "validation_accuracy":
            validation_metrics["accuracy"],
    }


def run_state_decoder_audit(
    dataset,
    device,
):
    seed_rows = []

    for index, seed in enumerate(
        DIAGNOSTIC_SEEDS,
        start=1,
    ):
        print(
            f"    decoder seed "
            f"{index}/{len(DIAGNOSTIC_SEEDS)}: "
            f"{seed}"
        )

        seed_rows.append(
            train_one_state_decoder(
                dataset=dataset,
                seed=seed,
                device=device,
            )
        )

    validation_accuracies = np.asarray(
        [
            row["validation_accuracy"]
            for row in seed_rows
        ],
        dtype=np.float64,
    )

    training_accuracies = np.asarray(
        [
            row["training_accuracy"]
            for row in seed_rows
        ],
        dtype=np.float64,
    )

    mean_validation_accuracy = float(
        validation_accuracies.mean()
    )

    worst_validation_accuracy = float(
        validation_accuracies.min()
    )

    passed = bool(
        mean_validation_accuracy
        >= MINIMUM_MEAN_VALIDATION_STATE_ACCURACY
        and worst_validation_accuracy
        >= MINIMUM_WORST_VALIDATION_STATE_ACCURACY
    )

    write_csv(
        OUTPUT_DIR
        / "state_decoder_seed_results.csv",
        seed_rows,
    )

    summary = {
        "diagnostic_model":
            "frozen residual convolutional state decoder",

        "diagnostic_seeds":
            list(DIAGNOSTIC_SEEDS),

        "seed_count":
            len(DIAGNOSTIC_SEEDS),

        "mean_training_accuracy":
            float(
                training_accuracies.mean()
            ),

        "mean_validation_accuracy":
            mean_validation_accuracy,

        "validation_accuracy_standard_deviation":
            float(
                validation_accuracies.std(
                    ddof=1
                )
            ),

        "worst_seed_validation_accuracy":
            worst_validation_accuracy,

        "best_seed_validation_accuracy":
            float(
                validation_accuracies.max()
            ),

        "minimum_required_mean_validation_accuracy":
            MINIMUM_MEAN_VALIDATION_STATE_ACCURACY,

        "minimum_required_worst_seed_validation_accuracy":
            MINIMUM_WORST_VALIDATION_STATE_ACCURACY,

        "state_separability_gate_passed":
            passed,

        "test_fields_used":
            False,

        "diagnostic_checkpoints_eligible_for_predictive_use":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "state_decoder_summary.json",
        summary,
    )

    return summary


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

            generators[str(name)] = tuple(
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

    if len(generators) != OPERATION_COUNT:
        raise AssertionError(
            "Primitive-operation count changed."
        )

    return generators


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

        semigroup.append(
            {
                "element_id":
                    str(
                        row["element_id"]
                    ),

                "mapping":
                    parse_mapping(
                        row["mapping"]
                    ),

                "word":
                    parse_word(
                        row["shortest_word"],
                        word_length,
                    ),

                "word_length":
                    word_length,
            }
        )

    if len(semigroup) != SEMIGROUP_SIZE:
        raise AssertionError(
            "Semigroup size changed."
        )

    return semigroup


def fit_affine_operator(
    source,
    target,
):
    ones = torch.ones(
        (
            source.shape[0],
            1,
        ),
        dtype=source.dtype,
        device=source.device,
    )

    augmented = torch.cat(
        [
            source,
            ones,
        ],
        dim=1,
    )

    gram = (
        augmented.T
        @ augmented
    )

    regularizer = torch.eye(
        gram.shape[0],
        dtype=gram.dtype,
        device=gram.device,
    )

    regularizer[-1, -1] = 0.0

    right_hand_side = (
        augmented.T
        @ target
    )

    solution = torch.linalg.solve(
        gram
        + AFFINE_RIDGE
        * regularizer,
        right_hand_side,
    )

    return (
        solution[:-1],
        solution[-1],
    )


def decode_same_carrier_states(
    predicted,
    candidates,
):
    distances = (
        predicted[
            :,
            :,
            None,
            :,
        ]
        - candidates[
            :,
            None,
            :,
            :,
        ]
    ).pow(
        2
    ).mean(
        dim=-1
    )

    return torch.argmin(
        distances,
        dim=-1,
    )


def evaluate_primitive_affine(
    latent_fields,
    flattened_fields,
    validation_indices,
    generators,
    matrices,
    biases,
    pca_mean,
    pca_basis,
):
    validation_tensor = torch.as_tensor(
        validation_indices,
        dtype=torch.long,
        device=latent_fields.device,
    )

    candidates = latent_fields[
        validation_tensor
    ]

    candidate_flattened = flattened_fields[
        validation_tensor
    ]

    mapping_correct = 0
    mapping_total = 0

    exact_count = 0
    exact_total = 0

    continuous_squared_error = 0.0
    continuous_value_count = 0

    carrier_operation_exact = np.zeros(
        (
            len(validation_indices),
            len(generators),
        ),
        dtype=bool,
    )

    for operation_index, (
        operation,
        mapping,
    ) in enumerate(
        generators.items()
    ):
        mapping_tensor = torch.as_tensor(
            mapping,
            dtype=torch.long,
            device=latent_fields.device,
        )

        predicted = (
            candidates
            @ matrices[operation]
            + biases[operation]
        )

        decoded = (
            decode_same_carrier_states(
                predicted=predicted,
                candidates=candidates,
            )
        )

        expected = mapping_tensor[
            None,
            :
        ].expand_as(
            decoded
        )

        correct = (
            decoded == expected
        )

        mapping_correct += int(
            correct.sum().item()
        )

        mapping_total += int(
            correct.numel()
        )

        exact = torch.all(
            correct,
            dim=1,
        )

        exact_count += int(
            exact.sum().item()
        )

        exact_total += len(
            validation_indices
        )

        carrier_operation_exact[
            :,
            operation_index,
        ] = (
            exact.detach()
            .cpu()
            .numpy()
        )

        reconstructed = (
            predicted
            @ pca_basis.T
            + pca_mean
        )

        target = candidate_flattened[
            :,
            mapping_tensor,
            :,
        ]

        squared_error = (
            reconstructed
            - target
        ).pow(
            2
        )

        continuous_squared_error += float(
            squared_error.sum().item()
        )

        continuous_value_count += int(
            squared_error.numel()
        )

    return {
        "mapping_accuracy":
            mapping_correct
            / mapping_total,

        "exact_primitive_operation_rate":
            exact_count
            / exact_total,

        "continuous_mse":
            continuous_squared_error
            / continuous_value_count,

        "fraction_carriers_all_primitives_exact":
            float(
                np.mean(
                    np.all(
                        carrier_operation_exact,
                        axis=1,
                    )
                )
            ),
    }


def evaluate_all_transformations(
    latent_fields,
    validation_indices,
    semigroup,
    matrices,
    biases,
):
    validation_tensor = torch.as_tensor(
        validation_indices,
        dtype=torch.long,
        device=latent_fields.device,
    )

    candidates = latent_fields[
        validation_tensor
    ]

    mapping_correct = 0
    mapping_total = 0

    exact_count = 0
    exact_total = 0

    carrier_transformation_exact = np.zeros(
        (
            len(validation_indices),
            len(semigroup),
        ),
        dtype=bool,
    )

    for transformation_index, transformation in enumerate(
        semigroup
    ):
        current = candidates

        for operation in transformation[
            "word"
        ]:
            current = (
                current
                @ matrices[operation]
                + biases[operation]
            )

        decoded = (
            decode_same_carrier_states(
                predicted=current,
                candidates=candidates,
            )
        )

        expected_mapping = torch.as_tensor(
            transformation["mapping"],
            dtype=torch.long,
            device=latent_fields.device,
        )

        expected = expected_mapping[
            None,
            :
        ].expand_as(
            decoded
        )

        correct = (
            decoded == expected
        )

        mapping_correct += int(
            correct.sum().item()
        )

        mapping_total += int(
            correct.numel()
        )

        exact = torch.all(
            correct,
            dim=1,
        )

        exact_count += int(
            exact.sum().item()
        )

        exact_total += len(
            validation_indices
        )

        carrier_transformation_exact[
            :,
            transformation_index,
        ] = (
            exact.detach()
            .cpu()
            .numpy()
        )

    return {
        "mapping_accuracy":
            mapping_correct
            / mapping_total,

        "exact_transformation_rate":
            exact_count
            / exact_total,

        "fraction_validation_carriers_all_104_exact":
            float(
                np.mean(
                    np.all(
                        carrier_transformation_exact,
                        axis=1,
                    )
                )
            ),
    }


def run_affine_shortcut_audit(
    normalized_bank,
    train_local_indices,
    validation_local_indices,
    generators,
    semigroup,
    device,
):
    flattened_numpy = np.asarray(
        normalized_bank,
        dtype=np.float32,
    ).reshape(
        DEVELOPMENT_CARRIER_COUNT,
        STATE_COUNT,
        FLATTENED_DIMENSION,
    ).copy()

    flattened = torch.from_numpy(
        flattened_numpy
    ).to(
        device=device,
        dtype=torch.float32,
    )

    train_tensor = torch.as_tensor(
        train_local_indices,
        dtype=torch.long,
        device=device,
    )

    training_flattened = flattened[
        train_tensor
    ].reshape(
        -1,
        FLATTENED_DIMENSION,
    )

    pca_mean = training_flattened.mean(
        dim=0
    )

    centered = (
        training_flattened
        - pca_mean
    )

    (
        _,
        singular_values,
        right_vectors,
    ) = torch.linalg.svd(
        centered,
        full_matrices=False,
    )

    pca_basis = (
        right_vectors[
            :AFFINE_PCA_DIMENSION
        ].T.contiguous()
    )

    latent_fields = (
        flattened
        - pca_mean[
            None,
            None,
            :,
        ]
    ) @ pca_basis

    matrices = {}
    biases = {}

    for operation, mapping in (
        generators.items()
    ):
        mapping_tensor = torch.as_tensor(
            mapping,
            dtype=torch.long,
            device=device,
        )

        source = latent_fields[
            train_tensor
        ].reshape(
            -1,
            AFFINE_PCA_DIMENSION,
        )

        target = latent_fields[
            train_tensor
        ][
            :,
            mapping_tensor,
            :,
        ].reshape(
            -1,
            AFFINE_PCA_DIMENSION,
        )

        matrix, bias = fit_affine_operator(
            source=source,
            target=target,
        )

        matrices[operation] = matrix
        biases[operation] = bias

    primitive = evaluate_primitive_affine(
        latent_fields=latent_fields,
        flattened_fields=flattened,
        validation_indices=(
            validation_local_indices
        ),
        generators=generators,
        matrices=matrices,
        biases=biases,
        pca_mean=pca_mean,
        pca_basis=pca_basis,
    )

    transformations = (
        evaluate_all_transformations(
            latent_fields=latent_fields,
            validation_indices=(
                validation_local_indices
            ),
            semigroup=semigroup,
            matrices=matrices,
            biases=biases,
        )
    )

    continuous_interpolation_fails = bool(
        primitive["continuous_mse"]
        > AFFINE_MINIMUM_VALIDATION_MSE
    )

    primitive_not_exact = bool(
        primitive[
            "exact_primitive_operation_rate"
        ] < 1.0
    )

    all_104_not_exact_everywhere = bool(
        transformations[
            "fraction_validation_carriers_all_104_exact"
        ] < 1.0
    )

    passed = bool(
        continuous_interpolation_fails
        and primitive_not_exact
        and all_104_not_exact_everywhere
    )

    np.savez_compressed(
        OUTPUT_DIR
        / "affine_diagnostic_parameters.npz",
        pca_mean=(
            pca_mean.detach()
            .cpu()
            .numpy()
        ),
        pca_basis=(
            pca_basis.detach()
            .cpu()
            .numpy()
        ),
        singular_values=(
            singular_values.detach()
            .cpu()
            .numpy()
        ),
        operation_names=np.asarray(
            list(generators),
            dtype=object,
        ),
        matrices=np.stack(
            [
                matrices[operation]
                .detach()
                .cpu()
                .numpy()
                for operation in generators
            ],
            axis=0,
        ),
        biases=np.stack(
            [
                biases[operation]
                .detach()
                .cpu()
                .numpy()
                for operation in generators
            ],
            axis=0,
        ),
    )

    result = {
        "diagnostic":
            (
                "Training-only PCA-128 plus ridge "
                "affine primitive operators."
            ),

        "pca_dimension":
            AFFINE_PCA_DIMENSION,

        "ridge":
            AFFINE_RIDGE,

        "training_carrier_count":
            TRAIN_CARRIER_COUNT,

        "validation_carrier_count":
            VALIDATION_CARRIER_COUNT,

        "validation_primitive_mapping_accuracy":
            primitive["mapping_accuracy"],

        "validation_exact_primitive_operation_rate":
            primitive[
                "exact_primitive_operation_rate"
            ],

        "validation_primitive_continuous_mse":
            primitive["continuous_mse"],

        "fraction_validation_carriers_all_primitives_exact":
            primitive[
                "fraction_carriers_all_primitives_exact"
            ],

        "validation_all_104_mapping_accuracy":
            transformations[
                "mapping_accuracy"
            ],

        "validation_all_104_exact_transformation_rate":
            transformations[
                "exact_transformation_rate"
            ],

        "fraction_validation_carriers_all_104_exact":
            transformations[
                "fraction_validation_carriers_all_104_exact"
            ],

        "validation_continuous_interpolation_fails":
            continuous_interpolation_fails,

        "validation_primitive_mapping_not_exact":
            primitive_not_exact,

        "all_104_not_exact_on_every_validation_carrier":
            all_104_not_exact_everywhere,

        "affine_shortcut_removed":
            passed,

        "test_fields_used":
            False,

        "eligible_for_predictive_use":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "affine_shortcut_audit.json",
        result,
    )

    del flattened
    del latent_fields
    del training_flattened
    del centered

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def audit_zero_noise_consistency(
    cell_id,
    normalized_bank,
    chunk_size,
):
    fields = np.load(
        VISIBLE_DIR
        / f"{cell_id}_fields.npy",
        mmap_mode="r",
    )

    point_indices = np.load(
        PRIVILEGED_DIR
        / f"{cell_id}_point_bank_indices.npy",
        mmap_mode="r",
    )

    flattened_bank = (
        normalized_bank.reshape(
            DEVELOPMENT_CARRIER_COUNT
            * STATE_COUNT,
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        )
    )

    exact = True
    maximum_absolute_error = 0.0

    for start in range(
        0,
        len(point_indices),
        chunk_size,
    ):
        end = min(
            len(point_indices),
            start + chunk_size,
        )

        expected = np.asarray(
            flattened_bank[
                point_indices[start:end]
            ],
            dtype=np.float16,
        )

        observed = np.asarray(
            fields[
                0,
                start:end,
            ],
            dtype=np.float16,
        )

        if not np.array_equal(
            expected,
            observed,
        ):
            exact = False

        difference = float(
            np.max(
                np.abs(
                    expected.astype(
                        np.float32
                    )
                    - observed.astype(
                        np.float32
                    )
                )
            )
        )

        maximum_absolute_error = max(
            maximum_absolute_error,
            difference,
        )

    return {
        "cell_id":
            cell_id,

        "point_count":
            int(
                len(point_indices)
            ),

        "zero_noise_exact_after_float16_storage":
            exact,

        "maximum_absolute_error":
            maximum_absolute_error,
    }


def audit_noise_independence():
    cell_id = "train_joint"

    fields = np.load(
        VISIBLE_DIR
        / f"{cell_id}_fields.npy",
        mmap_mode="r",
    )

    point_indices = np.load(
        PRIVILEGED_DIR
        / f"{cell_id}_point_bank_indices.npy",
        mmap_mode="r",
    )

    unique_values, counts = np.unique(
        point_indices,
        return_counts=True,
    )

    repeated_values = unique_values[
        counts >= 2
    ]

    tested_pairs = 0
    differing_pairs = 0

    for bank_index in repeated_values[
        :64
    ]:
        locations = np.flatnonzero(
            point_indices == bank_index
        )

        if len(locations) < 2:
            continue

        first = np.asarray(
            fields[
                2,
                locations[0],
            ],
            dtype=np.float16,
        )

        second = np.asarray(
            fields[
                2,
                locations[1],
            ],
            dtype=np.float16,
        )

        tested_pairs += 1

        if not np.array_equal(
            first,
            second,
        ):
            differing_pairs += 1

    return {
        "cell_id":
            cell_id,

        "noise_fraction":
            NOISE_LEVELS[2],

        "repeated_clean_field_pairs_tested":
            tested_pairs,

        "pairs_with_different_noisy_realizations":
            differing_pairs,

        "independent_noise_empirically_confirmed":
            bool(
                tested_pairs > 0
                and differing_pairs
                == tested_pairs
            ),
    }


def find_sealed_test_files():
    found = []

    for directory in (
        VISIBLE_DIR,
        PRIVILEGED_DIR,
        MANIFEST_DIR,
    ):
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


def audit_data_integrity(
    field_data,
    frozen_split_ids,
    zero_noise_chunk_size,
):
    development_carrier_ids = set(
        int(value)
        for value in field_data[
            "carrier_ids"
        ]
    )

    frozen_development_ids = (
        frozen_split_ids["train"]
        | frozen_split_ids["val"]
    )

    frozen_test_ids = (
        frozen_split_ids["test"]
    )

    development_ids_match = bool(
        development_carrier_ids
        == frozen_development_ids
    )

    test_ids_absent = bool(
        not (
            development_carrier_ids
            & frozen_test_ids
        )
    )

    parameter_rows = load_csv(
        PHASE4BR_DIR
        / "development_carrier_parameters.csv"
    )

    parameter_splits = {
        row["split"]
        for row in parameter_rows
    }

    test_parameters_absent = bool(
        "test" not in parameter_splits
        and len(parameter_rows)
        == DEVELOPMENT_CARRIER_COUNT
    )

    sealed_test_files = (
        find_sealed_test_files()
    )

    cell_rows = []
    all_cell_headers_valid = True
    all_carrier_assignments_valid = True
    all_markers_valid = True

    zero_noise_rows = []

    for cell_id in DEVELOPMENT_CELLS:
        marker_path = (
            MANIFEST_DIR
            / f"{cell_id}_complete.json"
        )

        marker = load_json(
            marker_path
        )

        marker_valid = bool(
            marker["status"]
            == "completed"
            and marker[
                "test_carriers_used"
            ]
            is False
            and marker[
                "test_metrics_computed"
            ]
            is False
        )

        all_markers_valid = bool(
            all_markers_valid
            and marker_valid
        )

        fields = np.load(
            VISIBLE_DIR
            / f"{cell_id}_fields.npy",
            mmap_mode="r",
        )

        expected_shape = (
            len(NOISE_LEVELS),
            int(marker["point_count"]),
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        )

        header_valid = bool(
            fields.shape == expected_shape
            and fields.dtype == np.float16
        )

        all_cell_headers_valid = bool(
            all_cell_headers_valid
            and header_valid
        )

        del fields

        metadata_rows = load_csv(
            PRIVILEGED_DIR
            / f"{cell_id}_trajectory_metadata.csv"
        )

        observed_carriers = {
            int(row["carrier_id"])
            for row in metadata_rows
        }

        expected_split = (
            CELL_CARRIER_SPLITS[
                cell_id
            ]
        )

        carrier_assignment_valid = bool(
            observed_carriers
            <= frozen_split_ids[
                expected_split
            ]
        )

        all_carrier_assignments_valid = bool(
            all_carrier_assignments_valid
            and carrier_assignment_valid
        )

        point_indices = np.load(
            PRIVILEGED_DIR
            / f"{cell_id}_point_bank_indices.npy",
            mmap_mode="r",
        )

        point_indices_valid = bool(
            len(point_indices)
            == int(marker["point_count"])
            and int(point_indices.min()) >= 0
            and int(point_indices.max())
            < DEVELOPMENT_CARRIER_COUNT
            * STATE_COUNT
        )

        zero_noise_result = (
            audit_zero_noise_consistency(
                cell_id=cell_id,
                normalized_bank=field_data[
                    "normalized_bank"
                ],
                chunk_size=(
                    zero_noise_chunk_size
                ),
            )
        )

        zero_noise_rows.append(
            zero_noise_result
        )

        cell_rows.append(
            {
                "cell_id":
                    cell_id,

                "expected_carrier_split":
                    expected_split,

                "observed_unique_carrier_count":
                    len(
                        observed_carriers
                    ),

                "carrier_assignment_valid":
                    carrier_assignment_valid,

                "field_header_valid":
                    header_valid,

                "marker_valid":
                    marker_valid,

                "point_bank_indices_valid":
                    point_indices_valid,

                "zero_noise_exact":
                    zero_noise_result[
                        "zero_noise_exact_after_float16_storage"
                    ],

                "test_fields_used":
                    False,
            }
        )

    transformation_rows = load_csv(
        PHASE1D_DIR
        / "transformation_split.csv"
    )

    transformation_counts = dict(
        Counter(
            row["split"]
            for row in transformation_rows
        )
    )

    transformation_split_unchanged = bool(
        transformation_counts
        == EXPECTED_TRANSFORMATION_SPLITS
    )

    zero_noise_exact = bool(
        all(
            row[
                "zero_noise_exact_after_float16_storage"
            ]
            for row in zero_noise_rows
        )
    )

    noise_independence = (
        audit_noise_independence()
    )

    passed = bool(
        development_ids_match
        and test_ids_absent
        and test_parameters_absent
        and not sealed_test_files
        and all_cell_headers_valid
        and all_carrier_assignments_valid
        and all_markers_valid
        and transformation_split_unchanged
        and zero_noise_exact
        and noise_independence[
            "independent_noise_empirically_confirmed"
        ]
    )

    write_csv(
        OUTPUT_DIR
        / "data_integrity_by_cell.csv",
        cell_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "zero_noise_consistency.csv",
        zero_noise_rows,
    )

    result = {
        "development_carrier_ids_match_frozen_split":
            development_ids_match,

        "frozen_test_carrier_ids_absent":
            test_ids_absent,

        "test_carrier_parameters_absent":
            test_parameters_absent,

        "sealed_test_files_found":
            sealed_test_files,

        "sealed_test_files_absent":
            bool(
                not sealed_test_files
            ),

        "all_development_cell_headers_valid":
            all_cell_headers_valid,

        "all_cell_carrier_assignments_valid":
            all_carrier_assignments_valid,

        "all_cell_markers_valid":
            all_markers_valid,

        "transformation_split_counts":
            transformation_counts,

        "transformation_split_unchanged":
            transformation_split_unchanged,

        "zero_noise_all_development_cells_exact":
            zero_noise_exact,

        "noise_independence_audit":
            noise_independence,

        "test_fields_read":
            False,

        "test_metrics_computed":
            False,

        "data_integrity_gate_passed":
            passed,
    }

    write_json(
        OUTPUT_DIR
        / "data_integrity_audit.json",
        result,
    )

    return result


def main():
    arguments = parse_arguments()

    if arguments.zero_noise_chunk_size <= 0:
        raise ValueError(
            "Zero-noise chunk size must be positive."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = validate_sources()

    device = resolve_device(
        arguments.device
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision(
            "high"
        )

    set_random_seed(
        DIAGNOSTIC_SEEDS[0]
    )

    carrier_rows, frozen_split_ids = (
        load_carrier_split()
    )

    field_data = (
        load_development_field_bank()
    )

    print(
        "[1/6] Auditing numerical physics"
    )

    physics = audit_physics()

    print(
        "[2/6] Auditing basic manifold quality"
    )

    manifold = audit_basic_manifold(
        normalized_bank=field_data[
            "normalized_bank"
        ],
        train_local_indices=field_data[
            "train_local_indices"
        ],
    )

    state_dataset = build_state_dataset(
        normalized_bank=field_data[
            "normalized_bank"
        ],
        train_local_indices=field_data[
            "train_local_indices"
        ],
        validation_local_indices=field_data[
            "validation_local_indices"
        ],
    )

    print(
        "[3/6] Training frozen global-statistics "
        "shortcut diagnostic"
    )

    global_statistics = (
        train_global_statistics_classifier(
            dataset=state_dataset,
            device=device,
        )
    )

    print(
        "[4/6] Training three frozen state "
        "separability diagnostics"
    )

    state_decoder = run_state_decoder_audit(
        dataset=state_dataset,
        device=device,
    )

    print(
        "[5/6] Auditing validation affine shortcut"
    )

    generators = load_generators()
    semigroup = load_semigroup()

    affine = run_affine_shortcut_audit(
        normalized_bank=field_data[
            "normalized_bank"
        ],
        train_local_indices=field_data[
            "train_local_indices"
        ],
        validation_local_indices=field_data[
            "validation_local_indices"
        ],
        generators=generators,
        semigroup=semigroup,
        device=device,
    )

    print(
        "[6/6] Auditing data integrity and "
        "sealed-test absence"
    )

    integrity = audit_data_integrity(
        field_data=field_data,
        frozen_split_ids=frozen_split_ids,
        zero_noise_chunk_size=(
            arguments.zero_noise_chunk_size
        ),
    )

    all_gates_passed = bool(
        physics["physics_gate_passed"]
        and manifold[
            "basic_manifold_gate_passed"
        ]
        and state_decoder[
            "state_separability_gate_passed"
        ]
        and global_statistics[
            "global_statistics_shortcut_absent"
        ]
        and affine[
            "affine_shortcut_removed"
        ]
        and integrity[
            "data_integrity_gate_passed"
        ]
    )

    gate_rows = [
        {
            "gate":
                "numerical_physics",

            "passed":
                physics[
                    "physics_gate_passed"
                ],
        },
        {
            "gate":
                "basic_manifold_quality",

            "passed":
                manifold[
                    "basic_manifold_gate_passed"
                ],
        },
        {
            "gate":
                "three_seed_state_separability",

            "passed":
                state_decoder[
                    "state_separability_gate_passed"
                ],
        },
        {
            "gate":
                "global_statistics_shortcut_absent",

            "passed":
                global_statistics[
                    "global_statistics_shortcut_absent"
                ],
        },
        {
            "gate":
                "validation_affine_shortcut_removed",

            "passed":
                affine[
                    "affine_shortcut_removed"
                ],
        },
        {
            "gate":
                "development_data_integrity_and_test_sealing",

            "passed":
                integrity[
                    "data_integrity_gate_passed"
                ],
        },
    ]

    write_csv(
        OUTPUT_DIR
        / "acceptance_gate_results.csv",
        gate_rows,
    )

    summary = {
        "phase":
            "4C-R Tier C v2 development-data acceptance audit",

        "protocol_version":
            PROTOCOL_VERSION,

        "source_phase_validation": {
            "phase4ar":
                sources[
                    "phase4ar"
                ][
                    "phase4ar_status"
                ]
                == "protocol_frozen",

            "phase4br_field_bank":
                sources[
                    "field_bank"
                ][
                    "field_bank_generation_status"
                ]
                == "completed",

            "phase4br_dataset":
                sources[
                    "dataset"
                ][
                    "phase4br_status"
                ]
                == "completed",
        },

        "training_carrier_count":
            TRAIN_CARRIER_COUNT,

        "validation_carrier_count":
            VALIDATION_CARRIER_COUNT,

        "test_carrier_count_generated":
            0,

        "development_field_count":
            DEVELOPMENT_CARRIER_COUNT
            * STATE_COUNT,

        "physics_gate_passed":
            physics[
                "physics_gate_passed"
            ],

        "energy_nonincrease_fraction":
            physics[
                "energy_nonincrease_fraction"
            ],

        "maximum_phase_mean_drift":
            physics[
                "maximum_phase_mean_drift"
            ],

        "phase_range_passed":
            physics[
                "phase_range_passed"
            ],

        "defect_range_passed":
            physics[
                "defect_range_passed"
            ],

        "basic_manifold_gate_passed":
            manifold[
                "basic_manifold_gate_passed"
            ],

        "exact_duplicate_field_count":
            manifold[
                "exact_duplicate_field_count"
            ],

        "state_decoder_seed_count":
            state_decoder[
                "seed_count"
            ],

        "mean_validation_state_accuracy":
            state_decoder[
                "mean_validation_accuracy"
            ],

        "worst_seed_validation_state_accuracy":
            state_decoder[
                "worst_seed_validation_accuracy"
            ],

        "state_separability_gate_passed":
            state_decoder[
                "state_separability_gate_passed"
            ],

        "global_statistics_validation_accuracy":
            global_statistics[
                "validation_accuracy"
            ],

        "global_statistics_shortcut_absent":
            global_statistics[
                "global_statistics_shortcut_absent"
            ],

        "affine_validation_continuous_mse":
            affine[
                "validation_primitive_continuous_mse"
            ],

        "affine_validation_exact_primitive_operation_rate":
            affine[
                "validation_exact_primitive_operation_rate"
            ],

        "affine_fraction_validation_carriers_all_104_exact":
            affine[
                "fraction_validation_carriers_all_104_exact"
            ],

        "affine_shortcut_removed":
            affine[
                "affine_shortcut_removed"
            ],

        "data_integrity_gate_passed":
            integrity[
                "data_integrity_gate_passed"
            ],

        "sealed_test_files_absent":
            integrity[
                "sealed_test_files_absent"
            ],

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_fields_read":
            False,

        "test_metrics_computed":
            False,

        "predictive_model_training_performed":
            False,

        "diagnostic_state_decoders_fitted":
            True,

        "diagnostic_global_statistics_classifier_fitted":
            True,

        "diagnostic_affine_operator_fitted":
            True,

        "diagnostic_models_eligible_for_predictive_use":
            False,

        "checkpoint_selection_for_predictive_models_performed":
            False,

        "tier_c_v1_outputs_modified":
            False,

        "phase2_outputs_modified":
            False,

        "phase3_outputs_modified":
            False,

        "phase4ar_outputs_modified":
            False,

        "phase4br_outputs_modified":
            False,

        "tier_c_v2_development_data_accepted":
            all_gates_passed,

        "phase4cr_status":
            (
                "passed"
                if all_gates_passed
                else "failed_acceptance_gates"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4cr_summary.json",
        summary,
    )

    print()
    print(
        "Phase 4C-R audit completed."
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

    if not all_gates_passed:
        raise SystemExit(
            "Phase 4C-R acceptance gates failed. "
            "Do not proceed to spatial-model implementation."
        )


if __name__ == "__main__":
    main()
