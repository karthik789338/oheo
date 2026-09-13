from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
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

PHASE3B_DIR = Path(
    "outputs/phase3b_tier_b_data"
)

PHASE4A_DIR = Path(
    "outputs/phase4a_tier_c_physical_protocol"
)

PHASE4B_DIR = Path(
    "outputs/phase4b_tier_c_data"
)

VISIBLE_DIR = PHASE4B_DIR / "visible"
PRIVILEGED_DIR = PHASE4B_DIR / "privileged"
MANIFEST_DIR = PHASE4B_DIR / "cell_manifests"

OUTPUT_DIR = Path(
    "outputs/phase4c_tier_c_audit"
)


STATE_COUNT = 8
CARRIER_COUNT = 160
OPERATION_COUNT = 6
SEMIGROUP_SIZE = 104

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

EXPECTED_CARRIER_SPLITS = {
    "train": 96,
    "val": 32,
    "test": 32,
}

EXPECTED_TRANSFORMATION_SPLITS = {
    "train": 86,
    "val": 9,
    "test": 9,
}

EXPECTED_CELLS = (
    "train_joint",
    "val_composition",
    "val_carrier",
    "val_joint",
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)

EXPECTED_TOTAL_POINT_COUNT = 280880

EXPECTED_CELL_CARRIER_SPLIT = {
    "train_joint": "train",
    "val_composition": "train",
    "val_carrier": "val",
    "val_joint": "val",
    "test_iid_pairing": "train",
    "test_composition": "train",
    "test_carrier": "test",
    "test_joint": "test",
}

PHYSICS_MAXIMUM_PHASE_MEAN_DRIFT = 5e-3
PHYSICS_MINIMUM_ENERGY_NONINCREASE_FRACTION = 0.95

MINIMUM_PRIVILEGED_STATE_ACCURACY = 0.80
MAXIMUM_GLOBAL_STATISTICS_TEST_ACCURACY = 0.98

AFFINE_PCA_DIMENSION = 128
AFFINE_RIDGE = 1e-4
AFFINE_NUMERICAL_TOLERANCE = 1e-10

DIAGNOSTIC_SEED = 42031

STATE_DECODER_MAXIMUM_EPOCHS = 200
STATE_DECODER_PATIENCE = 30
STATE_DECODER_BATCH_SIZE = 64

GLOBAL_CLASSIFIER_MAXIMUM_EPOCHS = 1000
GLOBAL_CLASSIFIER_PATIENCE = 100


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
        newline="",
        encoding="utf-8",
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
        newline="",
        encoding="utf-8",
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


def parse_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


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
            "Transformation mapping has the wrong size."
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
            "Semigroup word length changed."
        )

    return word


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
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


def validate_sources():
    phase4a = load_json(
        PHASE4A_DIR
        / "phase4a_summary.json"
    )

    phase4b = load_json(
        PHASE4B_DIR
        / "phase4b_summary.json"
    )

    field_bank = load_json(
        PHASE4B_DIR
        / "phase4b_field_bank_summary.json"
    )

    if (
        phase4a["phase4a_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A is not frozen."
        )

    if (
        phase4b["phase4b_status"]
        != "completed"
    ):
        raise AssertionError(
            "Phase 4B dataset generation is incomplete."
        )

    if (
        field_bank[
            "field_bank_generation_status"
        ]
        != "completed"
    ):
        raise AssertionError(
            "Phase 4B field bank is incomplete."
        )

    if (
        phase4b["total_point_count"]
        != EXPECTED_TOTAL_POINT_COUNT
    ):
        raise AssertionError(
            "Tier C point count changed."
        )

    if (
        phase4b["evaluation_cell_count"]
        != len(EXPECTED_CELLS)
    ):
        raise AssertionError(
            "Tier C evaluation-cell count changed."
        )

    if (
        phase4b["training_performed"]
        is not False
    ):
        raise AssertionError(
            "Training occurred before Phase 4C."
        )

    if (
        phase4b[
            "checkpoint_selection_performed"
        ]
        is not False
    ):
        raise AssertionError(
            "Checkpoint selection occurred before Phase 4C."
        )

    if (
        phase4b["test_metrics_computed"]
        is not False
    ):
        raise AssertionError(
            "Test metrics were computed before Phase 4C."
        )

    return {
        "phase4a": phase4a,
        "phase4b": phase4b,
        "field_bank": field_bank,
    }


def load_carrier_information():
    rows = load_csv(
        PHASE4A_DIR
        / "carrier_reuse_manifest.csv"
    )

    rows.sort(
        key=lambda row:
            int(row["carrier_id"])
    )

    if len(rows) != CARRIER_COUNT:
        raise AssertionError(
            "Carrier count changed."
        )

    counts = Counter(
        row["split"]
        for row in rows
    )

    if dict(counts) != EXPECTED_CARRIER_SPLITS:
        raise AssertionError(
            f"Carrier split counts changed: {dict(counts)}"
        )

    carrier_ids = {
        split:
            np.asarray(
                [
                    int(row["carrier_id"])
                    for row in rows
                    if row["split"] == split
                ],
                dtype=np.int64,
            )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    if (
        set(carrier_ids["train"])
        & set(carrier_ids["val"])
    ):
        raise AssertionError(
            "Train and validation carriers overlap."
        )

    if (
        set(carrier_ids["train"])
        & set(carrier_ids["test"])
    ):
        raise AssertionError(
            "Train and test carriers overlap."
        )

    if (
        set(carrier_ids["val"])
        & set(carrier_ids["test"])
    ):
        raise AssertionError(
            "Validation and test carriers overlap."
        )

    return rows, carrier_ids


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
                    str(row["element_id"]),

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

    for name, mapping in generators.items():
        if len(mapping) != STATE_COUNT:
            raise AssertionError(
                f"Generator {name} has an invalid mapping."
            )

    return generators


def load_transformation_split():
    rows = load_csv(
        PHASE1D_DIR
        / "transformation_split.csv"
    )

    counts = Counter(
        row["split"]
        for row in rows
    )

    if dict(counts) != EXPECTED_TRANSFORMATION_SPLITS:
        raise AssertionError(
            "Transformation split counts changed."
        )

    if len(rows) != SEMIGROUP_SIZE:
        raise AssertionError(
            "Transformation split size changed."
        )

    return rows


def audit_physics():
    rows = load_csv(
        PRIVILEGED_DIR
        / "physics_diagnostics.csv"
    )

    if len(rows) != (
        CARRIER_COUNT
        * STATE_COUNT
    ):
        raise AssertionError(
            "Physics diagnostic count changed."
        )

    finite_flags = [
        parse_bool(
            row["all_values_finite"]
        )
        for row in rows
    ]

    energy_flags = [
        parse_bool(
            row["energy_nonincrease"]
        )
        for row in rows
    ]

    phase_drifts = np.asarray(
        [
            float(
                row["phase_mean_drift"]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    phase_minimum = min(
        float(
            row["preclip_phase_minimum"]
        )
        for row in rows
    )

    phase_maximum = max(
        float(
            row["preclip_phase_maximum"]
        )
        for row in rows
    )

    defect_minimum = min(
        float(
            row["preclip_defect_minimum"]
        )
        for row in rows
    )

    defect_maximum = max(
        float(
            row["preclip_defect_maximum"]
        )
        for row in rows
    )

    energy_fraction = float(
        np.mean(
            energy_flags
        )
    )

    maximum_phase_drift = float(
        phase_drifts.max()
    )

    all_values_finite = bool(
        all(finite_flags)
    )

    phase_range_passed = bool(
        phase_minimum >= -1.05
        and phase_maximum <= 1.05
    )

    defect_range_passed = bool(
        defect_minimum >= -0.05
        and defect_maximum <= 1.05
    )

    passed = bool(
        all_values_finite
        and energy_fraction
        >= PHYSICS_MINIMUM_ENERGY_NONINCREASE_FRACTION
        and maximum_phase_drift
        <= PHYSICS_MAXIMUM_PHASE_MEAN_DRIFT
        and phase_range_passed
        and defect_range_passed
    )

    result = {
        "simulation_count":
            len(rows),

        "all_values_finite":
            all_values_finite,

        "energy_nonincrease_fraction":
            energy_fraction,

        "minimum_required_energy_nonincrease_fraction":
            PHYSICS_MINIMUM_ENERGY_NONINCREASE_FRACTION,

        "maximum_phase_mean_drift":
            maximum_phase_drift,

        "maximum_allowed_phase_mean_drift":
            PHYSICS_MAXIMUM_PHASE_MEAN_DRIFT,

        "preclip_phase_minimum":
            phase_minimum,

        "preclip_phase_maximum":
            phase_maximum,

        "phase_range_passed":
            phase_range_passed,

        "preclip_defect_minimum":
            defect_minimum,

        "preclip_defect_maximum":
            defect_maximum,

        "defect_range_passed":
            defect_range_passed,

        "physics_gate_passed":
            passed,
    }

    write_json(
        OUTPUT_DIR
        / "physics_audit.json",
        result,
    )

    return result


def load_field_bank():
    raw = np.load(
        PRIVILEGED_DIR
        / "carrier_state_fields_raw.npy",
        mmap_mode="r",
    )

    normalized = np.load(
        PRIVILEGED_DIR
        / "carrier_state_fields_normalized.npy",
        mmap_mode="r",
    )

    expected_shape = (
        CARRIER_COUNT,
        STATE_COUNT,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    if raw.shape != expected_shape:
        raise AssertionError(
            f"Raw field-bank shape changed: {raw.shape}"
        )

    if normalized.shape != expected_shape:
        raise AssertionError(
            "Normalized field-bank shape changed."
        )

    if raw.dtype != np.float32:
        raise AssertionError(
            "Raw clean fields are no longer float32."
        )

    if normalized.dtype != np.float32:
        raise AssertionError(
            "Normalized clean fields are no longer float32."
        )

    return raw, normalized


def audit_manifold(
    raw_bank,
    normalized_bank,
    carrier_ids,
):
    train = np.asarray(
        normalized_bank[
            carrier_ids["train"]
        ],
        dtype=np.float32,
    )

    channel_variances = train.var(
        axis=(
            0,
            1,
            3,
            4,
        ),
        dtype=np.float64,
    )

    all_channels_nonconstant = bool(
        np.all(
            channel_variances > 1e-8
        )
    )

    state_variation_rows = []

    every_state_varies = True

    for state_id in range(
        STATE_COUNT
    ):
        state_fields = np.asarray(
            normalized_bank[
                :,
                state_id,
            ],
            dtype=np.float32,
        )

        state_mean = state_fields.mean(
            axis=0,
            dtype=np.float64,
        )

        mean_squared_variation = float(
            np.mean(
                (
                    state_fields
                    - state_mean[
                        None,
                        ...,
                    ]
                ) ** 2
            )
        )

        positive_variation = bool(
            mean_squared_variation > 1e-8
        )

        every_state_varies = (
            every_state_varies
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
        CARRIER_COUNT
        * STATE_COUNT,
        FLATTENED_DIMENSION,
    )

    for index in range(
        CARRIER_COUNT
        * STATE_COUNT
    ):
        value = np.asarray(
            flattened[index],
            dtype=np.float32,
        )

        hashes.append(
            hashlib.sha256(
                value.tobytes()
            ).hexdigest()
        )

    duplicate_count = (
        len(hashes)
        - len(set(hashes))
    )

    no_exact_duplicates = bool(
        duplicate_count == 0
    )

    raw_minimum = np.asarray(
        raw_bank
    ).min(
        axis=(
            0,
            1,
            3,
            4,
        )
    )

    raw_maximum = np.asarray(
        raw_bank
    ).max(
        axis=(
            0,
            1,
            3,
            4,
        )
    )

    result = {
        "field_count":
            CARRIER_COUNT
            * STATE_COUNT,

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "training_channel_variances":
            channel_variances.tolist(),

        "all_training_channels_nonconstant":
            all_channels_nonconstant,

        "every_state_has_carrier_variation":
            every_state_varies,

        "exact_duplicate_field_count":
            duplicate_count,

        "no_exact_duplicate_fields":
            no_exact_duplicates,

        "raw_channel_minimum":
            raw_minimum.tolist(),

        "raw_channel_maximum":
            raw_maximum.tolist(),
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
    carrier_ids,
):
    fields = np.asarray(
        normalized_bank,
        dtype=np.float32,
    )

    labels = np.broadcast_to(
        np.arange(
            STATE_COUNT,
            dtype=np.int64,
        )[None, :],
        (
            CARRIER_COUNT,
            STATE_COUNT,
        ),
    )

    output = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        selected = carrier_ids[
            split
        ]

        output[split] = {
            "x":
                fields[
                    selected
                ].reshape(
                    -1,
                    FIELD_CHANNEL_COUNT,
                    GRID_HEIGHT,
                    GRID_WIDTH,
                ),

            "y":
                labels[
                    selected
                ].reshape(-1),
        }

    return output


def global_statistics_features(fields):
    channel_means = fields.mean(
        axis=(-2, -1),
        dtype=np.float64,
    )

    channel_variances = fields.var(
        axis=(-2, -1),
        dtype=np.float64,
    )

    return np.concatenate(
        [
            channel_means,
            channel_variances,
        ],
        axis=1,
    ).astype(
        np.float32
    )


class GlobalStatisticsClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.classifier = nn.Linear(
            FIELD_CHANNEL_COUNT * 2,
            STATE_COUNT,
        )

    def forward(self, features):
        return self.classifier(features)


class PrivilegedStateDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                FIELD_CHANNEL_COUNT,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
            nn.GELU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(
                output_size=1,
            ),
        )

        self.classifier = nn.Linear(
            128,
            STATE_COUNT,
        )

    def forward(self, fields):
        encoded = self.encoder(
            fields
        ).flatten(1)

        return self.classifier(
            encoded
        )


@torch.no_grad()
def evaluate_classifier(
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

    accuracy = (
        predictions
        == y
    ).float().mean()

    return {
        "loss":
            float(
                loss.item()
            ),

        "accuracy":
            float(
                accuracy.item()
            ),
    }


def train_global_statistics_classifier(
    dataset,
    device,
):
    set_random_seed(
        DIAGNOSTIC_SEED
    )

    train_features = (
        global_statistics_features(
            dataset["train"]["x"]
        )
    )

    val_features = (
        global_statistics_features(
            dataset["val"]["x"]
        )
    )

    test_features = (
        global_statistics_features(
            dataset["test"]["x"]
        )
    )

    feature_mean = train_features.mean(
        axis=0,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    feature_std = train_features.std(
        axis=0,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    feature_std = np.maximum(
        feature_std,
        1e-6,
    )

    train_features = (
        train_features
        - feature_mean
    ) / feature_std

    val_features = (
        val_features
        - feature_mean
    ) / feature_std

    test_features = (
        test_features
        - feature_mean
    ) / feature_std

    model = GlobalStatisticsClassifier().to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05,
        weight_decay=1e-4,
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
    best_val_loss = math.inf
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

        logits = model(
            train_x
        )

        loss = nn.functional.cross_entropy(
            logits,
            train_y,
        )

        loss.backward()
        optimizer.step()

        train_metrics = (
            evaluate_classifier(
                model=model,
                features=train_features,
                labels=dataset["train"]["y"],
                device=device,
            )
        )

        val_metrics = (
            evaluate_classifier(
                model=model,
                features=val_features,
                labels=dataset["val"]["y"],
                device=device,
            )
        )

        history.append(
            {
                "epoch": epoch,

                "training_loss":
                    train_metrics["loss"],

                "training_accuracy":
                    train_metrics["accuracy"],

                "validation_loss":
                    val_metrics["loss"],

                "validation_accuracy":
                    val_metrics["accuracy"],
            }
        )

        improvement = (
            best_val_loss
            - val_metrics["loss"]
        )

        if improvement > 1e-7:
            best_val_loss = (
                val_metrics["loss"]
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
            "No global-statistics checkpoint was selected."
        )

    model.load_state_dict(
        best_state
    )

    train_metrics = evaluate_classifier(
        model=model,
        features=train_features,
        labels=dataset["train"]["y"],
        device=device,
    )

    val_metrics = evaluate_classifier(
        model=model,
        features=val_features,
        labels=dataset["val"]["y"],
        device=device,
    )

    test_metrics = evaluate_classifier(
        model=model,
        features=test_features,
        labels=dataset["test"]["y"],
        device=device,
    )

    write_csv(
        OUTPUT_DIR
        / "global_statistics_classifier_history.csv",
        history,
    )

    torch.save(
        {
            "phase":
                "4C diagnostic global-statistics classifier",

            "seed":
                DIAGNOSTIC_SEED,

            "best_epoch":
                best_epoch,

            "feature_mean":
                feature_mean,

            "feature_standard_deviation":
                feature_std,

            "model_state_dict":
                best_state,

            "eligible_for_predictive_use":
                False,
        },
        OUTPUT_DIR
        / "global_statistics_classifier.pt",
    )

    result = {
        "feature_count":
            FIELD_CHANNEL_COUNT * 2,

        "feature_definition":
            (
                "Per-channel global mean and variance "
                "from clean normalized fields."
            ),

        "training_carrier_count":
            len(
                set(
                    dataset["train"]["y"]
                )
            ),

        "best_epoch":
            best_epoch,

        "epochs_executed":
            len(history),

        "train_accuracy":
            train_metrics["accuracy"],

        "validation_accuracy":
            val_metrics["accuracy"],

        "test_accuracy":
            test_metrics["accuracy"],

        "maximum_allowed_test_accuracy":
            MAXIMUM_GLOBAL_STATISTICS_TEST_ACCURACY,

        "global_statistics_shortcut_absent":
            bool(
                test_metrics["accuracy"]
                <= MAXIMUM_GLOBAL_STATISTICS_TEST_ACCURACY
            ),

        "used_for_predictive_training":
            False,

        "used_for_predictive_selection":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "global_statistics_classifier_metrics.json",
        result,
    )

    return result


def augment_spatial_batch(
    batch,
):
    if torch.rand(
        (),
        device=batch.device,
    ).item() < 0.5:
        batch = torch.flip(
            batch,
            dims=(-1,),
        )

    if torch.rand(
        (),
        device=batch.device,
    ).item() < 0.5:
        batch = torch.flip(
            batch,
            dims=(-2,),
        )

    rotation = int(
        torch.randint(
            low=0,
            high=4,
            size=(1,),
            device=batch.device,
        ).item()
    )

    if rotation:
        batch = torch.rot90(
            batch,
            k=rotation,
            dims=(-2, -1),
        )

    shift_y = int(
        torch.randint(
            low=0,
            high=GRID_HEIGHT,
            size=(1,),
            device=batch.device,
        ).item()
    )

    shift_x = int(
        torch.randint(
            low=0,
            high=GRID_WIDTH,
            size=(1,),
            device=batch.device,
        ).item()
    )

    return torch.roll(
        batch,
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
            fields[start:end]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        y = torch.from_numpy(
            labels[start:end]
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


def train_privileged_state_decoder(
    dataset,
    device,
):
    set_random_seed(
        DIAGNOSTIC_SEED
    )

    model = PrivilegedStateDecoder().to(
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
        DIAGNOSTIC_SEED
    )

    best_state = None
    best_val_loss = math.inf
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
                start
                + STATE_DECODER_BATCH_SIZE
            ]

            x = torch.from_numpy(
                train_x[indices]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            y = torch.from_numpy(
                train_y[indices]
            ).to(
                device=device,
                dtype=torch.long,
            )

            x = augment_spatial_batch(
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

            batch_size = len(x)

            total_loss += float(
                loss.item()
            ) * batch_size

            correct += int(
                (
                    torch.argmax(
                        logits,
                        dim=1,
                    )
                    == y
                ).sum().item()
            )

            total_count += batch_size

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
                "epoch":
                    epoch,

                "training_loss":
                    total_loss
                    / total_count,

                "training_accuracy":
                    correct
                    / total_count,

                "validation_loss":
                    validation_metrics[
                        "loss"
                    ],

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],
            }
        )

        improvement = (
            best_val_loss
            - validation_metrics[
                "loss"
            ]
        )

        if improvement > 1e-6:
            best_val_loss = (
                validation_metrics[
                    "loss"
                ]
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
            "No privileged state-decoder "
            "checkpoint was selected."
        )

    model.load_state_dict(
        best_state
    )

    train_metrics = evaluate_state_decoder(
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

    test_metrics = evaluate_state_decoder(
        model=model,
        fields=dataset["test"]["x"],
        labels=dataset["test"]["y"],
        device=device,
    )

    write_csv(
        OUTPUT_DIR
        / "privileged_state_decoder_history.csv",
        history,
    )

    torch.save(
        {
            "phase":
                "4C privileged diagnostic state decoder",

            "seed":
                DIAGNOSTIC_SEED,

            "best_epoch":
                best_epoch,

            "model_state_dict":
                best_state,

            "trained_on":
                "clean training-carrier fields only",

            "selected_on":
                "clean validation-carrier fields only",

            "eligible_for_predictive_use":
                False,
        },
        OUTPUT_DIR
        / "privileged_state_decoder.pt",
    )

    passed = bool(
        validation_metrics["accuracy"]
        >= MINIMUM_PRIVILEGED_STATE_ACCURACY
        and test_metrics["accuracy"]
        >= MINIMUM_PRIVILEGED_STATE_ACCURACY
    )

    result = {
        "model":
            "fixed convolutional diagnostic decoder",

        "training_carrier_count":
            96,

        "validation_carrier_count":
            32,

        "test_carrier_count":
            32,

        "best_epoch":
            best_epoch,

        "epochs_executed":
            len(history),

        "train_accuracy":
            train_metrics["accuracy"],

        "validation_accuracy":
            validation_metrics["accuracy"],

        "test_accuracy":
            test_metrics["accuracy"],

        "minimum_required_validation_accuracy":
            MINIMUM_PRIVILEGED_STATE_ACCURACY,

        "minimum_required_test_accuracy":
            MINIMUM_PRIVILEGED_STATE_ACCURACY,

        "state_separability_gate_passed":
            passed,

        "test_used_for_checkpoint_selection":
            False,

        "eligible_for_predictive_use":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "privileged_state_decoder_metrics.json",
        result,
    )

    return result


def fit_affine_operator(
    source,
    target,
    ridge,
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
        + float(ridge)
        * regularizer,
        right_hand_side,
    )

    matrix = solution[:-1]
    bias = solution[-1]

    return matrix, bias


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
    carrier_indices,
    generators,
    matrices,
    biases,
    pca_mean,
    pca_basis,
):
    carrier_indices_tensor = torch.as_tensor(
        carrier_indices,
        dtype=torch.long,
        device=latent_fields.device,
    )

    candidate_latent = latent_fields[
        carrier_indices_tensor
    ]

    candidate_flattened = flattened_fields[
        carrier_indices_tensor
    ]

    mapping_correct = 0
    mapping_total = 0
    exact_count = 0
    exact_total = 0

    continuous_squared_error = 0.0
    continuous_value_count = 0

    carrier_operation_exact = np.zeros(
        (
            len(carrier_indices),
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

        source = candidate_latent

        predicted = (
            source
            @ matrices[operation]
            + biases[operation]
        )

        decoded = (
            decode_same_carrier_states(
                predicted=predicted,
                candidates=candidate_latent,
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
            carrier_indices
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

        "carrier_operation_exact":
            carrier_operation_exact,
    }


def evaluate_all_affine_transformations(
    latent_fields,
    carrier_indices,
    semigroup,
    matrices,
    biases,
):
    carrier_indices_tensor = torch.as_tensor(
        carrier_indices,
        dtype=torch.long,
        device=latent_fields.device,
    )

    candidates = latent_fields[
        carrier_indices_tensor
    ]

    mapping_correct = 0
    mapping_total = 0
    exact_count = 0
    exact_total = 0

    carrier_transformation_exact = np.zeros(
        (
            len(carrier_indices),
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
            transformation[
                "mapping"
            ],
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
            carrier_indices
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

        "fraction_carriers_all_104_exact":
            float(
                np.mean(
                    np.all(
                        carrier_transformation_exact,
                        axis=1,
                    )
                )
            ),

        "carrier_transformation_exact":
            carrier_transformation_exact,
    }


def run_affine_shortcut_audit(
    normalized_bank,
    carrier_ids,
    semigroup,
    generators,
    device,
):
    flattened = torch.from_numpy(
        np.asarray(
            normalized_bank,
            dtype=np.float32,
        ).reshape(
            CARRIER_COUNT,
            STATE_COUNT,
            FLATTENED_DIMENSION,
        )
    ).to(
        device=device,
        dtype=torch.float32,
    )

    train_indices = torch.as_tensor(
        carrier_ids["train"],
        dtype=torch.long,
        device=device,
    )

    training_flattened = flattened[
        train_indices
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

    _, singular_values, right_vectors = (
        torch.linalg.svd(
            centered,
            full_matrices=False,
        )
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
            train_indices
        ].reshape(
            -1,
            AFFINE_PCA_DIMENSION,
        )

        target = latent_fields[
            train_indices
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
            ridge=AFFINE_RIDGE,
        )

        matrices[operation] = matrix
        biases[operation] = bias

    split_results = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        primitive = evaluate_primitive_affine(
            latent_fields=latent_fields,
            flattened_fields=flattened,
            carrier_indices=carrier_ids[split],
            generators=generators,
            matrices=matrices,
            biases=biases,
            pca_mean=pca_mean,
            pca_basis=pca_basis,
        )

        all_transformations = (
            evaluate_all_affine_transformations(
                latent_fields=latent_fields,
                carrier_indices=carrier_ids[split],
                semigroup=semigroup,
                matrices=matrices,
                biases=biases,
            )
        )

        split_results[split] = {
            "carrier_count":
                len(
                    carrier_ids[split]
                ),

            "primitive_mapping_accuracy":
                primitive[
                    "mapping_accuracy"
                ],

            "primitive_exact_operation_rate":
                primitive[
                    "exact_primitive_operation_rate"
                ],

            "primitive_continuous_mse":
                primitive[
                    "continuous_mse"
                ],

            "fraction_carriers_all_primitives_exact":
                primitive[
                    "fraction_carriers_all_primitives_exact"
                ],

            "all_104_mapping_accuracy":
                all_transformations[
                    "mapping_accuracy"
                ],

            "all_104_exact_transformation_rate":
                all_transformations[
                    "exact_transformation_rate"
                ],

            "fraction_carriers_all_104_exact":
                all_transformations[
                    "fraction_carriers_all_104_exact"
                ],
        }

    heldout_continuous_interpolation_fails = bool(
        split_results["val"][
            "primitive_continuous_mse"
        ] > AFFINE_NUMERICAL_TOLERANCE
        and split_results["test"][
            "primitive_continuous_mse"
        ] > AFFINE_NUMERICAL_TOLERANCE
    )

    at_least_one_heldout_primitive_mapping_fails = bool(
        split_results["val"][
            "primitive_exact_operation_rate"
        ] < 1.0
        or split_results["test"][
            "primitive_exact_operation_rate"
        ] < 1.0
    )

    all_104_not_exact_on_every_test_carrier = bool(
        split_results["test"][
            "fraction_carriers_all_104_exact"
        ] < 1.0
    )

    passed = bool(
        heldout_continuous_interpolation_fails
        and at_least_one_heldout_primitive_mapping_fails
        and all_104_not_exact_on_every_test_carrier
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
            "training-only PCA plus affine primitive operators",

        "pca_dimension":
            AFFINE_PCA_DIMENSION,

        "ridge":
            AFFINE_RIDGE,

        "training_carrier_count":
            len(
                carrier_ids["train"]
            ),

        "split_results":
            split_results,

        "heldout_continuous_interpolation_fails":
            heldout_continuous_interpolation_fails,

        "at_least_one_heldout_primitive_mapping_fails":
            at_least_one_heldout_primitive_mapping_fails,

        "all_104_transformations_not_exact_on_every_test_carrier":
            all_104_not_exact_on_every_test_carrier,

        "affine_shortcut_removed":
            passed,

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


def verify_metadata_copy(
    phase4b_path,
    phase3b_path,
):
    if not phase4b_path.exists():
        raise FileNotFoundError(
            phase4b_path
        )

    if not phase3b_path.exists():
        raise FileNotFoundError(
            phase3b_path
        )

    return bool(
        sha256_file(phase4b_path)
        == sha256_file(phase3b_path)
    )


def audit_zero_noise_consistency(
    cell_id,
    normalized_bank,
    chunk_size,
):
    field_path = (
        VISIBLE_DIR
        / f"{cell_id}_fields.npy"
    )

    index_path = (
        PRIVILEGED_DIR
        / f"{cell_id}_point_bank_indices.npy"
    )

    fields = np.load(
        field_path,
        mmap_mode="r",
    )

    point_indices = np.load(
        index_path,
        mmap_mode="r",
    )

    flattened_bank = (
        normalized_bank.reshape(
            CARRIER_COUNT
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

        difference = np.max(
            np.abs(
                expected.astype(
                    np.float32
                )
                - observed.astype(
                    np.float32
                )
            )
        )

        maximum_absolute_error = max(
            maximum_absolute_error,
            float(difference),
        )

    return {
        "cell_id":
            cell_id,

        "point_count":
            len(point_indices),

        "zero_noise_matches_field_bank_exactly_after_float16_storage":
            exact,

        "maximum_absolute_error":
            maximum_absolute_error,
    }


def audit_noise_independence(
    normalized_bank,
):
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

    values, counts = np.unique(
        point_indices,
        return_counts=True,
    )

    repeated_values = values[
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
            dtype=np.float32,
        )

        second = np.asarray(
            fields[
                2,
                locations[1],
            ],
            dtype=np.float32,
        )

        tested_pairs += 1

        if not np.array_equal(
            first,
            second,
        ):
            differing_pairs += 1

    passed = bool(
        tested_pairs > 0
        and differing_pairs == tested_pairs
    )

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
            passed,
    }


def audit_data_integrity(
    carrier_rows,
    carrier_ids,
    normalized_bank,
    zero_noise_chunk_size,
):
    visible_names = {
        path.name
        for path in VISIBLE_DIR.iterdir()
    }

    forbidden_visible_tokens = (
        "clean",
        "state_id",
        "carrier_state",
        "physics_diagnostic",
        "trajectory_metadata",
        "point_bank_indices",
    )

    visible_forbidden_files = [
        name
        for name in visible_names
        if any(
            token in name
            for token in forbidden_visible_tokens
        )
    ]

    metadata_hashes_match = True
    carrier_assignments_valid = True
    cell_headers_valid = True
    markers_valid = True

    cell_rows = []

    for cell_id in EXPECTED_CELLS:
        marker = load_json(
            MANIFEST_DIR
            / f"{cell_id}_complete.json"
        )

        marker_valid = bool(
            marker["status"] == "completed"
        )

        markers_valid = (
            markers_valid
            and marker_valid
        )

        field_path = (
            VISIBLE_DIR
            / f"{cell_id}_fields.npy"
        )

        fields = np.load(
            field_path,
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

        cell_headers_valid = (
            cell_headers_valid
            and header_valid
        )

        del fields

        copied_visible_trajectory = (
            VISIBLE_DIR
            / f"{cell_id}_trajectory_index.csv"
        )

        copied_visible_sequence = (
            VISIBLE_DIR
            / f"{cell_id}_sequence_manifest.csv"
        )

        copied_privileged_trajectory = (
            PRIVILEGED_DIR
            / f"{cell_id}_trajectory_metadata.csv"
        )

        source_visible_trajectory = (
            PHASE3B_DIR
            / "visible"
            / f"{cell_id}_trajectory_index.csv"
        )

        source_visible_sequence = (
            PHASE3B_DIR
            / "visible"
            / f"{cell_id}_sequence_manifest.csv"
        )

        source_privileged_trajectory = (
            PHASE3B_DIR
            / "privileged"
            / f"{cell_id}_trajectory_metadata.csv"
        )

        trajectory_hash_match = (
            verify_metadata_copy(
                copied_visible_trajectory,
                source_visible_trajectory,
            )
        )

        sequence_hash_match = (
            verify_metadata_copy(
                copied_visible_sequence,
                source_visible_sequence,
            )
        )

        privileged_hash_match = (
            verify_metadata_copy(
                copied_privileged_trajectory,
                source_privileged_trajectory,
            )
        )

        metadata_match = bool(
            trajectory_hash_match
            and sequence_hash_match
            and privileged_hash_match
        )

        metadata_hashes_match = (
            metadata_hashes_match
            and metadata_match
        )

        privileged_rows = load_csv(
            copied_privileged_trajectory
        )

        observed_carrier_ids = {
            int(row["carrier_id"])
            for row in privileged_rows
        }

        expected_split = (
            EXPECTED_CELL_CARRIER_SPLIT[
                cell_id
            ]
        )

        allowed_carriers = set(
            carrier_ids[
                expected_split
            ].tolist()
        )

        carrier_valid = bool(
            observed_carrier_ids
            <= allowed_carriers
        )

        carrier_assignments_valid = (
            carrier_assignments_valid
            and carrier_valid
        )

        point_indices = np.load(
            PRIVILEGED_DIR
            / f"{cell_id}_point_bank_indices.npy",
            mmap_mode="r",
        )

        point_index_valid = bool(
            len(point_indices)
            == int(marker["point_count"])
            and int(point_indices.min()) >= 0
            and int(point_indices.max())
            < CARRIER_COUNT
            * STATE_COUNT
        )

        cell_rows.append(
            {
                "cell_id":
                    cell_id,

                "point_count":
                    int(
                        marker["point_count"]
                    ),

                "trajectory_count":
                    int(
                        marker[
                            "trajectory_count"
                        ]
                    ),

                "expected_carrier_split":
                    expected_split,

                "observed_unique_carrier_count":
                    len(
                        observed_carrier_ids
                    ),

                "carrier_assignment_valid":
                    carrier_valid,

                "metadata_hashes_match_tier_b":
                    metadata_match,

                "field_header_valid":
                    header_valid,

                "point_bank_index_valid":
                    point_index_valid,

                "test_field_values_read":
                    False,
            }
        )

    zero_noise_results = []

    for cell_id in (
        "train_joint",
        "val_joint",
    ):
        zero_noise_results.append(
            audit_zero_noise_consistency(
                cell_id=cell_id,
                normalized_bank=normalized_bank,
                chunk_size=(
                    zero_noise_chunk_size
                ),
            )
        )

    zero_noise_consistent = bool(
        all(
            row[
                "zero_noise_matches_field_bank_exactly_after_float16_storage"
            ]
            for row in zero_noise_results
        )
    )

    noise_independence = (
        audit_noise_independence(
            normalized_bank
        )
    )

    transformation_rows = (
        load_transformation_split()
    )

    transformation_counts = dict(
        Counter(
            row["split"]
            for row in transformation_rows
        )
    )

    carrier_split_disjoint = bool(
        not (
            set(carrier_ids["train"])
            & set(carrier_ids["val"])
        )
        and not (
            set(carrier_ids["train"])
            & set(carrier_ids["test"])
        )
        and not (
            set(carrier_ids["val"])
            & set(carrier_ids["test"])
        )
    )

    passed = bool(
        not visible_forbidden_files
        and metadata_hashes_match
        and carrier_assignments_valid
        and cell_headers_valid
        and markers_valid
        and zero_noise_consistent
        and noise_independence[
            "independent_noise_empirically_confirmed"
        ]
        and carrier_split_disjoint
        and transformation_counts
        == EXPECTED_TRANSFORMATION_SPLITS
    )

    write_csv(
        OUTPUT_DIR
        / "data_integrity_by_cell.csv",
        cell_rows,
    )

    write_json(
        OUTPUT_DIR
        / "zero_noise_consistency.json",
        {
            "cells_audited":
                zero_noise_results,

            "test_trajectory_field_values_read":
                False,
        },
    )

    result = {
        "visible_forbidden_files":
            visible_forbidden_files,

        "visible_files_exclude_privileged_arrays":
            bool(
                not visible_forbidden_files
            ),

        "metadata_hashes_match_tier_b":
            metadata_hashes_match,

        "carrier_assignments_valid":
            carrier_assignments_valid,

        "carrier_splits_disjoint":
            carrier_split_disjoint,

        "transformation_split_counts":
            transformation_counts,

        "transformation_split_unchanged":
            bool(
                transformation_counts
                == EXPECTED_TRANSFORMATION_SPLITS
            ),

        "cell_headers_valid":
            cell_headers_valid,

        "all_cell_markers_valid":
            markers_valid,

        "zero_noise_train_and_validation_cells_consistent":
            zero_noise_consistent,

        "noise_independence_audit":
            noise_independence,

        "test_trajectory_field_values_read":
            False,

        "test_predictive_metrics_computed":
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
        DIAGNOSTIC_SEED
    )

    carrier_rows, carrier_ids = (
        load_carrier_information()
    )

    semigroup = load_semigroup()
    generators = load_generators()

    raw_bank, normalized_bank = (
        load_field_bank()
    )

    print(
        "[1/6] Auditing numerical physics"
    )

    physics = audit_physics()

    print(
        "[2/6] Auditing basic manifold quality"
    )

    manifold = audit_manifold(
        raw_bank=raw_bank,
        normalized_bank=normalized_bank,
        carrier_ids=carrier_ids,
    )

    state_dataset = build_state_dataset(
        normalized_bank=normalized_bank,
        carrier_ids=carrier_ids,
    )

    print(
        "[3/6] Training global-statistics "
        "shortcut diagnostic"
    )

    global_statistics = (
        train_global_statistics_classifier(
            dataset=state_dataset,
            device=device,
        )
    )

    print(
        "[4/6] Training privileged convolutional "
        "state decoder"
    )

    privileged_decoder = (
        train_privileged_state_decoder(
            dataset=state_dataset,
            device=device,
        )
    )

    print(
        "[5/6] Auditing low-rank affine shortcut"
    )

    affine = run_affine_shortcut_audit(
        normalized_bank=normalized_bank,
        carrier_ids=carrier_ids,
        semigroup=semigroup,
        generators=generators,
        device=device,
    )

    print(
        "[6/6] Auditing data and split integrity"
    )

    integrity = audit_data_integrity(
        carrier_rows=carrier_rows,
        carrier_ids=carrier_ids,
        normalized_bank=normalized_bank,
        zero_noise_chunk_size=(
            arguments.zero_noise_chunk_size
        ),
    )

    manifold_gate_passed = bool(
        manifold[
            "all_training_channels_nonconstant"
        ]
        and manifold[
            "every_state_has_carrier_variation"
        ]
        and manifold[
            "no_exact_duplicate_fields"
        ]
        and privileged_decoder[
            "state_separability_gate_passed"
        ]
        and global_statistics[
            "global_statistics_shortcut_absent"
        ]
    )

    all_gates_passed = bool(
        physics[
            "physics_gate_passed"
        ]
        and manifold_gate_passed
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
                "field_manifold_quality",

            "passed":
                manifold_gate_passed,
        },
        {
            "gate":
                "privileged_state_separability",

            "passed":
                privileged_decoder[
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
                "low_rank_affine_shortcut_removed",

            "passed":
                affine[
                    "affine_shortcut_removed"
                ],
        },
        {
            "gate":
                "data_integrity",

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
            "4C Tier C pretraining acceptance audit",

        "source_phase_validation": {
            "phase4a":
                sources[
                    "phase4a"
                ][
                    "phase4a_status"
                ]
                == "protocol_frozen",

            "phase4b":
                sources[
                    "phase4b"
                ][
                    "phase4b_status"
                ]
                == "completed",

            "field_bank":
                sources[
                    "field_bank"
                ][
                    "field_bank_generation_status"
                ]
                == "completed",
        },

        "field_count":
            CARRIER_COUNT
            * STATE_COUNT,

        "carrier_count":
            CARRIER_COUNT,

        "state_count":
            STATE_COUNT,

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

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

        "all_training_channels_nonconstant":
            manifold[
                "all_training_channels_nonconstant"
            ],

        "every_state_has_carrier_variation":
            manifold[
                "every_state_has_carrier_variation"
            ],

        "exact_duplicate_field_count":
            manifold[
                "exact_duplicate_field_count"
            ],

        "privileged_state_accuracy": {
            "train":
                privileged_decoder[
                    "train_accuracy"
                ],

            "validation":
                privileged_decoder[
                    "validation_accuracy"
                ],

            "test":
                privileged_decoder[
                    "test_accuracy"
                ],
        },

        "global_statistics_accuracy": {
            "train":
                global_statistics[
                    "train_accuracy"
                ],

            "validation":
                global_statistics[
                    "validation_accuracy"
                ],

            "test":
                global_statistics[
                    "test_accuracy"
                ],
        },

        "global_statistics_shortcut_absent":
            global_statistics[
                "global_statistics_shortcut_absent"
            ],

        "affine_shortcut_removed":
            affine[
                "affine_shortcut_removed"
            ],

        "heldout_continuous_interpolation_fails":
            affine[
                "heldout_continuous_interpolation_fails"
            ],

        "at_least_one_heldout_primitive_mapping_fails":
            affine[
                "at_least_one_heldout_primitive_mapping_fails"
            ],

        "all_104_transformations_not_exact_on_every_test_carrier":
            affine[
                "all_104_transformations_not_exact_on_every_test_carrier"
            ],

        "data_integrity_gate_passed":
            integrity[
                "data_integrity_gate_passed"
            ],

        "test_trajectory_field_values_read":
            False,

        "privileged_test_carrier_fields_used_for_dataset_audit":
            True,

        "test_predictive_metrics_computed":
            False,

        "predictive_model_training_performed":
            False,

        "diagnostic_reference_decoder_fitted":
            True,

        "diagnostic_global_statistics_classifier_fitted":
            True,

        "diagnostic_affine_operator_fitted":
            True,

        "diagnostic_models_eligible_for_predictive_use":
            False,

        "checkpoint_selection_performed":
            False,

        "phase2_outputs_modified":
            False,

        "phase3_outputs_modified":
            False,

        "phase4a_outputs_modified":
            False,

        "phase4b_outputs_modified":
            False,

        "tier_c_data_accepted":
            all_gates_passed,

        "phase4c_status":
            (
                "passed"
                if all_gates_passed
                else "failed_acceptance_gates"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4c_summary.json",
        summary,
    )

    print()
    print(
        "Phase 4C audit completed."
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
            "Phase 4C acceptance gates failed. "
            "Do not proceed to model training."
        )


if __name__ == "__main__":
    main()
