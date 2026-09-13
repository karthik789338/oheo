from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PHASE4AR3_DIR = Path(
    "outputs/phase4ar3_tier_c_v4_protocol"
)

FIELD_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_fields"
)

DATA_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_data"
)

AUDIT_DIR = Path(
    "outputs/phase4cr3_tier_c_v4_audit"
)

V2_AUDIT_DIR = Path(
    "outputs/phase4cr_tier_c_v2_audit"
)

OUTPUT_DIR = AUDIT_DIR


PROTOCOL_VERSION = "tier_c_v4"

STATE_COUNT = 8
FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32

TRAIN_CARRIER_COUNT = 96
VALIDATION_CARRIER_COUNT = 32

EXPECTED_TRAIN_SAMPLE_COUNT = (
    TRAIN_CARRIER_COUNT * STATE_COUNT
)

EXPECTED_VALIDATION_SAMPLE_COUNT = (
    VALIDATION_CARRIER_COUNT * STATE_COUNT
)

DIAGNOSTIC_SEEDS = (
    72011,
    72023,
    72037,
)

V2_ARCHITECTURE_REFERENCE_SEED = 52011

# This training schedule is frozen before any Tier C v4
# diagnostic outcome is observed.
EPOCH_COUNT = 200
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MINIMUM_LEARNING_RATE = 1e-5
DROPOUT_PROBABILITY = 0.10
LABEL_SMOOTHING = 0.0

MINIMUM_MEAN_VALIDATION_STATE_ACCURACY = 0.80
MINIMUM_WORST_VALIDATION_STATE_ACCURACY = 0.75

MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY = 0.90
MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY = 0.85

SEALED_TEST_CELLS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)

FACTOR_NAMES = (
    "coarsening_level",
    "anisotropy_level",
    "defect_recovery_level",
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
        "--force",
        action="store_true",
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
        return list(csv.DictReader(handle))


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


def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def find_sealed_test_files():
    found = []

    for root in (
        FIELD_DIR,
        DATA_DIR,
        AUDIT_DIR,
    ):
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            name = path.name.lower()

            if any(
                cell_id in name
                for cell_id in SEALED_TEST_CELLS
            ):
                found.append(str(path))

    return found


def validate_authorization():
    protocol = load_json(
        PHASE4AR3_DIR
        / "phase4ar3_summary.json"
    )

    predecoder = load_json(
        AUDIT_DIR
        / "phase4cr3_predecoder_summary.json"
    )

    field_summary = load_json(
        FIELD_DIR
        / "phase4br3_field_summary.json"
    )

    dataset_summary = load_json(
        DATA_DIR
        / "phase4br3_summary.json"
    )

    if (
        protocol["phase4ar3_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A-R3 is not frozen."
        )

    if (
        protocol["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Tier C v4 protocol version changed."
        )

    if (
        protocol["diagnostic_seeds"]
        != list(DIAGNOSTIC_SEEDS)
    ):
        raise AssertionError(
            "Tier C v4 diagnostic seeds changed."
        )

    if (
        protocol[
            "minimum_mean_validation_state_accuracy"
        ]
        != MINIMUM_MEAN_VALIDATION_STATE_ACCURACY
    ):
        raise AssertionError(
            "Mean state-accuracy threshold changed."
        )

    if (
        protocol[
            "minimum_worst_validation_state_accuracy"
        ]
        != MINIMUM_WORST_VALIDATION_STATE_ACCURACY
    ):
        raise AssertionError(
            "Worst state-accuracy threshold changed."
        )

    if (
        protocol[
            "minimum_mean_validation_coarsening_accuracy"
        ]
        != MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY
    ):
        raise AssertionError(
            "Mean coarsening threshold changed."
        )

    if (
        protocol[
            "minimum_worst_validation_coarsening_accuracy"
        ]
        != MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY
    ):
        raise AssertionError(
            "Worst coarsening threshold changed."
        )

    if (
        predecoder[
            "phase4cr3_predecoder_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 4C-R3 pre-decoder audit did not pass."
        )

    if (
        predecoder[
            "predecoder_acceptance_gate_passed"
        ]
        is not True
    ):
        raise AssertionError(
            "Pre-decoder acceptance gate failed."
        )

    if (
        predecoder[
            "diagnostic_decoder_fitting_authorized"
        ]
        is not True
    ):
        raise AssertionError(
            "Diagnostic decoder fitting is not authorized."
        )

    if (
        predecoder[
            "predictive_training_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "Predictive training was incorrectly authorized."
        )

    if (
        field_summary[
            "test_carrier_count_generated"
        ]
        != 0
    ):
        raise AssertionError(
            "Tier C v4 test carriers were generated."
        )

    if (
        dataset_summary[
            "sealed_test_files_absent"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v4 sealed test files are not absent."
        )

    if (
        dataset_summary[
            "diagnostic_decoder_fitted"
        ]
        is not False
    ):
        raise AssertionError(
            "A Tier C v4 decoder was already fitted."
        )

    sealed_test_files = find_sealed_test_files()

    if sealed_test_files:
        raise AssertionError(
            "Sealed Tier C v4 test files exist: "
            + ", ".join(sealed_test_files)
        )

    return {
        "protocol": protocol,
        "predecoder": predecoder,
        "field_summary": field_summary,
        "dataset_summary": dataset_summary,
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

        values = self.convolution_1(inputs)
        values = self.normalization_1(values)
        values = nn.functional.gelu(values)

        values = self.convolution_2(values)
        values = self.normalization_2(values)

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
                p=DROPOUT_PROBABILITY
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


def state_dict_signature(state_dict):
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in state_dict.items()
    ]


def validate_architecture_against_v2():
    checkpoint_path = (
        V2_AUDIT_DIR
        / (
            "state_decoder_seed_"
            f"{V2_ARCHITECTURE_REFERENCE_SEED}.pt"
        )
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "The frozen Tier C v2 architecture "
            f"reference is missing: {checkpoint_path}"
        )

    checkpoint = safe_torch_load(
        checkpoint_path,
        map_location="cpu",
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Tier C v2 checkpoint does not contain "
            "'model_state_dict'."
        )

    reference_signature = state_dict_signature(
        checkpoint["model_state_dict"]
    )

    current_model = (
        FrozenDiagnosticStateDecoder()
    )

    current_signature = state_dict_signature(
        current_model.state_dict()
    )

    signatures_match = (
        reference_signature
        == current_signature
    )

    if not signatures_match:
        raise AssertionError(
            "Tier C v4 diagnostic architecture does "
            "not match the frozen Tier C v2 architecture."
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in current_model.parameters()
    )

    result = {
        "reference_checkpoint":
            str(checkpoint_path),

        "reference_checkpoint_sha256":
            sha256_file(checkpoint_path),

        "architecture_signature_matches_v2":
            signatures_match,

        "parameter_count":
            parameter_count,

        "dropout_probability":
            DROPOUT_PROBABILITY,
    }

    write_json(
        OUTPUT_DIR
        / "state_decoder_architecture_audit.json",
        result,
    )

    return result


def load_state_registry():
    rows = load_csv(
        PHASE4AR3_DIR
        / "state_morphology_regimes_v4.csv"
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
            "Tier C v4 state IDs changed."
        )

    state_codes = [
        str(row["state_code"])
        for row in rows
    ]

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

    output_rows = []

    for row in rows:
        output_rows.append(
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
        / "state_decoder_state_registry.csv",
        output_rows,
    )

    return {
        "rows": output_rows,
        "state_codes": state_codes,
        "factor_values": factor_values,
    }


def load_clean_dataset():
    carrier_ids = np.load(
        FIELD_DIR
        / "privileged"
        / "development_carrier_ids.npy"
    ).astype(np.int64)

    carrier_splits = np.load(
        FIELD_DIR
        / "privileged"
        / "development_carrier_splits.npy"
    ).astype(str)

    fields = np.load(
        FIELD_DIR
        / "privileged"
        / "development_carrier_state_fields_normalized.npy",
        mmap_mode="r",
    )

    expected_shape = (
        TRAIN_CARRIER_COUNT
        + VALIDATION_CARRIER_COUNT,
        STATE_COUNT,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    if fields.shape != expected_shape:
        raise AssertionError(
            f"Tier C v4 normalized field shape "
            f"changed: {fields.shape}"
        )

    if fields.dtype != np.float32:
        raise AssertionError(
            "Tier C v4 normalized fields are not float32."
        )

    train_indices = np.where(
        carrier_splits == "train"
    )[0]

    validation_indices = np.where(
        carrier_splits == "val"
    )[0]

    if len(train_indices) != (
        TRAIN_CARRIER_COUNT
    ):
        raise AssertionError(
            "Training carrier count changed."
        )

    if len(validation_indices) != (
        VALIDATION_CARRIER_COUNT
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
            len(carrier_ids),
            STATE_COUNT,
        ),
    )

    train_fields = np.asarray(
        fields[train_indices],
        dtype=np.float32,
    ).reshape(
        -1,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    ).copy()

    validation_fields = np.asarray(
        fields[validation_indices],
        dtype=np.float32,
    ).reshape(
        -1,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    ).copy()

    train_labels = labels[
        train_indices
    ].reshape(-1).copy()

    validation_labels = labels[
        validation_indices
    ].reshape(-1).copy()

    if len(train_labels) != (
        EXPECTED_TRAIN_SAMPLE_COUNT
    ):
        raise AssertionError(
            "Training sample count changed."
        )

    if len(validation_labels) != (
        EXPECTED_VALIDATION_SAMPLE_COUNT
    ):
        raise AssertionError(
            "Validation sample count changed."
        )

    if not np.isfinite(
        train_fields
    ).all():
        raise FloatingPointError(
            "Training fields contain NaN or Inf."
        )

    if not np.isfinite(
        validation_fields
    ).all():
        raise FloatingPointError(
            "Validation fields contain NaN or Inf."
        )

    return {
        "train_fields": train_fields,
        "train_labels": train_labels,
        "validation_fields": validation_fields,
        "validation_labels": validation_labels,
        "train_carrier_ids":
            carrier_ids[train_indices].copy(),

        "validation_carrier_ids":
            carrier_ids[
                validation_indices
            ].copy(),
    }


def create_loader(
    fields,
    labels,
    batch_size,
    shuffle,
    seed,
):
    dataset = TensorDataset(
        torch.from_numpy(fields),
        torch.from_numpy(labels),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


@torch.no_grad()
def evaluate_model(
    model,
    fields,
    labels,
    device,
):
    model.eval()

    field_tensor = torch.from_numpy(
        fields
    )

    label_tensor = torch.from_numpy(
        labels
    )

    loader = DataLoader(
        TensorDataset(
            field_tensor,
            label_tensor,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    total_loss = 0.0
    total_count = 0

    prediction_batches = []
    probability_batches = []

    for batch_fields, batch_labels in loader:
        batch_fields = batch_fields.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        batch_labels = batch_labels.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )

        logits = model(batch_fields)

        loss = nn.functional.cross_entropy(
            logits,
            batch_labels,
            reduction="sum",
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        total_loss += float(
            loss.item()
        )

        total_count += int(
            batch_labels.shape[0]
        )

        prediction_batches.append(
            predictions.cpu().numpy()
        )

        probability_batches.append(
            probabilities.cpu()
            .numpy()
            .astype(np.float64)
        )

    predictions = np.concatenate(
        prediction_batches
    ).astype(np.int64)

    probabilities = np.concatenate(
        probability_batches,
        axis=0,
    )

    accuracy = float(
        np.mean(
            predictions == labels
        )
    )

    return {
        "loss":
            total_loss / total_count,

        "accuracy":
            accuracy,

        "predictions":
            predictions,

        "probabilities":
            probabilities,
    }


def clone_state_dict_to_cpu(model):
    return {
        name:
            value.detach()
            .cpu()
            .clone()
        for name, value
        in model.state_dict().items()
    }


def train_one_seed(
    seed,
    dataset,
    device,
):
    set_deterministic_seed(seed)

    model = (
        FrozenDiagnosticStateDecoder()
        .to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCH_COUNT,
            eta_min=MINIMUM_LEARNING_RATE,
        )
    )

    training_loader = create_loader(
        fields=dataset["train_fields"],
        labels=dataset["train_labels"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )

    best_epoch = None
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_state_dict = None

    history_rows = []

    for epoch in range(
        1,
        EPOCH_COUNT + 1,
    ):
        model.train()

        total_training_loss = 0.0
        total_training_count = 0

        for batch_fields, batch_labels in (
            training_loader
        ):
            batch_fields = batch_fields.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            batch_labels = batch_labels.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(batch_fields)

            loss = nn.functional.cross_entropy(
                logits,
                batch_labels,
                label_smoothing=LABEL_SMOOTHING,
            )

            loss.backward()

            optimizer.step()

            total_training_loss += float(
                loss.item()
            ) * int(
                batch_labels.shape[0]
            )

            total_training_count += int(
                batch_labels.shape[0]
            )

        validation_result = evaluate_model(
            model=model,
            fields=(
                dataset[
                    "validation_fields"
                ]
            ),
            labels=(
                dataset[
                    "validation_labels"
                ]
            ),
            device=device,
        )

        training_loss = (
            total_training_loss
            / total_training_count
        )

        current_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        history_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "training_loss":
                    training_loss,

                "validation_loss":
                    validation_result[
                        "loss"
                    ],

                "validation_accuracy":
                    validation_result[
                        "accuracy"
                    ],

                "learning_rate":
                    current_learning_rate,
            }
        )

        validation_accuracy = (
            validation_result[
                "accuracy"
            ]
        )

        validation_loss = (
            validation_result[
                "loss"
            ]
        )

        improved = bool(
            validation_accuracy
            > best_validation_accuracy
            + 1e-12
        )

        tied_with_lower_loss = bool(
            abs(
                validation_accuracy
                - best_validation_accuracy
            ) <= 1e-12
            and validation_loss
            < best_validation_loss
            - 1e-12
        )

        if (
            improved
            or tied_with_lower_loss
        ):
            best_epoch = epoch

            best_validation_accuracy = (
                validation_accuracy
            )

            best_validation_loss = (
                validation_loss
            )

            best_state_dict = (
                clone_state_dict_to_cpu(
                    model
                )
            )

        scheduler.step()

        if (
            epoch == 1
            or epoch % 20 == 0
            or epoch == EPOCH_COUNT
        ):
            print(
                f"seed={seed} "
                f"epoch={epoch:03d} "
                f"train_loss={training_loss:.6f} "
                f"val_loss={validation_loss:.6f} "
                f"val_acc={validation_accuracy:.6f} "
                f"best={best_validation_accuracy:.6f}"
            )

    if best_state_dict is None:
        raise RuntimeError(
            "No diagnostic checkpoint was selected."
        )

    model.load_state_dict(
        best_state_dict
    )

    training_result = evaluate_model(
        model=model,
        fields=dataset["train_fields"],
        labels=dataset["train_labels"],
        device=device,
    )

    validation_result = evaluate_model(
        model=model,
        fields=(
            dataset[
                "validation_fields"
            ]
        ),
        labels=(
            dataset[
                "validation_labels"
            ]
        ),
        device=device,
    )

    checkpoint = {
        "phase":
            (
                "4C-R3A Tier C v4 diagnostic "
                "state decoder"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "model_state_dict":
            best_state_dict,

        "training_configuration": {
            "epoch_count":
                EPOCH_COUNT,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "minimum_learning_rate":
                MINIMUM_LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "dropout_probability":
                DROPOUT_PROBABILITY,

            "label_smoothing":
                LABEL_SMOOTHING,

            "optimizer":
                "AdamW",

            "scheduler":
                "CosineAnnealingLR",

            "checkpoint_selection":
                (
                    "highest clean validation state "
                    "accuracy; validation loss tie-break"
                ),
        },

        "training_sample_count":
            EXPECTED_TRAIN_SAMPLE_COUNT,

        "validation_sample_count":
            EXPECTED_VALIDATION_SAMPLE_COUNT,

        "training_accuracy":
            training_result["accuracy"],

        "validation_accuracy":
            validation_result["accuracy"],

        "training_loss":
            training_result["loss"],

        "validation_loss":
            validation_result["loss"],

        "test_fields_used":
            False,

        "test_metrics_computed":
            False,

        "predictive_training":
            False,

        "eligible_for_predictive_use":
            False,
    }

    checkpoint_path = (
        OUTPUT_DIR
        / f"state_decoder_seed_{seed}.pt"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    write_csv(
        OUTPUT_DIR
        / f"state_decoder_seed_{seed}_history.csv",
        history_rows,
    )

    return {
        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "training_result":
            training_result,

        "validation_result":
            validation_result,

        "checkpoint_path":
            checkpoint_path,

        "history_rows":
            history_rows,
    }



def load_completed_seed_result(
    seed,
    dataset,
    device,
):
    """
    Load and independently reevaluate a completed diagnostic
    checkpoint so an interrupted aggregation step does not
    require retraining the seed.
    """

    checkpoint_path = (
        OUTPUT_DIR
        / f"state_decoder_seed_{seed}.pt"
    )

    history_path = (
        OUTPUT_DIR
        / f"state_decoder_seed_{seed}_history.csv"
    )

    if (
        not checkpoint_path.exists()
        or not history_path.exists()
    ):
        return None

    checkpoint = safe_torch_load(
        checkpoint_path,
        map_location="cpu",
    )

    if (
        checkpoint.get("protocol_version")
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            f"Existing seed {seed} checkpoint has "
            "the wrong protocol version."
        )

    if int(
        checkpoint.get("seed", -1)
    ) != seed:
        raise AssertionError(
            f"Existing checkpoint seed does not "
            f"match requested seed {seed}."
        )

    if (
        checkpoint.get("test_fields_used")
        is not False
        or checkpoint.get(
            "test_metrics_computed"
        )
        is not False
    ):
        raise AssertionError(
            f"Existing seed {seed} checkpoint "
            "does not preserve test sealing."
        )

    if (
        checkpoint.get(
            "eligible_for_predictive_use"
        )
        is not False
    ):
        raise AssertionError(
            f"Existing seed {seed} checkpoint was "
            "incorrectly marked for predictive use."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Existing seed {seed} checkpoint "
            "does not contain model_state_dict."
        )

    model = (
        FrozenDiagnosticStateDecoder()
        .to(device)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    training_result = evaluate_model(
        model=model,
        fields=dataset["train_fields"],
        labels=dataset["train_labels"],
        device=device,
    )

    validation_result = evaluate_model(
        model=model,
        fields=dataset[
            "validation_fields"
        ],
        labels=dataset[
            "validation_labels"
        ],
        device=device,
    )

    saved_training_accuracy = float(
        checkpoint["training_accuracy"]
    )

    saved_validation_accuracy = float(
        checkpoint["validation_accuracy"]
    )

    if abs(
        training_result["accuracy"]
        - saved_training_accuracy
    ) > 1e-12:
        raise AssertionError(
            f"Seed {seed} training accuracy did "
            "not reproduce from the saved checkpoint."
        )

    if abs(
        validation_result["accuracy"]
        - saved_validation_accuracy
    ) > 1e-12:
        raise AssertionError(
            f"Seed {seed} validation accuracy did "
            "not reproduce from the saved checkpoint."
        )

    print(
        f"seed={seed} reused checkpoint "
        f"best_epoch={checkpoint['best_epoch']} "
        f"train_acc={training_result['accuracy']:.6f} "
        f"val_acc={validation_result['accuracy']:.6f}"
    )

    return {
        "seed":
            seed,

        "best_epoch":
            int(
                checkpoint["best_epoch"]
            ),

        "training_result":
            training_result,

        "validation_result":
            validation_result,

        "checkpoint_path":
            checkpoint_path,

        "history_rows":
            load_csv(history_path),
    }

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


def confusion_rows(
    seed,
    split,
    labels,
    predictions,
    state_registry,
):
    matrix = confusion_matrix(
        labels=labels,
        predictions=predictions,
        class_count=STATE_COUNT,
    )

    rows = []

    for true_state in range(
        STATE_COUNT
    ):
        for predicted_state in range(
            STATE_COUNT
        ):
            rows.append(
                {
                    "seed":
                        seed,

                    "split":
                        split,

                    "true_state":
                        true_state,

                    "true_state_code":
                        state_registry[
                            "state_codes"
                        ][true_state],

                    "predicted_state":
                        predicted_state,

                    "predicted_state_code":
                        state_registry[
                            "state_codes"
                        ][predicted_state],

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


def factor_metric_rows(
    seed,
    split,
    labels,
    predictions,
    state_registry,
):
    rows = []

    for factor_name in FACTOR_NAMES:
        lookup = state_registry[
            "factor_values"
        ][factor_name]

        true_factor = lookup[labels]
        predicted_factor = lookup[
            predictions
        ]

        accuracy = float(
            np.mean(
                true_factor
                == predicted_factor
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

                "error_rate":
                    1.0 - accuracy,
            }
        )

    return rows


def per_state_rows(
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
        selected = labels == state_id

        sample_count = int(
            selected.sum()
        )

        correct_count = int(
            np.sum(
                predictions[selected]
                == state_id
            )
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
                    state_registry[
                        "state_codes"
                    ][state_id],

                "sample_count":
                    sample_count,

                "correct_count":
                    correct_count,

                "accuracy":
                    correct_count
                    / sample_count,
            }
        )

    return rows


def summarize_factor_rows(
    rows,
):
    output = []

    for split in (
        "train",
        "validation",
    ):
        for factor_name in FACTOR_NAMES:
            values = np.asarray(
                [
                    row["accuracy"]
                    for row in rows
                    if row["split"] == split
                    and row["factor"]
                    == factor_name
                ],
                dtype=np.float64,
            )

            output.append(
                {
                    "split":
                        split,

                    "factor":
                        factor_name,

                    "seed_count":
                        len(values),

                    "accuracy_mean":
                        float(values.mean()),

                    "accuracy_standard_deviation":
                        float(
                            values.std(
                                ddof=1
                            )
                        ),

                    "accuracy_minimum":
                        float(values.min()),

                    "accuracy_maximum":
                        float(values.max()),
                }
            )

    return output


def write_training_protocol():
    protocol = {
        "phase":
            (
                "4C-R3A Tier C v4 diagnostic "
                "state-decoder training protocol"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "protocol_frozen_before_training":
            True,

        "architecture":
            (
                "Residual convolutional state decoder "
                "identical in parameter structure to "
                "Tier C v2."
            ),

        "diagnostic_seeds":
            list(DIAGNOSTIC_SEEDS),

        "training_fields":
            (
                "Clean normalized Tier C v4 "
                "training-carrier fields only."
            ),

        "validation_fields":
            (
                "Clean normalized Tier C v4 "
                "validation-carrier fields only."
            ),

        "epoch_count":
            EPOCH_COUNT,

        "batch_size":
            BATCH_SIZE,

        "optimizer":
            "AdamW",

        "learning_rate":
            LEARNING_RATE,

        "minimum_learning_rate":
            MINIMUM_LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "scheduler":
            "CosineAnnealingLR",

        "dropout_probability":
            DROPOUT_PROBABILITY,

        "label_smoothing":
            LABEL_SMOOTHING,

        "checkpoint_selection_rule":
            (
                "Highest clean validation state "
                "accuracy, with lower validation loss "
                "as the sole tie-break."
            ),

        "data_augmentation":
            False,

        "test_fields_used":
            False,

        "diagnostic_checkpoints_eligible_for_predictive_use":
            False,

        "thresholds": {
            "minimum_mean_validation_state_accuracy":
                MINIMUM_MEAN_VALIDATION_STATE_ACCURACY,

            "minimum_worst_validation_state_accuracy":
                MINIMUM_WORST_VALIDATION_STATE_ACCURACY,

            "minimum_mean_validation_coarsening_accuracy":
                MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY,

            "minimum_worst_validation_coarsening_accuracy":
                MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY,
        },
    }

    write_json(
        OUTPUT_DIR
        / "state_decoder_training_protocol.json",
        protocol,
    )


def write_input_hashes():
    paths = {
        "phase4ar3_summary":
            PHASE4AR3_DIR
            / "phase4ar3_summary.json",

        "state_registry":
            PHASE4AR3_DIR
            / "state_morphology_regimes_v4.csv",

        "predecoder_summary":
            AUDIT_DIR
            / "phase4cr3_predecoder_summary.json",

        "development_carrier_ids":
            FIELD_DIR
            / "privileged"
            / "development_carrier_ids.npy",

        "development_carrier_splits":
            FIELD_DIR
            / "privileged"
            / "development_carrier_splits.npy",

        "normalized_field_bank":
            FIELD_DIR
            / "privileged"
            / "development_carrier_state_fields_normalized.npy",

        "v2_architecture_reference":
            V2_AUDIT_DIR
            / (
                "state_decoder_seed_"
                f"{V2_ARCHITECTURE_REFERENCE_SEED}.pt"
            ),
    }

    hashes = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        hashes[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    write_json(
        OUTPUT_DIR
        / "state_decoder_input_hashes.json",
        hashes,
    )


def main():
    arguments = parse_arguments()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_summary_path = (
        OUTPUT_DIR
        / "state_decoder_summary.json"
    )

    if (
        final_summary_path.exists()
        and not arguments.force
    ):
        existing = load_json(
            final_summary_path
        )

        if existing.get(
            "state_decoder_diagnostic_status"
        ) in {
            "passed",
            "failed_terminal_gate",
        }:
            print(
                "Tier C v4 state-decoder diagnostic "
                "already has a frozen result."
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            return

    validate_authorization()

    device = resolve_device(
        arguments.device
    )

    architecture_audit = (
        validate_architecture_against_v2()
    )

    state_registry = load_state_registry()

    dataset = load_clean_dataset()

    write_training_protocol()
    write_input_hashes()

    seed_results = []

    all_confusion_rows = []
    all_factor_rows = []
    all_per_state_rows = []

    for index, seed in enumerate(
        DIAGNOSTIC_SEEDS,
        start=1,
    ):
        print()

        result = None

        if not arguments.force:
            result = load_completed_seed_result(
                seed=seed,
                dataset=dataset,
                device=device,
            )

        if result is None:
            print(
                f"[{index}/{len(DIAGNOSTIC_SEEDS)}] "
                f"Training diagnostic seed {seed}"
            )

            result = train_one_seed(
                seed=seed,
                dataset=dataset,
                device=device,
            )

        else:
            print(
                f"[{index}/{len(DIAGNOSTIC_SEEDS)}] "
                f"Reused completed diagnostic seed {seed}"
            )

        seed_results.append(result)

        for (
            split,
            evaluation_key,
            labels_key,
        ) in (
            (
                "train",
                "training_result",
                "train_labels",
            ),
            (
                "validation",
                "validation_result",
                "validation_labels",
            ),
        ):
            labels = dataset[labels_key]

            evaluation = result[
                evaluation_key
            ]

            predictions = evaluation[
                "predictions"
            ]

            all_confusion_rows.extend(
                confusion_rows(
                    seed=seed,
                    split=split,
                    labels=labels,
                    predictions=predictions,
                    state_registry=state_registry,
                )
            )

            all_factor_rows.extend(
                factor_metric_rows(
                    seed=seed,
                    split=split,
                    labels=labels,
                    predictions=predictions,
                    state_registry=state_registry,
                )
            )

            all_per_state_rows.extend(
                per_state_rows(
                    seed=seed,
                    split=split,
                    labels=labels,
                    predictions=predictions,
                    state_registry=state_registry,
                )
            )

    seed_metric_rows = []

    for result in seed_results:
        seed = result["seed"]

        training_result = result[
            "training_result"
        ]

        validation_result = result[
            "validation_result"
        ]

        coarsening_row = next(
            row
            for row in all_factor_rows
            if row["seed"] == seed
            and row["split"] == "validation"
            and row["factor"]
            == "coarsening_level"
        )

        anisotropy_row = next(
            row
            for row in all_factor_rows
            if row["seed"] == seed
            and row["split"] == "validation"
            and row["factor"]
            == "anisotropy_level"
        )

        defect_row = next(
            row
            for row in all_factor_rows
            if row["seed"] == seed
            and row["split"] == "validation"
            and row["factor"]
            == "defect_recovery_level"
        )

        seed_metric_rows.append(
            {
                "seed":
                    seed,

                "best_epoch":
                    result["best_epoch"],

                "training_state_accuracy":
                    training_result[
                        "accuracy"
                    ],

                "validation_state_accuracy":
                    validation_result[
                        "accuracy"
                    ],

                "training_loss":
                    training_result[
                        "loss"
                    ],

                "validation_loss":
                    validation_result[
                        "loss"
                    ],

                "validation_coarsening_accuracy":
                    coarsening_row[
                        "accuracy"
                    ],

                "validation_anisotropy_accuracy":
                    anisotropy_row[
                        "accuracy"
                    ],

                "validation_defect_recovery_accuracy":
                    defect_row[
                        "accuracy"
                    ],

                "checkpoint_path":
                    str(
                        result[
                            "checkpoint_path"
                        ]
                    ),

                "eligible_for_predictive_use":
                    False,
            }
        )

    write_csv(
        OUTPUT_DIR
        / "state_decoder_seed_metrics.csv",
        seed_metric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "state_decoder_confusion_by_seed.csv",
        all_confusion_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "state_decoder_factor_metrics.csv",
        all_factor_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "state_decoder_per_state_metrics.csv",
        all_per_state_rows,
    )

    factor_summary_rows = (
        summarize_factor_rows(
            all_factor_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "state_decoder_factor_summary.csv",
        factor_summary_rows,
    )

    validation_state_accuracies = np.asarray(
        [
            row[
                "validation_state_accuracy"
            ]
            for row in seed_metric_rows
        ],
        dtype=np.float64,
    )

    training_state_accuracies = np.asarray(
        [
            row[
                "training_state_accuracy"
            ]
            for row in seed_metric_rows
        ],
        dtype=np.float64,
    )

    validation_coarsening_accuracies = np.asarray(
        [
            row[
                "validation_coarsening_accuracy"
            ]
            for row in seed_metric_rows
        ],
        dtype=np.float64,
    )

    validation_anisotropy_accuracies = np.asarray(
        [
            row[
                "validation_anisotropy_accuracy"
            ]
            for row in seed_metric_rows
        ],
        dtype=np.float64,
    )

    validation_defect_accuracies = np.asarray(
        [
            row[
                "validation_defect_recovery_accuracy"
            ]
            for row in seed_metric_rows
        ],
        dtype=np.float64,
    )

    mean_state_gate_passed = bool(
        validation_state_accuracies.mean()
        >= MINIMUM_MEAN_VALIDATION_STATE_ACCURACY
    )

    worst_state_gate_passed = bool(
        validation_state_accuracies.min()
        >= MINIMUM_WORST_VALIDATION_STATE_ACCURACY
    )

    mean_coarsening_gate_passed = bool(
        validation_coarsening_accuracies.mean()
        >= MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY
    )

    worst_coarsening_gate_passed = bool(
        validation_coarsening_accuracies.min()
        >= MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY
    )

    state_separability_gate_passed = bool(
        mean_state_gate_passed
        and worst_state_gate_passed
        and mean_coarsening_gate_passed
        and worst_coarsening_gate_passed
    )

    summary = {
        "phase":
            (
                "4C-R3A Tier C v4 three-seed "
                "state-separability diagnostic"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_predecoder_gate_passed":
            True,

        "architecture_signature_matches_v2":
            architecture_audit[
                "architecture_signature_matches_v2"
            ],

        "model_parameter_count":
            architecture_audit[
                "parameter_count"
            ],

        "device":
            str(device),

        "diagnostic_seed_count":
            len(DIAGNOSTIC_SEEDS),

        "diagnostic_seeds":
            list(DIAGNOSTIC_SEEDS),

        "training_sample_count":
            EXPECTED_TRAIN_SAMPLE_COUNT,

        "validation_sample_count":
            EXPECTED_VALIDATION_SAMPLE_COUNT,

        "mean_training_state_accuracy":
            float(
                training_state_accuracies.mean()
            ),

        "mean_validation_state_accuracy":
            float(
                validation_state_accuracies.mean()
            ),

        "validation_state_accuracy_standard_deviation":
            float(
                validation_state_accuracies.std(
                    ddof=1
                )
            ),

        "worst_seed_validation_state_accuracy":
            float(
                validation_state_accuracies.min()
            ),

        "best_seed_validation_state_accuracy":
            float(
                validation_state_accuracies.max()
            ),

        "required_mean_validation_state_accuracy":
            MINIMUM_MEAN_VALIDATION_STATE_ACCURACY,

        "required_worst_seed_validation_state_accuracy":
            MINIMUM_WORST_VALIDATION_STATE_ACCURACY,

        "mean_validation_coarsening_accuracy":
            float(
                validation_coarsening_accuracies.mean()
            ),

        "worst_seed_validation_coarsening_accuracy":
            float(
                validation_coarsening_accuracies.min()
            ),

        "required_mean_validation_coarsening_accuracy":
            MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY,

        "required_worst_seed_validation_coarsening_accuracy":
            MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY,

        "mean_validation_anisotropy_accuracy":
            float(
                validation_anisotropy_accuracies.mean()
            ),

        "worst_seed_validation_anisotropy_accuracy":
            float(
                validation_anisotropy_accuracies.min()
            ),

        "mean_validation_defect_recovery_accuracy":
            float(
                validation_defect_accuracies.mean()
            ),

        "worst_seed_validation_defect_recovery_accuracy":
            float(
                validation_defect_accuracies.min()
            ),

        "mean_state_accuracy_gate_passed":
            mean_state_gate_passed,

        "worst_state_accuracy_gate_passed":
            worst_state_gate_passed,

        "mean_coarsening_accuracy_gate_passed":
            mean_coarsening_gate_passed,

        "worst_coarsening_accuracy_gate_passed":
            worst_coarsening_gate_passed,

        "state_separability_gate_passed":
            state_separability_gate_passed,

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_fields_used":
            False,

        "test_metrics_computed":
            False,

        "diagnostic_decoder_fitted":
            True,

        "diagnostic_checkpoint_count":
            len(DIAGNOSTIC_SEEDS),

        "diagnostic_checkpoints_eligible_for_predictive_use":
            False,

        "predictive_training_performed":
            False,

        "predictive_checkpoint_selection_performed":
            False,

        "global_statistics_diagnostic_authorized":
            state_separability_gate_passed,

        "affine_shortcut_diagnostic_authorized":
            state_separability_gate_passed,

        "predictive_training_authorized":
            False,

        "tier_c_v5_authorized":
            False,

        "prior_outputs_modified":
            False,

        "state_decoder_diagnostic_status":
            (
                "passed"
                if state_separability_gate_passed
                else "failed_terminal_gate"
            ),
    }

    write_json(
        final_summary_path,
        summary,
    )

    print()
    print(
        "Phase 4C-R3A Tier C v4 state-decoder "
        "diagnostic completed."
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

    if not state_separability_gate_passed:
        raise SystemExit(
            "Tier C v4 failed the terminal state-"
            "separability gate. Do not run the global-"
            "statistics or affine diagnostics, do not "
            "train predictive models, and do not create "
            "Tier C v5."
        )


if __name__ == "__main__":
    main()
