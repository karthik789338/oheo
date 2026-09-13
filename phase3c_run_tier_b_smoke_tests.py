from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phase2b_operation_channel_model import (
    OperationChannelModel,
)

from phase2b_run_ocm_smoke_test import (
    ContinuousTrajectoryDataset,
    TrajectoryRecord,
    collate_trajectories,
    set_random_seed,
)

from phase2h_baseline_models import (
    OperationConditionedMLP,
    GRUOperationSequenceModel,
    TransformerOperationSequenceModel,
    ContinuousLatentOperatorModel,
)


PHASE3A_DIR = Path(
    "outputs/phase3a_tier_b_protocol"
)

PHASE3B_DATA_DIR = Path(
    "outputs/phase3b_tier_b_data"
)

PHASE3B_AUDIT_DIR = Path(
    "outputs/phase3b_tier_b_audit"
)

VISIBLE_DIR = (
    PHASE3B_DATA_DIR / "visible"
)

OUTPUT_DIR = Path(
    "outputs/phase3c_tier_b_smoke"
)

HISTORY_DIR = (
    OUTPUT_DIR / "histories"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR / "smoke_checkpoints"
)


OBSERVATION_DIMENSION = 32
LATENT_STATE_COUNT = 8

DEVELOPMENT_SEED = 11
DEVELOPMENT_NOISE = 0.25

BATCH_SIZE = 128
MAXIMUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-6

SMOKE_TRAIN_TRAJECTORY_LIMIT = 4096
SMOKE_VALIDATION_TRAJECTORY_LIMIT = 1152

OCM_TEMPERATURE = 0.5

EXPECTED_FULL_TRAJECTORY_COUNTS = {
    "train_joint": 11008,
    "val_joint": 1152,
}

MODEL_IDS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)


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
        "--epochs",
        type=int,
        default=MAXIMUM_EPOCHS,
    )

    parser.add_argument(
        "--force",
        action="store_true",
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

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
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
        writer.writerows(
            rows
        )


def validate_source_phases():
    phase3a = load_json(
        PHASE3A_DIR
        / "phase3a_summary.json"
    )

    phase3b = load_json(
        PHASE3B_AUDIT_DIR
        / "phase3b_summary.json"
    )

    if (
        phase3a[
            "status"
        ]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 3A is not frozen."
        )

    if (
        phase3b[
            "phase3b_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 3B has not passed."
        )

    if (
        phase3b[
            "tier_b_data_accepted"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier B data was not accepted."
        )

    if (
        phase3b[
            "affine_shortcut_removed"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier B affine shortcut remains present."
        )

    if (
        phase3b[
            "test_used_for_model_selection"
        ]
        is not False
    ):
        raise AssertionError(
            "Unexpected test-selection flag."
        )

    return {
        "phase3a":
            phase3a,

        "phase3b":
            phase3b,
    }


def load_operation_vocabulary():
    path = (
        Path(
            "outputs/phase2a_learning_protocol"
        )
        / "operation_vocabulary.json"
    )

    raw = load_json(
        path
    )

    if isinstance(
        raw,
        dict,
    ):
        vocabulary = {}

        for name, value in raw.items():
            if isinstance(
                value,
                dict,
            ):
                if "operation_id" in value:
                    operation_id = value[
                        "operation_id"
                    ]
                elif "index" in value:
                    operation_id = value[
                        "index"
                    ]
                else:
                    raise KeyError(
                        f"No operation index for {name}."
                    )
            else:
                operation_id = value

            vocabulary[
                str(
                    name
                )
            ] = int(
                operation_id
            )

    elif isinstance(
        raw,
        list,
    ):
        vocabulary = {}

        for index, row in enumerate(
            raw
        ):
            if isinstance(
                row,
                str,
            ):
                name = row
                operation_id = index
            else:
                name = row.get(
                    "name",
                    row.get(
                        "operation",
                    ),
                )

                operation_id = row.get(
                    "operation_id",
                    row.get(
                        "index",
                        index,
                    ),
                )

            vocabulary[
                str(
                    name
                )
            ] = int(
                operation_id
            )

    else:
        raise TypeError(
            "Unsupported operation-vocabulary format."
        )

    expected_ids = set(
        range(
            len(
                vocabulary
            )
        )
    )

    if set(
        vocabulary.values()
    ) != expected_ids:
        raise AssertionError(
            "Operation IDs are not contiguous."
        )

    return vocabulary


def parse_operations(
    value: str,
):
    parsed = json.loads(
        value
    )

    if not isinstance(
        parsed,
        list,
    ):
        raise TypeError(
            "Operation sequence must be a JSON list."
        )

    return tuple(
        str(
            item
        )
        for item in parsed
    )


def load_cell(
    cell_id: str,
    operation_vocabulary,
):
    if cell_id not in {
        "train_joint",
        "val_joint",
    }:
        raise ValueError(
            "Phase 3C may open only train_joint "
            "and val_joint."
        )

    observation_path = (
        VISIBLE_DIR
        / f"{cell_id}_observations.npz"
    )

    trajectory_path = (
        VISIBLE_DIR
        / f"{cell_id}_trajectory_index.csv"
    )

    sequence_path = (
        VISIBLE_DIR
        / f"{cell_id}_sequence_manifest.csv"
    )

    with np.load(
        observation_path
    ) as data:
        if set(
            data.files
        ) != {
            "noise_fractions",
            "vectors",
        }:
            raise AssertionError(
                f"{cell_id} visible NPZ has "
                "unexpected arrays."
            )

        noise_fractions = np.asarray(
            data[
                "noise_fractions"
            ],
            dtype=np.float64,
        )

        matches = np.where(
            np.isclose(
                noise_fractions,
                DEVELOPMENT_NOISE,
            )
        )[0]

        if len(
            matches
        ) != 1:
            raise AssertionError(
                "Could not locate development noise."
            )

        vectors = np.array(
            data[
                "vectors"
            ][
                int(
                    matches[0]
                )
            ],
            dtype=np.float32,
            copy=True,
        )

    if (
        vectors.ndim != 2
        or vectors.shape[1]
        != OBSERVATION_DIMENSION
    ):
        raise AssertionError(
            f"{cell_id} has invalid observation shape "
            f"{vectors.shape}."
        )

    if not np.isfinite(
        vectors
    ).all():
        raise FloatingPointError(
            f"{cell_id} contains non-finite values."
        )

    sequence_rows = load_csv(
        sequence_path
    )

    operations_by_sequence = {}

    for row in sequence_rows:
        sequence_index = int(
            row[
                "sequence_index"
            ]
        )

        names = parse_operations(
            row[
                "operation_sequence"
            ]
        )

        unknown = [
            name
            for name in names
            if name not in operation_vocabulary
        ]

        if unknown:
            raise AssertionError(
                "Unknown operations: "
                + ", ".join(
                    unknown
                )
            )

        operations_by_sequence[
            sequence_index
        ] = tuple(
            operation_vocabulary[
                name
            ]
            for name in names
        )

    trajectory_rows = load_csv(
        trajectory_path
    )

    records = []

    for row in trajectory_rows:
        sequence_index = int(
            row[
                "sequence_index"
            ]
        )

        if (
            sequence_index
            not in operations_by_sequence
        ):
            raise AssertionError(
                "Trajectory references an unknown "
                "sequence."
            )

        record = TrajectoryRecord(
            trajectory_index=int(
                row[
                    "trajectory_index"
                ]
            ),

            sequence_index=sequence_index,

            point_start=int(
                row[
                    "point_start"
                ]
            ),

            point_count=int(
                row[
                    "point_count"
                ]
            ),
        )

        expected_point_count = (
            len(
                operations_by_sequence[
                    sequence_index
                ]
            )
            + 1
        )

        if (
            record.point_count
            != expected_point_count
        ):
            raise AssertionError(
                "Trajectory length and operation "
                "count disagree."
            )

        records.append(
            record
        )

    if (
        len(
            records
        )
        != EXPECTED_FULL_TRAJECTORY_COUNTS[
            cell_id
        ]
    ):
        raise AssertionError(
            f"{cell_id} trajectory count changed."
        )

    return {
        "vectors":
            vectors,

        "records":
            records,

        "operations_by_sequence":
            operations_by_sequence,
    }


def evenly_spaced_subset(
    records,
    limit: int,
):
    if limit >= len(
        records
    ):
        return list(
            records
        )

    indices = np.linspace(
        0,
        len(
            records
        )
        - 1,
        num=limit,
        dtype=np.int64,
    )

    if len(
        np.unique(
            indices
        )
    ) != limit:
        raise AssertionError(
            "Subset selection produced duplicate "
            "indices."
        )

    return [
        records[
            int(
                index
            )
        ]
        for index in indices
    ]


def make_loader(
    vectors,
    records,
    operations_by_sequence,
    seed,
    shuffle,
):
    dataset = ContinuousTrajectoryDataset(
        vectors=vectors,
        records=records,
        operations_by_sequence=(
            operations_by_sequence
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_trajectories,
        generator=generator,
    )


def final_target(
    observations,
    lengths,
):
    indices = torch.arange(
        observations.shape[0],
        device=observations.device,
    )

    return observations[
        indices,
        lengths - 1,
        :,
    ]


def parameter_count(
    model,
):
    return int(
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
        )
    )


def state_is_finite(
    model,
):
    return bool(
        all(
            torch.isfinite(
                value
            ).all().item()
            for value
            in model.state_dict().values()
        )
    )


class AffineRolloutModel(nn.Module):

    def __init__(
        self,
        matrices,
        biases,
    ):
        super().__init__()

        self.register_buffer(
            "matrices",
            matrices,
        )

        self.register_buffer(
            "biases",
            biases,
        )

    def step(
        self,
        observations,
        operation_ids,
    ):
        matrices = self.matrices[
            operation_ids
        ]

        biases = self.biases[
            operation_ids
        ]

        return (
            torch.bmm(
                observations.unsqueeze(
                    1
                ),
                matrices,
            ).squeeze(
                1
            )
            + biases
        )

    def rollout(
        self,
        initial_observation,
        operation_ids,
        step_mask,
    ):
        current = initial_observation

        for step_index in range(
            operation_ids.shape[1]
        ):
            proposed = self.step(
                observations=current,
                operation_ids=operation_ids[
                    :,
                    step_index,
                ],
            )

            active = step_mask[
                :,
                step_index,
            ].unsqueeze(
                1
            )

            current = torch.where(
                active,
                proposed,
                current,
            )

        return current


def fit_affine_model(
    vectors,
    records,
    operations_by_sequence,
    operation_count,
    ridge,
):
    input_blocks = [
        []
        for _ in range(
            operation_count
        )
    ]

    target_blocks = [
        []
        for _ in range(
            operation_count
        )
    ]

    for record in records:
        start = record.point_start
        end = (
            start
            + record.point_count
        )

        observations = vectors[
            start:end
        ]

        operations = (
            operations_by_sequence[
                record.sequence_index
            ]
        )

        for step_index, operation_id in enumerate(
            operations
        ):
            input_blocks[
                operation_id
            ].append(
                observations[
                    step_index
                ]
            )

            target_blocks[
                operation_id
            ].append(
                observations[
                    step_index + 1
                ]
            )

    matrices = []
    biases = []

    for operation_id in range(
        operation_count
    ):
        inputs = np.asarray(
            input_blocks[
                operation_id
            ],
            dtype=np.float64,
        )

        targets = np.asarray(
            target_blocks[
                operation_id
            ],
            dtype=np.float64,
        )

        if len(
            inputs
        ) == 0:
            raise AssertionError(
                f"Operation {operation_id} has "
                "no affine training examples."
            )

        augmented = np.concatenate(
            [
                inputs,
                np.ones(
                    (
                        len(
                            inputs
                        ),
                        1,
                    ),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )

        penalty = np.eye(
            augmented.shape[1],
            dtype=np.float64,
        )

        penalty[
            -1,
            -1,
        ] = 0.0

        weights = np.linalg.solve(
            augmented.T
            @ augmented
            + ridge
            * penalty,
            augmented.T
            @ targets,
        )

        matrices.append(
            weights[
                :-1,
                :
            ]
        )

        biases.append(
            weights[
                -1,
                :
            ]
        )

    return AffineRolloutModel(
        matrices=torch.tensor(
            np.asarray(
                matrices
            ),
            dtype=torch.float32,
        ),

        biases=torch.tensor(
            np.asarray(
                biases
            ),
            dtype=torch.float32,
        ),
    )


def try_constructor(
    model_class,
    candidates,
):
    errors = []

    for kwargs in candidates:
        try:
            return model_class(
                **kwargs
            )

        except TypeError as error:
            errors.append(
                {
                    "kwargs":
                        sorted(
                            kwargs
                        ),

                    "error":
                        str(
                            error
                        ),
                }
            )

    signature = str(
        inspect.signature(
            model_class.__init__
        )
    )

    raise TypeError(
        f"Could not instantiate "
        f"{model_class.__name__}{signature}. "
        f"Attempts: {errors}"
    )


def build_baseline_model(
    model_id,
    operation_count,
    maximum_sequence_length,
):
    if model_id == "B2":
        return try_constructor(
            OperationConditionedMLP,
            [
                {
                    "observation_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "operation_embedding_dim":
                        16,

                    "hidden_width":
                        64,
                },
                {
                    "obs_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "operation_embedding_dim":
                        16,

                    "hidden_width":
                        64,
                },
            ],
        )

    if model_id == "B3":
        return try_constructor(
            GRUOperationSequenceModel,
            [
                {
                    "observation_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "operation_embedding_dim":
                        32,

                    "hidden_size":
                        64,
                },
                {
                    "obs_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "operation_embedding_dim":
                        32,

                    "hidden_size":
                        64,
                },
            ],
        )

    if model_id == "B4":
        return try_constructor(
            TransformerOperationSequenceModel,
            [
                {
                    "observation_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "model_dimension":
                        64,

                    "attention_heads":
                        4,

                    "layer_count":
                        2,

                    "feedforward_dimension":
                        128,

                    "dropout":
                        0.0,

                    "maximum_sequence_length":
                        maximum_sequence_length,
                },
                {
                    "observation_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "model_dimension":
                        64,

                    "attention_heads":
                        4,

                    "layer_count":
                        2,

                    "feedforward_dimension":
                        128,

                    "dropout":
                        0.0,
                },
                {
                    "obs_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "model_dimension":
                        64,

                    "attention_heads":
                        4,

                    "layer_count":
                        2,

                    "feedforward_dimension":
                        128,

                    "dropout":
                        0.0,

                    "maximum_sequence_length":
                        maximum_sequence_length,
                },
            ],
        )

    if model_id == "B5":
        return try_constructor(
            ContinuousLatentOperatorModel,
            [
                {
                    "observation_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "latent_dimension":
                        8,

                    "encoder_width":
                        64,

                    "operator_hidden_width":
                        32,
                },
                {
                    "obs_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "latent_dimension":
                        8,

                    "encoder_width":
                        64,

                    "operator_hidden_width":
                        32,
                },
            ],
        )

    raise ValueError(
        f"Unsupported baseline: {model_id}"
    )


def find_channel_logits(
    model,
    operation_count,
):
    candidates = []

    for name, value in model.named_parameters():
        if (
            value.ndim == 3
            and value.shape[0]
            == operation_count
            and value.shape[1]
            == LATENT_STATE_COUNT
            and value.shape[2]
            == LATENT_STATE_COUNT
        ):
            candidates.append(
                (
                    name,
                    value,
                )
            )

    if len(
        candidates
    ) == 1:
        return candidates[0]

    square_parameters = []

    for name, value in model.named_parameters():
        if (
            value.ndim == 2
            and value.shape
            == (
                LATENT_STATE_COUNT,
                LATENT_STATE_COUNT,
            )
            and (
                "operation" in name.lower()
                or "transition" in name.lower()
                or "channel" in name.lower()
            )
        ):
            square_parameters.append(
                (
                    name,
                    value,
                )
            )

    if len(
        square_parameters
    ) == operation_count:
        square_parameters.sort(
            key=lambda item:
                item[0]
        )

        return (
            "stacked_square_parameters",
            torch.stack(
                [
                    value
                    for _, value
                    in square_parameters
                ],
                dim=0,
            ),
        )

    return (
        None,
        None,
    )


def channel_entropy(
    model,
    operation_count,
    temperature,
):
    _, logits = find_channel_logits(
        model,
        operation_count,
    )

    if logits is None:
        return torch.zeros(
            (),
            device=next(
                model.parameters()
            ).device,
        )

    probabilities = torch.softmax(
        logits
        / temperature,
        dim=-1,
    )

    entropy = -(
        probabilities
        * torch.log(
            probabilities.clamp_min(
                1e-8
            )
        )
    ).sum(
        dim=-1
    )

    return entropy.mean()


def channel_row_sum_error(
    model,
    operation_count,
    temperature,
):
    _, logits = find_channel_logits(
        model,
        operation_count,
    )

    if logits is None:
        return None

    probabilities = torch.softmax(
        logits
        / temperature,
        dim=-1,
    )

    return float(
        (
            probabilities.sum(
                dim=-1
            )
            - 1.0
        )
        .abs()
        .max()
        .item()
    )


def js_divergence(
    first,
    second,
):
    first = first.clamp_min(
        1e-8
    )

    second = second.clamp_min(
        1e-8
    )

    middle = 0.5 * (
        first + second
    )

    return 0.5 * (
        (
            first
            * (
                torch.log(
                    first
                )
                - torch.log(
                    middle
                )
            )
        ).sum(
            dim=-1
        )
        + (
            second
            * (
                torch.log(
                    second
                )
                - torch.log(
                    middle
                )
            )
        ).sum(
            dim=-1
        )
    )


def masked_vector_mse(
    prediction,
    target,
    mask,
):
    squared = (
        prediction
        - target
    ).pow(
        2
    ).mean(
        dim=-1
    )

    weights = mask.to(
        squared.dtype
    )

    return (
        squared
        * weights
    ).sum() / weights.sum().clamp_min(
        1.0
    )


def ocm_forward_losses(
    model,
    batch,
    operation_count,
    temperature,
):
    observations = batch[
        "observations"
    ]

    operation_ids = batch[
        "operation_ids"
    ]

    step_mask = batch[
        "step_mask"
    ]

    lengths = batch[
        "lengths"
    ]

    batch_size, time_count, _ = (
        observations.shape
    )

    time_indices = torch.arange(
        time_count,
        device=observations.device,
    )[None, :]

    observation_mask = (
        time_indices
        < lengths[:, None]
    )

    flat_observations = (
        observations.reshape(
            -1,
            OBSERVATION_DIMENSION,
        )
    )

    encoded_flat = model.encode(
        flat_observations,
        temperature=temperature,
    )

    encoded = encoded_flat.reshape(
        batch_size,
        time_count,
        LATENT_STATE_COUNT,
    )

    reconstructed_flat = model.decode(
        encoded_flat
    )

    reconstructed = (
        reconstructed_flat.reshape(
            batch_size,
            time_count,
            OBSERVATION_DIMENSION,
        )
    )

    reconstruction_loss = (
        masked_vector_mse(
            prediction=reconstructed,
            target=observations,
            mask=observation_mask,
        )
    )

    current = encoded[
        :,
        0,
        :
    ]

    predicted_steps = []
    transition_losses = []

    for step_index in range(
        operation_ids.shape[1]
    ):
        proposed = model.apply_operation(
            latent_distribution=current,
            operation_ids=operation_ids[
                :,
                step_index,
            ],
            temperature=temperature,
        )

        predicted_observation = (
            model.decode(
                proposed
            )
        )

        predicted_steps.append(
            predicted_observation
        )

        target_distribution = encoded[
            :,
            step_index + 1,
            :
        ]

        transition_losses.append(
            js_divergence(
                proposed,
                target_distribution,
            )
        )

        active = step_mask[
            :,
            step_index,
        ].unsqueeze(
            1
        )

        current = torch.where(
            active,
            proposed,
            current,
        )

    predicted_steps = torch.stack(
        predicted_steps,
        dim=1,
    )

    transition_losses = torch.stack(
        transition_losses,
        dim=1,
    )

    step_loss = masked_vector_mse(
        prediction=predicted_steps,
        target=observations[
            :,
            1:,
            :,
        ],
        mask=step_mask,
    )

    transition_weights = step_mask.to(
        transition_losses.dtype
    )

    transition_loss = (
        transition_losses
        * transition_weights
    ).sum() / transition_weights.sum().clamp_min(
        1.0
    )

    final_prediction = model.decode(
        current
    )

    target = final_target(
        observations,
        lengths,
    )

    final_rollout_loss = F.mse_loss(
        final_prediction,
        target,
    )

    determinism_loss = channel_entropy(
        model=model,
        operation_count=operation_count,
        temperature=temperature,
    )

    total_loss = (
        reconstruction_loss
        + step_loss
        + final_rollout_loss
        + 0.1
        * transition_loss
        + 0.001
        * determinism_loss
    )

    return {
        "total_loss":
            total_loss,

        "reconstruction_loss":
            reconstruction_loss,

        "step_loss":
            step_loss,

        "final_rollout_loss":
            final_rollout_loss,

        "transition_loss":
            transition_loss,

        "determinism_loss":
            determinism_loss,

        "final_prediction":
            final_prediction,
    }


@torch.no_grad()
def evaluate_persistence(
    loader,
    device,
):
    squared_error_sum = 0.0
    value_count = 0

    for batch in loader:
        observations = batch[
            "observations"
        ].to(
            device
        )

        lengths = batch[
            "lengths"
        ].to(
            device
        )

        prediction = observations[
            :,
            0,
            :,
        ]

        target = final_target(
            observations,
            lengths,
        )

        squared_error_sum += float(
            (
                prediction
                - target
            )
            .pow(
                2
            )
            .sum()
            .item()
        )

        value_count += int(
            target.numel()
        )

    return (
        squared_error_sum
        / value_count
    )


@torch.no_grad()
def evaluate_affine_or_baseline(
    model,
    loader,
    device,
):
    model.eval()

    squared_error_sum = 0.0
    value_count = 0

    for batch in loader:
        observations = batch[
            "observations"
        ].to(
            device
        )

        operation_ids = batch[
            "operation_ids"
        ].to(
            device
        )

        step_mask = batch[
            "step_mask"
        ].to(
            device
        )

        lengths = batch[
            "lengths"
        ].to(
            device
        )

        prediction = model.rollout(
            initial_observation=(
                observations[
                    :,
                    0,
                    :,
                ]
            ),
            operation_ids=operation_ids,
            step_mask=step_mask,
        )

        target = final_target(
            observations,
            lengths,
        )

        squared_error_sum += float(
            (
                prediction
                - target
            )
            .pow(
                2
            )
            .sum()
            .item()
        )

        value_count += int(
            target.numel()
        )

    return (
        squared_error_sum
        / value_count
    )


@torch.no_grad()
def evaluate_ocm(
    model,
    loader,
    device,
    temperature,
):
    model.eval()

    squared_error_sum = 0.0
    value_count = 0

    for batch in loader:
        observations = batch[
            "observations"
        ].to(
            device
        )

        operation_ids = batch[
            "operation_ids"
        ].to(
            device
        )

        step_mask = batch[
            "step_mask"
        ].to(
            device
        )

        lengths = batch[
            "lengths"
        ].to(
            device
        )

        current = model.encode(
            observations[
                :,
                0,
                :,
            ],
            temperature=temperature,
        )

        for step_index in range(
            operation_ids.shape[1]
        ):
            proposed = model.apply_operation(
                latent_distribution=current,
                operation_ids=operation_ids[
                    :,
                    step_index,
                ],
                temperature=temperature,
            )

            active = step_mask[
                :,
                step_index,
            ].unsqueeze(
                1
            )

            current = torch.where(
                active,
                proposed,
                current,
            )

        prediction = model.decode(
            current
        )

        target = final_target(
            observations,
            lengths,
        )

        squared_error_sum += float(
            (
                prediction
                - target
            )
            .pow(
                2
            )
            .sum()
            .item()
        )

        value_count += int(
            target.numel()
        )

    return (
        squared_error_sum
        / value_count
    )


def train_baseline(
    model_id,
    model,
    train_loader,
    validation_loader,
    device,
    maximum_epochs,
):
    set_random_seed(
        DEVELOPMENT_SEED
    )

    model = model.to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    initial_validation_mse = (
        evaluate_affine_or_baseline(
            model=model,
            loader=validation_loader,
            device=device,
        )
    )

    best_validation_mse = (
        initial_validation_mse
    )

    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{model_id}_smoke_best.pt"
    )

    history = []

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        model.train()

        train_sse = 0.0
        train_value_count = 0

        for batch in train_loader:
            observations = batch[
                "observations"
            ].to(
                device
            )

            operation_ids = batch[
                "operation_ids"
            ].to(
                device
            )

            step_mask = batch[
                "step_mask"
            ].to(
                device
            )

            lengths = batch[
                "lengths"
            ].to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model.rollout(
                initial_observation=(
                    observations[
                        :,
                        0,
                        :,
                    ]
                ),
                operation_ids=operation_ids,
                step_mask=step_mask,
            )

            target = final_target(
                observations,
                lengths,
            )

            loss = F.mse_loss(
                prediction,
                target,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            train_sse += float(
                (
                    prediction.detach()
                    - target
                )
                .pow(
                    2
                )
                .sum()
                .item()
            )

            train_value_count += int(
                target.numel()
            )

        train_mse = (
            train_sse
            / train_value_count
        )

        validation_mse = (
            evaluate_affine_or_baseline(
                model=model,
                loader=validation_loader,
                device=device,
            )
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_rollout_mse":
                    train_mse,

                "validation_rollout_mse":
                    validation_mse,
            }
        )

        print(
            f"{model_id} | epoch={epoch:02d} | "
            f"train={train_mse:.6f} | "
            f"val={validation_mse:.6f}"
        )

        improvement = (
            best_validation_mse
            - validation_mse
        )

        if (
            improvement
            > EARLY_STOPPING_MIN_DELTA
        ):
            best_validation_mse = (
                validation_mse
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "phase":
                        "3C smoke only",

                    "eligible_for_final_use":
                        False,

                    "model_id":
                        model_id,

                    "epoch":
                        epoch,

                    "validation_rollout_mse":
                        validation_mse,

                    "model_state_dict":
                        model.state_dict(),
                },
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    return {
        "model":
            model,

        "history":
            history,

        "initial_validation_mse":
            float(
                initial_validation_mse
            ),

        "best_validation_mse":
            float(
                best_validation_mse
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "epochs_executed":
            len(
                history
            ),

        "improved_over_initial":
            bool(
                best_validation_mse
                <
                initial_validation_mse
                - EARLY_STOPPING_MIN_DELTA
            ),
    }


def train_ocm(
    model,
    train_loader,
    validation_loader,
    operation_count,
    device,
    maximum_epochs,
):
    set_random_seed(
        DEVELOPMENT_SEED
    )

    model = model.to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    initial_validation_mse = (
        evaluate_ocm(
            model=model,
            loader=validation_loader,
            device=device,
            temperature=OCM_TEMPERATURE,
        )
    )

    best_validation_mse = (
        initial_validation_mse
    )

    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "OCM_smoke_best.pt"
    )

    history = []

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        model.train()

        metric_sums = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "step_loss": 0.0,
            "final_rollout_loss": 0.0,
            "transition_loss": 0.0,
            "determinism_loss": 0.0,
        }

        batch_count = 0

        for batch in train_loader:
            batch = {
                key:
                    (
                        value.to(
                            device
                        )
                        if torch.is_tensor(
                            value
                        )
                        else value
                    )
                for key, value in batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            losses = ocm_forward_losses(
                model=model,
                batch=batch,
                operation_count=operation_count,
                temperature=OCM_TEMPERATURE,
            )

            losses[
                "total_loss"
            ].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            for metric_name in metric_sums:
                metric_sums[
                    metric_name
                ] += float(
                    losses[
                        metric_name
                    ]
                    .detach()
                    .item()
                )

            batch_count += 1

        validation_mse = evaluate_ocm(
            model=model,
            loader=validation_loader,
            device=device,
            temperature=OCM_TEMPERATURE,
        )

        row = {
            "epoch":
                epoch,

            "validation_rollout_mse":
                validation_mse,
        }

        for metric_name, value in (
            metric_sums.items()
        ):
            row[
                "train_"
                + metric_name
            ] = (
                value
                / batch_count
            )

        history.append(
            row
        )

        print(
            f"OCM | epoch={epoch:02d} | "
            f"train_total="
            f"{row['train_total_loss']:.6f} | "
            f"val={validation_mse:.6f}"
        )

        improvement = (
            best_validation_mse
            - validation_mse
        )

        if (
            improvement
            > EARLY_STOPPING_MIN_DELTA
        ):
            best_validation_mse = (
                validation_mse
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "phase":
                        "3C smoke only",

                    "eligible_for_final_use":
                        False,

                    "model_id":
                        "OCM",

                    "epoch":
                        epoch,

                    "temperature":
                        OCM_TEMPERATURE,

                    "validation_rollout_mse":
                        validation_mse,

                    "model_state_dict":
                        model.state_dict(),
                },
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    return {
        "model":
            model,

        "history":
            history,

        "initial_validation_mse":
            float(
                initial_validation_mse
            ),

        "best_validation_mse":
            float(
                best_validation_mse
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "epochs_executed":
            len(
                history
            ),

        "improved_over_initial":
            bool(
                best_validation_mse
                <
                initial_validation_mse
                - EARLY_STOPPING_MIN_DELTA
            ),
    }


def main():
    arguments = parse_arguments()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "phase3c_summary.json"
    )

    if (
        summary_path.exists()
        and not arguments.force
    ):
        existing = load_json(
            summary_path
        )

        if (
            existing.get(
                "phase3c_status"
            )
            == "passed"
        ):
            print(
                "Phase 3C already passed. "
                "Use --force to rerun."
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            return

    sources = validate_source_phases()

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

    operation_count = len(
        operation_vocabulary
    )

    train_cell = load_cell(
        cell_id="train_joint",
        operation_vocabulary=(
            operation_vocabulary
        ),
    )

    validation_cell = load_cell(
        cell_id="val_joint",
        operation_vocabulary=(
            operation_vocabulary
        ),
    )

    train_records = evenly_spaced_subset(
        train_cell[
            "records"
        ],
        SMOKE_TRAIN_TRAJECTORY_LIMIT,
    )

    validation_records = (
        evenly_spaced_subset(
            validation_cell[
                "records"
            ],
            SMOKE_VALIDATION_TRAJECTORY_LIMIT,
        )
    )

    maximum_sequence_length = max(
        record.point_count - 1
        for record in (
            train_records
            + validation_records
        )
    )

    train_loader = make_loader(
        vectors=train_cell[
            "vectors"
        ],
        records=train_records,
        operations_by_sequence=(
            train_cell[
                "operations_by_sequence"
            ]
        ),
        seed=DEVELOPMENT_SEED,
        shuffle=True,
    )

    validation_loader = make_loader(
        vectors=validation_cell[
            "vectors"
        ],
        records=validation_records,
        operations_by_sequence=(
            validation_cell[
                "operations_by_sequence"
            ]
        ),
        seed=DEVELOPMENT_SEED,
        shuffle=False,
    )

    persistence_validation_mse = (
        evaluate_persistence(
            loader=validation_loader,
            device=device,
        )
    )

    results = [
        {
            "model_id":
                "B0",

            "fit_type":
                "none",

            "parameter_count":
                0,

            "initial_validation_mse":
                persistence_validation_mse,

            "best_validation_mse":
                persistence_validation_mse,

            "best_epoch":
                None,

            "epochs_executed":
                0,

            "improved_over_initial":
                False,

            "beat_persistence":
                False,

            "all_state_values_finite":
                True,

            "numerically_valid":
                bool(
                    np.isfinite(
                        persistence_validation_mse
                    )
                ),
        }
    ]

    # B1: deterministic affine smoke fit.
    affine_model = fit_affine_model(
        vectors=train_cell[
            "vectors"
        ],
        records=train_records,
        operations_by_sequence=(
            train_cell[
                "operations_by_sequence"
            ]
        ),
        operation_count=operation_count,
        ridge=1e-4,
    ).to(
        device
    )

    affine_validation_mse = (
        evaluate_affine_or_baseline(
            model=affine_model,
            loader=validation_loader,
            device=device,
        )
    )

    torch.save(
        {
            "phase":
                "3C smoke only",

            "eligible_for_final_use":
                False,

            "model_id":
                "B1",

            "validation_rollout_mse":
                affine_validation_mse,

            "model_state_dict":
                affine_model.state_dict(),
        },
        CHECKPOINT_DIR
        / "B1_smoke_best.pt",
    )

    results.append(
        {
            "model_id":
                "B1",

            "fit_type":
                "deterministic",

            "parameter_count":
                int(
                    affine_model.matrices.numel()
                    + affine_model.biases.numel()
                ),

            "initial_validation_mse":
                None,

            "best_validation_mse":
                float(
                    affine_validation_mse
                ),

            "best_epoch":
                None,

            "epochs_executed":
                0,

            "improved_over_initial":
                None,

            "beat_persistence":
                bool(
                    affine_validation_mse
                    <
                    persistence_validation_mse
                ),

            "all_state_values_finite":
                state_is_finite(
                    affine_model
                ),

            "numerically_valid":
                bool(
                    state_is_finite(
                        affine_model
                    )
                    and np.isfinite(
                        affine_validation_mse
                    )
                ),
        }
    )

    # B2-B5 neural smoke fits.
    for model_id in (
        "B2",
        "B3",
        "B4",
        "B5",
    ):
        set_random_seed(
            DEVELOPMENT_SEED
        )

        model = build_baseline_model(
            model_id=model_id,
            operation_count=operation_count,
            maximum_sequence_length=(
                maximum_sequence_length
            ),
        )

        output = train_baseline(
            model_id=model_id,
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            maximum_epochs=arguments.epochs,
        )

        write_csv(
            HISTORY_DIR
            / f"{model_id}_history.csv",
            output[
                "history"
            ],
        )

        finite = state_is_finite(
            output[
                "model"
            ]
        )

        result = {
            "model_id":
                model_id,

            "fit_type":
                "neural",

            "parameter_count":
                parameter_count(
                    output[
                        "model"
                    ]
                ),

            "initial_validation_mse":
                output[
                    "initial_validation_mse"
                ],

            "best_validation_mse":
                output[
                    "best_validation_mse"
                ],

            "best_epoch":
                output[
                    "best_epoch"
                ],

            "epochs_executed":
                output[
                    "epochs_executed"
                ],

            "improved_over_initial":
                output[
                    "improved_over_initial"
                ],

            "beat_persistence":
                bool(
                    output[
                        "best_validation_mse"
                    ]
                    <
                    persistence_validation_mse
                ),

            "all_state_values_finite":
                finite,

            "numerically_valid":
                bool(
                    finite
                    and np.isfinite(
                        output[
                            "best_validation_mse"
                        ]
                    )
                ),
        }

        results.append(
            result
        )

    # OCM smoke fit.
    set_random_seed(
        DEVELOPMENT_SEED
    )

    ocm = OperationChannelModel(
        observation_dim=(
            OBSERVATION_DIMENSION
        ),
        latent_state_count=(
            LATENT_STATE_COUNT
        ),
        operation_count=(
            operation_count
        ),
    )

    ocm_output = train_ocm(
        model=ocm,
        train_loader=train_loader,
        validation_loader=validation_loader,
        operation_count=operation_count,
        device=device,
        maximum_epochs=arguments.epochs,
    )

    write_csv(
        HISTORY_DIR
        / "OCM_history.csv",
        ocm_output[
            "history"
        ],
    )

    ocm_finite = state_is_finite(
        ocm_output[
            "model"
        ]
    )

    row_sum_error = (
        channel_row_sum_error(
            model=ocm_output[
                "model"
            ],
            operation_count=operation_count,
            temperature=OCM_TEMPERATURE,
        )
    )

    ocm_result = {
        "model_id":
            "OCM",

        "fit_type":
            "neural_structured",

        "parameter_count":
            parameter_count(
                ocm_output[
                    "model"
                ]
            ),

        "temperature":
            OCM_TEMPERATURE,

        "initial_validation_mse":
            ocm_output[
                "initial_validation_mse"
            ],

        "best_validation_mse":
            ocm_output[
                "best_validation_mse"
            ],

        "best_epoch":
            ocm_output[
                "best_epoch"
            ],

        "epochs_executed":
            ocm_output[
                "epochs_executed"
            ],

        "improved_over_initial":
            ocm_output[
                "improved_over_initial"
            ],

        "beat_persistence":
            bool(
                ocm_output[
                    "best_validation_mse"
                ]
                <
                persistence_validation_mse
            ),

        "all_state_values_finite":
            ocm_finite,

        "maximum_channel_row_sum_error":
            row_sum_error,

        "numerically_valid":
            bool(
                ocm_finite
                and np.isfinite(
                    ocm_output[
                        "best_validation_mse"
                    ]
                )
                and (
                    row_sum_error is None
                    or row_sum_error
                    < 1e-6
                )
            ),
    }

    results.append(
        ocm_result
    )

    write_csv(
        OUTPUT_DIR
        / "smoke_results.csv",
        results,
    )

    all_models_present = (
        {
            row[
                "model_id"
            ]
            for row in results
        }
        == set(
            MODEL_IDS
        )
    )

    all_numerically_valid = bool(
        all(
            row[
                "numerically_valid"
            ]
            for row in results
        )
    )

    neural_baseline_improvement_count = sum(
        row[
            "model_id"
        ]
        in {
            "B2",
            "B3",
            "B4",
            "B5",
        }
        and row[
            "improved_over_initial"
        ]
        is True
        for row in results
    )

    ocm_improved = bool(
        ocm_result[
            "improved_over_initial"
        ]
    )

    phase_passed = bool(
        all_models_present
        and all_numerically_valid
        and neural_baseline_improvement_count
        >= 3
        and ocm_improved
    )

    summary = {
        "phase":
            "3C Tier B leakage-safe smoke tests",

        "development_seed":
            DEVELOPMENT_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE,

        "training_cell":
            "train_joint",

        "validation_cell":
            "val_joint",

        "full_training_trajectory_count":
            len(
                train_cell[
                    "records"
                ]
            ),

        "smoke_training_trajectory_count":
            len(
                train_records
            ),

        "full_validation_trajectory_count":
            len(
                validation_cell[
                    "records"
                ]
            ),

        "smoke_validation_trajectory_count":
            len(
                validation_records
            ),

        "maximum_sequence_length":
            maximum_sequence_length,

        "observation_dimension":
            OBSERVATION_DIMENSION,

        "operation_count":
            operation_count,

        "model_result_count":
            len(
                results
            ),

        "persistence_validation_mse":
            float(
                persistence_validation_mse
            ),

        "all_models_present":
            all_models_present,

        "all_models_numerically_valid":
            all_numerically_valid,

        "neural_baseline_improvement_count":
            neural_baseline_improvement_count,

        "minimum_required_neural_baseline_improvements":
            3,

        "ocm_improved_over_initial":
            ocm_improved,

        "ocm_maximum_channel_row_sum_error":
            row_sum_error,

        "test_cells_opened":
            False,

        "privileged_arrays_opened":
            False,

        "clean_targets_read":
            False,

        "clean_targets_used_for_training":
            False,

        "clean_targets_used_for_selection":
            False,

        "structural_labels_read":
            False,

        "smoke_checkpoints_eligible_for_final_use":
            False,

        "phase2_outputs_modified":
            False,

        "phase3c_status":
            (
                "passed"
                if phase_passed
                else "failed"
            ),

        "model_results": {
            row[
                "model_id"
            ]:
                row
            for row in results
        },
    }

    write_json(
        summary_path,
        summary,
    )

    print()
    print(
        "Phase 3C Tier B smoke tests completed."
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
