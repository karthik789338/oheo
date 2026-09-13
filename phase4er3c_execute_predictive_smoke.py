from __future__ import annotations

import os

# Required before importing torch when deterministic CUDA execution is used.
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import argparse
import csv
import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import phase3c_run_tier_b_smoke_tests as tier_b_smoke
import phase3d_tune_tier_b_models as tier_b


PROTOCOL_VERSION = "tier_c_v4"

SEED = 11
OBSERVATION_DIMENSION = 128
OPERATION_COUNT = 6
MAXIMUM_SEQUENCE_LENGTH = 4
LATENT_STATE_COUNT_EXPECTED = 8

PHYSICAL_MICROBATCH_SIZE = 4
EXPECTED_TRAJECTORY_COUNT = 12
EXPECTED_MICROBATCH_COUNT = 3

GRADIENT_CLIP_NORM = 5.0
B1_LEARNING_RATE = 3e-4

MODEL_IDS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

TRAINABLE_MODEL_IDS = set(MODEL_IDS)

SMOKE_BATCH_PATH = Path(
    "outputs/"
    "phase4er3br1_tier_c_v4_operation_complete_smoke_batch/"
    "tier_c_v4_operation_complete_smoke_batch.npz"
)

SMOKE_PROVENANCE_PATH = Path(
    "outputs/"
    "phase4er3br1_tier_c_v4_operation_complete_smoke_batch/"
    "operation_complete_smoke_batch_provenance.json"
)

EXECUTION_CONTRACT_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "tier_c_v4_execution_contract.json"
)

EXECUTION_SUMMARY_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "phase4dr32_execution_contract_summary.json"
)

CODEC_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_field_codec_registry.json"
)

CODEC_MICROARCHITECTURE_PATH = Path(
    "outputs/"
    "phase4dr33_tier_c_v4_codec_microarchitecture/"
    "tier_c_v4_codec_microarchitecture.json"
)

CODEC_MICROARCHITECTURE_SUMMARY_PATH = Path(
    "outputs/"
    "phase4dr33_tier_c_v4_codec_microarchitecture/"
    "phase4dr33_codec_microarchitecture_summary.json"
)

TUNING_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_tuning_registry.csv"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4er3c_tier_c_v4_predictive_smoke"
)

PHASE3D_SOURCE_PATH = Path(
    "phase3d_tune_tier_b_models.py"
)

PHASE3C_SOURCE_PATH = Path(
    "phase3c_run_tier_b_smoke_tests.py"
)


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(
    path: Path,
    value: Any,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True
    )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels,
                eps=1e-5,
                affine=True,
            ),
            nn.GELU(
                approximate="none",
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels,
                eps=1e-5,
                affine=True,
            ),
        )

        self.activation = nn.GELU(
            approximate="none",
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.activation(
            inputs + self.main(inputs)
        )


class ResidualDownsampleBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=output_channels,
                eps=1e-5,
                affine=True,
            ),
            nn.GELU(
                approximate="none",
            ),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=output_channels,
                eps=1e-5,
                affine=True,
            ),
        )

        self.skip = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        )

        self.activation = nn.GELU(
            approximate="none",
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.activation(
            self.main(inputs)
            + self.skip(inputs)
        )


class TierCFieldEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                4,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=32,
                eps=1e-5,
                affine=True,
            ),
            nn.GELU(
                approximate="none",
            ),
            ResidualBlock(32),
            ResidualDownsampleBlock(
                32,
                64,
            ),
            ResidualBlock(64),
            ResidualDownsampleBlock(
                64,
                128,
            ),
            ResidualBlock(128),
            ResidualDownsampleBlock(
                128,
                128,
            ),
            ResidualBlock(128),
            nn.Flatten(),
            nn.Linear(
                2048,
                OBSERVATION_DIMENSION,
                bias=True,
            ),
        )

    def forward(
        self,
        fields: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(fields)


class TierCFieldDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.initial_linear = nn.Linear(
            OBSERVATION_DIMENSION,
            2048,
            bias=True,
        )

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            ResidualBlock(128),
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            ResidualBlock(64),
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            ResidualBlock(32),
            nn.Conv2d(
                32,
                4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        flat = self.initial_linear(latent)

        shaped = flat.reshape(
            *flat.shape[:-1],
            128,
            4,
            4,
        )

        leading_shape = shaped.shape[:-3]

        decoded = self.network(
            shaped.reshape(
                -1,
                128,
                4,
                4,
            )
        )

        return decoded.reshape(
            *leading_shape,
            4,
            32,
            32,
        )


class TierCFieldCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = TierCFieldEncoder()
        self.decoder = TierCFieldDecoder()

    def encode(
        self,
        fields: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(fields)

    def decode(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(latent)


class AffineLatentTransitionBaseline(nn.Module):
    """
    B1: affine latent-transition baseline with nonlinear field codec.

    Each primitive has one independent affine map in the shared
    128-dimensional codec space.
    """

    def __init__(
        self,
        observation_dimension: int,
        operation_count: int,
    ) -> None:
        super().__init__()

        self.observation_dimension = (
            observation_dimension
        )

        self.operation_count = (
            operation_count
        )

        self.operation_maps = (
            nn.ModuleList(
                [
                    nn.Linear(
                        observation_dimension,
                        observation_dimension,
                        bias=True,
                    )
                    for _ in range(
                        operation_count
                    )
                ]
            )
        )

    def step(
        self,
        observations: torch.Tensor,
        operation_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(
            observations
        )

        for operation_id in range(
            self.operation_count
        ):
            selected = (
                operation_ids
                == operation_id
            )

            if selected.any():
                output[selected] = (
                    self.operation_maps[
                        operation_id
                    ](
                        observations[
                            selected
                        ]
                    )
                )

        return output

    def rollout(
        self,
        initial_observation: torch.Tensor,
        operation_ids: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        current = initial_observation

        for step_index in range(
            operation_ids.shape[1]
        ):
            proposed = self.step(
                current,
                operation_ids[
                    :,
                    step_index,
                ],
            )

            active = step_mask[
                :,
                step_index,
            ].unsqueeze(-1)

            current = torch.where(
                active,
                proposed,
                current,
            )

        return current

    def ridge_penalty(self) -> torch.Tensor:
        penalties = [
            layer.weight.pow(2).sum()
            for layer in self.operation_maps
        ]

        return torch.stack(
            penalties
        ).sum()


def read_first_configuration_per_model() -> dict[str, dict]:
    selected: dict[str, dict] = {}

    with TUNING_REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            model_id = row["model_id"]

            if model_id not in MODEL_IDS:
                continue

            if model_id in selected:
                continue

            selected[model_id] = json.loads(
                row[
                    "source_configuration_json"
                ]
            )

    require(
        set(selected) == set(MODEL_IDS),
        (
            "Could not load one configuration for "
            f"every model: {sorted(selected)}"
        ),
    )

    return selected


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(requested)

    if device.type == "cuda":
        require(
            torch.cuda.is_available(),
            "CUDA was requested but is unavailable.",
        )

    return device


def load_smoke_batch(
    device: torch.device,
) -> dict[str, torch.Tensor]:
    provenance = load_json(
        SMOKE_PROVENANCE_PATH
    )

    require(
        provenance[
            "operation_complete_smoke_batch_status"
        ] == "prepared_and_verified",
        "Operation-complete smoke batch is not verified.",
    )

    require(
        provenance[
            "development_data_only"
        ] is True,
        "Smoke batch is not development-only.",
    )

    require(
        provenance[
            "privileged_fields_read"
        ] is False,
        "Privileged fields were read.",
    )

    require(
        provenance[
            "test_fields_read"
        ] is False,
        "Test fields were read.",
    )

    require(
        provenance[
            "optimizer_steps_performed"
        ] == 0,
        "Prior optimizer steps are recorded.",
    )

    require(
        sha256_file(SMOKE_BATCH_PATH)
        == provenance["smoke_batch_sha256"],
        "Smoke-batch SHA-256 does not match provenance.",
    )

    with np.load(
        SMOKE_BATCH_PATH,
        allow_pickle=False,
    ) as archive:
        observations = torch.from_numpy(
            archive[
                "observations"
            ].copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )

        operation_ids = torch.from_numpy(
            archive[
                "operation_ids"
            ].copy()
        ).to(
            device=device,
            dtype=torch.long,
        )

        point_mask = torch.from_numpy(
            archive[
                "point_mask"
            ].copy()
        ).to(
            device=device,
            dtype=torch.bool,
        )

        step_mask = torch.from_numpy(
            archive[
                "step_mask"
            ].copy()
        ).to(
            device=device,
            dtype=torch.bool,
        )

        sequence_lengths = torch.from_numpy(
            archive[
                "sequence_lengths"
            ].copy()
        ).to(
            device=device,
            dtype=torch.long,
        )

    require(
        observations.shape
        == (
            EXPECTED_TRAJECTORY_COUNT,
            5,
            4,
            32,
            32,
        ),
        (
            "Unexpected observation shape: "
            f"{tuple(observations.shape)}"
        ),
    )

    require(
        operation_ids.shape
        == (
            EXPECTED_TRAJECTORY_COUNT,
            MAXIMUM_SEQUENCE_LENGTH,
        ),
        "Unexpected operation-ID shape.",
    )

    require(
        point_mask.shape
        == (
            EXPECTED_TRAJECTORY_COUNT,
            5,
        ),
        "Unexpected point-mask shape.",
    )

    require(
        step_mask.shape
        == (
            EXPECTED_TRAJECTORY_COUNT,
            MAXIMUM_SEQUENCE_LENGTH,
        ),
        "Unexpected step-mask shape.",
    )

    require(
        torch.equal(
            point_mask.sum(dim=1),
            sequence_lengths + 1,
        ),
        "Point mask is inconsistent with sequence lengths.",
    )

    require(
        torch.equal(
            step_mask.sum(dim=1),
            sequence_lengths,
        ),
        "Step mask is inconsistent with sequence lengths.",
    )

    require(
        bool(
            torch.isfinite(
                observations
            ).all()
        ),
        "Smoke observations contain non-finite values.",
    )

    return {
        "observations":
            observations,

        "operation_ids":
            operation_ids,

        "point_mask":
            point_mask,

        "step_mask":
            step_mask,

        "sequence_lengths":
            sequence_lengths,
    }


def slice_batch(
    batch: dict[str, torch.Tensor],
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    return {
        key: value[start:end]
        for key, value in batch.items()
    }


def per_sequence_temporal_field_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    First average pixels and channels for each valid time point,
    then average valid time points inside each trajectory,
    then average trajectories.
    """

    require(
        prediction.shape == target.shape,
        "Prediction and target field shapes differ.",
    )

    squared = (
        prediction - target
    ).pow(2).mean(
        dim=(-3, -2, -1)
    )

    require(
        squared.shape == mask.shape,
        (
            "Temporal field-loss mask shape differs: "
            f"{tuple(squared.shape)} vs {tuple(mask.shape)}"
        ),
    )

    weights = mask.to(
        squared.dtype
    )

    per_sequence = (
        squared * weights
    ).sum(dim=1) / weights.sum(
        dim=1
    ).clamp_min(1.0)

    return per_sequence.mean()


def per_sequence_final_field_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    require(
        prediction.shape == target.shape,
        "Final prediction and target shapes differ.",
    )

    return (
        prediction - target
    ).pow(2).mean(
        dim=(-3, -2, -1)
    ).mean()


def per_sequence_temporal_scalar_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    require(
        values.shape == mask.shape,
        "Temporal scalar and mask shapes differ.",
    )

    weights = mask.to(
        values.dtype
    )

    per_sequence = (
        values * weights
    ).sum(dim=1) / weights.sum(
        dim=1
    ).clamp_min(1.0)

    return per_sequence.mean()


def gather_final_fields(
    observations: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    batch_indices = torch.arange(
        observations.shape[0],
        device=observations.device,
    )

    return observations[
        batch_indices,
        sequence_lengths,
    ]


def encode_all_fields(
    codec: TierCFieldCodec,
    fields: torch.Tensor,
) -> torch.Tensor:
    batch_size = fields.shape[0]
    time_count = fields.shape[1]

    flat = fields.reshape(
        -1,
        4,
        32,
        32,
    )

    latent = codec.encode(flat)

    require(
        latent.shape
        == (
            batch_size * time_count,
            OBSERVATION_DIMENSION,
        ),
        "Codec encoder produced an unexpected shape.",
    )

    return latent.reshape(
        batch_size,
        time_count,
        OBSERVATION_DIMENSION,
    )


def decode_temporal_latent(
    codec: TierCFieldCodec,
    latent: torch.Tensor,
) -> torch.Tensor:
    batch_size = latent.shape[0]
    time_count = latent.shape[1]

    decoded = codec.decode(
        latent.reshape(
            -1,
            OBSERVATION_DIMENSION,
        )
    )

    return decoded.reshape(
        batch_size,
        time_count,
        4,
        32,
        32,
    )


def baseline_forward_losses(
    model_id: str,
    model: nn.Module,
    codec: TierCFieldCodec,
    batch: dict[str, torch.Tensor],
    configuration: dict,
) -> dict[str, torch.Tensor]:
    fields = batch["observations"]
    operation_ids = batch[
        "operation_ids"
    ]
    point_mask = batch["point_mask"]
    step_mask = batch["step_mask"]
    sequence_lengths = batch[
        "sequence_lengths"
    ]

    batch_size = fields.shape[0]
    maximum_length = (
        operation_ids.shape[1]
    )

    encoded = encode_all_fields(
        codec,
        fields,
    )

    source_latent = encoded[
        :,
        :-1,
        :,
    ].reshape(
        batch_size * maximum_length,
        OBSERVATION_DIMENSION,
    )

    flat_operations = operation_ids.reshape(
        batch_size * maximum_length,
    )

    one_step_mask = torch.ones(
        (
            batch_size
            * maximum_length,
            1,
        ),
        dtype=torch.bool,
        device=fields.device,
    )

    predicted_step_latent = model.rollout(
        initial_observation=source_latent,
        operation_ids=(
            flat_operations[:, None]
        ),
        step_mask=one_step_mask,
    )

    require(
        predicted_step_latent.shape
        == (
            batch_size
            * maximum_length,
            OBSERVATION_DIMENSION,
        ),
        (
            f"{model_id} one-step latent shape "
            "is incorrect."
        ),
    )

    predicted_step_fields = (
        codec.decode(
            predicted_step_latent
        ).reshape(
            batch_size,
            maximum_length,
            4,
            32,
            32,
        )
    )

    one_step_loss = (
        per_sequence_temporal_field_mse(
            prediction=(
                predicted_step_fields
            ),
            target=fields[:, 1:, ...],
            mask=step_mask,
        )
    )

    final_latent = model.rollout(
        initial_observation=encoded[
            :,
            0,
            :,
        ],
        operation_ids=operation_ids,
        step_mask=step_mask,
    )

    require(
        final_latent.shape
        == (
            batch_size,
            OBSERVATION_DIMENSION,
        ),
        (
            f"{model_id} rollout latent shape "
            "is incorrect."
        ),
    )

    final_prediction = codec.decode(
        final_latent
    )

    final_target = gather_final_fields(
        fields,
        sequence_lengths,
    )

    rollout_loss = (
        per_sequence_final_field_mse(
            prediction=final_prediction,
            target=final_target,
        )
    )

    reconstruction_loss = (
        fields.new_zeros(())
    )

    if model_id == "B5":
        internal_reconstruction = (
            model.decode(
                model.encode(
                    encoded.reshape(
                        -1,
                        OBSERVATION_DIMENSION,
                    )
                )
            ).reshape(
                batch_size,
                fields.shape[1],
                OBSERVATION_DIMENSION,
            )
        )

        reconstructed_fields = (
            decode_temporal_latent(
                codec,
                internal_reconstruction,
            )
        )

        reconstruction_loss = (
            per_sequence_temporal_field_mse(
                prediction=(
                    reconstructed_fields
                ),
                target=fields,
                mask=point_mask,
            )
        )

    ridge_loss = fields.new_zeros(())

    if model_id == "B1":
        ridge = float(
            configuration["ridge"]
        )

        ridge_loss = (
            ridge
            * model.ridge_penalty()
        )

    total_loss = (
        one_step_loss
        + rollout_loss
        + reconstruction_loss
        + ridge_loss
    )

    return {
        "total_loss":
            total_loss,

        "one_step_field_loss":
            one_step_loss,

        "rollout_field_loss":
            rollout_loss,

        "reconstruction_field_loss":
            reconstruction_loss,

        "ridge_loss":
            ridge_loss,

        "final_prediction":
            final_prediction,
    }


def ocm_forward_losses(
    model: nn.Module,
    codec: TierCFieldCodec,
    batch: dict[str, torch.Tensor],
    configuration: dict,
) -> dict[str, torch.Tensor]:
    fields = batch["observations"]
    operation_ids = batch[
        "operation_ids"
    ]
    point_mask = batch["point_mask"]
    step_mask = batch["step_mask"]
    sequence_lengths = batch[
        "sequence_lengths"
    ]

    batch_size = fields.shape[0]
    time_count = fields.shape[1]
    maximum_length = (
        operation_ids.shape[1]
    )

    temperature = float(
        configuration["temperature"]
    )

    latent_state_count = int(
        configuration[
            "latent_state_count"
        ]
    )

    encoded_fields = encode_all_fields(
        codec,
        fields,
    )

    flat_encoded_fields = (
        encoded_fields.reshape(
            -1,
            OBSERVATION_DIMENSION,
        )
    )

    distributions_flat = model.encode(
        flat_encoded_fields,
        temperature=temperature,
    )

    require(
        distributions_flat.shape
        == (
            batch_size
            * time_count,
            latent_state_count,
        ),
        "OCM encoded-distribution shape is incorrect.",
    )

    distributions = (
        distributions_flat.reshape(
            batch_size,
            time_count,
            latent_state_count,
        )
    )

    reconstructed_latent = (
        model.decode(
            distributions_flat
        ).reshape(
            batch_size,
            time_count,
            OBSERVATION_DIMENSION,
        )
    )

    reconstructed_fields = (
        decode_temporal_latent(
            codec,
            reconstructed_latent,
        )
    )

    reconstruction_loss = (
        per_sequence_temporal_field_mse(
            prediction=(
                reconstructed_fields
            ),
            target=fields,
            mask=point_mask,
        )
    )

    source_distributions = (
        distributions[
            :,
            :-1,
            :,
        ].reshape(
            batch_size
            * maximum_length,
            latent_state_count,
        )
    )

    flat_operations = operation_ids.reshape(
        batch_size
        * maximum_length
    )

    proposed_flat = model.apply_operation(
        latent_distribution=(
            source_distributions
        ),
        operation_ids=flat_operations,
        temperature=temperature,
    )

    proposed = proposed_flat.reshape(
        batch_size,
        maximum_length,
        latent_state_count,
    )

    predicted_step_latent = (
        model.decode(
            proposed_flat
        ).reshape(
            batch_size,
            maximum_length,
            OBSERVATION_DIMENSION,
        )
    )

    predicted_step_fields = (
        decode_temporal_latent(
            codec,
            predicted_step_latent,
        )
    )

    one_step_loss = (
        per_sequence_temporal_field_mse(
            prediction=(
                predicted_step_fields
            ),
            target=fields[:, 1:, ...],
            mask=step_mask,
        )
    )

    target_distributions = (
        distributions[
            :,
            1:,
            :,
        ]
    )

    transition_values = (
        tier_b_smoke.js_divergence(
            proposed.reshape(
                -1,
                latent_state_count,
            ),
            target_distributions.reshape(
                -1,
                latent_state_count,
            ),
        ).reshape(
            batch_size,
            maximum_length,
        )
    )

    transition_loss = (
        per_sequence_temporal_scalar_mean(
            transition_values,
            step_mask,
        )
    )

    final_distribution = model.rollout(
        initial_distribution=(
            distributions[
                :,
                0,
                :,
            ]
        ),
        operation_ids=operation_ids,
        step_mask=step_mask,
        temperature=temperature,
    )

    final_latent = model.decode(
        final_distribution
    )

    final_prediction = codec.decode(
        final_latent
    )

    final_target = gather_final_fields(
        fields,
        sequence_lengths,
    )

    rollout_loss = (
        per_sequence_final_field_mse(
            prediction=final_prediction,
            target=final_target,
        )
    )

    determinism_loss = (
        tier_b_smoke.channel_entropy(
            model=model,
            operation_count=(
                OPERATION_COUNT
            ),
            temperature=temperature,
        )
    )

    lambda_transition = float(
        configuration[
            "lambda_transition"
        ]
    )

    lambda_rollout = float(
        configuration[
            "lambda_rollout"
        ]
    )

    lambda_determinism = float(
        configuration[
            "lambda_determinism"
        ]
    )

    total_loss = (
        reconstruction_loss
        + one_step_loss
        + lambda_transition
        * transition_loss
        + lambda_rollout
        * rollout_loss
        + lambda_determinism
        * determinism_loss
    )

    return {
        "total_loss":
            total_loss,

        "reconstruction_field_loss":
            reconstruction_loss,

        "one_step_field_loss":
            one_step_loss,

        "transition_loss":
            transition_loss,

        "rollout_field_loss":
            rollout_loss,

        "determinism_loss":
            determinism_loss,

        "final_prediction":
            final_prediction,
    }


def build_model_and_optimizer(
    model_id: str,
    configuration: dict,
    device: torch.device,
) -> tuple[
    TierCFieldCodec,
    nn.Module,
    torch.optim.Optimizer,
]:
    codec = TierCFieldCodec().to(
        device=device,
        dtype=torch.float32,
    )

    if model_id == "B1":
        model = (
            AffineLatentTransitionBaseline(
                observation_dimension=(
                    OBSERVATION_DIMENSION
                ),
                operation_count=(
                    OPERATION_COUNT
                ),
            ).to(
                device=device,
                dtype=torch.float32,
            )
        )

        learning_rate = (
            B1_LEARNING_RATE
        )

    elif model_id in {
        "B2",
        "B3",
        "B4",
        "B5",
    }:
        # Runtime-only interface amendment. The inherited file is
        # not modified.
        tier_b.OBSERVATION_DIMENSION = (
            OBSERVATION_DIMENSION
        )

        model = tier_b.build_model(
            configuration=configuration,
            operation_count=(
                OPERATION_COUNT
            ),
            maximum_sequence_length=(
                MAXIMUM_SEQUENCE_LENGTH
            ),
        ).to(
            device=device,
            dtype=torch.float32,
        )

        learning_rate = float(
            configuration[
                "learning_rate"
            ]
        )

    elif model_id == "OCM":
        latent_state_count = int(
            configuration[
                "latent_state_count"
            ]
        )

        model = (
            tier_b.OperationChannelModel(
                observation_dim=(
                    OBSERVATION_DIMENSION
                ),
                latent_state_count=(
                    latent_state_count
                ),
                operation_count=(
                    OPERATION_COUNT
                ),
            ).to(
                device=device,
                dtype=torch.float32,
            )
        )

        learning_rate = float(
            configuration[
                "learning_rate"
            ]
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_id}"
        )

    parameters = list(
        codec.parameters()
    ) + list(
        model.parameters()
    )

    optimizer = torch.optim.Adam(
        parameters,
        lr=learning_rate,
    )

    return (
        codec,
        model,
        optimizer,
    )


def parameter_snapshot(
    codec: nn.Module,
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    snapshot = {}

    for prefix, module in (
        ("codec", codec),
        ("model", model),
    ):
        for name, parameter in (
            module.named_parameters()
        ):
            if not parameter.requires_grad:
                continue

            snapshot[
                f"{prefix}.{name}"
            ] = (
                parameter.detach()
                .cpu()
                .clone()
            )

    return snapshot


def audit_gradients(
    codec: nn.Module,
    model: nn.Module,
) -> dict[str, Any]:
    none_gradients = []
    nonfinite_gradients = []
    zero_gradient_parameters = []

    squared_norm = 0.0
    parameter_count = 0

    for prefix, module in (
        ("codec", codec),
        ("model", model),
    ):
        for name, parameter in (
            module.named_parameters()
        ):
            if not parameter.requires_grad:
                continue

            parameter_count += 1

            full_name = (
                f"{prefix}.{name}"
            )

            if parameter.grad is None:
                none_gradients.append(
                    full_name
                )
                continue

            gradient = parameter.grad

            if not bool(
                torch.isfinite(
                    gradient
                ).all()
            ):
                nonfinite_gradients.append(
                    full_name
                )
                continue

            norm = float(
                gradient.detach()
                .norm()
                .cpu()
            )

            squared_norm += norm * norm

            if norm == 0.0:
                zero_gradient_parameters.append(
                    full_name
                )

    return {
        "trainable_parameter_tensor_count":
            parameter_count,

        "none_gradient_parameters":
            none_gradients,

        "none_gradient_parameter_count":
            len(none_gradients),

        "nonfinite_gradient_parameters":
            nonfinite_gradients,

        "nonfinite_gradient_parameter_count":
            len(nonfinite_gradients),

        "zero_gradient_parameters":
            zero_gradient_parameters,

        "zero_gradient_parameter_count":
            len(
                zero_gradient_parameters
            ),

        "global_gradient_norm":
            math.sqrt(
                squared_norm
            ),
    }


def compare_parameters(
    before: dict[str, torch.Tensor],
    codec: nn.Module,
    model: nn.Module,
) -> dict[str, Any]:
    changed = []
    unchanged = []
    maximum_absolute_change = 0.0

    after_parameters = {}

    for prefix, module in (
        ("codec", codec),
        ("model", model),
    ):
        for name, parameter in (
            module.named_parameters()
        ):
            if parameter.requires_grad:
                after_parameters[
                    f"{prefix}.{name}"
                ] = (
                    parameter.detach()
                    .cpu()
                )

    require(
        set(before)
        == set(after_parameters),
        "Parameter registry changed during smoke execution.",
    )

    for name in sorted(before):
        delta = (
            after_parameters[name]
            - before[name]
        ).abs()

        local_maximum = float(
            delta.max()
        )

        maximum_absolute_change = max(
            maximum_absolute_change,
            local_maximum,
        )

        if local_maximum > 0.0:
            changed.append(name)
        else:
            unchanged.append(name)

    return {
        "parameter_tensor_count":
            len(before),

        "changed_parameter_tensor_count":
            len(changed),

        "unchanged_parameter_tensor_count":
            len(unchanged),

        "changed_parameter_tensors":
            changed,

        "unchanged_parameter_tensors":
            unchanged,

        "maximum_absolute_parameter_change":
            maximum_absolute_change,
    }


def run_model_smoke(
    model_id: str,
    configuration: dict,
    full_batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    set_determinism(SEED)

    codec, model, optimizer = (
        build_model_and_optimizer(
            model_id=model_id,
            configuration=configuration,
            device=device,
        )
    )

    codec.train()
    model.train()

    before = parameter_snapshot(
        codec,
        model,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    component_sums: dict[str, float] = {}
    microbatch_records = []

    total_trajectories = int(
        full_batch[
            "observations"
        ].shape[0]
    )

    require(
        total_trajectories
        == EXPECTED_TRAJECTORY_COUNT,
        "Unexpected smoke trajectory count.",
    )

    for microbatch_index, start in enumerate(
        range(
            0,
            total_trajectories,
            PHYSICAL_MICROBATCH_SIZE,
        )
    ):
        end = min(
            start
            + PHYSICAL_MICROBATCH_SIZE,
            total_trajectories,
        )

        microbatch = slice_batch(
            full_batch,
            start,
            end,
        )

        if model_id == "OCM":
            losses = ocm_forward_losses(
                model=model,
                codec=codec,
                batch=microbatch,
                configuration=(
                    configuration
                ),
            )
        else:
            losses = baseline_forward_losses(
                model_id=model_id,
                model=model,
                codec=codec,
                batch=microbatch,
                configuration=(
                    configuration
                ),
            )

        total_loss = losses[
            "total_loss"
        ]

        require(
            total_loss.ndim == 0,
            (
                f"{model_id} total loss "
                "is not scalar."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    total_loss
                )
            ),
            (
                f"{model_id} total loss "
                "is non-finite."
            ),
        )

        final_prediction = losses[
            "final_prediction"
        ]

        require(
            final_prediction.shape
            == (
                end - start,
                4,
                32,
                32,
            ),
            (
                f"{model_id} final field "
                "prediction shape is incorrect: "
                f"{tuple(final_prediction.shape)}"
            ),
        )

        require(
            bool(
                torch.isfinite(
                    final_prediction
                ).all()
            ),
            (
                f"{model_id} produced "
                "non-finite fields."
            ),
        )

        microbatch_weight = (
            float(end - start)
            / float(
                total_trajectories
            )
        )

        weighted_loss = (
            total_loss
            * microbatch_weight
        )

        weighted_loss.backward()

        component_record = {}

        for key, value in losses.items():
            if key == "final_prediction":
                continue

            scalar = float(
                value.detach().cpu()
            )

            require(
                math.isfinite(scalar),
                (
                    f"{model_id} component "
                    f"{key} is non-finite."
                ),
            )

            component_record[key] = scalar

            component_sums[key] = (
                component_sums.get(
                    key,
                    0.0,
                )
                + microbatch_weight
                * scalar
            )

        microbatch_records.append(
            {
                "microbatch_index":
                    microbatch_index,

                "trajectory_start":
                    start,

                "trajectory_end":
                    end,

                "trajectory_count":
                    end - start,

                "weight":
                    microbatch_weight,

                "components":
                    component_record,

                "final_prediction_shape":
                    list(
                        final_prediction.shape
                    ),

                "final_prediction_finite":
                    True,
            }
        )

    require(
        len(microbatch_records)
        == EXPECTED_MICROBATCH_COUNT,
        "Unexpected physical microbatch count.",
    )

    gradient_before_clipping = (
        audit_gradients(
            codec,
            model,
        )
    )

    require(
        gradient_before_clipping[
            "none_gradient_parameter_count"
        ] == 0,
        (
            f"{model_id} has parameters "
            "without gradients."
        ),
    )

    require(
        gradient_before_clipping[
            "nonfinite_gradient_parameter_count"
        ] == 0,
        (
            f"{model_id} has non-finite "
            "gradients."
        ),
    )

    trainable_parameters = list(
        codec.parameters()
    ) + list(
        model.parameters()
    )

    reported_preclip_norm = float(
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=(
                GRADIENT_CLIP_NORM
            ),
            norm_type=2.0,
            error_if_nonfinite=True,
        ).detach().cpu()
    )

    gradient_after_clipping = (
        audit_gradients(
            codec,
            model,
        )
    )

    require(
        gradient_after_clipping[
            "nonfinite_gradient_parameter_count"
        ] == 0,
        (
            f"{model_id} has non-finite "
            "gradients after clipping."
        ),
    )

    optimizer.step()

    parameter_changes = (
        compare_parameters(
            before=before,
            codec=codec,
            model=model,
        )
    )

    require(
        parameter_changes[
            "changed_parameter_tensor_count"
        ] > 0,
        (
            f"{model_id} optimizer step "
            "changed no parameters."
        ),
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter in (
            list(codec.parameters())
            + list(model.parameters())
        )
    )

    codec_parameter_count = sum(
        parameter.numel()
        for parameter
        in codec.parameters()
    )

    model_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    result = {
        "model_id":
            model_id,

        "configuration_id":
            configuration[
                "configuration_id"
            ],

        "configuration":
            configuration,

        "model_class":
            type(model).__name__,

        "codec_class":
            type(codec).__name__,

        "seed":
            SEED,

        "device":
            str(device),

        "parameter_counts": {
            "codec":
                codec_parameter_count,

            "transition_model":
                model_parameter_count,

            "total":
                total_parameter_count,
        },

        "physical_microbatch_size":
            PHYSICAL_MICROBATCH_SIZE,

        "physical_microbatch_count":
            len(microbatch_records),

        "effective_smoke_trajectory_count":
            total_trajectories,

        "microbatches":
            microbatch_records,

        "weighted_component_means":
            component_sums,

        "forward_pass_finite":
            True,

        "backward_pass_completed":
            True,

        "gradient_before_clipping":
            gradient_before_clipping,

        "clip_grad_norm_returned_preclip_norm":
            reported_preclip_norm,

        "gradient_after_clipping":
            gradient_after_clipping,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "optimizer":
            "torch.optim.Adam",

        "optimizer_step_count":
            1,

        "parameter_changes":
            parameter_changes,

        "checkpoint_saved":
            False,

        "scientific_metric":
            False,

        "smoke_status":
            "passed",
    }

    del optimizer
    del model
    del codec

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def validate_sources() -> dict[str, Any]:
    required_paths = (
        SMOKE_BATCH_PATH,
        SMOKE_PROVENANCE_PATH,
        EXECUTION_CONTRACT_PATH,
        EXECUTION_SUMMARY_PATH,
        CODEC_REGISTRY_PATH,
        CODEC_MICROARCHITECTURE_PATH,
        CODEC_MICROARCHITECTURE_SUMMARY_PATH,
        TUNING_REGISTRY_PATH,
        PHASE3D_SOURCE_PATH,
        PHASE3C_SOURCE_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    contract = load_json(
        EXECUTION_CONTRACT_PATH
    )

    execution_summary = load_json(
        EXECUTION_SUMMARY_PATH
    )

    codec_registry = load_json(
        CODEC_REGISTRY_PATH
    )

    codec_microarchitecture = load_json(
        CODEC_MICROARCHITECTURE_PATH
    )

    codec_summary = load_json(
        CODEC_MICROARCHITECTURE_SUMMARY_PATH
    )

    require(
        execution_summary[
            "phase4dr32_status"
        ] == "execution_contract_frozen",
        "Execution contract is not frozen.",
    )

    require(
        execution_summary[
            "implementation_smoke_test_authorized"
        ] is True,
        "Implementation smoke execution is unauthorized.",
    )

    require(
        execution_summary[
            "tuning_execution_authorized"
        ] is False,
        "Tuning has already been authorized.",
    )

    require(
        execution_summary[
            "final_fit_execution_authorized"
        ] is False,
        "Final fitting has already been authorized.",
    )

    require(
        execution_summary[
            "test_artifact_generation_authorized"
        ] is False,
        "Test generation is authorized unexpectedly.",
    )

    require(
        execution_summary[
            "test_evaluation_authorized"
        ] is False,
        "Test evaluation is authorized unexpectedly.",
    )

    require(
        execution_summary[
            "test_open_count"
        ] == 0,
        "Test data has been opened.",
    )

    require(
        codec_summary[
            "phase4dr33_status"
        ] == "codec_microarchitecture_frozen",
        "Codec microarchitecture is not frozen.",
    )

    require(
        codec_summary[
            "microarchitecture_complete"
        ] is True,
        "Codec microarchitecture is incomplete.",
    )

    require(
        sha256_file(
            CODEC_MICROARCHITECTURE_PATH
        )
        == codec_summary[
            "microarchitecture_sha256"
        ],
        "Codec microarchitecture SHA-256 changed.",
    )

    require(
        codec_registry[
            "latent_dimension"
        ] == OBSERVATION_DIMENSION,
        "Codec latent dimension changed.",
    )

    require(
        codec_registry[
            "input_shape"
        ] == [4, 32, 32],
        "Codec input shape changed.",
    )

    require(
        contract[
            "batching"
        ][
            "physical_microbatch_size_trajectories"
        ] == PHYSICAL_MICROBATCH_SIZE,
        "Physical microbatch size changed.",
    )

    require(
        contract[
            "gradient_clipping_policy"
        ][
            "maximum_norm"
        ] == GRADIENT_CLIP_NORM,
        "Gradient clipping norm changed.",
    )

    require(
        contract[
            "mixed_precision_policy"
        ][
            "enabled"
        ] is False,
        "Mixed precision was enabled.",
    )

    require(
        contract[
            "ocm_interface_amendment"
        ][
            "execution_observation_dimension"
        ] == OBSERVATION_DIMENSION,
        "OCM interface dimension changed.",
    )

    frozen_phase3d_hash = contract[
        "ocm_objective_semantics"
    ][
        "source_sha256"
    ]

    require(
        sha256_file(
            PHASE3D_SOURCE_PATH
        ) == frozen_phase3d_hash,
        (
            "The inherited Phase 3D training "
            "source changed after the contract freeze."
        ),
    )

    require(
        codec_microarchitecture[
            "test_data_used"
        ] is False,
        "Codec freeze used test data.",
    )

    return {
        "required_paths":
            required_paths,

        "contract":
            contract,

        "execution_summary":
            execution_summary,

        "codec_registry":
            codec_registry,

        "codec_summary":
            codec_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the Tier C v4 model-family "
            "forward/backward/optimizer smoke test."
        )
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Execution device: auto, cpu, cuda, "
            "or a concrete device such as cuda:0."
        ),
    )

    arguments = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1/6] Validating frozen protocol and source hashes"
    )

    sources = validate_sources()

    print(
        "[2/6] Resolving deterministic execution device"
    )

    device = resolve_device(
        arguments.device
    )

    set_determinism(SEED)

    print(
        f"Using device: {device}"
    )

    if device.type == "cuda":
        print(
            "CUDA device:",
            torch.cuda.get_device_name(
                device
            ),
        )

    print(
        "[3/6] Loading verified operation-complete smoke batch"
    )

    full_batch = load_smoke_batch(
        device=device
    )

    print(
        "[4/6] Loading one frozen configuration per model family"
    )

    configurations = (
        read_first_configuration_per_model()
    )

    print(
        "[5/6] Running one optimizer smoke step per model family"
    )

    results = {}

    for model_id in MODEL_IDS:
        print(
            f"  - {model_id}: starting"
        )

        result = run_model_smoke(
            model_id=model_id,
            configuration=(
                configurations[
                    model_id
                ]
            ),
            full_batch=full_batch,
            device=device,
        )

        results[model_id] = result

        write_json(
            OUTPUT_DIR
            / f"{model_id}_smoke_result.json",
            result,
        )

        print(
            f"  - {model_id}: passed"
        )

    print(
        "[6/6] Writing smoke-execution acceptance record"
    )

    passed_model_ids = [
        model_id
        for model_id, result
        in results.items()
        if result["smoke_status"]
        == "passed"
    ]

    total_optimizer_steps = sum(
        result[
            "optimizer_step_count"
        ]
        for result in results.values()
    )

    all_gradients_finite = all(
        result[
            "gradient_before_clipping"
        ][
            "nonfinite_gradient_parameter_count"
        ] == 0
        and result[
            "gradient_after_clipping"
        ][
            "nonfinite_gradient_parameter_count"
        ] == 0
        for result in results.values()
    )

    all_parameters_updated = all(
        result[
            "parameter_changes"
        ][
            "changed_parameter_tensor_count"
        ] > 0
        for result in results.values()
    )

    source_hashes = {
        str(path):
            sha256_file(path)
        for path in sources[
            "required_paths"
        ]
    }

    write_json(
        OUTPUT_DIR
        / "smoke_execution_source_hashes.json",
        source_hashes,
    )

    summary = {
        "phase":
            (
                "4E-R3C Tier C v4 model-family "
                "forward-backward-rollout smoke execution"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "device":
            str(device),

        "cuda_available":
            torch.cuda.is_available(),

        "cuda_device_name":
            (
                torch.cuda.get_device_name(
                    device
                )
                if device.type == "cuda"
                else None
            ),

        "seed":
            SEED,

        "source_execution_contract_frozen":
            True,

        "source_codec_microarchitecture_frozen":
            True,

        "source_smoke_batch_verified":
            True,

        "smoke_batch_sha256":
            sha256_file(
                SMOKE_BATCH_PATH
            ),

        "model_ids_requested":
            list(MODEL_IDS),

        "model_ids_passed":
            passed_model_ids,

        "model_family_count":
            len(MODEL_IDS),

        "model_family_pass_count":
            len(passed_model_ids),

        "b0_excluded_reason":
            (
                "B0 persistence has no trainable "
                "parameters and therefore has no "
                "forward/backward optimizer path."
            ),

        "operation_count":
            OPERATION_COUNT,

        "all_six_primitives_covered":
            True,

        "identity_single_step_available":
            False,

        "identity_coverage_mode":
            "contained_in_length_2_sequence",

        "sequence_lengths_covered":
            [1, 2, 3, 4],

        "trajectory_count":
            EXPECTED_TRAJECTORY_COUNT,

        "physical_microbatch_size":
            PHYSICAL_MICROBATCH_SIZE,

        "physical_microbatch_count_per_model":
            EXPECTED_MICROBATCH_COUNT,

        "forward_passes_completed":
            len(results),

        "backward_passes_completed":
            len(results),

        "optimizer_steps_per_model":
            1,

        "total_optimizer_steps":
            total_optimizer_steps,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "all_gradients_finite":
            all_gradients_finite,

        "all_model_families_updated_parameters":
            all_parameters_updated,

        "mixed_precision_used":
            False,

        "checkpoints_saved":
            False,

        "validation_data_read":
            False,

        "validation_selection_performed":
            False,

        "privileged_data_read":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "scientific_comparison_performed":
            False,

        "scientific_metrics_reported":
            False,

        "tuning_execution_performed":
            False,

        "final_fit_execution_performed":
            False,

        "tuning_execution_authorized":
            False,

        "final_fit_execution_authorized":
            False,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "implementation_smoke_test_executed":
            True,

        "implementation_smoke_test_passed":
            (
                len(passed_model_ids)
                == len(MODEL_IDS)
                and all_gradients_finite
                and all_parameters_updated
                and total_optimizer_steps
                == len(MODEL_IDS)
            ),

        "phase4er3c_status":
            (
                "passed"
                if (
                    len(passed_model_ids)
                    == len(MODEL_IDS)
                    and all_gradients_finite
                    and all_parameters_updated
                    and total_optimizer_steps
                    == len(MODEL_IDS)
                )
                else "failed"
            ),

        "next_phase":
            (
                "4E-R3D Tier C v4 predictive "
                "implementation acceptance freeze"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4er3c_predictive_smoke_summary.json",
        summary,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    require(
        summary[
            "implementation_smoke_test_passed"
        ] is True,
        "Tier C v4 predictive smoke execution failed.",
    )

    print(
        "Phase 4E-R3C predictive smoke execution passed."
    )

    print(
        f"Outputs written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
