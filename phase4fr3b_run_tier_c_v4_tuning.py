from __future__ import annotations

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import phase3d_tune_tier_b_models as tier_b
import phase4er3c_execute_predictive_smoke as accepted_impl


PROTOCOL_VERSION = "tier_c_v4"

OPERATION_VOCABULARY = {
    "identity": 0,
    "cycle": 1,
    "pair_collapse": 2,
    "half_collapse": 3,
    "parity_collapse": 4,
    "reset": 5,
}

OPERATION_COUNT = 6
OBSERVATION_DIMENSION = 128
NOISE_INDEX = 2
NOISE_FRACTION = 0.25

PHYSICAL_MICROBATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 64
EFFECTIVE_BATCH_SIZE = 256

MAXIMUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 25
GRADIENT_CLIP_NORM = 5.0

EXPECTED_TRAIN_TRAJECTORIES = 11008
EXPECTED_VAL_TRAJECTORIES = 1152
EXPECTED_CONFIGURATION_COUNT = 43

MODEL_IDS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

EXPECTED_CONFIGURATION_COUNTS = {
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
    "OCM": 24,
}

DATA_ROOT = Path(
    "outputs/"
    "phase4br3_tier_c_v4_development_data/"
    "visible"
)

STRUCTURAL_RELEASE_ROOT = Path(
    "outputs/"
    "phase4fr3ar1_tier_c_v4_visible_trajectory_indexes/"
    "visible"
)

REPAIRED_PREFLIGHT_PATH = Path(
    "outputs/"
    "phase4fr3ar1_tier_c_v4_visible_trajectory_indexes/"
    "phase4fr3ar1_repaired_tuning_preflight_summary.json"
)

IMPLEMENTATION_ACCEPTANCE_PATH = Path(
    "outputs/"
    "phase4er3d_tier_c_v4_implementation_acceptance/"
    "phase4er3d_implementation_acceptance_summary.json"
)

EXECUTION_CONTRACT_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "tier_c_v4_execution_contract.json"
)

TUNING_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_tuning_registry.csv"
)

ACCEPTED_IMPLEMENTATION_PATH = Path(
    "phase4er3c_execute_predictive_smoke.py"
)

OUTPUT_ROOT = Path(
    "outputs/"
    "phase4fr3b_tier_c_v4_development_tuning"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def reject_privileged_path(
    path: Path,
) -> None:
    normalized = str(path).replace(
        "\\",
        "/",
    ).lower()

    require(
        "/privileged/" not in normalized,
        f"Privileged path rejected: {path}",
    )


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def atomic_write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )

    os.replace(
        temporary_path,
        path,
    )


def atomic_torch_save(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        value,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
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


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def set_global_determinism(
    seed: int,
) -> None:
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


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        requested = (
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


class TrajectoryCell:
    def __init__(
        self,
        cell_id: str,
        expected_trajectory_count: int,
    ) -> None:
        self.cell_id = cell_id

        self.fields_path = (
            DATA_ROOT
            / f"{cell_id}_fields.npy"
        )

        self.manifest_path = (
            DATA_ROOT
            / f"{cell_id}_sequence_manifest.csv"
        )

        self.index_path = (
            STRUCTURAL_RELEASE_ROOT
            / f"{cell_id}_trajectory_index.csv"
        )

        for path in (
            self.fields_path,
            self.manifest_path,
            self.index_path,
        ):
            reject_privileged_path(path)

            if not path.exists():
                raise FileNotFoundError(path)

        self.fields = np.load(
            self.fields_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        require(
            self.fields.ndim == 5,
            (
                f"{cell_id} field array has shape "
                f"{self.fields.shape}."
            ),
        )

        require(
            self.fields.shape[0] == 5,
            f"{cell_id} does not have five noise levels.",
        )

        require(
            list(
                self.fields.shape[-3:]
            ) == [4, 32, 32],
            f"{cell_id} field shape changed.",
        )

        require(
            str(self.fields.dtype)
            == "float16",
            f"{cell_id} on-disk dtype changed.",
        )

        self.sequence_lookup = (
            self._load_manifest()
        )

        self.rows = self._load_index(
            expected_trajectory_count
        )

        self.lengths = np.asarray(
            [
                row["sequence_length"]
                for row in self.rows
            ],
            dtype=np.int64,
        )

        self.maximum_sequence_length = int(
            self.lengths.max()
        )

        self.length_counts = {
            str(length):
                int(count)
            for length, count
            in sorted(
                Counter(
                    int(value)
                    for value
                    in self.lengths
                ).items()
            )
        }

        self.zero_length_count = int(
            np.sum(
                self.lengths == 0
            )
        )

    def _load_manifest(
        self,
    ) -> dict[int, dict[str, Any]]:
        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            columns = set(
                reader.fieldnames or []
            )

            required_columns = {
                "cell_sequence_index",
                "source_sequence_id",
                "sequence_length",
                "operation_sequence",
            }

            missing = sorted(
                required_columns
                - columns
            )

            require(
                not missing,
                (
                    f"{self.manifest_path} lacks "
                    f"columns {missing}."
                ),
            )

            rows = list(reader)

        lookup: dict[int, dict[str, Any]] = {}

        for row in rows:
            cell_sequence_index = int(
                row[
                    "cell_sequence_index"
                ]
            )

            require(
                cell_sequence_index
                not in lookup,
                (
                    "Duplicate cell_sequence_index "
                    f"{cell_sequence_index}."
                ),
            )

            operation_names = json.loads(
                row[
                    "operation_sequence"
                ]
            )

            require(
                isinstance(
                    operation_names,
                    list,
                ),
                "operation_sequence is not a JSON list.",
            )

            sequence_length = int(
                row[
                    "sequence_length"
                ]
            )

            require(
                len(operation_names)
                == sequence_length,
                (
                    "Operation-sequence length mismatch "
                    f"for {cell_sequence_index}."
                ),
            )

            unknown = sorted(
                set(operation_names)
                - set(
                    OPERATION_VOCABULARY
                )
            )

            require(
                not unknown,
                (
                    "Unknown operations in manifest: "
                    f"{unknown}"
                ),
            )

            operation_ids = np.asarray(
                [
                    OPERATION_VOCABULARY[
                        operation_name
                    ]
                    for operation_name
                    in operation_names
                ],
                dtype=np.int64,
            )

            lookup[
                cell_sequence_index
            ] = {
                "source_sequence_id":
                    str(
                        row[
                            "source_sequence_id"
                        ]
                    ),

                "sequence_length":
                    sequence_length,

                "operation_names":
                    operation_names,

                "operation_ids":
                    operation_ids,
            }

        return lookup

    def _load_index(
        self,
        expected_trajectory_count: int,
    ) -> list[dict[str, Any]]:
        with self.index_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            expected_columns = {
                "trajectory_index",
                "cell_sequence_index",
                "source_sequence_id",
                "point_start",
                "point_count",
                "sequence_length",
            }

            observed_columns = set(
                reader.fieldnames or []
            )

            require(
                observed_columns
                == expected_columns,
                (
                    f"Released index schema changed for "
                    f"{self.cell_id}: {observed_columns}"
                ),
            )

            raw_rows = list(reader)

        require(
            len(raw_rows)
            == expected_trajectory_count,
            (
                f"{self.cell_id} expected "
                f"{expected_trajectory_count} trajectories; "
                f"found {len(raw_rows)}."
            ),
        )

        rows: list[dict[str, Any]] = []
        expected_point_start = 0

        for expected_index, raw in enumerate(
            raw_rows
        ):
            row = {
                "trajectory_index":
                    int(
                        raw[
                            "trajectory_index"
                        ]
                    ),

                "cell_sequence_index":
                    int(
                        raw[
                            "cell_sequence_index"
                        ]
                    ),

                "source_sequence_id":
                    str(
                        raw[
                            "source_sequence_id"
                        ]
                    ),

                "point_start":
                    int(
                        raw[
                            "point_start"
                        ]
                    ),

                "point_count":
                    int(
                        raw[
                            "point_count"
                        ]
                    ),

                "sequence_length":
                    int(
                        raw[
                            "sequence_length"
                        ]
                    ),
            }

            require(
                row["trajectory_index"]
                == expected_index,
                (
                    f"{self.cell_id} trajectory indexes "
                    "are not contiguous."
                ),
            )

            require(
                row["point_start"]
                == expected_point_start,
                (
                    f"{self.cell_id} point coverage "
                    "is not contiguous."
                ),
            )

            require(
                row["point_count"]
                == (
                    row[
                        "sequence_length"
                    ]
                    + 1
                ),
                (
                    f"{self.cell_id} point_count differs "
                    "from sequence_length + 1."
                ),
            )

            sequence_index = row[
                "cell_sequence_index"
            ]

            require(
                sequence_index
                in self.sequence_lookup,
                (
                    f"{self.cell_id} references unknown "
                    f"sequence {sequence_index}."
                ),
            )

            manifest = self.sequence_lookup[
                sequence_index
            ]

            require(
                row[
                    "source_sequence_id"
                ]
                == manifest[
                    "source_sequence_id"
                ],
                (
                    f"{self.cell_id} source sequence "
                    "mismatch."
                ),
            )

            require(
                row[
                    "sequence_length"
                ]
                == manifest[
                    "sequence_length"
                ],
                (
                    f"{self.cell_id} sequence length "
                    "mismatch."
                ),
            )

            expected_point_start += row[
                "point_count"
            ]

            rows.append(row)

        require(
            expected_point_start
            == self.fields.shape[1],
            (
                f"{self.cell_id} index covers "
                f"{expected_point_start} points but "
                f"field array has {self.fields.shape[1]}."
            ),
        )

        return rows

    def make_batch(
        self,
        trajectory_indices: list[int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        selected_rows = [
            self.rows[index]
            for index
            in trajectory_indices
        ]

        lengths = np.asarray(
            [
                row[
                    "sequence_length"
                ]
                for row
                in selected_rows
            ],
            dtype=np.int64,
        )

        maximum_length = int(
            lengths.max()
        )

        require(
            maximum_length >= 1,
            (
                "A zero-step-only physical microbatch "
                "reached collation."
            ),
        )

        batch_size = len(
            selected_rows
        )

        observations = np.zeros(
            (
                batch_size,
                maximum_length + 1,
                4,
                32,
                32,
            ),
            dtype=np.float32,
        )

        operation_ids = np.zeros(
            (
                batch_size,
                maximum_length,
            ),
            dtype=np.int64,
        )

        point_mask = np.zeros(
            (
                batch_size,
                maximum_length + 1,
            ),
            dtype=np.bool_,
        )

        step_mask = np.zeros(
            (
                batch_size,
                maximum_length,
            ),
            dtype=np.bool_,
        )

        for batch_index, row in enumerate(
            selected_rows
        ):
            point_start = row[
                "point_start"
            ]

            point_count = row[
                "point_count"
            ]

            sequence_length = row[
                "sequence_length"
            ]

            fields = np.asarray(
                self.fields[
                    NOISE_INDEX,
                    point_start:
                    point_start
                    + point_count,
                ],
                dtype=np.float32,
            )

            require(
                fields.shape
                == (
                    point_count,
                    4,
                    32,
                    32,
                ),
                "Unexpected trajectory field shape.",
            )

            require(
                np.isfinite(
                    fields
                ).all(),
                "Non-finite field encountered.",
            )

            observations[
                batch_index,
                :point_count,
            ] = fields

            point_mask[
                batch_index,
                :point_count,
            ] = True

            manifest = self.sequence_lookup[
                row[
                    "cell_sequence_index"
                ]
            ]

            if sequence_length > 0:
                operation_ids[
                    batch_index,
                    :sequence_length,
                ] = manifest[
                    "operation_ids"
                ]

                step_mask[
                    batch_index,
                    :sequence_length,
                ] = True

        return {
            "observations":
                torch.from_numpy(
                    observations
                ).to(
                    device=device,
                    dtype=torch.float32,
                ),

            "operation_ids":
                torch.from_numpy(
                    operation_ids
                ).to(
                    device=device,
                    dtype=torch.long,
                ),

            "point_mask":
                torch.from_numpy(
                    point_mask
                ).to(
                    device=device,
                    dtype=torch.bool,
                ),

            "step_mask":
                torch.from_numpy(
                    step_mask
                ).to(
                    device=device,
                    dtype=torch.bool,
                ),

            "sequence_lengths":
                torch.from_numpy(
                    lengths
                ).to(
                    device=device,
                    dtype=torch.long,
                ),
        }

    def audit_record(
        self,
    ) -> dict[str, Any]:
        operations_seen = set()

        for manifest in (
            self.sequence_lookup.values()
        ):
            operations_seen.update(
                manifest[
                    "operation_names"
                ]
            )

        return {
            "cell_id":
                self.cell_id,

            "field_path":
                str(
                    self.fields_path
                ),

            "manifest_path":
                str(
                    self.manifest_path
                ),

            "trajectory_index_path":
                str(
                    self.index_path
                ),

            "trajectory_count":
                len(self.rows),

            "sequence_count":
                len(
                    self.sequence_lookup
                ),

            "point_count":
                int(
                    self.fields.shape[1]
                ),

            "noise_index":
                NOISE_INDEX,

            "noise_fraction":
                NOISE_FRACTION,

            "maximum_sequence_length":
                self.maximum_sequence_length,

            "zero_length_trajectory_count":
                self.zero_length_count,

            "sequence_length_counts":
                self.length_counts,

            "operations_seen":
                sorted(
                    operations_seen
                ),

            "all_six_operations_seen":
                operations_seen
                == set(
                    OPERATION_VOCABULARY
                ),

            "privileged_path_used":
                False,
        }


def repair_zero_only_microbatches(
    order: np.ndarray,
    lengths: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
]:
    repaired = np.asarray(
        order,
        dtype=np.int64,
    ).copy()

    require(
        len(repaired)
        % PHYSICAL_MICROBATCH_SIZE
        == 0,
        (
            "Trajectory count is not divisible by "
            "the physical microbatch size."
        ),
    )

    repair_count = 0

    for start in range(
        0,
        len(repaired),
        PHYSICAL_MICROBATCH_SIZE,
    ):
        end = start + PHYSICAL_MICROBATCH_SIZE

        batch_positions = repaired[
            start:end
        ]

        if np.any(
            lengths[
                batch_positions
            ] > 0
        ):
            continue

        donor_position = None

        for donor_start in range(
            0,
            len(repaired),
            PHYSICAL_MICROBATCH_SIZE,
        ):
            donor_end = (
                donor_start
                + PHYSICAL_MICROBATCH_SIZE
            )

            if donor_start == start:
                continue

            donor_positions = repaired[
                donor_start:
                donor_end
            ]

            nonzero_offsets = np.where(
                lengths[
                    donor_positions
                ] > 0
            )[0]

            if len(nonzero_offsets) >= 2:
                donor_position = (
                    donor_start
                    + int(
                        nonzero_offsets[0]
                    )
                )

                break

        require(
            donor_position is not None,
            (
                "Could not repair a zero-step-only "
                "physical microbatch."
            ),
        )

        repaired[
            start
        ], repaired[
            donor_position
        ] = (
            repaired[
                donor_position
            ],
            repaired[
                start
            ],
        )

        repair_count += 1

    for start in range(
        0,
        len(repaired),
        PHYSICAL_MICROBATCH_SIZE,
    ):
        batch_positions = repaired[
            start:
            start
            + PHYSICAL_MICROBATCH_SIZE
        ]

        require(
            np.any(
                lengths[
                    batch_positions
                ] > 0
            ),
            (
                "A zero-step-only microbatch remains "
                "after repair."
            ),
        )

    return repaired, repair_count


def load_configurations() -> list[dict[str, Any]]:
    records = []

    with TUNING_REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for registry_index, row in enumerate(
            reader
        ):
            configuration = json.loads(
                row[
                    "source_configuration_json"
                ]
            )

            require(
                configuration[
                    "configuration_id"
                ]
                == row[
                    "configuration_id"
                ],
                "Configuration ID mismatch.",
            )

            require(
                row[
                    "training_cell"
                ] == "train_joint",
                "Training cell changed.",
            )

            require(
                row[
                    "checkpoint_selection_cell"
                ] == "val_joint",
                "Validation cell changed.",
            )

            require(
                float(
                    row[
                        "tuning_noise_fraction"
                    ]
                ) == NOISE_FRACTION,
                "Tuning noise fraction changed.",
            )

            raw_checkpoint_direction = (
                row[
                    "checkpoint_selection_direction"
                ]
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            minimization_direction_labels = {
                "minimize",
                "minimum",
                "min",
                "lower",
                "lower_is_better",
                "smaller_is_better",
                "ascending",
            }

            require(
                raw_checkpoint_direction
                in minimization_direction_labels,
                (
                    "Checkpoint-selection direction is "
                    "not a recognized minimization label: "
                    f"{row['checkpoint_selection_direction']!r}"
                ),
            )

            require(
                row[
                    "test_access_allowed"
                ].strip().lower()
                in {
                    "false",
                    "0",
                },
                "A tuning row permits test access.",
            )

            records.append(
                {
                    "registry_index":
                        registry_index,

                    "model_id":
                        row[
                            "model_id"
                        ],

                    "configuration_id":
                        row[
                            "configuration_id"
                        ],

                    "tuning_seed":
                        int(
                            row[
                                "tuning_seed"
                            ]
                        ),

                    "configuration":
                        configuration,

                    "registry_row":
                        row,
                }
            )

    require(
        len(records)
        == EXPECTED_CONFIGURATION_COUNT,
        "Expected 43 tuning configurations.",
    )

    counts = Counter(
        record[
            "model_id"
        ]
        for record in records
    )

    require(
        dict(counts)
        == EXPECTED_CONFIGURATION_COUNTS,
        (
            "Configuration-count mismatch: "
            f"{dict(counts)}"
        ),
    )

    return records


def build_model_bundle(
    model_id: str,
    configuration: dict[str, Any],
    maximum_sequence_length: int,
    device: torch.device,
) -> tuple[
    nn.Module,
    nn.Module,
    torch.optim.Optimizer,
]:
    codec = (
        accepted_impl
        .TierCFieldCodec()
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    if model_id == "B1":
        model = (
            accepted_impl
            .AffineLatentTransitionBaseline(
                observation_dimension=(
                    OBSERVATION_DIMENSION
                ),
                operation_count=(
                    OPERATION_COUNT
                ),
            )
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        learning_rate = 3e-4

    elif model_id in {
        "B2",
        "B3",
        "B4",
        "B5",
    }:
        tier_b.OBSERVATION_DIMENSION = (
            OBSERVATION_DIMENSION
        )

        model = tier_b.build_model(
            configuration=configuration,
            operation_count=(
                OPERATION_COUNT
            ),
            maximum_sequence_length=(
                maximum_sequence_length
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
        model = (
            tier_b.OperationChannelModel(
                observation_dim=(
                    OBSERVATION_DIMENSION
                ),
                latent_state_count=int(
                    configuration[
                        "latent_state_count"
                    ]
                ),
                operation_count=(
                    OPERATION_COUNT
                ),
            )
            .to(
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

    optimizer = torch.optim.Adam(
        list(
            codec.parameters()
        )
        + list(
            model.parameters()
        ),
        lr=learning_rate,
    )

    return codec, model, optimizer


def compute_losses(
    model_id: str,
    model: nn.Module,
    codec: nn.Module,
    batch: dict[str, torch.Tensor],
    configuration: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if model_id == "OCM":
        return accepted_impl.ocm_forward_losses(
            model=model,
            codec=codec,
            batch=batch,
            configuration=configuration,
        )

    return accepted_impl.baseline_forward_losses(
        model_id=model_id,
        model=model,
        codec=codec,
        batch=batch,
        configuration=configuration,
    )


def scale_gradients(
    parameters: list[nn.Parameter],
    divisor: int,
) -> None:
    require(
        divisor > 0,
        "Gradient divisor must be positive.",
    )

    for parameter in parameters:
        if parameter.grad is None:
            continue

        parameter.grad.div_(
            float(divisor)
        )

        require(
            bool(
                torch.isfinite(
                    parameter.grad
                ).all()
            ),
            "Non-finite gradient before clipping.",
        )


def summarize_components(
    component_sums: dict[str, float],
    trajectory_count: int,
) -> dict[str, float]:
    require(
        trajectory_count > 0,
        "No trajectories were aggregated.",
    )

    return {
        key:
            value
            / float(
                trajectory_count
            )
        for key, value
        in component_sums.items()
    }


def run_training_epoch(
    model_id: str,
    model: nn.Module,
    codec: nn.Module,
    optimizer: torch.optim.Optimizer,
    configuration: dict[str, Any],
    train_cell: TrajectoryCell,
    device: torch.device,
    epoch: int,
    tuning_seed: int,
) -> dict[str, Any]:
    model.train()
    codec.train()

    torch.manual_seed(
        tuning_seed + epoch
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            tuning_seed + epoch
        )

    rng = np.random.default_rng(
        tuning_seed + epoch
    )

    order = rng.permutation(
        len(
            train_cell.rows
        )
    )

    order, zero_batch_repairs = (
        repair_zero_only_microbatches(
            order,
            train_cell.lengths,
        )
    )

    parameters = list(
        codec.parameters()
    ) + list(
        model.parameters()
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    component_sums: dict[str, float] = {}

    accumulated_trajectories = 0
    accumulated_microbatches = 0

    optimizer_steps = 0
    microbatch_count = 0

    preclip_norms = []

    epoch_start = time.time()

    for start in range(
        0,
        len(order),
        PHYSICAL_MICROBATCH_SIZE,
    ):
        selected = order[
            start:
            start
            + PHYSICAL_MICROBATCH_SIZE
        ].tolist()

        batch = train_cell.make_batch(
            selected,
            device,
        )

        batch_size = len(selected)

        losses = compute_losses(
            model_id=model_id,
            model=model,
            codec=codec,
            batch=batch,
            configuration=configuration,
        )

        total_loss = losses[
            "total_loss"
        ]

        require(
            bool(
                torch.isfinite(
                    total_loss
                )
            ),
            "Non-finite training loss.",
        )

        (
            total_loss
            * float(batch_size)
        ).backward()

        accumulated_trajectories += (
            batch_size
        )

        accumulated_microbatches += 1
        microbatch_count += 1

        for key, value in losses.items():
            if key == "final_prediction":
                continue

            scalar = float(
                value.detach().cpu()
            )

            require(
                math.isfinite(scalar),
                (
                    "Non-finite training component "
                    f"{key}."
                ),
            )

            component_sums[key] = (
                component_sums.get(
                    key,
                    0.0,
                )
                + scalar
                * float(batch_size)
            )

        is_accumulation_boundary = (
            accumulated_microbatches
            == GRADIENT_ACCUMULATION_STEPS
        )

        is_final_microbatch = (
            start
            + PHYSICAL_MICROBATCH_SIZE
            >= len(order)
        )

        if (
            is_accumulation_boundary
            or is_final_microbatch
        ):
            scale_gradients(
                parameters,
                accumulated_trajectories,
            )

            preclip_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=(
                        GRADIENT_CLIP_NORM
                    ),
                    norm_type=2.0,
                    error_if_nonfinite=True,
                ).detach().cpu()
            )

            preclip_norms.append(
                preclip_norm
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_steps += 1

            accumulated_trajectories = 0
            accumulated_microbatches = 0

    require(
        accumulated_trajectories == 0,
        "Unflushed training trajectories remain.",
    )

    require(
        accumulated_microbatches == 0,
        "Unflushed microbatches remain.",
    )

    expected_steps = (
        len(train_cell.rows)
        // EFFECTIVE_BATCH_SIZE
    )

    require(
        optimizer_steps == expected_steps,
        (
            f"Expected {expected_steps} optimizer "
            f"steps; observed {optimizer_steps}."
        ),
    )

    return {
        "epoch":
            epoch,

        "trajectory_count":
            len(train_cell.rows),

        "microbatch_count":
            microbatch_count,

        "optimizer_step_count":
            optimizer_steps,

        "zero_step_only_microbatch_repairs":
            zero_batch_repairs,

        "component_means":
            summarize_components(
                component_sums,
                len(train_cell.rows),
            ),

        "preclip_gradient_norm_minimum":
            float(
                min(preclip_norms)
            ),

        "preclip_gradient_norm_maximum":
            float(
                max(preclip_norms)
            ),

        "preclip_gradient_norm_mean":
            float(
                np.mean(
                    preclip_norms
                )
            ),

        "elapsed_seconds":
            float(
                time.time()
                - epoch_start
            ),
    }


@torch.no_grad()
def evaluate_validation(
    model_id: str,
    model: nn.Module,
    codec: nn.Module,
    configuration: dict[str, Any],
    validation_cell: TrajectoryCell,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    codec.eval()

    order = np.arange(
        len(
            validation_cell.rows
        ),
        dtype=np.int64,
    )

    order, zero_batch_repairs = (
        repair_zero_only_microbatches(
            order,
            validation_cell.lengths,
        )
    )

    component_sums: dict[str, float] = {}
    microbatch_count = 0

    evaluation_start = time.time()

    for start in range(
        0,
        len(order),
        PHYSICAL_MICROBATCH_SIZE,
    ):
        selected = order[
            start:
            start
            + PHYSICAL_MICROBATCH_SIZE
        ].tolist()

        batch = validation_cell.make_batch(
            selected,
            device,
        )

        losses = compute_losses(
            model_id=model_id,
            model=model,
            codec=codec,
            batch=batch,
            configuration=configuration,
        )

        batch_size = len(selected)

        for key, value in losses.items():
            if key == "final_prediction":
                continue

            scalar = float(
                value.detach().cpu()
            )

            require(
                math.isfinite(scalar),
                (
                    "Non-finite validation component "
                    f"{key}."
                ),
            )

            component_sums[key] = (
                component_sums.get(
                    key,
                    0.0,
                )
                + scalar
                * float(batch_size)
            )

        microbatch_count += 1

    component_means = summarize_components(
        component_sums,
        len(validation_cell.rows),
    )

    require(
        "rollout_field_loss"
        in component_means,
        (
            "Validation rollout field loss "
            "was not computed."
        ),
    )

    selection_metric = float(
        component_means[
            "rollout_field_loss"
        ]
    )

    require(
        math.isfinite(
            selection_metric
        ),
        "Validation selection metric is non-finite.",
    )

    return {
        "trajectory_count":
            len(
                validation_cell.rows
            ),

        "microbatch_count":
            microbatch_count,

        "zero_step_only_microbatch_repairs":
            zero_batch_repairs,

        "component_means":
            component_means,

        "selection_metric_name":
            (
                "val_joint noisy-target "
                "rollout field MSE"
            ),

        "selection_metric_direction":
            "minimize",

        "selection_metric":
            selection_metric,

        "elapsed_seconds":
            float(
                time.time()
                - evaluation_start
            ),
    }


def build_source_signature(
    record: dict[str, Any],
    train_cell: TrajectoryCell,
    validation_cell: TrajectoryCell,
) -> dict[str, Any]:
    signature = {
        "protocol_version":
            PROTOCOL_VERSION,

        "configuration_id":
            record[
                "configuration_id"
            ],

        "configuration":
            record[
                "configuration"
            ],

        "tuning_seed":
            record[
                "tuning_seed"
            ],

        "noise_index":
            NOISE_INDEX,

        "noise_fraction":
            NOISE_FRACTION,

        "physical_microbatch_size":
            PHYSICAL_MICROBATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_batch_size":
            EFFECTIVE_BATCH_SIZE,

        "maximum_epochs":
            MAXIMUM_EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "training_field_path":
            str(
                train_cell.fields_path
            ),

        "validation_field_path":
            str(
                validation_cell.fields_path
            ),

        "training_index_sha256":
            sha256_file(
                train_cell.index_path
            ),

        "validation_index_sha256":
            sha256_file(
                validation_cell.index_path
            ),

        "training_manifest_sha256":
            sha256_file(
                train_cell.manifest_path
            ),

        "validation_manifest_sha256":
            sha256_file(
                validation_cell.manifest_path
            ),

        "accepted_implementation_sha256":
            sha256_file(
                ACCEPTED_IMPLEMENTATION_PATH
            ),

        "runner_sha256":
            sha256_file(
                Path(__file__)
            ),
    }

    signature[
        "signature_sha256"
    ] = sha256_json(signature)

    return signature


def save_last_checkpoint(
    path: Path,
    record: dict[str, Any],
    source_signature: dict[str, Any],
    codec: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int | None,
    best_metric: float,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
) -> None:
    atomic_torch_save(
        path,
        {
            "protocol_version":
                PROTOCOL_VERSION,

            "configuration_id":
                record[
                    "configuration_id"
                ],

            "model_id":
                record[
                    "model_id"
                ],

            "source_signature":
                source_signature,

            "epoch_completed":
                epoch,

            "best_epoch":
                best_epoch,

            "best_metric":
                best_metric,

            "epochs_without_improvement":
                epochs_without_improvement,

            "history":
                history,

            "codec_state_dict":
                codec.state_dict(),

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),
        },
    )


def run_configuration(
    record: dict[str, Any],
    train_cell: TrajectoryCell,
    validation_cell: TrajectoryCell,
    device: torch.device,
    overwrite_completed: bool,
) -> dict[str, Any]:
    model_id = record[
        "model_id"
    ]

    configuration_id = record[
        "configuration_id"
    ]

    configuration = record[
        "configuration"
    ]

    tuning_seed = record[
        "tuning_seed"
    ]

    configuration_dir = (
        OUTPUT_ROOT
        / model_id
        / configuration_id
    )

    configuration_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        configuration_dir
        / "tuning_result.json"
    )

    last_checkpoint_path = (
        configuration_dir
        / "last_checkpoint.pt"
    )

    best_checkpoint_path = (
        configuration_dir
        / "best_checkpoint.pt"
    )

    history_path = (
        configuration_dir
        / "epoch_history.json"
    )

    source_signature = (
        build_source_signature(
            record,
            train_cell,
            validation_cell,
        )
    )

    if (
        result_path.exists()
        and not overwrite_completed
    ):
        existing_result = load_json(
            result_path
        )

        require(
            existing_result[
                "source_signature_sha256"
            ]
            == source_signature[
                "signature_sha256"
            ],
            (
                f"Completed result source signature "
                f"changed for {configuration_id}."
            ),
        )

        if (
            existing_result[
                "tuning_status"
            ] == "completed"
        ):
            print(
                f"{configuration_id}: already complete"
            )

            return existing_result

    set_global_determinism(
        tuning_seed
    )

    codec, model, optimizer = (
        build_model_bundle(
            model_id=model_id,
            configuration=configuration,
            maximum_sequence_length=max(
                train_cell.maximum_sequence_length,
                validation_cell.maximum_sequence_length,
            ),
            device=device,
        )
    )

    start_epoch = 1
    best_epoch: int | None = None
    best_metric = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if (
        last_checkpoint_path.exists()
        and not overwrite_completed
    ):
        checkpoint = torch.load(
            last_checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        require(
            checkpoint[
                "source_signature"
            ][
                "signature_sha256"
            ]
            == source_signature[
                "signature_sha256"
            ],
            (
                f"Checkpoint source signature changed "
                f"for {configuration_id}."
            ),
        )

        codec.load_state_dict(
            checkpoint[
                "codec_state_dict"
            ]
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        start_epoch = (
            int(
                checkpoint[
                    "epoch_completed"
                ]
            )
            + 1
        )

        best_epoch = checkpoint[
            "best_epoch"
        ]

        best_metric = float(
            checkpoint[
                "best_metric"
            ]
        )

        epochs_without_improvement = int(
            checkpoint[
                "epochs_without_improvement"
            ]
        )

        history = list(
            checkpoint[
                "history"
            ]
        )

        print(
            f"{configuration_id}: resuming at "
            f"epoch {start_epoch}"
        )

    run_start = time.time()
    stop_reason = "maximum_epochs_reached"

    for epoch in range(
        start_epoch,
        MAXIMUM_EPOCHS + 1,
    ):
        training_record = (
            run_training_epoch(
                model_id=model_id,
                model=model,
                codec=codec,
                optimizer=optimizer,
                configuration=configuration,
                train_cell=train_cell,
                device=device,
                epoch=epoch,
                tuning_seed=tuning_seed,
            )
        )

        validation_record = (
            evaluate_validation(
                model_id=model_id,
                model=model,
                codec=codec,
                configuration=configuration,
                validation_cell=(
                    validation_cell
                ),
                device=device,
            )
        )

        validation_metric = float(
            validation_record[
                "selection_metric"
            ]
        )

        improved = (
            validation_metric
            < best_metric
        )

        if improved:
            best_metric = (
                validation_metric
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            atomic_torch_save(
                best_checkpoint_path,
                {
                    "protocol_version":
                        PROTOCOL_VERSION,

                    "model_id":
                        model_id,

                    "configuration_id":
                        configuration_id,

                    "configuration":
                        configuration,

                    "tuning_seed":
                        tuning_seed,

                    "noise_fraction":
                        NOISE_FRACTION,

                    "training_cell":
                        "train_joint",

                    "checkpoint_selection_cell":
                        "val_joint",

                    "checkpoint_selection_metric":
                        (
                            "val_joint noisy-target "
                            "rollout field MSE"
                        ),

                    "checkpoint_selection_direction":
                        "minimize",

                    "best_epoch":
                        best_epoch,

                    "best_validation_rollout_mse":
                        best_metric,

                    "source_signature":
                        source_signature,

                    "codec_state_dict":
                        codec.state_dict(),

                    "model_state_dict":
                        model.state_dict(),

                    "test_data_read":
                        False,
                },
            )

        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch":
                epoch,

            "training":
                training_record,

            "validation":
                validation_record,

            "improved":
                improved,

            "best_epoch":
                best_epoch,

            "best_validation_rollout_mse":
                best_metric,

            "epochs_without_improvement":
                epochs_without_improvement,
        }

        history.append(
            epoch_record
        )

        atomic_write_json(
            history_path,
            history,
        )

        save_last_checkpoint(
            path=last_checkpoint_path,
            record=record,
            source_signature=(
                source_signature
            ),
            codec=codec,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_epoch=best_epoch,
            best_metric=best_metric,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
            history=history,
        )

        print(
            f"{configuration_id} | "
            f"epoch={epoch:03d} | "
            f"train_total="
            f"{training_record['component_means']['total_loss']:.8f} | "
            f"val_rollout="
            f"{validation_metric:.8f} | "
            f"best={best_metric:.8f} | "
            f"bad_epochs="
            f"{epochs_without_improvement}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            stop_reason = (
                "early_stopping_patience_reached"
            )

            break

    require(
        best_epoch is not None,
        (
            f"{configuration_id} never produced "
            "a valid checkpoint."
        ),
    )

    require(
        best_checkpoint_path.exists(),
        (
            f"{configuration_id} best checkpoint "
            "is missing."
        ),
    )

    total_optimizer_steps = sum(
        epoch_record[
            "training"
        ][
            "optimizer_step_count"
        ]
        for epoch_record
        in history
    )

    result = {
        "phase":
            (
                "4F-R3B Tier C v4 predictive "
                "development tuning"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "model_id":
            model_id,

        "configuration_id":
            configuration_id,

        "configuration":
            configuration,

        "tuning_seed":
            tuning_seed,

        "training_cell":
            "train_joint",

        "validation_cell":
            "val_joint",

        "noise_fraction":
            NOISE_FRACTION,

        "noise_index":
            NOISE_INDEX,

        "checkpoint_selection_metric":
            (
                "val_joint noisy-target "
                "rollout field MSE"
            ),

        "checkpoint_selection_direction":
            "minimize",

        "best_epoch":
            best_epoch,

        "best_validation_rollout_mse":
            best_metric,

        "epochs_completed":
            len(history),

        "stop_reason":
            stop_reason,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "maximum_epochs":
            MAXIMUM_EPOCHS,

        "physical_microbatch_size":
            PHYSICAL_MICROBATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_batch_size":
            EFFECTIVE_BATCH_SIZE,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "optimizer":
            "torch.optim.Adam",

        "total_optimizer_steps":
            total_optimizer_steps,

        "best_checkpoint_path":
            str(
                best_checkpoint_path
            ),

        "best_checkpoint_sha256":
            sha256_file(
                best_checkpoint_path
            ),

        "last_checkpoint_path":
            str(
                last_checkpoint_path
            ),

        "history_path":
            str(
                history_path
            ),

        "source_signature_sha256":
            source_signature[
                "signature_sha256"
            ],

        "elapsed_seconds":
            float(
                time.time()
                - run_start
            ),

        "validation_selection_performed":
            True,

        "validation_selection_cell":
            "val_joint",

        "privileged_directory_read":
            False,

        "clean_targets_read":
            False,

        "privileged_state_ids_read":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "final_fit_performed":
            False,

        "tuning_status":
            "completed",
    }

    atomic_write_json(
        result_path,
        result,
    )

    return result


def validate_protocol() -> None:
    required_paths = (
        REPAIRED_PREFLIGHT_PATH,
        IMPLEMENTATION_ACCEPTANCE_PATH,
        EXECUTION_CONTRACT_PATH,
        TUNING_REGISTRY_PATH,
        ACCEPTED_IMPLEMENTATION_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    repaired_preflight = load_json(
        REPAIRED_PREFLIGHT_PATH
    )

    acceptance = load_json(
        IMPLEMENTATION_ACCEPTANCE_PATH
    )

    contract = load_json(
        EXECUTION_CONTRACT_PATH
    )

    require(
        repaired_preflight[
            "phase4fr3ar1_status"
        ]
        == (
            "ready_for_development_tuning_"
            "with_derived_visible_structural_indexes"
        ),
        "Repaired tuning preflight is not accepted.",
    )

    require(
        repaired_preflight[
            "training_must_use_released_visible_indexes"
        ] is True,
        "Visible structural indexes are not mandatory.",
    )

    require(
        repaired_preflight[
            "training_may_not_read_privileged_directory"
        ] is True,
        "Privileged-directory prohibition is missing.",
    )

    require(
        acceptance[
            "tuning_execution_authorized"
        ] is True,
        "Tuning is not authorized.",
    )

    require(
        acceptance[
            "final_fit_execution_authorized"
        ] is False,
        "Final fitting is already authorized.",
    )

    require(
        acceptance[
            "test_artifact_generation_authorized"
        ] is False,
        "Test artifact generation is authorized.",
    )

    require(
        acceptance[
            "test_evaluation_authorized"
        ] is False,
        "Test evaluation is authorized.",
    )

    require(
        acceptance[
            "test_open_count"
        ] == 0,
        "Test data was opened.",
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
            "batching"
        ][
            "gradient_accumulation_steps"
        ] == GRADIENT_ACCUMULATION_STEPS,
        "Gradient accumulation changed.",
    )

    require(
        contract[
            "batching"
        ][
            "effective_batch_size_trajectories"
        ] == EFFECTIVE_BATCH_SIZE,
        "Effective batch size changed.",
    )

    require(
        contract[
            "gradient_clipping_policy"
        ][
            "maximum_norm"
        ] == GRADIENT_CLIP_NORM,
        "Gradient clipping changed.",
    )

    require(
        contract[
            "mixed_precision_policy"
        ][
            "enabled"
        ] is False,
        "Mixed precision was enabled.",
    )


def write_data_audit(
    train_cell: TrajectoryCell,
    validation_cell: TrajectoryCell,
    device: torch.device,
) -> dict[str, Any]:
    train_order = np.arange(
        len(
            train_cell.rows
        ),
        dtype=np.int64,
    )

    train_order, train_repairs = (
        repair_zero_only_microbatches(
            train_order,
            train_cell.lengths,
        )
    )

    validation_order = np.arange(
        len(
            validation_cell.rows
        ),
        dtype=np.int64,
    )

    validation_order, val_repairs = (
        repair_zero_only_microbatches(
            validation_order,
            validation_cell.lengths,
        )
    )

    sample_train_batch = (
        train_cell.make_batch(
            train_order[
                :PHYSICAL_MICROBATCH_SIZE
            ].tolist(),
            device,
        )
    )

    sample_val_batch = (
        validation_cell.make_batch(
            validation_order[
                :PHYSICAL_MICROBATCH_SIZE
            ].tolist(),
            device,
        )
    )

    audit = {
        "phase":
            (
                "4F-R3B Tier C v4 tuning "
                "data-loader audit"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "train_joint":
            train_cell.audit_record(),

        "val_joint":
            validation_cell.audit_record(),

        "train_zero_step_microbatch_repairs":
            train_repairs,

        "val_zero_step_microbatch_repairs":
            val_repairs,

        "sample_train_batch_shapes": {
            key:
                list(value.shape)
            for key, value
            in sample_train_batch.items()
        },

        "sample_val_batch_shapes": {
            key:
                list(value.shape)
            for key, value
            in sample_val_batch.items()
        },

        "sample_train_batch_finite":
            bool(
                torch.isfinite(
                    sample_train_batch[
                        "observations"
                    ]
                ).all()
            ),

        "sample_val_batch_finite":
            bool(
                torch.isfinite(
                    sample_val_batch[
                        "observations"
                    ]
                ).all()
            ),

        "privileged_directory_read":
            False,

        "model_initialized":
            False,

        "optimizer_initialized":
            False,

        "optimizer_steps_performed":
            0,

        "validation_selection_performed":
            False,

        "test_data_read":
            False,

        "data_loader_audit_passed":
            True,
    }

    atomic_write_json(
        OUTPUT_ROOT
        / "data_loader_audit.json",
        audit,
    )

    return audit


def write_global_summary(
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = []
    incomplete = []

    for record in configurations:
        result_path = (
            OUTPUT_ROOT
            / record[
                "model_id"
            ]
            / record[
                "configuration_id"
            ]
            / "tuning_result.json"
        )

        if result_path.exists():
            result = load_json(
                result_path
            )

            if (
                result.get(
                    "tuning_status"
                )
                == "completed"
            ):
                completed.append(
                    record[
                        "configuration_id"
                    ]
                )

                continue

        incomplete.append(
            record[
                "configuration_id"
            ]
        )

    summary = {
        "phase":
            (
                "4F-R3B Tier C v4 resumable "
                "development tuning status"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "configuration_count":
            len(configurations),

        "completed_configuration_count":
            len(completed),

        "incomplete_configuration_count":
            len(incomplete),

        "completed_configuration_ids":
            completed,

        "incomplete_configuration_ids":
            incomplete,

        "all_configurations_completed":
            len(incomplete) == 0,

        "training_cell":
            "train_joint",

        "validation_cell":
            "val_joint",

        "noise_fraction":
            NOISE_FRACTION,

        "checkpoint_selection_metric":
            (
                "val_joint noisy-target "
                "rollout field MSE"
            ),

        "privileged_directory_read":
            False,

        "test_data_read":
            False,

        "test_open_count":
            0,

        "final_fit_performed":
            False,

        "phase4fr3b_status":
            (
                "completed"
                if len(incomplete) == 0
                else "in_progress"
            ),
    }

    atomic_write_json(
        OUTPUT_ROOT
        / "phase4fr3b_tuning_status.json",
        summary,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--configuration-id",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--model-id",
        action="append",
        choices=MODEL_IDS,
        default=[],
    )

    parser.add_argument(
        "--max-configurations",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite-completed",
        action="store_true",
    )

    parser.add_argument(
        "--data-audit-only",
        action="store_true",
    )

    parser.add_argument(
        "--list-configurations",
        action="store_true",
    )

    arguments = parser.parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_protocol()

    device = resolve_device(
        arguments.device
    )

    require(
        device.type == "cuda",
        (
            "The frozen 43-configuration tuning "
            "execution must run on CUDA."
        ),
    )

    print(
        "CUDA device:",
        torch.cuda.get_device_name(
            device
        ),
    )

    configurations = (
        load_configurations()
    )

    if arguments.list_configurations:
        for record in configurations:
            print(
                record[
                    "registry_index"
                ],
                record[
                    "model_id"
                ],
                record[
                    "configuration_id"
                ],
            )

        return

    train_cell = TrajectoryCell(
        cell_id="train_joint",
        expected_trajectory_count=(
            EXPECTED_TRAIN_TRAJECTORIES
        ),
    )

    validation_cell = TrajectoryCell(
        cell_id="val_joint",
        expected_trajectory_count=(
            EXPECTED_VAL_TRAJECTORIES
        ),
    )

    require(
        train_cell.maximum_sequence_length
        == validation_cell.maximum_sequence_length
        or train_cell.maximum_sequence_length
        >= validation_cell.maximum_sequence_length,
        (
            "Validation maximum sequence length exceeds "
            "the training model interface."
        ),
    )

    audit = write_data_audit(
        train_cell,
        validation_cell,
        device,
    )

    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    if arguments.data_audit_only:
        print(
            "Data-loader audit passed; no model "
            "or optimizer was initialized."
        )

        return

    selected = configurations

    if arguments.configuration_id:
        requested = set(
            arguments.configuration_id
        )

        selected = [
            record
            for record
            in selected
            if record[
                "configuration_id"
            ]
            in requested
        ]

        found = {
            record[
                "configuration_id"
            ]
            for record
            in selected
        }

        require(
            found == requested,
            (
                "Unknown configuration IDs: "
                f"{sorted(requested - found)}"
            ),
        )

    if arguments.model_id:
        requested_models = set(
            arguments.model_id
        )

        selected = [
            record
            for record
            in selected
            if record[
                "model_id"
            ]
            in requested_models
        ]

    if (
        arguments.max_configurations
        is not None
    ):
        require(
            arguments.max_configurations
            >= 1,
            "--max-configurations must be positive.",
        )

        selected = selected[
            :arguments.max_configurations
        ]

    require(
        selected,
        "No tuning configurations were selected.",
    )

    print(
        f"Selected {len(selected)} configuration(s)."
    )

    for position, record in enumerate(
        selected,
        start=1,
    ):
        print(
            "=" * 88
        )

        print(
            f"[{position}/{len(selected)}] "
            f"{record['model_id']} | "
            f"{record['configuration_id']}"
        )

        result = run_configuration(
            record=record,
            train_cell=train_cell,
            validation_cell=(
                validation_cell
            ),
            device=device,
            overwrite_completed=(
                arguments.overwrite_completed
            ),
        )

        print(
            json.dumps(
                {
                    "model_id":
                        result[
                            "model_id"
                        ],

                    "configuration_id":
                        result[
                            "configuration_id"
                        ],

                    "best_epoch":
                        result[
                            "best_epoch"
                        ],

                    "best_validation_rollout_mse":
                        result[
                            "best_validation_rollout_mse"
                        ],

                    "epochs_completed":
                        result[
                            "epochs_completed"
                        ],

                    "stop_reason":
                        result[
                            "stop_reason"
                        ],

                    "tuning_status":
                        result[
                            "tuning_status"
                        ],
                },
                indent=2,
            )
        )

        write_global_summary(
            configurations
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_summary = write_global_summary(
        configurations
    )

    print(
        json.dumps(
            final_summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
