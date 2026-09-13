from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phase2b_operation_channel_model import (
    OperationChannelModel,
)

from phase2b_run_ocm_smoke_test import (
    TrajectoryRecord,
)

from phase3c_run_tier_b_smoke_tests import (
    AffineRolloutModel,
    OBSERVATION_DIMENSION,
    LATENT_STATE_COUNT,
    channel_row_sum_error,
    load_operation_vocabulary,
    parameter_count,
    state_is_finite,
)

from phase3d_tune_tier_b_models import (
    build_model,
)


PHASE3B_DATA_DIR = Path(
    "outputs/phase3b_tier_b_data"
)

VISIBLE_DIR = (
    PHASE3B_DATA_DIR / "visible"
)

PRIVILEGED_DIR = (
    PHASE3B_DATA_DIR / "privileged"
)

PHASE3E_DIR = Path(
    "outputs/phase3e_tier_b_final_fits"
)

PHASE3F_DIR = Path(
    "outputs/phase3f_tier_b_prediction"
)

RUN_OUTPUT_DIR = (
    PHASE3F_DIR / "runs"
)


TEST_CELLS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)

CELL_DESCRIPTIONS = {
    "test_iid_pairing":
        "Seen composition and carrier distributions with new pairings.",

    "test_composition":
        "Unseen composite transformations with seen carriers.",

    "test_carrier":
        "Seen transformations with unseen carriers.",

    "test_joint":
        "Unseen composite transformations and unseen carriers.",
}

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

ALL_MODELS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

CHECKPOINT_MODELS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

PAIRED_BASELINES = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
)

EXPECTED_FINAL_FIT_COUNTS = {
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
    "OCM": 25,
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

EXPECTED_CHECKPOINT_COUNT = 130
EXPECTED_EVALUATION_RUN_COUNT = 135
EXPECTED_CELL_RESULT_COUNT = 540
EXPECTED_TRAJECTORIES_PER_CELL = 1152

BATCH_SIZE = 128
MAXIMUM_SEQUENCE_LENGTH = 25


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


def summarize(
    values,
):
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
            float(
                values.mean()
            ),

        "std":
            (
                float(
                    values.std(
                        ddof=1
                    )
                )
                if len(values) > 1
                else 0.0
            ),

        "minimum":
            float(
                values.min()
            ),

        "maximum":
            float(
                values.max()
            ),
    }


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
            operation
        )
        for operation in parsed
    )


def validate_phase3e():
    summary = load_json(
        PHASE3E_DIR
        / "phase3e_summary.json"
    )

    registry = load_json(
        PHASE3E_DIR
        / "final_fit_registry.json"
    )

    if (
        summary[
            "phase3e_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 3E has not passed."
        )

    if (
        summary[
            "completed_run_count"
        ]
        != EXPECTED_CHECKPOINT_COUNT
    ):
        raise AssertionError(
            "Phase 3E checkpoint count changed."
        )

    if (
        summary[
            "all_runs_numerically_valid"
        ]
        is not True
    ):
        raise AssertionError(
            "Phase 3E contains invalid runs."
        )

    if (
        summary[
            "final_checkpoints_eligible_for_evaluation"
        ]
        is not True
    ):
        raise AssertionError(
            "Phase 3E checkpoints are not eligible."
        )

    if (
        summary[
            "test_cells_opened"
        ]
        is not False
    ):
        raise AssertionError(
            "Test cells were opened before Phase 3F."
        )

    if (
        summary[
            "test_metrics_computed"
        ]
        is not False
    ):
        raise AssertionError(
            "Test metrics were computed before Phase 3F."
        )

    if (
        summary[
            "privileged_arrays_opened"
        ]
        is not False
    ):
        raise AssertionError(
            "Privileged arrays were opened before Phase 3F."
        )

    if (
        registry[
            "registry_status"
        ]
        != "frozen"
    ):
        raise AssertionError(
            "Phase 3E registry is not frozen."
        )

    if (
        registry[
            "run_count"
        ]
        != EXPECTED_CHECKPOINT_COUNT
    ):
        raise AssertionError(
            "Phase 3E registry count changed."
        )

    counts = Counter(
        row[
            "model_id"
        ]
        for row in registry[
            "runs"
        ]
    )

    if dict(
        counts
    ) != EXPECTED_FINAL_FIT_COUNTS:
        raise AssertionError(
            "Phase 3E model counts changed: "
            f"{dict(counts)}"
        )

    return summary, registry


def create_evaluation_registry(
    final_fit_registry,
):
    rows = []

    for noise in FROZEN_NOISE_LEVELS:
        rows.append(
            {
                "run_id":
                    f"B0_noise_{noise_slug(noise)}",

                "model_id":
                    "B0",

                "configuration_id":
                    "persistence",

                "seed":
                    "deterministic",

                "noise_fraction":
                    noise,

                "checkpoint_required":
                    False,
            }
        )

    for run in final_fit_registry[
        "runs"
    ]:
        rows.append(
            {
                "run_id":
                    run[
                        "run_id"
                    ],

                "model_id":
                    run[
                        "model_id"
                    ],

                "configuration_id":
                    run[
                        "configuration_id"
                    ],

                "seed":
                    run[
                        "seed"
                    ],

                "noise_fraction":
                    float(
                        run[
                            "noise_fraction"
                        ]
                    ),

                "checkpoint_required":
                    True,
            }
        )

    if (
        len(
            rows
        )
        != EXPECTED_EVALUATION_RUN_COUNT
    ):
        raise AssertionError(
            "Expected 135 predictive evaluation runs."
        )

    counts = Counter(
        row[
            "model_id"
        ]
        for row in rows
    )

    if dict(
        counts
    ) != EXPECTED_EVALUATION_RUN_COUNTS:
        raise AssertionError(
            "Evaluation model counts changed: "
            f"{dict(counts)}"
        )

    run_ids = [
        row[
            "run_id"
        ]
        for row in rows
    ]

    if len(
        run_ids
    ) != len(
        set(
            run_ids
        )
    ):
        raise AssertionError(
            "Evaluation run IDs are not unique."
        )

    rows.sort(
        key=lambda row: (
            row[
                "noise_fraction"
            ],
            ALL_MODELS.index(
                row[
                    "model_id"
                ]
            ),
            str(
                row[
                    "seed"
                ]
            ),
        )
    )

    return rows


class TierBTestDataset(Dataset):

    def __init__(
        self,
        noisy_vectors,
        clean_vectors,
        records,
        operations_by_sequence,
    ):
        self.noisy_vectors = noisy_vectors
        self.clean_vectors = clean_vectors
        self.records = records
        self.operations_by_sequence = (
            operations_by_sequence
        )

    def __len__(self):
        return len(
            self.records
        )

    def __getitem__(
        self,
        index,
    ):
        record = self.records[
            index
        ]

        start = record.point_start
        end = (
            start
            + record.point_count
        )

        noisy = self.noisy_vectors[
            start:end
        ]

        clean = self.clean_vectors[
            start:end
        ]

        operations = (
            self.operations_by_sequence[
                record.sequence_index
            ]
        )

        if len(
            operations
        ) + 1 != len(
            noisy
        ):
            raise AssertionError(
                "Trajectory and operation lengths disagree."
            )

        return {
            "trajectory_index":
                int(
                    record.trajectory_index
                ),

            "sequence_index":
                int(
                    record.sequence_index
                ),

            "observations":
                torch.from_numpy(
                    noisy
                ).to(
                    torch.float32
                ),

            "operation_ids":
                torch.tensor(
                    operations,
                    dtype=torch.long,
                ),

            "clean_final_target":
                torch.from_numpy(
                    clean[-1]
                ).to(
                    torch.float32
                ),

            "sequence_length":
                len(
                    operations
                ),
        }


def collate_test_trajectories(
    batch,
):
    batch_size = len(
        batch
    )

    maximum_point_count = max(
        item[
            "observations"
        ].shape[0]
        for item in batch
    )

    maximum_step_count = (
        maximum_point_count
        - 1
    )

    observations = torch.zeros(
        (
            batch_size,
            maximum_point_count,
            OBSERVATION_DIMENSION,
        ),
        dtype=torch.float32,
    )

    operation_ids = torch.zeros(
        (
            batch_size,
            maximum_step_count,
        ),
        dtype=torch.long,
    )

    step_mask = torch.zeros(
        (
            batch_size,
            maximum_step_count,
        ),
        dtype=torch.bool,
    )

    lengths = torch.zeros(
        batch_size,
        dtype=torch.long,
    )

    sequence_lengths = torch.zeros(
        batch_size,
        dtype=torch.long,
    )

    clean_final_targets = torch.zeros(
        (
            batch_size,
            OBSERVATION_DIMENSION,
        ),
        dtype=torch.float32,
    )

    trajectory_indices = torch.zeros(
        batch_size,
        dtype=torch.long,
    )

    sequence_indices = torch.zeros(
        batch_size,
        dtype=torch.long,
    )

    for batch_index, item in enumerate(
        batch
    ):
        point_count = item[
            "observations"
        ].shape[0]

        step_count = item[
            "operation_ids"
        ].shape[0]

        observations[
            batch_index,
            :point_count,
            :,
        ] = item[
            "observations"
        ]

        operation_ids[
            batch_index,
            :step_count,
        ] = item[
            "operation_ids"
        ]

        step_mask[
            batch_index,
            :step_count,
        ] = True

        lengths[
            batch_index
        ] = point_count

        sequence_lengths[
            batch_index
        ] = item[
            "sequence_length"
        ]

        clean_final_targets[
            batch_index
        ] = item[
            "clean_final_target"
        ]

        trajectory_indices[
            batch_index
        ] = item[
            "trajectory_index"
        ]

        sequence_indices[
            batch_index
        ] = item[
            "sequence_index"
        ]

    return {
        "observations":
            observations,

        "operation_ids":
            operation_ids,

        "step_mask":
            step_mask,

        "lengths":
            lengths,

        "sequence_lengths":
            sequence_lengths,

        "clean_final_targets":
            clean_final_targets,

        "trajectory_indices":
            trajectory_indices,

        "sequence_indices":
            sequence_indices,
    }


def load_test_cell(
    cell_id,
    noise_fraction,
    operation_vocabulary,
):
    if cell_id not in TEST_CELLS:
        raise ValueError(
            f"Unsupported test cell: {cell_id}"
        )

    visible_path = (
        VISIBLE_DIR
        / f"{cell_id}_observations.npz"
    )

    clean_path = (
        PRIVILEGED_DIR
        / f"{cell_id}_clean_vectors.npy"
    )

    trajectory_path = (
        VISIBLE_DIR
        / f"{cell_id}_trajectory_index.csv"
    )

    sequence_path = (
        VISIBLE_DIR
        / f"{cell_id}_sequence_manifest.csv"
    )

    forbidden_paths = (
        PRIVILEGED_DIR
        / f"{cell_id}_state_ids.npy",

        PRIVILEGED_DIR
        / f"{cell_id}_trajectory_metadata.csv",
    )

    for forbidden_path in forbidden_paths:
        if not forbidden_path.exists():
            raise FileNotFoundError(
                forbidden_path
            )

    with np.load(
        visible_path
    ) as data:
        if set(
            data.files
        ) != {
            "noise_fractions",
            "vectors",
        }:
            raise AssertionError(
                f"{cell_id} visible file contains "
                "unexpected arrays."
            )

        noise_levels = np.asarray(
            data[
                "noise_fractions"
            ],
            dtype=np.float64,
        )

        matches = np.where(
            np.isclose(
                noise_levels,
                noise_fraction,
            )
        )[0]

        if len(
            matches
        ) != 1:
            raise AssertionError(
                f"Noise {noise_fraction} not found "
                f"for {cell_id}."
            )

        noisy_vectors = np.array(
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

    clean_vectors = np.asarray(
        np.load(
            clean_path
        ),
        dtype=np.float32,
    )

    if (
        noisy_vectors.shape
        != clean_vectors.shape
    ):
        raise AssertionError(
            f"{cell_id} noisy and clean shapes differ."
        )

    if (
        noisy_vectors.ndim != 2
        or noisy_vectors.shape[1]
        != OBSERVATION_DIMENSION
    ):
        raise AssertionError(
            f"{cell_id} observation shape changed: "
            f"{noisy_vectors.shape}"
        )

    if not (
        np.isfinite(
            noisy_vectors
        ).all()
        and np.isfinite(
            clean_vectors
        ).all()
    ):
        raise FloatingPointError(
            f"{cell_id} contains NaN or Inf."
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

        operation_names = parse_operations(
            row[
                "operation_sequence"
            ]
        )

        unknown = [
            operation
            for operation in operation_names
            if operation
            not in operation_vocabulary
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
                operation
            ]
            for operation
            in operation_names
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
                "Trajectory references an unknown sequence."
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
                "Trajectory point count changed."
            )

        records.append(
            record
        )

    if (
        len(
            records
        )
        != EXPECTED_TRAJECTORIES_PER_CELL
    ):
        raise AssertionError(
            f"{cell_id} trajectory count changed."
        )

    maximum_length = max(
        record.point_count - 1
        for record in records
    )

    if (
        maximum_length
        > MAXIMUM_SEQUENCE_LENGTH
    ):
        raise AssertionError(
            "Test sequence length exceeds the frozen maximum."
        )

    dataset = TierBTestDataset(
        noisy_vectors=noisy_vectors,
        clean_vectors=clean_vectors,
        records=records,
        operations_by_sequence=(
            operations_by_sequence
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_test_trajectories,
    )

    return {
        "cell_id":
            cell_id,

        "noise_fraction":
            float(
                noise_fraction
            ),

        "loader":
            loader,

        "trajectory_count":
            len(
                records
            ),

        "point_count":
            int(
                len(
                    noisy_vectors
                )
            ),

        "maximum_sequence_length":
            maximum_length,
    }


def validate_run_result(
    run,
):
    run_dir = (
        PHASE3E_DIR
        / "runs"
        / run[
            "run_id"
        ]
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

    result = load_json(
        result_path
    )

    if (
        result[
            "status"
        ]
        != "completed"
    ):
        raise AssertionError(
            f"{run['run_id']} is incomplete."
        )

    if (
        result[
            "numerically_valid"
        ]
        is not True
    ):
        raise AssertionError(
            f"{run['run_id']} is invalid."
        )

    if (
        result[
            "eligible_for_final_evaluation"
        ]
        is not True
    ):
        raise AssertionError(
            f"{run['run_id']} is not evaluation eligible."
        )

    if (
        result[
            "configuration_id"
        ]
        != run[
            "configuration_id"
        ]
    ):
        raise AssertionError(
            f"{run['run_id']} configuration mismatch."
        )

    if not np.isclose(
        float(
            result[
                "noise_fraction"
            ]
        ),
        float(
            run[
                "noise_fraction"
            ]
        ),
    ):
        raise AssertionError(
            f"{run['run_id']} noise mismatch."
        )

    if (
        str(
            result[
                "seed"
            ]
        )
        != str(
            run[
                "seed"
            ]
        )
    ):
        raise AssertionError(
            f"{run['run_id']} seed mismatch."
        )

    if (
        int(
            result[
                "batch_size"
            ]
        )
        != BATCH_SIZE
    ):
        raise AssertionError(
            f"{run['run_id']} batch-size mismatch."
        )

    if (
        result[
            "test_cells_opened"
        ]
        is not False
    ):
        raise AssertionError(
            f"{run['run_id']} opened test data during fitting."
        )

    if (
        result[
            "privileged_arrays_opened"
        ]
        is not False
    ):
        raise AssertionError(
            f"{run['run_id']} opened privileged arrays."
        )

    return (
        result,
        checkpoint_path,
    )


def load_checkpoint_model(
    run,
    operation_count,
    device,
):
    if run[
        "model_id"
    ] == "B0":
        return {
            "model":
                None,

            "temperature":
                None,

            "parameter_count":
                0,

            "checkpoint_path":
                None,

            "result_path":
                None,

            "maximum_channel_row_sum_error":
                None,
        }

    result, checkpoint_path = (
        validate_run_result(
            run
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        checkpoint.get(
            "eligible_for_final_evaluation"
        )
        is not True
    ):
        raise AssertionError(
            f"{run['run_id']} checkpoint is not eligible."
        )

    model_id = run[
        "model_id"
    ]

    state = checkpoint[
        "model_state_dict"
    ]

    temperature = None
    row_sum_error = None

    if model_id == "B1":
        if (
            "matrices" not in state
            or "biases" not in state
        ):
            raise AssertionError(
                "B1 checkpoint lacks affine parameters."
            )

        model = AffineRolloutModel(
            matrices=state[
                "matrices"
            ],
            biases=state[
                "biases"
            ],
        ).to(
            device
        )

        model.load_state_dict(
            state
        )

    elif model_id == "OCM":
        model = OperationChannelModel(
            observation_dim=(
                OBSERVATION_DIMENSION
            ),
            latent_state_count=(
                LATENT_STATE_COUNT
            ),
            operation_count=(
                operation_count
            ),
        ).to(
            device
        )

        model.load_state_dict(
            state
        )

        temperature = float(
            checkpoint.get(
                "temperature",
                result[
                    "configuration"
                ][
                    "temperature"
                ],
            )
        )

        row_sum_error = (
            channel_row_sum_error(
                model=model,
                operation_count=(
                    operation_count
                ),
                temperature=temperature,
            )
        )

        if (
            row_sum_error is not None
            and row_sum_error >= 1e-6
        ):
            raise AssertionError(
                f"{run['run_id']} channel rows are invalid."
            )

    else:
        configuration = result[
            "configuration"
        ]

        model = build_model(
            configuration=configuration,
            operation_count=operation_count,
            maximum_sequence_length=(
                MAXIMUM_SEQUENCE_LENGTH
            ),
        ).to(
            device
        )

        model.load_state_dict(
            state
        )

    model.eval()

    if not state_is_finite(
        model
    ):
        raise FloatingPointError(
            f"{run['run_id']} contains non-finite values."
        )

    observed_parameter_count = (
        parameter_count(
            model
        )
    )

    if model_id == "B1":
        observed_parameter_count = int(
            model.matrices.numel()
            + model.biases.numel()
        )

    if (
        observed_parameter_count
        != int(
            result[
                "parameter_count"
            ]
        )
    ):
        raise AssertionError(
            f"{run['run_id']} parameter count changed."
        )

    return {
        "model":
            model,

        "temperature":
            temperature,

        "parameter_count":
            observed_parameter_count,

        "checkpoint_path":
            str(
                checkpoint_path
            ),

        "result_path":
            str(
                PHASE3E_DIR
                / "runs"
                / run[
                    "run_id"
                ]
                / "result.json"
            ),

        "maximum_channel_row_sum_error":
            row_sum_error,
    }


def noisy_final_target(
    observations,
    lengths,
):
    batch_indices = torch.arange(
        observations.shape[0],
        device=observations.device,
    )

    return observations[
        batch_indices,
        lengths - 1,
        :,
    ]


@torch.no_grad()
def predict_batch(
    model_id,
    model,
    observations,
    operation_ids,
    step_mask,
    temperature,
):
    if model_id == "B0":
        return observations[
            :,
            0,
            :,
        ]

    if model_id == "OCM":
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

        return model.decode(
            current
        )

    return model.rollout(
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


@torch.no_grad()
def evaluate_one_cell(
    model_id,
    model,
    temperature,
    cell_data,
    device,
):
    if model is not None:
        model.eval()

    totals = {
        "model_noisy_sse": 0.0,
        "model_clean_sse": 0.0,
        "persistence_noisy_sse": 0.0,
        "persistence_clean_sse": 0.0,
        "value_count": 0,
        "trajectory_count": 0,
    }

    by_length = defaultdict(
        lambda: {
            "model_noisy_sum": 0.0,
            "model_clean_sum": 0.0,
            "persistence_noisy_sum": 0.0,
            "persistence_clean_sum": 0.0,
            "trajectory_count": 0,
        }
    )

    for batch in cell_data[
        "loader"
    ]:
        observations = batch[
            "observations"
        ].to(
            device,
            non_blocking=True,
        )

        operation_ids = batch[
            "operation_ids"
        ].to(
            device,
            non_blocking=True,
        )

        step_mask = batch[
            "step_mask"
        ].to(
            device,
            non_blocking=True,
        )

        lengths = batch[
            "lengths"
        ].to(
            device,
            non_blocking=True,
        )

        sequence_lengths = batch[
            "sequence_lengths"
        ].to(
            device,
            non_blocking=True,
        )

        clean_targets = batch[
            "clean_final_targets"
        ].to(
            device,
            non_blocking=True,
        )

        noisy_targets = noisy_final_target(
            observations,
            lengths,
        )

        persistence = observations[
            :,
            0,
            :,
        ]

        prediction = predict_batch(
            model_id=model_id,
            model=model,
            observations=observations,
            operation_ids=operation_ids,
            step_mask=step_mask,
            temperature=temperature,
        )

        if not torch.isfinite(
            prediction
        ).all():
            raise FloatingPointError(
                f"{model_id} produced non-finite predictions."
            )

        model_noisy_error = (
            prediction
            - noisy_targets
        ).pow(
            2
        )

        model_clean_error = (
            prediction
            - clean_targets
        ).pow(
            2
        )

        persistence_noisy_error = (
            persistence
            - noisy_targets
        ).pow(
            2
        )

        persistence_clean_error = (
            persistence
            - clean_targets
        ).pow(
            2
        )

        totals[
            "model_noisy_sse"
        ] += float(
            model_noisy_error.sum().item()
        )

        totals[
            "model_clean_sse"
        ] += float(
            model_clean_error.sum().item()
        )

        totals[
            "persistence_noisy_sse"
        ] += float(
            persistence_noisy_error.sum().item()
        )

        totals[
            "persistence_clean_sse"
        ] += float(
            persistence_clean_error.sum().item()
        )

        totals[
            "value_count"
        ] += int(
            prediction.numel()
        )

        totals[
            "trajectory_count"
        ] += int(
            prediction.shape[0]
        )

        per_example_model_noisy = (
            model_noisy_error.mean(
                dim=1
            )
        )

        per_example_model_clean = (
            model_clean_error.mean(
                dim=1
            )
        )

        per_example_persistence_noisy = (
            persistence_noisy_error.mean(
                dim=1
            )
        )

        per_example_persistence_clean = (
            persistence_clean_error.mean(
                dim=1
            )
        )

        unique_lengths = torch.unique(
            sequence_lengths
        )

        for length_tensor in unique_lengths:
            length = int(
                length_tensor.item()
            )

            mask = (
                sequence_lengths
                == length
            )

            item = by_length[
                length
            ]

            item[
                "model_noisy_sum"
            ] += float(
                per_example_model_noisy[
                    mask
                ].sum().item()
            )

            item[
                "model_clean_sum"
            ] += float(
                per_example_model_clean[
                    mask
                ].sum().item()
            )

            item[
                "persistence_noisy_sum"
            ] += float(
                per_example_persistence_noisy[
                    mask
                ].sum().item()
            )

            item[
                "persistence_clean_sum"
            ] += float(
                per_example_persistence_clean[
                    mask
                ].sum().item()
            )

            item[
                "trajectory_count"
            ] += int(
                mask.sum().item()
            )

    if (
        totals[
            "trajectory_count"
        ]
        != EXPECTED_TRAJECTORIES_PER_CELL
    ):
        raise AssertionError(
            "Evaluated trajectory count changed."
        )

    value_count = totals[
        "value_count"
    ]

    result = {
        "trajectory_count":
            totals[
                "trajectory_count"
            ],

        "value_count":
            value_count,

        "model_noisy_target_mse":
            totals[
                "model_noisy_sse"
            ]
            / value_count,

        "model_clean_target_mse":
            totals[
                "model_clean_sse"
            ]
            / value_count,

        "persistence_noisy_target_mse":
            totals[
                "persistence_noisy_sse"
            ]
            / value_count,

        "persistence_clean_target_mse":
            totals[
                "persistence_clean_sse"
            ]
            / value_count,
    }

    length_rows = []

    trajectory_total = 0

    for length in sorted(
        by_length
    ):
        item = by_length[
            length
        ]

        count = item[
            "trajectory_count"
        ]

        trajectory_total += count

        length_rows.append(
            {
                "sequence_length":
                    length,

                "trajectory_count":
                    count,

                "model_noisy_target_mse":
                    item[
                        "model_noisy_sum"
                    ]
                    / count,

                "model_clean_target_mse":
                    item[
                        "model_clean_sum"
                    ]
                    / count,

                "persistence_noisy_target_mse":
                    item[
                        "persistence_noisy_sum"
                    ]
                    / count,

                "persistence_clean_target_mse":
                    item[
                        "persistence_clean_sum"
                    ]
                    / count,
            }
        )

    if (
        trajectory_total
        != EXPECTED_TRAJECTORIES_PER_CELL
    ):
        raise AssertionError(
            "Sequence-length totals changed."
        )

    return result, length_rows


def evaluate_run(
    run,
    loaded_cells,
    operation_count,
    device,
):
    loaded_model = load_checkpoint_model(
        run=run,
        operation_count=operation_count,
        device=device,
    )

    model = loaded_model[
        "model"
    ]

    temperature = loaded_model[
        "temperature"
    ]

    cell_results = {}
    sequence_length_results = []

    for cell_id in TEST_CELLS:
        metrics, length_rows = (
            evaluate_one_cell(
                model_id=run[
                    "model_id"
                ],
                model=model,
                temperature=temperature,
                cell_data=loaded_cells[
                    cell_id
                ],
                device=device,
            )
        )

        cell_results[
            cell_id
        ] = {
            "cell_description":
                CELL_DESCRIPTIONS[
                    cell_id
                ],

            **metrics,
        }

        for row in length_rows:
            sequence_length_results.append(
                {
                    "cell_id":
                        cell_id,

                    **row,
                }
            )

        print(
            f"{run['run_id']} | "
            f"{cell_id} | "
            f"noisy={metrics['model_noisy_target_mse']:.6f} | "
            f"clean={metrics['model_clean_target_mse']:.6f}"
        )

    result = {
        "phase":
            "3F final Tier B predictive evaluation",

        "run_id":
            run[
                "run_id"
            ],

        "model_id":
            run[
                "model_id"
            ],

        "configuration_id":
            run[
                "configuration_id"
            ],

        "seed":
            run[
                "seed"
            ],

        "noise_fraction":
            float(
                run[
                    "noise_fraction"
                ]
            ),

        "parameter_count":
            loaded_model[
                "parameter_count"
            ],

        "temperature":
            temperature,

        "maximum_channel_row_sum_error":
            loaded_model[
                "maximum_channel_row_sum_error"
            ],

        "checkpoint_path":
            loaded_model[
                "checkpoint_path"
            ],

        "source_result_path":
            loaded_model[
                "result_path"
            ],

        "cell_results":
            cell_results,

        "sequence_length_results":
            sequence_length_results,

        "training_performed":
            False,

        "checkpoint_selection_reopened":
            False,

        "checkpoint_modified":
            False,

        "test_cells_opened":
            True,

        "privileged_clean_targets_read":
            True,

        "privileged_state_ids_read":
            False,

        "privileged_trajectory_metadata_read":
            False,

        "clean_targets_used_for_evaluation_only":
            True,

        "test_metrics_used_for_selection":
            False,

        "structural_labels_read":
            False,

        "status":
            "completed",
    }

    if model is not None:
        del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def flatten_results(
    evaluation_registry,
):
    cell_rows = []
    length_rows = []

    for run in evaluation_registry:
        result_path = (
            RUN_OUTPUT_DIR
            / run[
                "run_id"
            ]
            / "result.json"
        )

        if not result_path.exists():
            return None

        result = load_json(
            result_path
        )

        if (
            result.get(
                "status"
            )
            != "completed"
        ):
            return None

        for cell_id in TEST_CELLS:
            metrics = result[
                "cell_results"
            ][
                cell_id
            ]

            cell_rows.append(
                {
                    "run_id":
                        result[
                            "run_id"
                        ],

                    "model_id":
                        result[
                            "model_id"
                        ],

                    "configuration_id":
                        result[
                            "configuration_id"
                        ],

                    "seed":
                        result[
                            "seed"
                        ],

                    "noise_fraction":
                        result[
                            "noise_fraction"
                        ],

                    "parameter_count":
                        result[
                            "parameter_count"
                        ],

                    "cell_id":
                        cell_id,

                    "trajectory_count":
                        metrics[
                            "trajectory_count"
                        ],

                    "model_noisy_target_mse":
                        metrics[
                            "model_noisy_target_mse"
                        ],

                    "model_clean_target_mse":
                        metrics[
                            "model_clean_target_mse"
                        ],

                    "persistence_noisy_target_mse":
                        metrics[
                            "persistence_noisy_target_mse"
                        ],

                    "persistence_clean_target_mse":
                        metrics[
                            "persistence_clean_target_mse"
                        ],
                }
            )

        for row in result[
            "sequence_length_results"
        ]:
            length_rows.append(
                {
                    "run_id":
                        result[
                            "run_id"
                        ],

                    "model_id":
                        result[
                            "model_id"
                        ],

                    "configuration_id":
                        result[
                            "configuration_id"
                        ],

                    "seed":
                        result[
                            "seed"
                        ],

                    "noise_fraction":
                        result[
                            "noise_fraction"
                        ],

                    "cell_id":
                        row[
                            "cell_id"
                        ],

                    "sequence_length":
                        row[
                            "sequence_length"
                        ],

                    "trajectory_count":
                        row[
                            "trajectory_count"
                        ],

                    "model_noisy_target_mse":
                        row[
                            "model_noisy_target_mse"
                        ],

                    "model_clean_target_mse":
                        row[
                            "model_clean_target_mse"
                        ],

                    "persistence_noisy_target_mse":
                        row[
                            "persistence_noisy_target_mse"
                        ],

                    "persistence_clean_target_mse":
                        row[
                            "persistence_clean_target_mse"
                        ],
                }
            )

    if (
        len(
            cell_rows
        )
        != EXPECTED_CELL_RESULT_COUNT
    ):
        raise AssertionError(
            "Expected 540 cell-level result rows."
        )

    return cell_rows, length_rows


def aggregate_model_cell_noise(
    cell_rows,
):
    grouped = defaultdict(
        list
    )

    for row in cell_rows:
        key = (
            row[
                "model_id"
            ],
            row[
                "cell_id"
            ],
            float(
                row[
                    "noise_fraction"
                ]
            ),
        )

        grouped[
            key
        ].append(
            row
        )

    output = []

    for (
        model_id,
        cell_id,
        noise,
    ), group in sorted(
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

        if (
            len(
                group
            )
            != expected_count
        ):
            raise AssertionError(
                f"{model_id}/{cell_id}/{noise} "
                f"has {len(group)} rows."
            )

        noisy = summarize(
            [
                row[
                    "model_noisy_target_mse"
                ]
                for row in group
            ]
        )

        clean = summarize(
            [
                row[
                    "model_clean_target_mse"
                ]
                for row in group
            ]
        )

        persistence_noisy = summarize(
            [
                row[
                    "persistence_noisy_target_mse"
                ]
                for row in group
            ]
        )

        persistence_clean = summarize(
            [
                row[
                    "persistence_clean_target_mse"
                ]
                for row in group
            ]
        )

        parameter_counts = {
            int(
                row[
                    "parameter_count"
                ]
            )
            for row in group
        }

        if len(
            parameter_counts
        ) != 1:
            raise AssertionError(
                "Parameter count changed across seeds."
            )

        output.append(
            {
                "model_id":
                    model_id,

                "cell_id":
                    cell_id,

                "noise_fraction":
                    noise,

                "fit_count":
                    len(
                        group
                    ),

                "parameter_count":
                    next(
                        iter(
                            parameter_counts
                        )
                    ),

                "noisy_mse_mean":
                    noisy[
                        "mean"
                    ],

                "noisy_mse_std":
                    noisy[
                        "std"
                    ],

                "noisy_mse_minimum":
                    noisy[
                        "minimum"
                    ],

                "noisy_mse_maximum":
                    noisy[
                        "maximum"
                    ],

                "clean_mse_mean":
                    clean[
                        "mean"
                    ],

                "clean_mse_std":
                    clean[
                        "std"
                    ],

                "clean_mse_minimum":
                    clean[
                        "minimum"
                    ],

                "clean_mse_maximum":
                    clean[
                        "maximum"
                    ],

                "persistence_noisy_mse_mean":
                    persistence_noisy[
                        "mean"
                    ],

                "persistence_clean_mse_mean":
                    persistence_clean[
                        "mean"
                    ],
            }
        )

    if len(
        output
    ) != (
        len(
            ALL_MODELS
        )
        * len(
            TEST_CELLS
        )
        * len(
            FROZEN_NOISE_LEVELS
        )
    ):
        raise AssertionError(
            "Model/cell/noise summary count changed."
        )

    return output


def create_rankings(
    summary_rows,
):
    output = []

    for cell_id in TEST_CELLS:
        for noise in FROZEN_NOISE_LEVELS:
            group = [
                row
                for row in summary_rows
                if (
                    row[
                        "cell_id"
                    ]
                    == cell_id
                    and np.isclose(
                        row[
                            "noise_fraction"
                        ],
                        noise,
                    )
                )
            ]

            noisy_sorted = sorted(
                group,
                key=lambda row: (
                    row[
                        "noisy_mse_mean"
                    ],
                    row[
                        "model_id"
                    ],
                ),
            )

            clean_sorted = sorted(
                group,
                key=lambda row: (
                    row[
                        "clean_mse_mean"
                    ],
                    row[
                        "model_id"
                    ],
                ),
            )

            noisy_ranks = {
                row[
                    "model_id"
                ]:
                    rank
                for rank, row in enumerate(
                    noisy_sorted,
                    start=1,
                )
            }

            clean_ranks = {
                row[
                    "model_id"
                ]:
                    rank
                for rank, row in enumerate(
                    clean_sorted,
                    start=1,
                )
            }

            for row in group:
                output.append(
                    {
                        "cell_id":
                            cell_id,

                        "noise_fraction":
                            noise,

                        "model_id":
                            row[
                                "model_id"
                            ],

                        "noisy_target_rank":
                            noisy_ranks[
                                row[
                                    "model_id"
                                ]
                            ],

                        "clean_target_rank":
                            clean_ranks[
                                row[
                                    "model_id"
                                ]
                            ],

                        "noisy_mse_mean":
                            row[
                                "noisy_mse_mean"
                            ],

                        "clean_mse_mean":
                            row[
                                "clean_mse_mean"
                            ],

                        "rank_tiebreak":
                            "model_id",
                    }
                )

    return output


def create_generalization_gaps(
    summary_rows,
):
    lookup = {
        (
            row[
                "model_id"
            ],
            float(
                row[
                    "noise_fraction"
                ]
            ),
            row[
                "cell_id"
            ],
        ):
            row
        for row in summary_rows
    }

    output = []

    for model_id in ALL_MODELS:
        for noise in FROZEN_NOISE_LEVELS:
            iid = lookup[
                (
                    model_id,
                    noise,
                    "test_iid_pairing",
                )
            ]

            composition = lookup[
                (
                    model_id,
                    noise,
                    "test_composition",
                )
            ]

            carrier = lookup[
                (
                    model_id,
                    noise,
                    "test_carrier",
                )
            ]

            joint = lookup[
                (
                    model_id,
                    noise,
                    "test_joint",
                )
            ]

            noisy_iid = iid[
                "noisy_mse_mean"
            ]

            noisy_composition = composition[
                "noisy_mse_mean"
            ]

            noisy_carrier = carrier[
                "noisy_mse_mean"
            ]

            noisy_joint = joint[
                "noisy_mse_mean"
            ]

            clean_iid = iid[
                "clean_mse_mean"
            ]

            clean_composition = composition[
                "clean_mse_mean"
            ]

            clean_carrier = carrier[
                "clean_mse_mean"
            ]

            clean_joint = joint[
                "clean_mse_mean"
            ]

            output.append(
                {
                    "model_id":
                        model_id,

                    "noise_fraction":
                        noise,

                    "iid_noisy_mse":
                        noisy_iid,

                    "composition_noisy_mse":
                        noisy_composition,

                    "carrier_noisy_mse":
                        noisy_carrier,

                    "joint_noisy_mse":
                        noisy_joint,

                    "composition_noisy_gap":
                        noisy_composition
                        - noisy_iid,

                    "carrier_noisy_gap":
                        noisy_carrier
                        - noisy_iid,

                    "joint_noisy_gap":
                        noisy_joint
                        - noisy_iid,

                    "composition_carrier_noisy_interaction":
                        (
                            noisy_joint
                            - noisy_composition
                            - noisy_carrier
                            + noisy_iid
                        ),

                    "iid_clean_mse":
                        clean_iid,

                    "composition_clean_mse":
                        clean_composition,

                    "carrier_clean_mse":
                        clean_carrier,

                    "joint_clean_mse":
                        clean_joint,

                    "composition_clean_gap":
                        clean_composition
                        - clean_iid,

                    "carrier_clean_gap":
                        clean_carrier
                        - clean_iid,

                    "joint_clean_gap":
                        clean_joint
                        - clean_iid,

                    "composition_carrier_clean_interaction":
                        (
                            clean_joint
                            - clean_composition
                            - clean_carrier
                            + clean_iid
                        ),
                }
            )

    return output


def build_seed_lookup(
    cell_rows,
    model_id,
    cell_id,
    noise,
):
    rows = [
        row
        for row in cell_rows
        if (
            row[
                "model_id"
            ]
            == model_id
            and row[
                "cell_id"
            ]
            == cell_id
            and np.isclose(
                float(
                    row[
                        "noise_fraction"
                    ]
                ),
                noise,
            )
        )
    ]

    if model_id in {
        "B0",
        "B1",
    }:
        if len(
            rows
        ) != 1:
            raise AssertionError(
                f"{model_id} should have one deterministic row."
            )

        return {
            seed:
                rows[0]
            for seed in FROZEN_SEEDS
        }

    lookup = {
        int(
            row[
                "seed"
            ]
        ):
            row
        for row in rows
    }

    if set(
        lookup
    ) != set(
        FROZEN_SEEDS
    ):
        raise AssertionError(
            f"{model_id} seed coverage changed."
        )

    return lookup


def create_paired_ocm_comparisons(
    cell_rows,
):
    output = []

    for baseline_id in PAIRED_BASELINES:
        for cell_id in TEST_CELLS:
            for noise in FROZEN_NOISE_LEVELS:
                ocm_lookup = build_seed_lookup(
                    cell_rows=cell_rows,
                    model_id="OCM",
                    cell_id=cell_id,
                    noise=noise,
                )

                baseline_lookup = (
                    build_seed_lookup(
                        cell_rows=cell_rows,
                        model_id=baseline_id,
                        cell_id=cell_id,
                        noise=noise,
                    )
                )

                noisy_differences = []
                clean_differences = []

                noisy_ocm_wins = 0
                clean_ocm_wins = 0

                tolerance = 1e-12

                for seed in FROZEN_SEEDS:
                    ocm = ocm_lookup[
                        seed
                    ]

                    baseline = baseline_lookup[
                        seed
                    ]

                    noisy_difference = (
                        baseline[
                            "model_noisy_target_mse"
                        ]
                        - ocm[
                            "model_noisy_target_mse"
                        ]
                    )

                    clean_difference = (
                        baseline[
                            "model_clean_target_mse"
                        ]
                        - ocm[
                            "model_clean_target_mse"
                        ]
                    )

                    noisy_differences.append(
                        noisy_difference
                    )

                    clean_differences.append(
                        clean_difference
                    )

                    if (
                        noisy_difference
                        > tolerance
                    ):
                        noisy_ocm_wins += 1

                    if (
                        clean_difference
                        > tolerance
                    ):
                        clean_ocm_wins += 1

                noisy_summary = summarize(
                    noisy_differences
                )

                clean_summary = summarize(
                    clean_differences
                )

                output.append(
                    {
                        "baseline_id":
                            baseline_id,

                        "cell_id":
                            cell_id,

                        "noise_fraction":
                            noise,

                        "paired_seed_count":
                            len(
                                FROZEN_SEEDS
                            ),

                        "ocm_noisy_advantage_mean":
                            noisy_summary[
                                "mean"
                            ],

                        "ocm_noisy_advantage_std":
                            noisy_summary[
                                "std"
                            ],

                        "ocm_noisy_advantage_minimum":
                            noisy_summary[
                                "minimum"
                            ],

                        "ocm_noisy_advantage_maximum":
                            noisy_summary[
                                "maximum"
                            ],

                        "ocm_clean_advantage_mean":
                            clean_summary[
                                "mean"
                            ],

                        "ocm_clean_advantage_std":
                            clean_summary[
                                "std"
                            ],

                        "ocm_clean_advantage_minimum":
                            clean_summary[
                                "minimum"
                            ],

                        "ocm_clean_advantage_maximum":
                            clean_summary[
                                "maximum"
                            ],

                        "ocm_noisy_win_count":
                            noisy_ocm_wins,

                        "baseline_noisy_win_count":
                            len(
                                FROZEN_SEEDS
                            )
                            - noisy_ocm_wins,

                        "ocm_clean_win_count":
                            clean_ocm_wins,

                        "baseline_clean_win_count":
                            len(
                                FROZEN_SEEDS
                            )
                            - clean_ocm_wins,

                        "positive_advantage_means":
                            "OCM has lower MSE",
                    }
                )

    return output


def aggregate_sequence_lengths(
    length_rows,
):
    grouped = defaultdict(
        list
    )

    for row in length_rows:
        key = (
            row[
                "model_id"
            ],
            row[
                "cell_id"
            ],
            float(
                row[
                    "noise_fraction"
                ]
            ),
            int(
                row[
                    "sequence_length"
                ]
            ),
        )

        grouped[
            key
        ].append(
            row
        )

    output = []

    for (
        model_id,
        cell_id,
        noise,
        sequence_length,
    ), group in sorted(
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

        if (
            len(
                group
            )
            != expected_count
        ):
            raise AssertionError(
                "Sequence-length seed count changed."
            )

        noisy = summarize(
            [
                row[
                    "model_noisy_target_mse"
                ]
                for row in group
            ]
        )

        clean = summarize(
            [
                row[
                    "model_clean_target_mse"
                ]
                for row in group
            ]
        )

        trajectory_counts = {
            int(
                row[
                    "trajectory_count"
                ]
            )
            for row in group
        }

        if len(
            trajectory_counts
        ) != 1:
            raise AssertionError(
                "Trajectory count differs across seeds."
            )

        output.append(
            {
                "model_id":
                    model_id,

                "cell_id":
                    cell_id,

                "noise_fraction":
                    noise,

                "sequence_length":
                    sequence_length,

                "fit_count":
                    len(
                        group
                    ),

                "trajectory_count_per_fit":
                    next(
                        iter(
                            trajectory_counts
                        )
                    ),

                "noisy_mse_mean":
                    noisy[
                        "mean"
                    ],

                "noisy_mse_std":
                    noisy[
                        "std"
                    ],

                "clean_mse_mean":
                    clean[
                        "mean"
                    ],

                "clean_mse_std":
                    clean[
                        "std"
                    ],
            }
        )

    return output


def create_best_model_summary(
    summary_rows,
):
    output = {}

    for cell_id in TEST_CELLS:
        output[
            cell_id
        ] = {}

        for noise in FROZEN_NOISE_LEVELS:
            candidates = [
                row
                for row in summary_rows
                if (
                    row[
                        "cell_id"
                    ]
                    == cell_id
                    and np.isclose(
                        row[
                            "noise_fraction"
                        ],
                        noise,
                    )
                )
            ]

            best_noisy = min(
                candidates,
                key=lambda row: (
                    row[
                        "noisy_mse_mean"
                    ],
                    row[
                        "model_id"
                    ],
                ),
            )

            best_clean = min(
                candidates,
                key=lambda row: (
                    row[
                        "clean_mse_mean"
                    ],
                    row[
                        "model_id"
                    ],
                ),
            )

            output[
                cell_id
            ][
                str(
                    noise
                )
            ] = {
                "best_noisy_target_model":
                    best_noisy[
                        "model_id"
                    ],

                "best_noisy_target_mse":
                    best_noisy[
                        "noisy_mse_mean"
                    ],

                "best_clean_target_model":
                    best_clean[
                        "model_id"
                    ],

                "best_clean_target_mse":
                    best_clean[
                        "clean_mse_mean"
                    ],
            }

    return output


def finalize_phase(
    evaluation_registry,
):
    flattened = flatten_results(
        evaluation_registry
    )

    if flattened is None:
        return None

    cell_rows, length_rows = (
        flattened
    )

    model_summary = (
        aggregate_model_cell_noise(
            cell_rows
        )
    )

    rankings = create_rankings(
        model_summary
    )

    gaps = create_generalization_gaps(
        model_summary
    )

    paired = (
        create_paired_ocm_comparisons(
            cell_rows
        )
    )

    length_summary = (
        aggregate_sequence_lengths(
            length_rows
        )
    )

    best_models = (
        create_best_model_summary(
            model_summary
        )
    )

    write_csv(
        PHASE3F_DIR
        / "all_predictive_run_results.csv",
        cell_rows,
    )

    write_csv(
        PHASE3F_DIR
        / "all_sequence_length_results.csv",
        length_rows,
    )

    write_csv(
        PHASE3F_DIR
        / "model_cell_noise_summary.csv",
        model_summary,
    )

    write_csv(
        PHASE3F_DIR
        / "rankings_by_cell_noise.csv",
        rankings,
    )

    write_csv(
        PHASE3F_DIR
        / "generalization_gap_summary.csv",
        gaps,
    )

    write_csv(
        PHASE3F_DIR
        / "paired_ocm_comparisons.csv",
        paired,
    )

    write_csv(
        PHASE3F_DIR
        / "sequence_length_summary.csv",
        length_summary,
    )

    write_json(
        PHASE3F_DIR
        / "best_models_by_cell_and_noise.json",
        best_models,
    )

    summary = {
        "phase":
            "3F final Tier B predictive evaluation",

        "source_checkpoint_count":
            EXPECTED_CHECKPOINT_COUNT,

        "persistence_condition_count":
            len(
                FROZEN_NOISE_LEVELS
            ),

        "evaluation_run_count":
            len(
                evaluation_registry
            ),

        "cell_count":
            len(
                TEST_CELLS
            ),

        "cell_result_count":
            len(
                cell_rows
            ),

        "model_cell_noise_summary_count":
            len(
                model_summary
            ),

        "paired_comparison_count":
            len(
                paired
            ),

        "test_cells":
            list(
                TEST_CELLS
            ),

        "test_cells_opened_after_checkpoint_freeze":
            True,

        "training_performed":
            False,

        "checkpoint_selection_reopened":
            False,

        "checkpoints_modified":
            False,

        "test_metrics_used_for_selection":
            False,

        "privileged_clean_targets_read":
            True,

        "privileged_clean_targets_use":
            "final evaluation only",

        "privileged_state_ids_read":
            False,

        "privileged_trajectory_metadata_read":
            False,

        "structural_labels_read":
            False,

        "phase2_outputs_modified":
            False,

        "phase3e_outputs_modified":
            False,

        "all_results_numerically_valid":
            bool(
                all(
                    np.isfinite(
                        row[
                            "model_noisy_target_mse"
                        ]
                    )
                    and np.isfinite(
                        row[
                            "model_clean_target_mse"
                        ]
                    )
                    for row in cell_rows
                )
            ),

        "best_models_by_cell_and_noise":
            best_models,

        "phase3f_status":
            "passed",
    }

    write_json(
        PHASE3F_DIR
        / "phase3f_summary.json",
        summary,
    )

    return summary


def main():
    arguments = parse_arguments()

    PHASE3F_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, final_fit_registry = (
        validate_phase3e()
    )

    evaluation_registry = (
        create_evaluation_registry(
            final_fit_registry
        )
    )

    write_json(
        PHASE3F_DIR
        / "evaluation_registry.json",
        {
            "phase":
                "3F evaluation registry",

            "test_cells":
                list(
                    TEST_CELLS
                ),

            "evaluation_run_count":
                len(
                    evaluation_registry
                ),

            "cell_result_count":
                (
                    len(
                        evaluation_registry
                    )
                    * len(
                        TEST_CELLS
                    )
                ),

            "runs":
                evaluation_registry,

            "registry_status":
                "frozen",
        },
    )

    requested_runs = [
        run
        for run in evaluation_registry
        if (
            (
                arguments.run_id
                == "all"
                or run[
                    "run_id"
                ]
                == arguments.run_id
            )
            and (
                arguments.model_id
                == "all"
                or run[
                    "model_id"
                ]
                == arguments.model_id
            )
        )
    ]

    if not requested_runs:
        raise ValueError(
            "No predictive evaluation runs match "
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

    operation_count = len(
        operation_vocabulary
    )

    current_noise = None
    loaded_cells = None

    for index, run in enumerate(
        requested_runs,
        start=1,
    ):
        run_dir = (
            RUN_OUTPUT_DIR
            / run[
                "run_id"
            ]
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_path = (
            run_dir
            / "result.json"
        )

        if (
            result_path.exists()
            and not arguments.force
        ):
            existing = load_json(
                result_path
            )

            if (
                existing.get(
                    "status"
                )
                == "completed"
            ):
                print(
                    f"[{index}/{len(requested_runs)}] "
                    f"{run['run_id']} already completed."
                )
                continue

        noise = float(
            run[
                "noise_fraction"
            ]
        )

        if (
            current_noise is None
            or not np.isclose(
                current_noise,
                noise,
            )
        ):
            print()
            print(
                "Opening frozen test cells for "
                f"noise={noise}"
            )

            loaded_cells = {
                cell_id:
                    load_test_cell(
                        cell_id=cell_id,
                        noise_fraction=noise,
                        operation_vocabulary=(
                            operation_vocabulary
                        ),
                    )
                for cell_id in TEST_CELLS
            }

            current_noise = noise

        print()
        print(
            f"[{index}/{len(requested_runs)}] "
            f"Evaluating {run['run_id']}"
        )

        result = evaluate_run(
            run=run,
            loaded_cells=loaded_cells,
            operation_count=operation_count,
            device=device,
        )

        write_json(
            result_path,
            result,
        )

    summary = finalize_phase(
        evaluation_registry
    )

    if summary is None:
        completed_count = sum(
            (
                RUN_OUTPUT_DIR
                / run[
                    "run_id"
                ]
                / "result.json"
            ).exists()
            for run in evaluation_registry
        )

        print()
        print(
            "Phase 3F progress: "
            f"{completed_count}/"
            f"{EXPECTED_EVALUATION_RUN_COUNT} "
            "run results present."
        )

    else:
        print()
        print(
            "Phase 3F predictive evaluation completed."
        )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

        print(
            f"Outputs written to: {PHASE3F_DIR}"
        )


if __name__ == "__main__":
    main()
