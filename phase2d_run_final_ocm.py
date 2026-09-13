from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phase2b_operation_channel_model import (
    OperationChannelModel,
)

from phase2b_run_ocm_smoke_test import (
    ContinuousTrajectoryDataset,
    TrajectoryRecord,
    collate_trajectories,
    set_random_seed,
    load_operation_vocabulary,
    kmeans_training_centers,
    maximum_channel_row_sum_error,
)

from phase2c_run_ocm_hyperparameter_sweep import (
    BATCH_SIZE,
    MAXIMUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    INITIAL_TEMPERATURE,
    temperature_for_epoch,
    train_one_epoch,
)


PHASE1C_DIR = Path(
    "outputs/phase1c_continuous_observations"
)

PHASE2A_DIR = Path(
    "outputs/phase2a_learning_protocol"
)

PHASE2C_DIR = Path(
    "outputs/phase2c_hyperparameter_sweep"
)

OUTPUT_DIR = Path(
    "outputs/phase2d_final_ocm"
)


EXPECTED_CONFIGURATION_ID = "c05"

EXPECTED_CONFIGURATION = {
    "learning_rate": 0.001,
    "lambda_transition": 0.1,
    "lambda_rollout": 1.0,
    "lambda_determinism": 0.001,
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

EXPECTED_SEQUENCE_COUNTS = {
    "train": 1376,
    "val": 144,
    "test": 144,
}

EXPECTED_TRAJECTORY_COUNTS = {
    "train": 11008,
    "val": 1152,
    "test": 1152,
}

OBSERVATION_DIM = 12
LATENT_STATE_COUNT = 8


@dataclass
class EvaluationExample:
    observations: torch.Tensor
    operations: torch.Tensor
    clean_final_target: torch.Tensor


class EvaluationTrajectoryDataset(Dataset):
    """
    Evaluation dataset containing noisy model inputs and a clean
    final target.

    No symbolic state IDs or structural labels are used.
    """

    def __init__(
        self,
        noisy_vectors: np.ndarray,
        clean_vectors: np.ndarray,
        records: list[TrajectoryRecord],
        operations_by_sequence: dict[
            int,
            tuple[int, ...],
        ],
    ) -> None:

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

        observations = (
            self.noisy_vectors[
                start:end
            ]
        )

        clean_final = (
            self.clean_vectors[
                end - 1
            ]
        )

        operation_ids = (
            self.operations_by_sequence[
                record.sequence_index
            ]
        )

        if (
            len(operation_ids)
            != record.point_count - 1
        ):
            raise AssertionError(
                "Operation count does not match "
                "trajectory length."
            )

        return {
            "observations":
                torch.from_numpy(
                    observations
                ).float(),

            "operations":
                torch.tensor(
                    operation_ids,
                    dtype=torch.long,
                ),

            "clean_final_target":
                torch.from_numpy(
                    clean_final
                ).float(),
        }


def collate_evaluation(
    examples,
):
    base_examples = [
        {
            "observations":
                example[
                    "observations"
                ],

            "operations":
                example[
                    "operations"
                ],
        }
        for example in examples
    ]

    batch = collate_trajectories(
        base_examples
    )

    batch[
        "clean_final_target"
    ] = torch.stack(
        [
            example[
                "clean_final_target"
            ]
            for example in examples
        ],
        dim=0,
    )

    return batch


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--noise",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
            "auto",
        ],
        default="auto",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def resolve_device(
    requested,
):
    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
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


def validate_protocol(
    seed,
    noise,
):
    if seed not in FROZEN_SEEDS:
        raise ValueError(
            f"Seed {seed} is not frozen."
        )

    if not any(
        np.isclose(
            noise,
            frozen,
        )
        for frozen in FROZEN_NOISE_LEVELS
    ):
        raise ValueError(
            f"Noise {noise} is not frozen."
        )

    selected = load_json(
        PHASE2C_DIR
        / "selected_configuration.json"
    )

    if (
        selected[
            "configuration_id"
        ]
        != EXPECTED_CONFIGURATION_ID
    ):
        raise AssertionError(
            "Selected Phase 2C configuration changed."
        )

    for name, expected in (
        EXPECTED_CONFIGURATION.items()
    ):

        if not np.isclose(
            selected[name],
            expected,
        ):
            raise AssertionError(
                f"Frozen {name} changed."
            )

    phase2c_summary = load_json(
        PHASE2C_DIR
        / "phase2c_summary.json"
    )

    if (
        phase2c_summary[
            "phase2c_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2C is not passed."
        )

    return selected


def load_manifest(
    operation_vocabulary,
):
    rows = load_csv(
        PHASE2A_DIR
        / "ood_balanced_sequence_manifest.csv"
    )

    operations_by_sequence = {}
    split_by_sequence = {}

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for row in rows:

        sequence_index = int(
            row[
                "sequence_index"
            ]
        )

        split = row[
            "split"
        ]

        counts[
            split
        ] += 1

        names = json.loads(
            row[
                "operation_sequence"
            ]
        )

        operations_by_sequence[
            sequence_index
        ] = tuple(
            operation_vocabulary[
                name
            ]
            for name in names
        )

        split_by_sequence[
            sequence_index
        ] = split

    if counts != EXPECTED_SEQUENCE_COUNTS:
        raise AssertionError(
            "Frozen sequence split changed: "
            f"{counts}"
        )

    return (
        operations_by_sequence,
        split_by_sequence,
        counts,
    )


def load_trajectory_records(
    split_by_sequence,
):
    rows = load_csv(
        PHASE2A_DIR
        / "visible_trajectory_index.csv"
    )

    records = {
        "train": [],
        "val": [],
        "test": [],
    }

    for row in rows:

        sequence_index = int(
            row[
                "sequence_index"
            ]
        )

        if (
            sequence_index
            not in split_by_sequence
        ):
            continue

        split = split_by_sequence[
            sequence_index
        ]

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

        records[
            split
        ].append(
            record
        )

    counts = {
        split:
            len(
                split_records
            )
        for split, split_records
        in records.items()
    }

    if counts != EXPECTED_TRAJECTORY_COUNTS:
        raise AssertionError(
            "Frozen trajectory split changed: "
            f"{counts}"
        )

    return records


def load_vectors(
    requested_noise,
):
    path = (
        PHASE1C_DIR
        / "continuous_observations.npz"
    )

    with np.load(
        path
    ) as data:

        noise_fractions = np.asarray(
            data[
                "noise_fractions"
            ],
            dtype=np.float64,
        )

        requested_indices = np.where(
            np.isclose(
                noise_fractions,
                requested_noise,
            )
        )[0]

        clean_indices = np.where(
            np.isclose(
                noise_fractions,
                0.0,
            )
        )[0]

        if len(
            requested_indices
        ) != 1:
            raise AssertionError(
                "Could not uniquely identify "
                "requested noise condition."
            )

        if len(
            clean_indices
        ) != 1:
            raise AssertionError(
                "Could not uniquely identify "
                "clean observations."
            )

        requested_index = int(
            requested_indices[
                0
            ]
        )

        clean_index = int(
            clean_indices[
                0
            ]
        )

        noisy_vectors = np.array(
            data[
                "vectors"
            ][
                requested_index
            ],
            dtype=np.float32,
            copy=True,
        )

        clean_vectors = np.array(
            data[
                "vectors"
            ][
                clean_index
            ],
            dtype=np.float32,
            copy=True,
        )

    return (
        noisy_vectors,
        clean_vectors,
        requested_index,
    )


def make_training_loader(
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
        collate_fn=(
            collate_trajectories
        ),
        generator=generator,
    )


def make_evaluation_loader(
    noisy_vectors,
    clean_vectors,
    records,
    operations_by_sequence,
):
    dataset = (
        EvaluationTrajectoryDataset(
            noisy_vectors=(
                noisy_vectors
            ),
            clean_vectors=(
                clean_vectors
            ),
            records=records,
            operations_by_sequence=(
                operations_by_sequence
            ),
        )
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=(
            collate_evaluation
        ),
    )


@torch.no_grad()
def evaluate_predictions(
    model,
    loader,
    temperature,
    device,
):
    model.eval()

    model_noisy_sse = 0.0
    model_clean_sse = 0.0

    persistence_noisy_sse = 0.0
    persistence_clean_sse = 0.0

    value_count = 0

    by_length = {}

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

        clean_target = batch[
            "clean_final_target"
        ].to(
            device
        )

        initial_latent = model.encode(
            observations[
                :,
                0,
                :
            ],
            temperature=temperature,
        )

        final_latent = model.rollout(
            initial_distribution=(
                initial_latent
            ),
            operation_ids=(
                operation_ids
            ),
            step_mask=(
                step_mask
            ),
            temperature=temperature,
        )

        prediction = model.decode(
            final_latent
        )

        batch_indices = torch.arange(
            observations.shape[0],
            device=device,
        )

        noisy_target = observations[
            batch_indices,
            lengths - 1,
            :
        ]

        initial_observation = (
            observations[
                :,
                0,
                :
            ]
        )

        model_noisy_error = (
            prediction
            - noisy_target
        ).pow(
            2
        )

        model_clean_error = (
            prediction
            - clean_target
        ).pow(
            2
        )

        persistence_noisy_error = (
            initial_observation
            - noisy_target
        ).pow(
            2
        )

        persistence_clean_error = (
            initial_observation
            - clean_target
        ).pow(
            2
        )

        model_noisy_sse += float(
            model_noisy_error.sum().item()
        )

        model_clean_sse += float(
            model_clean_error.sum().item()
        )

        persistence_noisy_sse += float(
            persistence_noisy_error.sum().item()
        )

        persistence_clean_sse += float(
            persistence_clean_error.sum().item()
        )

        value_count += int(
            noisy_target.numel()
        )

        sequence_lengths = (
            lengths - 1
        ).detach().cpu().numpy()

        sample_model_noisy = (
            model_noisy_error
            .mean(
                dim=1
            )
            .detach()
            .cpu()
            .numpy()
        )

        sample_model_clean = (
            model_clean_error
            .mean(
                dim=1
            )
            .detach()
            .cpu()
            .numpy()
        )

        for (
            sequence_length,
            noisy_mse,
            clean_mse,
        ) in zip(
            sequence_lengths,
            sample_model_noisy,
            sample_model_clean,
        ):

            sequence_length = int(
                sequence_length
            )

            item = by_length.setdefault(
                sequence_length,
                {
                    "count": 0,
                    "noisy_mse_sum": 0.0,
                    "clean_mse_sum": 0.0,
                },
            )

            item[
                "count"
            ] += 1

            item[
                "noisy_mse_sum"
            ] += float(
                noisy_mse
            )

            item[
                "clean_mse_sum"
            ] += float(
                clean_mse
            )

    metrics = {
        "model_noisy_target_mse":
            model_noisy_sse
            / value_count,

        "model_clean_target_mse":
            model_clean_sse
            / value_count,

        "persistence_noisy_target_mse":
            persistence_noisy_sse
            / value_count,

        "persistence_clean_target_mse":
            persistence_clean_sse
            / value_count,
    }

    length_rows = []

    for sequence_length in sorted(
        by_length
    ):

        item = by_length[
            sequence_length
        ]

        length_rows.append(
            {
                "sequence_length":
                    sequence_length,

                "trajectory_count":
                    item[
                        "count"
                    ],

                "model_noisy_target_mse":
                    item[
                        "noisy_mse_sum"
                    ]
                    / item[
                        "count"
                    ],

                "model_clean_target_mse":
                    item[
                        "clean_mse_sum"
                    ]
                    / item[
                        "count"
                    ],
            }
        )

    return (
        metrics,
        length_rows,
    )


def write_history(
    path,
    rows,
):
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


def write_length_metrics(
    path,
    rows,
):
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


def main():
    args = parse_arguments()

    selected_configuration = (
        validate_protocol(
            seed=args.seed,
            noise=args.noise,
        )
    )

    device = resolve_device(
        args.device
    )

    run_name = (
        f"seed_{args.seed}"
        f"_noise_{noise_slug(args.noise)}"
    )

    run_dir = (
        OUTPUT_DIR
        / "runs"
        / run_name
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
        and not args.force
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
                f"{run_name} already completed."
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            return

    set_random_seed(
        args.seed
    )

    operation_vocabulary = (
        load_operation_vocabulary()
    )

    (
        operations_by_sequence,
        split_by_sequence,
        sequence_counts,
    ) = load_manifest(
        operation_vocabulary
    )

    records = (
        load_trajectory_records(
            split_by_sequence
        )
    )

    (
        noisy_vectors,
        clean_vectors,
        noise_index,
    ) = load_vectors(
        args.noise
    )

    codebook_centers = (
        kmeans_training_centers(
            vectors=noisy_vectors,
            train_records=records[
                "train"
            ],
            seed=args.seed,
        )
    )

    train_loader = (
        make_training_loader(
            vectors=noisy_vectors,
            records=records[
                "train"
            ],
            operations_by_sequence=(
                operations_by_sequence
            ),
            seed=args.seed,
            shuffle=True,
        )
    )

    val_loader = (
        make_training_loader(
            vectors=noisy_vectors,
            records=records[
                "val"
            ],
            operations_by_sequence=(
                operations_by_sequence
            ),
            seed=args.seed,
            shuffle=False,
        )
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

    model.initialize_codebook(
        codebook_centers
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=selected_configuration[
            "learning_rate"
        ],
    )

    best_validation_mse = float(
        "inf"
    )

    best_epoch = None
    best_temperature = None

    epochs_without_improvement = 0

    checkpoint_path = (
        run_dir
        / "best_model.pt"
    )

    history = []

    print(
        f"Phase 2D run: {run_name}"
    )

    print(
        json.dumps(
            {
                "device":
                    str(device),

                "configuration":
                    EXPECTED_CONFIGURATION_ID,

                "seed":
                    args.seed,

                "noise":
                    args.noise,

                "noise_index":
                    noise_index,

                "train_sequences":
                    sequence_counts[
                        "train"
                    ],

                "validation_sequences":
                    sequence_counts[
                        "val"
                    ],

                "test_sequences_reserved":
                    sequence_counts[
                        "test"
                    ],
            },
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # Training and validation only.
    # The test loader does not exist yet.
    # ---------------------------------------------------------

    for epoch in range(
        1,
        MAXIMUM_EPOCHS + 1,
    ):

        temperature = (
            temperature_for_epoch(
                epoch
            )
        )

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                temperature=temperature,
                device=device,
                configuration=(
                    selected_configuration
                ),
            )
        )

        validation_loader = (
            make_evaluation_loader(
                noisy_vectors=(
                    noisy_vectors
                ),
                clean_vectors=(
                    clean_vectors
                ),
                records=records[
                    "val"
                ],
                operations_by_sequence=(
                    operations_by_sequence
                ),
            )
        )

        (
            validation_metrics,
            _,
        ) = evaluate_predictions(
            model=model,
            loader=validation_loader,
            temperature=temperature,
            device=device,
        )

        validation_mse = float(
            validation_metrics[
                "model_noisy_target_mse"
            ]
        )

        history.append(
            {
                "epoch":
                    epoch,

                "temperature":
                    temperature,

                "train_total_loss":
                    train_metrics[
                        "total"
                    ],

                "train_reconstruction_loss":
                    train_metrics[
                        "reconstruction"
                    ],

                "train_one_step_loss":
                    train_metrics[
                        "one_step"
                    ],

                "train_transition_loss":
                    train_metrics[
                        "transition"
                    ],

                "train_rollout_loss":
                    train_metrics[
                        "rollout"
                    ],

                "train_determinism_loss":
                    train_metrics[
                        "determinism"
                    ],

                "validation_rollout_mse":
                    validation_mse,
            }
        )

        print(
            f"epoch={epoch:03d} | "
            f"T={temperature:.4f} | "
            f"train={train_metrics['total']:.6f} | "
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

            best_temperature = (
                temperature
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "configuration":
                        selected_configuration,

                    "seed":
                        args.seed,

                    "noise":
                        args.noise,

                    "epoch":
                        best_epoch,

                    "temperature":
                        best_temperature,

                    "validation_rollout_mse":
                        best_validation_mse,

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
            print(
                "Early stopping."
            )
            break

    write_history(
        run_dir
        / "training_history.csv",
        history,
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            "No validation checkpoint saved."
        )

    # ---------------------------------------------------------
    # Freeze checkpoint before opening test evaluation.
    # ---------------------------------------------------------

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

    best_epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    best_temperature = float(
        checkpoint[
            "temperature"
        ]
    )

    row_sum_error = float(
        maximum_channel_row_sum_error(
            model=model,
            temperature=best_temperature,
        )
    )

    parameters_finite = bool(
        all(
            torch.isfinite(
                parameter
            ).all().item()
            for parameter
            in model.parameters()
        )
    )

    # ---------------------------------------------------------
    # First test evaluation for this frozen run.
    # ---------------------------------------------------------

    test_loader = (
        make_evaluation_loader(
            noisy_vectors=noisy_vectors,
            clean_vectors=clean_vectors,
            records=records[
                "test"
            ],
            operations_by_sequence=(
                operations_by_sequence
            ),
        )
    )

    (
        test_metrics,
        test_length_rows,
    ) = evaluate_predictions(
        model=model,
        loader=test_loader,
        temperature=best_temperature,
        device=device,
    )

    write_length_metrics(
        run_dir
        / "test_mse_by_sequence_length.csv",
        test_length_rows,
    )

    numerically_valid = bool(
        parameters_finite
        and row_sum_error < 1e-6
        and all(
            np.isfinite(
                value
            )
            for value in (
                test_metrics.values()
            )
        )
    )

    result = {
        "phase":
            "2D final OCM predictive evaluation",

        "status":
            "completed",

        "configuration_id":
            EXPECTED_CONFIGURATION_ID,

        "configuration": {
            name:
                float(
                    value
                )
            for name, value
            in EXPECTED_CONFIGURATION.items()
        },

        "seed":
            int(
                args.seed
            ),

        "noise_fraction":
            float(
                args.noise
            ),

        "noise_index":
            int(
                noise_index
            ),

        "device":
            str(
                device
            ),

        "parameter_count":
            int(
                parameter_count
            ),

        "train_sequence_count":
            sequence_counts[
                "train"
            ],

        "validation_sequence_count":
            sequence_counts[
                "val"
            ],

        "test_sequence_count":
            sequence_counts[
                "test"
            ],

        "best_epoch":
            best_epoch,

        "best_temperature":
            best_temperature,

        "best_validation_rollout_mse":
            float(
                checkpoint[
                    "validation_rollout_mse"
                ]
            ),

        "test_metrics":
            {
                name:
                    float(
                        value
                    )
                for name, value
                in test_metrics.items()
            },

        "maximum_channel_row_sum_error":
            row_sum_error,

        "all_parameters_finite":
            parameters_finite,

        "numerically_valid":
            numerically_valid,

        "checkpoint_selected_using":
            "validation rollout MSE only",

        "test_used_for_checkpoint_selection":
            False,

        "structural_labels_read":
            False,

        "blackwell_labels_read":
            False,

        "information_classes_read":
            False,

        "symbolic_states_read":
            False,

        "structural_metrics_computed":
            False,
    }

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=2,
        )

    print(
        "Phase 2D run completed."
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
