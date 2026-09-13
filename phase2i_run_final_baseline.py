from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phase2b_run_ocm_smoke_test import (
    ContinuousTrajectoryDataset,
    TrajectoryRecord,
    collate_trajectories,
    load_operation_vocabulary,
    set_random_seed,
)

from phase2h_tune_predictive_baselines import (
    build_neural_model,
    collect_affine_training_pairs,
    fit_affine_model,
    state_is_finite,
    train_one_epoch,
)


PHASE1C_DIR = Path(
    "outputs/phase1c_continuous_observations"
)

PHASE2A_DIR = Path(
    "outputs/phase2a_learning_protocol"
)

PHASE2H_DIR = Path(
    "outputs/phase2h_baseline_tuning"
)

OUTPUT_DIR = Path(
    "outputs/phase2i_final_baselines"
)


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

NEURAL_BASELINES = (
    "B2",
    "B3",
    "B4",
    "B5",
)

EXPECTED_CONFIGURATION_IDS = {
    "B1": "B1_c02",
    "B2": "B2_c01",
    "B3": "B3_c01",
    "B4": "B4_c02",
    "B5": "B5_c02",
}

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

BATCH_SIZE = 128
MAXIMUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-5


class BaselineEvaluationDataset(Dataset):
    """
    Noisy model inputs and noisy/clean final observation targets.

    Clean observations are exposed only during post-selection
    validation reporting and final test evaluation. They are never
    used in optimization or checkpoint selection.
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

        operations = (
            self.operations_by_sequence[
                record.sequence_index
            ]
        )

        if (
            len(operations)
            != record.point_count - 1
        ):
            raise AssertionError(
                "Operation count and trajectory "
                "length do not match."
            )

        return {
            "observations":
                torch.from_numpy(
                    self.noisy_vectors[
                        start:end
                    ]
                ).float(),

            "operations":
                torch.tensor(
                    operations,
                    dtype=torch.long,
                ),

            "clean_final_target":
                torch.from_numpy(
                    self.clean_vectors[
                        end - 1
                    ]
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
        "--baseline",
        choices=[
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
        ],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--noise",
        type=float,
        required=True,
    )

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
                "CUDA was requested but unavailable."
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


def noise_slug(
    noise,
):
    return (
        f"{noise:g}"
        .replace(
            ".",
            "p",
        )
    )


def validate_arguments(
    baseline,
    seed,
    noise,
):
    if not any(
        np.isclose(
            noise,
            allowed,
        )
        for allowed in FROZEN_NOISE_LEVELS
    ):
        raise ValueError(
            f"Noise {noise} is not frozen."
        )

    if baseline == "B1":
        if seed is not None:
            raise ValueError(
                "B1 is deterministic and must "
                "not receive a seed."
            )

    else:
        if seed not in FROZEN_SEEDS:
            raise ValueError(
                f"Seed {seed} is not frozen."
            )


def load_selected_configuration(
    baseline,
):
    phase2h_summary = load_json(
        PHASE2H_DIR
        / "phase2h_summary.json"
    )

    if (
        phase2h_summary[
            "phase2h_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2H is not passed."
        )

    selected = load_json(
        PHASE2H_DIR
        / "selected_baseline_configurations.json"
    )

    configuration = selected[
        baseline
    ]

    if (
        configuration[
            "configuration_id"
        ]
        != EXPECTED_CONFIGURATION_IDS[
            baseline
        ]
    ):
        raise AssertionError(
            f"Frozen {baseline} configuration changed."
        )

    return configuration


def load_manifest(
    operation_vocabulary,
):
    rows = load_csv(
        PHASE2A_DIR
        / "ood_balanced_sequence_manifest.csv"
    )

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    operations_by_sequence = {}
    split_by_sequence = {}

    for row in rows:
        split = row[
            "split"
        ]

        counts[
            split
        ] += 1

        sequence_index = int(
            row[
                "sequence_index"
            ]
        )

        operation_names = json.loads(
            row[
                "operation_sequence"
            ]
        )

        operations_by_sequence[
            sequence_index
        ] = tuple(
            operation_vocabulary[
                operation_name
            ]
            for operation_name
            in operation_names
        )

        split_by_sequence[
            sequence_index
        ] = split

    if counts != EXPECTED_SEQUENCE_COUNTS:
        raise AssertionError(
            "Frozen sequence counts changed: "
            f"{counts}"
        )

    return (
        operations_by_sequence,
        split_by_sequence,
        counts,
    )


def load_records(
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

        records[
            split
        ].append(
            TrajectoryRecord(
                trajectory_index=int(
                    row[
                        "trajectory_index"
                    ]
                ),

                sequence_index=(
                    sequence_index
                ),

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
            "Frozen trajectory counts changed: "
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
                "Could not uniquely locate "
                "the requested noise condition."
            )

        if len(
            clean_indices
        ) != 1:
            raise AssertionError(
                "Could not uniquely locate "
                "the clean condition."
            )

        noise_index = int(
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
                noise_index
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
        noise_index,
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
    dataset = BaselineEvaluationDataset(
        noisy_vectors=noisy_vectors,
        clean_vectors=clean_vectors,
        records=records,
        operations_by_sequence=(
            operations_by_sequence
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_evaluation,
    )


@torch.no_grad()
def evaluate_predictions(
    model,
    loader,
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

        initial_observation = observations[
            :,
            0,
            :
        ]

        prediction = model.rollout(
            initial_observation=(
                initial_observation
            ),
            operation_ids=(
                operation_ids
            ),
            step_mask=(
                step_mask
            ),
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

        sample_noisy_mse = (
            model_noisy_error
            .mean(
                dim=1
            )
            .detach()
            .cpu()
            .numpy()
        )

        sample_clean_mse = (
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
            sample_noisy_mse,
            sample_clean_mse,
        ):
            sequence_length = int(
                sequence_length
            )

            item = by_length.setdefault(
                sequence_length,
                {
                    "trajectory_count": 0,
                    "noisy_mse_sum": 0.0,
                    "clean_mse_sum": 0.0,
                },
            )

            item[
                "trajectory_count"
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
                        "trajectory_count"
                    ],

                "model_noisy_target_mse":
                    item[
                        "noisy_mse_sum"
                    ]
                    / item[
                        "trajectory_count"
                    ],

                "model_clean_target_mse":
                    item[
                        "clean_mse_sum"
                    ]
                    / item[
                        "trajectory_count"
                    ],
            }
        )

    return (
        metrics,
        length_rows,
    )


def parameter_count(
    model,
):
    return int(
        sum(
            value.numel()
            for value
            in model.state_dict().values()
        )
    )


def run_name_for(
    baseline,
    seed,
    noise,
):
    if baseline == "B1":
        return (
            f"B1_noise_"
            f"{noise_slug(noise)}"
        )

    return (
        f"{baseline}_seed_{seed}"
        f"_noise_{noise_slug(noise)}"
    )


def run_affine(
    configuration,
    noisy_vectors,
    clean_vectors,
    records,
    operations_by_sequence,
    operation_count,
    device,
    run_dir,
):
    training_pairs = (
        collect_affine_training_pairs(
            vectors=noisy_vectors,
            records=records[
                "train"
            ],
            operations_by_sequence=(
                operations_by_sequence
            ),
            operation_count=(
                operation_count
            ),
        )
    )

    model = fit_affine_model(
        training_pairs=training_pairs,
        ridge=float(
            configuration[
                "ridge"
            ]
        ),
    ).to(
        device
    )

    validation_loader = (
        make_evaluation_loader(
            noisy_vectors=noisy_vectors,
            clean_vectors=clean_vectors,
            records=records[
                "val"
            ],
            operations_by_sequence=(
                operations_by_sequence
            ),
        )
    )

    validation_metrics, _ = (
        evaluate_predictions(
            model=model,
            loader=validation_loader,
            device=device,
        )
    )

    checkpoint_path = (
        run_dir / "best_model.pt"
    )

    torch.save(
        {
            "configuration":
                configuration,

            "model_state_dict":
                model.state_dict(),

            "validation_rollout_mse":
                validation_metrics[
                    "model_noisy_target_mse"
                ],
        },
        checkpoint_path,
    )

    # The deterministic model is now frozen.
    test_loader = make_evaluation_loader(
        noisy_vectors=noisy_vectors,
        clean_vectors=clean_vectors,
        records=records[
            "test"
        ],
        operations_by_sequence=(
            operations_by_sequence
        ),
    )

    test_metrics, length_rows = (
        evaluate_predictions(
            model=model,
            loader=test_loader,
            device=device,
        )
    )

    return {
        "model":
            model,

        "best_epoch":
            None,

        "epochs_executed":
            0,

        "best_validation_rollout_mse":
            float(
                validation_metrics[
                    "model_noisy_target_mse"
                ]
            ),

        "test_metrics":
            test_metrics,

        "length_rows":
            length_rows,

        "checkpoint_path":
            checkpoint_path,
    }


def run_neural(
    baseline,
    configuration,
    seed,
    noisy_vectors,
    clean_vectors,
    records,
    operations_by_sequence,
    operation_count,
    device,
    run_dir,
):
    set_random_seed(
        seed
    )

    train_loader = make_training_loader(
        vectors=noisy_vectors,
        records=records[
            "train"
        ],
        operations_by_sequence=(
            operations_by_sequence
        ),
        seed=seed,
        shuffle=True,
    )

    validation_loader = (
        make_evaluation_loader(
            noisy_vectors=noisy_vectors,
            clean_vectors=clean_vectors,
            records=records[
                "val"
            ],
            operations_by_sequence=(
                operations_by_sequence
            ),
        )
    )

    model = build_neural_model(
        configuration=configuration,
        operation_count=operation_count,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            configuration[
                "learning_rate"
            ]
        ),
    )

    best_validation_mse = float(
        "inf"
    )

    best_epoch = None
    epochs_without_improvement = 0

    checkpoint_path = (
        run_dir / "best_model.pt"
    )

    history = []

    for epoch in range(
        1,
        MAXIMUM_EPOCHS + 1,
    ):
        train_metrics = train_one_epoch(
            baseline_id=baseline,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        validation_metrics, _ = (
            evaluate_predictions(
                model=model,
                loader=validation_loader,
                device=device,
            )
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

                "train_total_loss":
                    train_metrics[
                        "total"
                    ],

                "train_primary_loss":
                    train_metrics[
                        "primary"
                    ],

                "train_auxiliary_loss":
                    train_metrics[
                        "auxiliary"
                    ],

                "train_rollout_loss":
                    train_metrics[
                        "rollout"
                    ],

                "validation_rollout_mse":
                    validation_mse,
            }
        )

        print(
            f"epoch={epoch:03d} | "
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
            epochs_without_improvement = 0

            torch.save(
                {
                    "configuration":
                        configuration,

                    "baseline":
                        baseline,

                    "seed":
                        seed,

                    "epoch":
                        best_epoch,

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

    write_csv(
        run_dir
        / "training_history.csv",
        history,
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            "No validation-selected checkpoint saved."
        )

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

    # Test data is opened only after loading the selected checkpoint.
    test_loader = make_evaluation_loader(
        noisy_vectors=noisy_vectors,
        clean_vectors=clean_vectors,
        records=records[
            "test"
        ],
        operations_by_sequence=(
            operations_by_sequence
        ),
    )

    test_metrics, length_rows = (
        evaluate_predictions(
            model=model,
            loader=test_loader,
            device=device,
        )
    )

    return {
        "model":
            model,

        "best_epoch":
            int(
                checkpoint[
                    "epoch"
                ]
            ),

        "epochs_executed":
            len(
                history
            ),

        "best_validation_rollout_mse":
            float(
                checkpoint[
                    "validation_rollout_mse"
                ]
            ),

        "test_metrics":
            test_metrics,

        "length_rows":
            length_rows,

        "checkpoint_path":
            checkpoint_path,
    }


def main():
    arguments = parse_arguments()

    validate_arguments(
        baseline=arguments.baseline,
        seed=arguments.seed,
        noise=arguments.noise,
    )

    configuration = (
        load_selected_configuration(
            arguments.baseline
        )
    )

    device = resolve_device(
        arguments.device
    )

    run_name = run_name_for(
        baseline=arguments.baseline,
        seed=arguments.seed,
        noise=arguments.noise,
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
        run_dir / "result.json"
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
                f"{run_name} already completed."
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            return

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

    records = load_records(
        split_by_sequence
    )

    (
        noisy_vectors,
        clean_vectors,
        noise_index,
    ) = load_vectors(
        arguments.noise
    )

    print(
        f"Phase 2I run: {run_name}"
    )

    print(
        json.dumps(
            {
                "baseline":
                    arguments.baseline,

                "configuration_id":
                    configuration[
                        "configuration_id"
                    ],

                "seed":
                    (
                        arguments.seed
                        if arguments.seed
                        is not None
                        else "deterministic"
                    ),

                "noise_fraction":
                    arguments.noise,

                "noise_index":
                    noise_index,

                "device":
                    str(
                        device
                    ),

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

    if arguments.baseline == "B1":
        run_output = run_affine(
            configuration=configuration,
            noisy_vectors=noisy_vectors,
            clean_vectors=clean_vectors,
            records=records,
            operations_by_sequence=(
                operations_by_sequence
            ),
            operation_count=len(
                operation_vocabulary
            ),
            device=device,
            run_dir=run_dir,
        )

        result_seed = "deterministic"

    else:
        run_output = run_neural(
            baseline=arguments.baseline,
            configuration=configuration,
            seed=arguments.seed,
            noisy_vectors=noisy_vectors,
            clean_vectors=clean_vectors,
            records=records,
            operations_by_sequence=(
                operations_by_sequence
            ),
            operation_count=len(
                operation_vocabulary
            ),
            device=device,
            run_dir=run_dir,
        )

        result_seed = int(
            arguments.seed
        )

    model = run_output[
        "model"
    ]

    finite = state_is_finite(
        model
    )

    metrics_finite = bool(
        all(
            np.isfinite(
                value
            )
            for value in run_output[
                "test_metrics"
            ].values()
        )
    )

    numerically_valid = bool(
        finite
        and metrics_finite
    )

    length_rows = [
        {
            "baseline_id":
                arguments.baseline,

            "seed":
                result_seed,

            "noise_fraction":
                float(
                    arguments.noise
                ),

            **row,
        }
        for row in run_output[
            "length_rows"
        ]
    ]

    write_csv(
        run_dir
        / "test_mse_by_sequence_length.csv",
        length_rows,
    )

    result = {
        "phase":
            "2I final predictive baseline evaluation",

        "status":
            "completed",

        "baseline_id":
            arguments.baseline,

        "configuration_id":
            configuration[
                "configuration_id"
            ],

        "configuration":
            configuration,

        "seed":
            result_seed,

        "noise_fraction":
            float(
                arguments.noise
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
            parameter_count(
                model
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
            run_output[
                "best_epoch"
            ],

        "epochs_executed":
            run_output[
                "epochs_executed"
            ],

        "best_validation_rollout_mse":
            run_output[
                "best_validation_rollout_mse"
            ],

        "test_metrics": {
            name:
                float(
                    value
                )
            for name, value
            in run_output[
                "test_metrics"
            ].items()
        },

        "all_state_values_finite":
            finite,

        "numerically_valid":
            numerically_valid,

        "checkpoint_selected_using":
            (
                "validation noisy-target rollout MSE only"
                if arguments.baseline != "B1"
                else (
                    "not applicable: deterministic "
                    "closed-form training"
                )
            ),

        "test_used_for_checkpoint_selection":
            False,

        "clean_targets_used_for_training":
            False,

        "structural_metrics_computed":
            False,

        "structural_labels_read":
            False,

        "blackwell_labels_read":
            False,

        "symbolic_states_read":
            False,
    }

    write_json(
        result_path,
        result,
    )

    print(
        "Phase 2I baseline run completed."
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
