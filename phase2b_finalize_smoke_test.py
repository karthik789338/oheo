from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from phase2b_operation_channel_model import (
    OperationChannelModel,
)

from phase2b_run_ocm_smoke_test import (
    SMOKE_SEED,
    PRIMARY_NOISE_FRACTION,
    OBSERVATION_DIM,
    LATENT_STATE_COUNT,
    LEARNING_RATE,
    LAMBDA_TRANS,
    LAMBDA_ROLL,
    LAMBDA_DET,
    BATCH_SIZE,
    SMOKE_MAX_EPOCHS,
    SMOKE_PATIENCE,
    INITIAL_TEMPERATURE,
    FINAL_TEMPERATURE,
    OUTPUT_DIR,
    set_random_seed,
    load_operation_vocabulary,
    load_manifest,
    load_visible_trajectory_records,
    load_continuous_vectors,
    kmeans_training_centers,
    make_loader,
    evaluate_persistence_mse,
    evaluate_rollout_mse,
    maximum_channel_row_sum_error,
)


def main() -> None:
    """
    Finalize the Phase 2B smoke test without retraining.

    The original run completed training and saved its best checkpoint.
    It failed only while serializing NumPy boolean values to JSON.
    """

    checkpoint_path = (
        OUTPUT_DIR
        / "best_model.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "training_history.csv"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    if not history_path.exists():
        raise FileNotFoundError(
            f"Missing training history: {history_path}"
        )

    set_random_seed(
        SMOKE_SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    operation_vocabulary = (
        load_operation_vocabulary()
    )

    (
        operations_by_sequence,
        split_by_sequence,
        split_counts,
    ) = load_manifest(
        operation_vocabulary
    )

    (
        train_records,
        val_records,
    ) = load_visible_trajectory_records(
        split_by_sequence
    )

    (
        vectors,
        noise_index,
    ) = load_continuous_vectors()

    # Reproduce the exact unsupervised codebook initialization
    # used at the beginning of the original smoke test.
    codebook_centers = (
        kmeans_training_centers(
            vectors=vectors,
            train_records=train_records,
            seed=SMOKE_SEED,
        )
    )

    val_loader = make_loader(
        vectors=vectors,
        records=val_records,
        operations_by_sequence=(
            operations_by_sequence
        ),
        shuffle=False,
    )

    # ------------------------------------------------------------
    # Reconstruct the initial model deterministically.
    # ------------------------------------------------------------

    initial_model = OperationChannelModel(
        observation_dim=OBSERVATION_DIM,
        latent_state_count=LATENT_STATE_COUNT,
        operation_count=len(
            operation_vocabulary
        ),
    ).to(
        device
    )

    initial_model.initialize_codebook(
        codebook_centers
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in initial_model.parameters()
    )

    persistence_val_mse = (
        evaluate_persistence_mse(
            val_loader,
            device,
        )
    )

    initial_val_rollout_mse = (
        evaluate_rollout_mse(
            model=initial_model,
            loader=val_loader,
            temperature=(
                INITIAL_TEMPERATURE
            ),
            device=device,
        )
    )

    # ------------------------------------------------------------
    # Load the already-trained best checkpoint.
    # ------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = OperationChannelModel(
        observation_dim=OBSERVATION_DIM,
        latent_state_count=LATENT_STATE_COUNT,
        operation_count=len(
            operation_vocabulary
        ),
    ).to(
        device
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

    best_val_mse = float(
        evaluate_rollout_mse(
            model=model,
            loader=val_loader,
            temperature=best_temperature,
            device=device,
        )
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

    implementation_checks = bool(
        parameters_finite
        and row_sum_error < 1e-6
        and bool(
            np.isfinite(
                best_val_mse
            )
        )
    )

    improves_over_initial = bool(
        best_val_mse
        < initial_val_rollout_mse
    )

    beats_persistence = bool(
        best_val_mse
        < persistence_val_mse
    )

    smoke_passed = bool(
        implementation_checks
        and improves_over_initial
        and beats_persistence
    )

    summary = {
        "phase":
            "2B OCM implementation smoke test",

        "seed":
            SMOKE_SEED,

        "device":
            str(device),

        "noise_fraction":
            PRIMARY_NOISE_FRACTION,

        "noise_index":
            int(noise_index),

        "train_sequence_count":
            int(
                split_counts[
                    "train"
                ]
            ),

        "validation_sequence_count":
            int(
                split_counts[
                    "val"
                ]
            ),

        "test_sequence_count_not_used":
            int(
                split_counts[
                    "test"
                ]
            ),

        "train_trajectory_count":
            int(
                len(
                    train_records
                )
            ),

        "validation_trajectory_count":
            int(
                len(
                    val_records
                )
            ),

        "parameter_count":
            int(
                parameter_count
            ),

        "fixed_smoke_hyperparameters": {
            "learning_rate":
                float(
                    LEARNING_RATE
                ),

            "lambda_transition":
                float(
                    LAMBDA_TRANS
                ),

            "lambda_rollout":
                float(
                    LAMBDA_ROLL
                ),

            "lambda_determinism":
                float(
                    LAMBDA_DET
                ),

            "batch_size":
                int(
                    BATCH_SIZE
                ),

            "maximum_epochs":
                int(
                    SMOKE_MAX_EPOCHS
                ),

            "patience":
                int(
                    SMOKE_PATIENCE
                ),

            "initial_temperature":
                float(
                    INITIAL_TEMPERATURE
                ),

            "final_temperature":
                float(
                    FINAL_TEMPERATURE
                ),
        },

        "persistence_validation_mse":
            float(
                persistence_val_mse
            ),

        "initial_validation_rollout_mse":
            float(
                initial_val_rollout_mse
            ),

        "best_epoch":
            best_epoch,

        "best_temperature":
            best_temperature,

        "best_validation_rollout_mse":
            best_val_mse,

        "improves_over_initial_model":
            bool(
                improves_over_initial
            ),

        "beats_persistence_validation":
            bool(
                beats_persistence
            ),

        "maximum_channel_row_sum_error":
            row_sum_error,

        "all_parameters_finite":
            bool(
                parameters_finite
            ),

        "implementation_checks_passed":
            bool(
                implementation_checks
            ),

        "structural_labels_read":
            False,

        "blackwell_labels_read":
            False,

        "test_split_evaluated":
            False,

        "checkpoint_metric":
            "validation rollout MSE only",

        "smoke_test_status":
            (
                "passed"
                if smoke_passed
                else "failed"
            ),

        "interpretation": (
            "This is an implementation smoke test only. "
            "The model was trained without structural labels, "
            "Blackwell labels, or OOD test evaluation."
        ),

        "original_run_note": (
            "The original training run completed successfully "
            "but failed while serializing a NumPy boolean to JSON. "
            "This file finalizes that already-trained checkpoint "
            "without retraining."
        ),
    }

    output_path = (
        OUTPUT_DIR
        / "phase2b_smoke_summary.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "Phase 2B smoke-test summary "
        "recovered successfully."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"Summary written to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
