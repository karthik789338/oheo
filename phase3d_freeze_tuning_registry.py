from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PHASE3A_DIR = Path(
    "outputs/phase3a_tier_b_protocol"
)

PHASE3B_DIR = Path(
    "outputs/phase3b_tier_b_audit"
)

PHASE3C_DIR = Path(
    "outputs/phase3c_tier_b_smoke"
)

OUTPUT_DIR = Path(
    "outputs/phase3d_tier_b_tuning"
)


DEVELOPMENT_SEED = 11
DEVELOPMENT_NOISE = 0.25

EXPECTED_COUNTS = {
    "OCM": 24,
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
}


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(path: Path, value):
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def validate_sources():
    phase3a = load_json(
        PHASE3A_DIR
        / "phase3a_summary.json"
    )

    phase3b = load_json(
        PHASE3B_DIR
        / "phase3b_summary.json"
    )

    phase3c = load_json(
        PHASE3C_DIR
        / "phase3c_summary.json"
    )

    if phase3a["status"] != "protocol_frozen":
        raise AssertionError(
            "Phase 3A is not frozen."
        )

    if phase3b["phase3b_status"] != "passed":
        raise AssertionError(
            "Phase 3B has not passed."
        )

    if phase3b["tier_b_data_accepted"] is not True:
        raise AssertionError(
            "Tier B data was not accepted."
        )

    if phase3c["phase3c_status"] != "passed":
        raise AssertionError(
            "Phase 3C has not passed."
        )

    return {
        "phase3a": phase3a,
        "phase3b": phase3b,
        "phase3c": phase3c,
    }


def create_ocm_configurations():
    configurations = []
    counter = 1

    for learning_rate in (
        0.001,
        0.0005,
    ):
        for transition_weight in (
            0.05,
            0.1,
        ):
            for rollout_weight in (
                1.0,
                2.0,
            ):
                for temperature in (
                    0.25,
                    0.5,
                    0.75,
                ):
                    configurations.append(
                        {
                            "baseline_id":
                                "OCM",

                            "configuration_id":
                                f"OCM_c{counter:02d}",

                            "learning_rate":
                                learning_rate,

                            "lambda_transition":
                                transition_weight,

                            "lambda_rollout":
                                rollout_weight,

                            "lambda_determinism":
                                0.001,

                            "temperature":
                                temperature,

                            "latent_state_count":
                                8,

                            "observation_dimension":
                                32,
                        }
                    )

                    counter += 1

    if len(configurations) != 24:
        raise AssertionError(
            "Expected 24 OCM configurations."
        )

    return configurations


def create_baseline_configurations():
    configurations = [
        {
            "baseline_id": "B1",
            "configuration_id": "B1_c01",
            "ridge": 1e-6,
        },
        {
            "baseline_id": "B1",
            "configuration_id": "B1_c02",
            "ridge": 1e-4,
        },
        {
            "baseline_id": "B1",
            "configuration_id": "B1_c03",
            "ridge": 1e-2,
        },

        {
            "baseline_id": "B2",
            "configuration_id": "B2_c01",
            "learning_rate": 0.001,
            "hidden_width": 64,
            "operation_embedding_dim": 16,
        },
        {
            "baseline_id": "B2",
            "configuration_id": "B2_c02",
            "learning_rate": 0.001,
            "hidden_width": 128,
            "operation_embedding_dim": 16,
        },
        {
            "baseline_id": "B2",
            "configuration_id": "B2_c03",
            "learning_rate": 0.0005,
            "hidden_width": 128,
            "operation_embedding_dim": 32,
        },
        {
            "baseline_id": "B2",
            "configuration_id": "B2_c04",
            "learning_rate": 0.0005,
            "hidden_width": 256,
            "operation_embedding_dim": 32,
        },

        {
            "baseline_id": "B3",
            "configuration_id": "B3_c01",
            "learning_rate": 0.001,
            "hidden_size": 64,
            "operation_embedding_dim": 32,
        },
        {
            "baseline_id": "B3",
            "configuration_id": "B3_c02",
            "learning_rate": 0.001,
            "hidden_size": 128,
            "operation_embedding_dim": 32,
        },
        {
            "baseline_id": "B3",
            "configuration_id": "B3_c03",
            "learning_rate": 0.0005,
            "hidden_size": 128,
            "operation_embedding_dim": 64,
        },
        {
            "baseline_id": "B3",
            "configuration_id": "B3_c04",
            "learning_rate": 0.0005,
            "hidden_size": 256,
            "operation_embedding_dim": 64,
        },

        {
            "baseline_id": "B4",
            "configuration_id": "B4_c01",
            "learning_rate": 0.001,
            "model_dimension": 64,
            "attention_heads": 4,
            "layer_count": 2,
            "feedforward_dimension": 128,
            "dropout": 0.0,
        },
        {
            "baseline_id": "B4",
            "configuration_id": "B4_c02",
            "learning_rate": 0.0005,
            "model_dimension": 128,
            "attention_heads": 4,
            "layer_count": 2,
            "feedforward_dimension": 256,
            "dropout": 0.0,
        },
        {
            "baseline_id": "B4",
            "configuration_id": "B4_c03",
            "learning_rate": 0.0005,
            "model_dimension": 128,
            "attention_heads": 8,
            "layer_count": 3,
            "feedforward_dimension": 256,
            "dropout": 0.1,
        },
        {
            "baseline_id": "B4",
            "configuration_id": "B4_c04",
            "learning_rate": 0.0003,
            "model_dimension": 192,
            "attention_heads": 8,
            "layer_count": 3,
            "feedforward_dimension": 384,
            "dropout": 0.1,
        },

        {
            "baseline_id": "B5",
            "configuration_id": "B5_c01",
            "learning_rate": 0.001,
            "latent_dimension": 8,
            "encoder_width": 64,
            "operator_hidden_width": 32,
        },
        {
            "baseline_id": "B5",
            "configuration_id": "B5_c02",
            "learning_rate": 0.001,
            "latent_dimension": 16,
            "encoder_width": 128,
            "operator_hidden_width": 64,
        },
        {
            "baseline_id": "B5",
            "configuration_id": "B5_c03",
            "learning_rate": 0.0005,
            "latent_dimension": 32,
            "encoder_width": 128,
            "operator_hidden_width": 64,
        },
        {
            "baseline_id": "B5",
            "configuration_id": "B5_c04",
            "learning_rate": 0.0005,
            "latent_dimension": 32,
            "encoder_width": 256,
            "operator_hidden_width": 128,
        },
    ]

    if len(configurations) != 19:
        raise AssertionError(
            "Expected 19 baseline configurations."
        )

    return configurations


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_sources()

    configurations = (
        create_ocm_configurations()
        + create_baseline_configurations()
    )

    counts = Counter(
        configuration["baseline_id"]
        for configuration in configurations
    )

    if dict(counts) != EXPECTED_COUNTS:
        raise AssertionError(
            f"Configuration counts changed: {dict(counts)}"
        )

    configuration_ids = [
        configuration["configuration_id"]
        for configuration in configurations
    ]

    if len(configuration_ids) != len(
        set(configuration_ids)
    ):
        raise AssertionError(
            "Configuration IDs are not unique."
        )

    registry = {
        "phase":
            "3D Tier B leakage-safe tuning registry",

        "development_seed":
            DEVELOPMENT_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE,

        "training_cell":
            "train_joint",

        "checkpoint_selection_cell":
            "val_joint",

        "selection_metric":
            "val_joint noisy-target rollout MSE only",

        "maximum_epochs":
            200,

        "early_stopping_patience":
            25,

        "early_stopping_min_delta":
            1e-5,

        "batch_size":
            256,

        "configuration_count":
            len(configurations),

        "configuration_counts":
            dict(counts),

        "configurations":
            configurations,

        "test_cells_opened":
            False,

        "privileged_arrays_opened":
            False,

        "clean_targets_read":
            False,

        "structural_labels_read":
            False,

        "tuning_checkpoints_eligible_for_final_use":
            False,

        "status":
            "registry_frozen",
    }

    write_json(
        OUTPUT_DIR
        / "tuning_registry.json",
        registry,
    )

    summary = {
        "phase":
            "3D Tier B tuning registry freeze",

        "configuration_count":
            len(configurations),

        "configuration_counts":
            dict(counts),

        "development_seed":
            DEVELOPMENT_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE,

        "training_cell":
            "train_joint",

        "checkpoint_selection_cell":
            "val_joint",

        "selection_metric":
            registry["selection_metric"],

        "test_cells_opened":
            False,

        "privileged_arrays_opened":
            False,

        "clean_targets_read":
            False,

        "structural_labels_read":
            False,

        "training_performed":
            False,

        "phase3d_registry_status":
            "frozen",
    }

    write_json(
        OUTPUT_DIR
        / "phase3d_registry_summary.json",
        summary,
    )

    print(
        "Phase 3D tuning registry frozen."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
