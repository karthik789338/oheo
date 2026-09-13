from __future__ import annotations

import csv
import hashlib
import json
from itertools import product
from pathlib import Path


PHASE2A_DIR = Path(
    "outputs/phase2a_learning_protocol"
)

PHASE2C_DIR = Path(
    "outputs/phase2c_hyperparameter_sweep"
)

PHASE2F_DIR = Path(
    "outputs/phase2f_ood_structural_audit"
)

OUTPUT_DIR = Path(
    "outputs/phase2g_baseline_protocol"
)


TUNING_SEED = 11
DEVELOPMENT_NOISE = 0.25

FINAL_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

FINAL_NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

EXPECTED_OOD_COUNTS = {
    "train": 1376,
    "val": 144,
    "test": 144,
}

NEURAL_LEARNING_RATES = (
    0.001,
    0.0003,
)

MAXIMUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-5
BATCH_SIZE = 128


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(
    path: Path,
):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def write_json(
    path: Path,
    value,
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


def write_csv(
    path: Path,
    rows,
) -> None:
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
        writer.writerows(rows)


def file_sha256(
    path: Path,
):
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_completed_phases():
    phase2a = load_json(
        PHASE2A_DIR
        / "phase2a_summary.json"
    )

    phase2c = load_json(
        PHASE2C_DIR
        / "phase2c_summary.json"
    )

    selected = load_json(
        PHASE2C_DIR
        / "selected_configuration.json"
    )

    phase2f = load_json(
        PHASE2F_DIR
        / "phase2f_summary.json"
    )

    if (
        phase2a[
            "status"
        ]
        != "protocol_frozen"
        or phase2a[
            "sanity_checks"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2A is not frozen."
        )

    if (
        phase2c[
            "phase2c_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2C is not passed."
        )

    if (
        selected[
            "configuration_id"
        ]
        != "c05"
    ):
        raise AssertionError(
            "Frozen OCM configuration changed."
        )

    if (
        phase2f[
            "phase2f_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2F is not passed."
        )


def validate_phase2a_registry():
    registry = load_json(
        PHASE2A_DIR
        / "baseline_registry.json"
    )

    expected = {
        "B0_persistence",
        "B1_linear_operation",
        "B2_operation_mlp",
        "B3_gru",
        "B4_transformer",
        "B5_continuous_latent_operator",
    }

    if set(
        registry
    ) != expected:
        raise AssertionError(
            "Phase 2A baseline registry changed."
        )


def load_ood_manifest():
    rows = load_csv(
        PHASE2A_DIR
        / "ood_balanced_sequence_manifest.csv"
    )

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    maximum_length = 0

    for row in rows:
        split = row[
            "split"
        ]

        counts[
            split
        ] += 1

        maximum_length = max(
            maximum_length,
            int(
                row[
                    "sequence_length"
                ]
            ),
        )

    if counts != EXPECTED_OOD_COUNTS:
        raise AssertionError(
            "Frozen OOD split changed: "
            f"{counts}"
        )

    return (
        counts,
        maximum_length,
    )


def build_search_registry():
    configurations = []

    # ---------------------------------------------------------
    # B1: deterministic affine operators fitted by ridge.
    # ---------------------------------------------------------

    for index, ridge in enumerate(
        (
            1e-6,
            1e-4,
            1e-2,
        ),
        start=1,
    ):
        configurations.append(
            {
                "baseline_id":
                    "B1",

                "configuration_id":
                    f"B1_c{index:02d}",

                "ridge":
                    ridge,
            }
        )

    # ---------------------------------------------------------
    # B2: recursively applied operation-conditioned MLP.
    # ---------------------------------------------------------

    index = 1

    for learning_rate, width in product(
        NEURAL_LEARNING_RATES,
        (
            64,
            128,
        ),
    ):
        configurations.append(
            {
                "baseline_id":
                    "B2",

                "configuration_id":
                    f"B2_c{index:02d}",

                "learning_rate":
                    learning_rate,

                "hidden_width":
                    width,

                "operation_embedding_dim":
                    16,
            }
        )

        index += 1

    # ---------------------------------------------------------
    # B3: initial-observation-conditioned GRU.
    # ---------------------------------------------------------

    index = 1

    for learning_rate, hidden_size in product(
        NEURAL_LEARNING_RATES,
        (
            64,
            128,
        ),
    ):
        configurations.append(
            {
                "baseline_id":
                    "B3",

                "configuration_id":
                    f"B3_c{index:02d}",

                "learning_rate":
                    learning_rate,

                "hidden_size":
                    hidden_size,

                "operation_embedding_dim":
                    32,
            }
        )

        index += 1

    # ---------------------------------------------------------
    # B4: Transformer over initial-observation and operation
    # tokens.
    # ---------------------------------------------------------

    index = 1

    for learning_rate, layer_count in product(
        NEURAL_LEARNING_RATES,
        (
            1,
            2,
        ),
    ):
        configurations.append(
            {
                "baseline_id":
                    "B4",

                "configuration_id":
                    f"B4_c{index:02d}",

                "learning_rate":
                    learning_rate,

                "model_dimension":
                    64,

                "attention_heads":
                    4,

                "layer_count":
                    layer_count,

                "feedforward_dimension":
                    128,

                "dropout":
                    0.0,
            }
        )

        index += 1

    # ---------------------------------------------------------
    # B5: composable continuous latent operators.
    # ---------------------------------------------------------

    index = 1

    for learning_rate, operator_width in product(
        NEURAL_LEARNING_RATES,
        (
            16,
            32,
        ),
    ):
        configurations.append(
            {
                "baseline_id":
                    "B5",

                "configuration_id":
                    f"B5_c{index:02d}",

                "learning_rate":
                    learning_rate,

                "latent_dimension":
                    8,

                "encoder_width":
                    64,

                "operator_hidden_width":
                    operator_width,
            }
        )

        index += 1

    expected_counts = {
        "B1": 3,
        "B2": 4,
        "B3": 4,
        "B4": 4,
        "B5": 4,
    }

    actual_counts = {
        baseline_id:
            sum(
                row[
                    "baseline_id"
                ]
                == baseline_id
                for row in configurations
            )
        for baseline_id in expected_counts
    }

    if actual_counts != expected_counts:
        raise AssertionError(
            "Baseline search counts changed: "
            f"{actual_counts}"
        )

    if len(
        configurations
    ) != 19:
        raise AssertionError(
            "Expected exactly 19 tuning configurations."
        )

    return configurations


def build_final_run_matrix():
    rows = []

    # B1 is deterministic. Fit once per noise level.
    for noise in FINAL_NOISE_LEVELS:
        rows.append(
            {
                "baseline_id":
                    "B1",

                "seed":
                    "deterministic",

                "noise_fraction":
                    noise,
            }
        )

    # Neural baselines use all frozen seeds and noise levels.
    for baseline_id in (
        "B2",
        "B3",
        "B4",
        "B5",
    ):
        for seed in FINAL_SEEDS:
            for noise in FINAL_NOISE_LEVELS:
                rows.append(
                    {
                        "baseline_id":
                            baseline_id,

                        "seed":
                            seed,

                        "noise_fraction":
                            noise,
                    }
                )

    if len(rows) != 105:
        raise AssertionError(
            "Expected exactly 105 final baseline fits."
        )

    return rows


def build_protocol(
    maximum_sequence_length,
):
    return {
        "phase":
            "2G baseline comparison protocol freeze",

        "primary_question": (
            "Does explicit finite operation-channel structure "
            "improve prediction on unseen composite transformations "
            "relative to generic continuous and sequence models?"
        ),

        "development_condition": {
            "seed":
                TUNING_SEED,

            "noise_fraction":
                DEVELOPMENT_NOISE,

            "selection_metric":
                "OOD validation noisy-target rollout MSE only",

            "test_split_evaluated":
                False,

            "structural_metrics_used_for_selection":
                False,
        },

        "shared_data_protocol": {
            "train_sequence_count":
                1376,

            "validation_sequence_count":
                144,

            "test_sequence_count":
                144,

            "maximum_sequence_length":
                maximum_sequence_length,

            "training_inputs": [
                "continuous observations",
                "primitive operation labels",
                "sequence order",
            ],

            "forbidden_during_training": [
                "symbolic state IDs",
                "transformation IDs",
                "information classes",
                "Blackwell relation labels",
                "Hasse edges",
                "clean targets at nonzero noise",
            ],
        },

        "shared_neural_training": {
            "batch_size":
                BATCH_SIZE,

            "maximum_epochs":
                MAXIMUM_EPOCHS,

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,

            "early_stopping_minimum_delta":
                EARLY_STOPPING_MIN_DELTA,

            "optimizer":
                "Adam",

            "gradient_clip_norm":
                5.0,

            "learning_rates":
                list(
                    NEURAL_LEARNING_RATES
                ),
        },

        "baselines": {
            "B0": {
                "name":
                    "Persistence",

                "status":
                    "already evaluated in Phase 2D",

                "prediction":
                    "final observation equals initial observation",
            },

            "B1": {
                "name":
                    "Per-operation affine operator",

                "model": (
                    "One ridge-regularized affine map in "
                    "observation space per primitive operation"
                ),

                "training_loss":
                    "closed-form one-step squared error",

                "rollout":
                    "recursive affine composition",

                "search_configuration_count":
                    3,
            },

            "B2": {
                "name":
                    "Operation-conditioned MLP",

                "model": (
                    "A shared observation-space MLP conditioned "
                    "on an operation embedding"
                ),

                "training_loss": (
                    "one-step MSE plus recursive rollout MSE"
                ),

                "rollout":
                    "recursive predicted-observation rollout",

                "search_configuration_count":
                    4,
            },

            "B3": {
                "name":
                    "GRU sequence predictor",

                "model": (
                    "Initial observation initializes the recurrent "
                    "state; operation tokens drive the GRU"
                ),

                "training_loss": (
                    "prefix prediction MSE plus final rollout MSE"
                ),

                "search_configuration_count":
                    4,
            },

            "B4": {
                "name":
                    "Transformer sequence predictor",

                "model": (
                    "Initial-observation token followed by operation "
                    "tokens with learned positional embeddings"
                ),

                "training_loss": (
                    "prefix prediction MSE plus final rollout MSE"
                ),

                "search_configuration_count":
                    4,
            },

            "B5": {
                "name":
                    "Continuous latent operator",

                "model": (
                    "Continuous encoder and decoder with one "
                    "unrestricted nonlinear latent map per operation"
                ),

                "training_loss": (
                    "reconstruction MSE plus one-step latent-operator "
                    "prediction MSE plus final rollout MSE"
                ),

                "rollout":
                    "recursive nonlinear latent-operator composition",

                "importance": (
                    "Primary capacity-oriented comparison against OCM"
                ),

                "search_configuration_count":
                    4,
            },
        },

        "final_evaluation": {
            "noise_levels":
                list(
                    FINAL_NOISE_LEVELS
                ),

            "neural_seeds":
                list(
                    FINAL_SEEDS
                ),

            "noisy_target_metric":
                "OOD final-observation MSE",

            "clean_target_metric": (
                "OOD clean final-observation MSE, evaluated only "
                "after checkpoint selection"
            ),

            "results_reported_by_sequence_length":
                True,

            "test_used_for_checkpoint_selection":
                False,
        },

        "fairness_notes": [
            (
                "Baselines are allowed equal or greater parameter "
                "capacity than OCM."
            ),
            (
                "Every baseline receives the same continuous "
                "observations and primitive operation labels."
            ),
            (
                "The OOD split holds out composite transformations, "
                "not primitive operation identities."
            ),
            (
                "B5 is the main test of whether the finite stochastic "
                "channel constraint adds value beyond generic "
                "composable latent dynamics."
            ),
        ],
    }


def save_input_hashes():
    paths = {
        "phase2a_summary":
            PHASE2A_DIR
            / "phase2a_summary.json",

        "phase2a_baseline_registry":
            PHASE2A_DIR
            / "baseline_registry.json",

        "phase2a_ood_manifest":
            PHASE2A_DIR
            / "ood_balanced_sequence_manifest.csv",

        "phase2c_selected_configuration":
            PHASE2C_DIR
            / "selected_configuration.json",

        "phase2c_summary":
            PHASE2C_DIR
            / "phase2c_summary.json",

        "phase2f_summary":
            PHASE2F_DIR
            / "phase2f_summary.json",
    }

    output = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                path
            )

        output[
            name
        ] = {
            "path":
                str(
                    path
                ),

            "sha256":
                file_sha256(
                    path
                ),
        }

    write_json(
        OUTPUT_DIR
        / "input_hashes.json",
        output,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_completed_phases()
    validate_phase2a_registry()

    (
        split_counts,
        maximum_sequence_length,
    ) = load_ood_manifest()

    search_registry = (
        build_search_registry()
    )

    final_run_matrix = (
        build_final_run_matrix()
    )

    protocol = build_protocol(
        maximum_sequence_length
    )

    save_input_hashes()

    write_json(
        OUTPUT_DIR
        / "baseline_search_registry.json",
        {
            "configuration_count":
                len(
                    search_registry
                ),

            "tuning_seed":
                TUNING_SEED,

            "development_noise_fraction":
                DEVELOPMENT_NOISE,

            "selection_metric":
                "OOD validation noisy-target rollout MSE only",

            "configurations":
                search_registry,
        },
    )

    write_csv(
        OUTPUT_DIR
        / "final_run_matrix.csv",
        final_run_matrix,
    )

    write_json(
        OUTPUT_DIR
        / "baseline_protocol.json",
        protocol,
    )

    summary = {
        "phase":
            "2G baseline comparison protocol freeze",

        "ood_sequence_counts":
            split_counts,

        "maximum_sequence_length":
            maximum_sequence_length,

        "tuning_configuration_count":
            len(
                search_registry
            ),

        "tuning_configuration_counts": {
            "B1": 3,
            "B2": 4,
            "B3": 4,
            "B4": 4,
            "B5": 4,
        },

        "final_baseline_fit_count":
            len(
                final_run_matrix
            ),

        "final_deterministic_fit_count":
            5,

        "final_neural_fit_count":
            100,

        "development_seed":
            TUNING_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE,

        "selection_metric":
            "OOD validation noisy-target rollout MSE only",

        "structural_metrics_used_for_selection":
            False,

        "test_split_used_during_tuning":
            False,

        "clean_targets_used_during_training":
            False,

        "status":
            "protocol_frozen",

        "sanity_checks":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase2g_summary.json",
        summary,
    )

    print(
        "Phase 2G baseline protocol frozen successfully."
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
