from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PHASE4D_DIR = Path(
    "outputs/phase4dr3_tier_c_v4_predictive_protocol"
)

AUDIT_DIR = Path(
    "outputs/phase4dr31_tier_c_v4_execution_contract_audit"
)

PHASE3_REGISTRY_PATH = Path(
    "outputs/phase3d_tier_b_tuning/tuning_registry.json"
)

TIER_B_TRAINING_SOURCE = Path(
    "phase3d_tune_tier_b_models.py"
)

TIER_B_SMOKE_SOURCE = Path(
    "phase3c_run_tier_b_smoke_tests.py"
)

OUTPUT_DIR = Path(
    "outputs/phase4dr32_tier_c_v4_execution_contract"
)


PROTOCOL_SUMMARY_PATH = (
    PHASE4D_DIR
    / "phase4dr3_predictive_protocol_summary.json"
)

FIELD_CODEC_PATH = (
    PHASE4D_DIR
    / "tier_c_v4_field_codec_registry.json"
)

TUNING_REGISTRY_PATH = (
    PHASE4D_DIR
    / "tier_c_v4_tuning_registry.csv"
)

FINAL_RUN_REGISTRY_PATH = (
    PHASE4D_DIR
    / "tier_c_v4_final_run_registry.csv"
)

TEST_POLICY_PATH = (
    PHASE4D_DIR
    / "tier_c_v4_test_opening_policy.json"
)

AUDIT_SUMMARY_PATH = (
    AUDIT_DIR
    / "phase4dr31_execution_contract_audit_summary.json"
)


PROTOCOL_VERSION = "tier_c_v4"

EXPECTED_CONFIGURATION_COUNTS = {
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
    "OCM": 24,
}

EXPECTED_CONFIGURATION_COUNT = 43

FIELD_CODEC_LATENT_DIMENSION = 128

B1_FIXED_SEED = 11
B1_FIXED_LEARNING_RATE = 3e-4

PHYSICAL_MICROBATCH_SIZE = 4
GRADIENT_CLIP_NORM = 5.0

ADDED_CODEC_RECONSTRUCTION_WEIGHT = 0.0
PREDICTIVE_FIELD_LOSS_WEIGHT = 1.0

MIXED_PRECISION_ENABLED = False

EXPECTED_MISSING_FIELDS = {
    "field_codec_training_scope",
    "field_codec_reconstruction_objective",
    "field_codec_reconstruction_loss_weight",
    "predictive_field_loss_weight",
    "latent_transition_loss_weight",
    "rollout_loss_application",
    "teacher_forcing_policy",
    "variable_length_batching_policy",
    "optimizer_scope",
    "learning_rate_scope",
    "weight_decay_scope",
    "gradient_clipping_policy",
    "mixed_precision_policy",
    "codec_initialization_policy",
    "checkpoint_contents",
    "b1_reporting_name",
}


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


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


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
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
                fieldnames.append(key)

    with path.open(
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
    condition,
    message,
):
    if not condition:
        raise AssertionError(message)


def detect_optimizer_families(
    source_path: Path,
):
    source = source_path.read_text(
        encoding="utf-8"
    )

    patterns = (
        r"torch\.optim\.([A-Za-z0-9_]+)\s*\(",
        r"optim\.([A-Za-z0-9_]+)\s*\(",
    )

    detected = set()

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            source,
        ):
            detected.add(
                match.group(1)
            )

    return sorted(detected)


def load_configuration_registry():
    rows = read_csv(
        TUNING_REGISTRY_PATH
    )

    require(
        len(rows)
        == EXPECTED_CONFIGURATION_COUNT,
        "Expected 43 tuning configurations.",
    )

    counts = Counter(
        row["model_id"]
        for row in rows
    )

    require(
        dict(counts)
        == EXPECTED_CONFIGURATION_COUNTS,
        (
            "Tuning configuration counts changed: "
            f"{dict(counts)}"
        ),
    )

    configurations = []

    for row in rows:
        configuration = json.loads(
            row[
                "source_configuration_json"
            ]
        )

        configurations.append(
            {
                "model_id":
                    row["model_id"],

                "configuration_id":
                    row[
                        "configuration_id"
                    ],

                "configuration":
                    configuration,
            }
        )

    return configurations


def validate_sources():
    required_paths = (
        PROTOCOL_SUMMARY_PATH,
        FIELD_CODEC_PATH,
        TUNING_REGISTRY_PATH,
        FINAL_RUN_REGISTRY_PATH,
        TEST_POLICY_PATH,
        AUDIT_SUMMARY_PATH,
        PHASE3_REGISTRY_PATH,
        TIER_B_TRAINING_SOURCE,
        TIER_B_SMOKE_SOURCE,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = load_json(
        PROTOCOL_SUMMARY_PATH
    )

    codec = load_json(
        FIELD_CODEC_PATH
    )

    audit = load_json(
        AUDIT_SUMMARY_PATH
    )

    phase3_registry = load_json(
        PHASE3_REGISTRY_PATH
    )

    test_policy = load_json(
        TEST_POLICY_PATH
    )

    require(
        protocol[
            "protocol_version"
        ] == PROTOCOL_VERSION,
        "Protocol version changed.",
    )

    require(
        protocol[
            "phase4dr3_status"
        ] == "protocol_frozen",
        "Phase 4D-R3 is not frozen.",
    )

    require(
        protocol[
            "implementation_smoke_test_authorized"
        ] is True,
        "Smoke-test preparation is not authorized.",
    )

    require(
        protocol[
            "tuning_execution_authorized"
        ] is False,
        "Tuning was already authorized.",
    )

    require(
        protocol[
            "final_fit_execution_authorized"
        ] is False,
        "Final fitting was already authorized.",
    )

    require(
        protocol[
            "test_open_count"
        ] == 0,
        "Tier C test data was opened.",
    )

    require(
        protocol[
            "test_fields_generated"
        ] is False,
        "Tier C test fields were generated.",
    )

    require(
        audit[
            "audit_status"
        ]
        == (
            "blocked_missing_tier_c_"
            "execution_fields"
        ),
        "Unexpected Phase 4D-R3.1 audit status.",
    )

    require(
        audit[
            "execution_contract_complete"
        ] is False,
        "Execution contract was already complete.",
    )

    require(
        set(
            audit[
                "tier_c_execution_fields_missing"
            ]
        ) == EXPECTED_MISSING_FIELDS,
        "Missing execution-field registry changed.",
    )

    require(
        codec[
            "latent_dimension"
        ]
        == FIELD_CODEC_LATENT_DIMENSION,
        "Field-codec latent dimension changed.",
    )

    require(
        codec[
            "pretraining"
        ] is False,
        "Field-codec pretraining was enabled.",
    )

    require(
        codec[
            "shared_weights_across_models"
        ] is False,
        "Field-codec weights were made shared.",
    )

    require(
        test_policy[
            "test_open_count"
        ] == 0,
        "Test-opening policy records test access.",
    )

    for key in (
        "batch_size",
        "maximum_epochs",
        "early_stopping_patience",
    ):
        require(
            key in phase3_registry,
            f"Phase 3 registry lacks {key}.",
        )

    return {
        "protocol":
            protocol,

        "codec":
            codec,

        "audit":
            audit,

        "phase3_registry":
            phase3_registry,

        "test_policy":
            test_policy,

        "required_paths":
            required_paths,
    }


def summarize_configuration_compatibility(
    configurations,
):
    rows = []

    original_ocm_dimensions = set()
    b1_ridge_values = set()
    b5_internal_dimensions = set()

    for item in configurations:
        model_id = item[
            "model_id"
        ]

        configuration_id = item[
            "configuration_id"
        ]

        configuration = item[
            "configuration"
        ]

        compatibility_change = "none"
        execution_value = None

        if model_id == "OCM":
            original_dimension = int(
                configuration[
                    "observation_dimension"
                ]
            )

            original_ocm_dimensions.add(
                original_dimension
            )

            compatibility_change = (
                "Replace inherited observation_dimension "
                "with frozen field-codec output dimension 128. "
                "This is a non-tuned interface compatibility "
                "override; all other OCM values remain unchanged."
            )

            execution_value = (
                FIELD_CODEC_LATENT_DIMENSION
            )

        elif model_id == "B1":
            ridge = float(
                configuration["ridge"]
            )

            b1_ridge_values.add(ridge)

            compatibility_change = (
                "Interpret ridge as the L2 coefficient on "
                "the affine latent-transition weight matrix. "
                "The nonlinear encoder and decoder are trained "
                "jointly from scratch."
            )

            execution_value = ridge

        elif model_id == "B5":
            latent_dimension = int(
                configuration[
                    "latent_dimension"
                ]
            )

            b5_internal_dimensions.add(
                latent_dimension
            )

            compatibility_change = (
                "Retain B5 latent_dimension as its internal "
                "bottleneck after receiving the 128-dimensional "
                "field-codec representation."
            )

            execution_value = (
                latent_dimension
            )

        rows.append(
            {
                "model_id":
                    model_id,

                "configuration_id":
                    configuration_id,

                "compatibility_change":
                    compatibility_change,

                "execution_value":
                    (
                        ""
                        if execution_value is None
                        else execution_value
                    ),

                "configuration_count_changed":
                    False,

                "validation_outcome_used":
                    False,

                "test_outcome_used":
                    False,
            }
        )

    return {
        "rows":
            rows,

        "original_ocm_observation_dimensions":
            sorted(
                original_ocm_dimensions
            ),

        "b1_ridge_values":
            sorted(
                b1_ridge_values
            ),

        "b5_internal_latent_dimensions":
            sorted(
                b5_internal_dimensions
            ),
    }


def create_execution_contract(
    sources,
    compatibility,
    optimizer_families,
):
    phase3_registry = sources[
        "phase3_registry"
    ]

    effective_batch_size = int(
        phase3_registry[
            "batch_size"
        ]
    )

    require(
        effective_batch_size
        % PHYSICAL_MICROBATCH_SIZE
        == 0,
        (
            "Effective batch size is not divisible "
            "by the frozen microbatch size."
        ),
    )

    accumulation_steps = (
        effective_batch_size
        // PHYSICAL_MICROBATCH_SIZE
    )

    contract = {
        "phase":
            (
                "4D-R3.2 Tier C v4 predictive "
                "execution-contract freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "contract_frozen_before_smoke_test":
            True,

        "field_codec_training_scope": {
            "mode":
                "joint_end_to_end_from_random_initialization",

            "encoder_transition_decoder_jointly_optimized":
                True,

            "shared_architecture_across_models":
                True,

            "shared_weights_across_models":
                False,

            "pretraining":
                False,

            "diagnostic_weight_reuse":
                False,

            "one_independent_codec_per_model_run":
                True,
        },

        "field_codec_reconstruction_objective": {
            "additional_codec_only_reconstruction_loss":
                False,

            "added_reconstruction_loss_weight":
                ADDED_CODEC_RECONSTRUCTION_WEIGHT,

            "model_intrinsic_reconstruction_terms":
                (
                    "Retain reconstruction terms already "
                    "present in the frozen Tier B model-family "
                    "objective, including OCM and any B5 "
                    "autoencoding term. Do not add a new "
                    "standalone codec objective."
                ),
        },

        "predictive_field_objective": {
            "target":
                "stored noisy normalized field",

            "loss":
                "mean_squared_error",

            "base_weight":
                PREDICTIVE_FIELD_LOSS_WEIGHT,

            "adaptation_rule":
                (
                    "Every inherited Tier B observation-space "
                    "MSE term is replaced at the same position "
                    "in the objective by decoded normalized "
                    "field-space MSE. Existing model-specific "
                    "weights remain unchanged."
                ),

            "clean_targets_used_for_training":
                False,

            "privileged_state_ids_used_for_training":
                False,
        },

        "latent_transition_loss_weight": {
            "B1":
                (
                    "No direct latent target loss. Apply the "
                    "configuration ridge value only as L2 "
                    "penalty on the affine transition matrix."
                ),

            "B2":
                "No added Tier C latent loss.",

            "B3":
                "No added Tier C latent loss.",

            "B4":
                "No added Tier C latent loss.",

            "B5":
                (
                    "Retain only latent or reconstruction terms "
                    "already intrinsic to the frozen Tier B B5 "
                    "objective."
                ),

            "OCM":
                (
                    "Retain each configuration's frozen "
                    "lambda_transition, lambda_rollout, and "
                    "lambda_determinism values."
                ),
        },

        "ocm_objective_semantics": {
            "reconstruction_coefficient":
                1.0,

            "one_step_coefficient":
                1.0,

            "transition_coefficient":
                "configuration.lambda_transition",

            "rollout_coefficient":
                "configuration.lambda_rollout",

            "determinism_coefficient":
                "configuration.lambda_determinism",

            "temperature":
                "configuration.temperature",

            "source":
                str(
                    TIER_B_TRAINING_SOURCE
                ),

            "source_sha256":
                sha256_file(
                    TIER_B_TRAINING_SOURCE
                ),
        },

        "rollout_policy": {
            "one_step_branch":
                (
                    "Use the encoded observed source field "
                    "for each valid one-step transition."
                ),

            "rollout_branch":
                (
                    "Begin from the encoded initial observed "
                    "field and proceed autoregressively through "
                    "the operation sequence."
                ),

            "scheduled_sampling":
                False,

            "teacher_forcing_inside_autoregressive_rollout":
                False,

            "rollout_weights":
                (
                    "Use the inherited model-family objective "
                    "and inherited configuration weights."
                ),
        },

        "variable_length_batching_policy": {
            "padding":
                "pad_to_longest_trajectory_in_physical_microbatch",

            "step_mask_required":
                True,

            "padded_steps_contribute_to_loss":
                False,

            "per_sequence_aggregation":
                (
                    "Average valid pixels, channels, and time "
                    "steps within each trajectory first."
                ),

            "batch_aggregation":
                "mean_of_per_sequence_losses",
        },

        "batching": {
            "effective_batch_size_trajectories":
                effective_batch_size,

            "physical_microbatch_size_trajectories":
                PHYSICAL_MICROBATCH_SIZE,

            "gradient_accumulation_steps":
                accumulation_steps,

            "optimizer_step_frequency":
                (
                    "One optimizer step after the frozen "
                    "number of accumulated physical "
                    "microbatches."
                ),

            "partial_final_effective_batch":
                (
                    "Scale accumulated gradients by the actual "
                    "number of trajectories contributing to the "
                    "final optimizer step."
                ),
        },

        "optimizer_scope": {
            "B2_B3_B4_B5_OCM":
                (
                    "Use the exact optimizer construction in "
                    "the SHA-256-pinned Phase 3D training source "
                    "with each configuration's learning rate. "
                    "Apply it jointly to encoder, transition "
                    "model, and decoder parameters."
                ),

            "detected_optimizer_families_in_source":
                optimizer_families,

            "B1":
                {
                    "optimizer":
                        "torch.optim.Adam",

                    "learning_rate":
                        B1_FIXED_LEARNING_RATE,

                    "betas":
                        [
                            0.9,
                            0.999,
                        ],

                    "epsilon":
                        1e-8,

                    "weight_decay":
                        0.0,
                },
        },

        "learning_rate_scope": {
            "B2_B3_B4_B5_OCM":
                (
                    "The configuration learning_rate applies "
                    "uniformly to all trainable encoder, "
                    "transition-model, and decoder parameters."
                ),

            "B1":
                B1_FIXED_LEARNING_RATE,

            "separate_codec_learning_rate":
                False,

            "learning_rate_scheduler_added":
                False,
        },

        "weight_decay_scope": {
            "global_optimizer_weight_decay_added":
                False,

            "B1_ridge_penalty_scope":
                (
                    "Affine latent-transition weight matrix "
                    "only; exclude bias, encoder, decoder, and "
                    "normalization parameters."
                ),

            "other_models":
                (
                    "Retain only regularization already present "
                    "in the SHA-256-pinned Tier B implementation."
                ),
        },

        "gradient_clipping_policy": {
            "method":
                "global_l2_norm",

            "maximum_norm":
                GRADIENT_CLIP_NORM,

            "applied_after_gradient_accumulation":
                True,

            "applied_before_optimizer_step":
                True,
        },

        "mixed_precision_policy": {
            "enabled":
                MIXED_PRECISION_ENABLED,

            "dataset_float16_cast_to_compute_dtype":
                "float32",

            "autocast":
                False,

            "gradient_scaler":
                False,
        },

        "codec_initialization_policy": {
            "seed_before_model_construction":
                True,

            "seed_source":
                (
                    "Frozen run seed; B1 uses seed 11 for "
                    "every noise condition."
                ),

            "convolution_and_linear_initialization":
                "PyTorch module defaults",

            "normalization_initialization":
                "PyTorch module defaults",

            "manual_post_initialization_rewrite":
                False,
        },

        "training_duration": {
            "maximum_epochs":
                int(
                    phase3_registry[
                        "maximum_epochs"
                    ]
                ),

            "early_stopping_patience":
                int(
                    phase3_registry[
                        "early_stopping_patience"
                    ]
                ),

            "checkpoint_selection_cell":
                "val_joint",

            "checkpoint_selection_metric":
                "noisy_target_rollout_mse",

            "selection_direction":
                "lower_is_better",
        },

        "checkpoint_contents": [
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "best_validation_metric",
            "full_configuration",
            "execution_contract_sha256",
            "normalization_artifact_sha256",
            "training_manifest_sha256",
            "Python_random_state",
            "NumPy_random_state",
            "PyTorch_CPU_random_state",
            "PyTorch_CUDA_random_state_if_available",
        ],

        "b1_contract_amendment": {
            "previous_description":
                "deterministic fitted affine baseline",

            "authoritative_description":
                (
                    "single-seed end-to-end affine "
                    "latent-transition baseline with "
                    "nonlinear field codec"
                ),

            "global_field_space_affine":
                False,

            "latent_transition_affine":
                True,

            "trainable_encoder":
                True,

            "trainable_decoder":
                True,

            "fixed_seed":
                B1_FIXED_SEED,

            "one_fit_per_noise_condition":
                True,

            "ridge_values":
                compatibility[
                    "b1_ridge_values"
                ],

            "planned_run_count_changed":
                False,
        },

        "ocm_interface_amendment": {
            "original_observation_dimensions":
                compatibility[
                    "original_ocm_observation_dimensions"
                ],

            "execution_observation_dimension":
                FIELD_CODEC_LATENT_DIMENSION,

            "override_is_tuned":
                False,

            "other_ocm_configuration_values_changed":
                False,

            "configuration_count_changed":
                False,
        },

        "b5_interface_semantics": {
            "field_codec_output_dimension":
                FIELD_CODEC_LATENT_DIMENSION,

            "internal_b5_latent_dimensions":
                compatibility[
                    "b5_internal_latent_dimensions"
                ],

            "internal_latent_dimension_retained":
                True,
        },

        "determinism": {
            "Python_seeded":
                True,

            "NumPy_seeded":
                True,

            "PyTorch_CPU_seeded":
                True,

            "PyTorch_CUDA_seeded":
                True,

            "cudnn_benchmark":
                False,

            "cudnn_deterministic":
                True,

            "torch_deterministic_algorithms":
                True,

            "deterministic_failure_policy":
                (
                    "Stop rather than silently falling back "
                    "to a nondeterministic implementation."
                ),
        },

        "test_access": {
            "test_generation_authorized":
                False,

            "test_evaluation_authorized":
                False,

            "test_open_count":
                0,
        },
    }

    return contract


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "phase4dr32_execution_contract_summary.json"
    )

    if summary_path.exists():
        existing = load_json(
            summary_path
        )

        if existing.get(
            "phase4dr32_status"
        ) == "execution_contract_frozen":
            print(
                "Phase 4D-R3.2 execution contract "
                "is already frozen."
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            return

    print(
        "[1/5] Validating Phase 4D-R3 and "
        "Phase 4D-R3.1 sources"
    )

    sources = validate_sources()

    print(
        "[2/5] Loading the 43 frozen configurations"
    )

    configurations = (
        load_configuration_registry()
    )

    print(
        "[3/5] Freezing compatibility amendments"
    )

    compatibility = (
        summarize_configuration_compatibility(
            configurations
        )
    )

    write_csv(
        OUTPUT_DIR
        / "configuration_compatibility_registry.csv",
        compatibility["rows"],
    )

    optimizer_families = (
        detect_optimizer_families(
            TIER_B_TRAINING_SOURCE
        )
    )

    print(
        "[4/5] Freezing the complete execution contract"
    )

    contract = create_execution_contract(
        sources=sources,
        compatibility=compatibility,
        optimizer_families=(
            optimizer_families
        ),
    )

    contract_path = (
        OUTPUT_DIR
        / "tier_c_v4_execution_contract.json"
    )

    write_json(
        contract_path,
        contract,
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
        / "execution_contract_source_hashes.json",
        source_hashes,
    )

    print(
        "[5/5] Freezing authorization state"
    )

    execution_contract_sha256 = (
        sha256_file(
            contract_path
        )
    )

    summary = {
        "phase":
            (
                "4D-R3.2 Tier C v4 predictive "
                "execution-contract freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_phase4dr3_status":
            sources[
                "protocol"
            ][
                "phase4dr3_status"
            ],

        "source_phase4dr31_audit_status":
            sources[
                "audit"
            ][
                "audit_status"
            ],

        "source_missing_execution_field_count":
            len(
                EXPECTED_MISSING_FIELDS
            ),

        "resolved_execution_field_count":
            len(
                EXPECTED_MISSING_FIELDS
            ),

        "unresolved_execution_fields":
            [],

        "configuration_count":
            len(configurations),

        "configuration_counts":
            dict(
                Counter(
                    item["model_id"]
                    for item
                    in configurations
                )
            ),

        "effective_batch_size":
            int(
                sources[
                    "phase3_registry"
                ][
                    "batch_size"
                ]
            ),

        "physical_microbatch_size":
            PHYSICAL_MICROBATCH_SIZE,

        "gradient_accumulation_steps":
            (
                int(
                    sources[
                        "phase3_registry"
                    ][
                        "batch_size"
                    ]
                )
                // PHYSICAL_MICROBATCH_SIZE
            ),

        "maximum_epochs":
            int(
                sources[
                    "phase3_registry"
                ][
                    "maximum_epochs"
                ]
            ),

        "early_stopping_patience":
            int(
                sources[
                    "phase3_registry"
                ][
                    "early_stopping_patience"
                ]
            ),

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "mixed_precision_enabled":
            MIXED_PRECISION_ENABLED,

        "added_codec_reconstruction_weight":
            ADDED_CODEC_RECONSTRUCTION_WEIGHT,

        "predictive_field_loss_weight":
            PREDICTIVE_FIELD_LOSS_WEIGHT,

        "b1_reporting_name":
            (
                "single-seed end-to-end affine "
                "latent-transition baseline with "
                "nonlinear field codec"
            ),

        "b1_global_field_space_affine":
            False,

        "ocm_execution_observation_dimension":
            FIELD_CODEC_LATENT_DIMENSION,

        "execution_contract_path":
            str(contract_path),

        "execution_contract_sha256":
            execution_contract_sha256,

        "execution_contract_complete":
            True,

        "implementation_code_written":
            False,

        "implementation_smoke_test_authorized":
            True,

        "implementation_smoke_test_executed":
            False,

        "tuning_execution_authorized":
            False,

        "final_fit_execution_authorized":
            False,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "test_open_count":
            0,

        "tier_c_v5_authorized":
            False,

        "prior_protocol_files_modified":
            False,

        "phase4dr32_status":
            "execution_contract_frozen",

        "next_phase":
            (
                "4E-R3 Tier C v4 predictive "
                "implementation smoke test"
            ),
    }

    write_json(
        summary_path,
        summary,
    )

    print(
        "Phase 4D-R3.2 Tier C v4 execution "
        "contract frozen."
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
