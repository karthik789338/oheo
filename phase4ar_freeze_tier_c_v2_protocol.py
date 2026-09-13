from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


PHASE1D_DIR = Path(
    "outputs/phase1d_compositional_splits"
)

PHASE3H_DIR = Path(
    "outputs/phase3h_tier_b_synthesis"
)

PHASE4A_V1_DIR = Path(
    "outputs/phase4a_tier_c_physical_protocol"
)

PHASE4C_V1_DIR = Path(
    "outputs/phase4c_tier_c_audit"
)

OUTPUT_DIR = Path(
    "outputs/phase4ar_tier_c_v2_protocol"
)


PROTOCOL_VERSION = "tier_c_v2"

STATE_COUNT = 8
PRIMITIVE_OPERATION_COUNT = 6
SEMIGROUP_SIZE = 104
INFORMATION_CLASS_COUNT = 9

CARRIER_COUNT = 160

CARRIER_SPLIT_COUNTS = {
    "train": 96,
    "val": 32,
    "test": 32,
}

TRANSFORMATION_SPLIT_COUNTS = {
    "train": 86,
    "val": 9,
    "test": 9,
}

FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32

PHYSICAL_CARRIER_DIMENSION = 8

FROZEN_NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

FROZEN_FINAL_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

DEVELOPMENT_SEED = 11
DEVELOPMENT_NOISE_FRACTION = 0.25

CARRIER_SPLIT_SEED = 51026
CARRIER_PARAMETER_SEED = 51027
FIELD_INITIALIZATION_SEED = 51028
SOLVER_SEED = 51029
NOISE_SEED = 51030
DEVELOPMENT_MANIFEST_SEED = 51032
SEALED_TEST_MANIFEST_SEED = 51033

DIAGNOSTIC_SEEDS = (
    52011,
    52023,
    52037,
)

INITIAL_TIME_STEP = 0.005
MINIMUM_TIME_STEP = (
    INITIAL_TIME_STEP / 32.0
)
MAXIMUM_STEP_HALVINGS = 5

TRAINING_SEQUENCE_COUNT = 1376
VALIDATION_SEQUENCE_COUNT = 144
TEST_SEQUENCE_COUNT = 144

TRAJECTORIES_PER_SEQUENCE = 8

OCM_TUNING_CONFIGURATION_COUNT = 24

BASELINE_TUNING_CONFIGURATION_COUNTS = {
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
}

FINAL_FIT_COUNTS = {
    "OCM": 25,
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
}


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
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


def sha256_file(path: Path):
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


def validate_prior_results():
    phase3h = load_json(
        PHASE3H_DIR
        / "phase3h_summary.json"
    )

    phase4a_v1 = load_json(
        PHASE4A_V1_DIR
        / "phase4a_summary.json"
    )

    rejection = load_json(
        PHASE4C_V1_DIR
        / "phase4c_rejection_summary.json"
    )

    rejection_record = load_json(
        PHASE4C_V1_DIR
        / "phase4c_rejection_record.json"
    )

    if (
        phase3h["phase3h_status"]
        != "passed"
    ):
        raise AssertionError(
            "Phase 3H has not passed."
        )

    if (
        phase3h["phase3_results_frozen"]
        is not True
    ):
        raise AssertionError(
            "Phase 3 is not frozen."
        )

    if (
        phase4a_v1["phase4a_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Tier C v1 protocol was not frozen."
        )

    if (
        rejection[
            "phase4c_rejection_freeze_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Tier C v1 rejection is not frozen."
        )

    if (
        rejection["tier_c_v1_rejected"]
        is not True
    ):
        raise AssertionError(
            "Tier C v1 was not rejected."
        )

    if (
        rejection[
            "phase4d_training_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "Tier C v1 incorrectly authorizes training."
        )

    if (
        rejection_record[
            "rejection_is_final"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v1 rejection is not final."
        )

    expected_failed_gates = {
        "numerical_physics",
        "field_manifold_quality",
        "privileged_state_separability",
    }

    if set(
        rejection[
            "failed_acceptance_gates"
        ]
    ) != expected_failed_gates:
        raise AssertionError(
            "Unexpected Tier C v1 rejection gates."
        )

    return {
        "phase3h": phase3h,
        "phase4a_v1": phase4a_v1,
        "rejection": rejection,
        "rejection_record": rejection_record,
    }


def load_v1_carrier_split():
    rows = load_csv(
        PHASE4A_V1_DIR
        / "carrier_reuse_manifest.csv"
    )

    if len(rows) != CARRIER_COUNT:
        raise AssertionError(
            "Tier C v1 carrier count changed."
        )

    split_lookup = {
        int(row["carrier_id"]):
            row["split"]
        for row in rows
    }

    if set(split_lookup) != set(
        range(CARRIER_COUNT)
    ):
        raise AssertionError(
            "Tier C v1 carrier IDs changed."
        )

    return split_lookup


def create_fresh_v2_carrier_split(
    v1_split_lookup,
):
    generator = np.random.default_rng(
        CARRIER_SPLIT_SEED
    )

    v1_test_ids = np.asarray(
        [
            carrier_id
            for carrier_id, split
            in v1_split_lookup.items()
            if split == "test"
        ],
        dtype=np.int64,
    )

    v1_non_test_ids = np.asarray(
        [
            carrier_id
            for carrier_id, split
            in v1_split_lookup.items()
            if split != "test"
        ],
        dtype=np.int64,
    )

    if len(v1_test_ids) != 32:
        raise AssertionError(
            "Tier C v1 test-carrier count changed."
        )

    if len(v1_non_test_ids) != 128:
        raise AssertionError(
            "Tier C v1 non-test-carrier count changed."
        )

    shuffled_non_test = generator.permutation(
        v1_non_test_ids
    )

    v2_test_ids = shuffled_non_test[:32]

    remaining_ids = np.concatenate(
        [
            shuffled_non_test[32:],
            generator.permutation(
                v1_test_ids
            ),
        ]
    )

    remaining_ids = generator.permutation(
        remaining_ids
    )

    v2_val_ids = remaining_ids[:32]
    v2_train_ids = remaining_ids[32:]

    if len(v2_train_ids) != 96:
        raise AssertionError(
            "Tier C v2 training-carrier count changed."
        )

    if len(v2_val_ids) != 32:
        raise AssertionError(
            "Tier C v2 validation-carrier count changed."
        )

    if len(v2_test_ids) != 32:
        raise AssertionError(
            "Tier C v2 test-carrier count changed."
        )

    if set(v2_test_ids) & set(v1_test_ids):
        raise AssertionError(
            "Tier C v2 test IDs overlap v1 test IDs."
        )

    split_members = {
        "train":
            set(
                int(value)
                for value in v2_train_ids
            ),

        "val":
            set(
                int(value)
                for value in v2_val_ids
            ),

        "test":
            set(
                int(value)
                for value in v2_test_ids
            ),
    }

    if (
        split_members["train"]
        & split_members["val"]
    ):
        raise AssertionError(
            "Tier C v2 train and validation overlap."
        )

    if (
        split_members["train"]
        & split_members["test"]
    ):
        raise AssertionError(
            "Tier C v2 train and test overlap."
        )

    if (
        split_members["val"]
        & split_members["test"]
    ):
        raise AssertionError(
            "Tier C v2 validation and test overlap."
        )

    rows = []

    for split in (
        "train",
        "val",
        "test",
    ):
        ordered_ids = sorted(
            split_members[split]
        )

        for split_position, carrier_id in enumerate(
            ordered_ids
        ):
            rows.append(
                {
                    "carrier_id":
                        carrier_id,

                    "split":
                        split,

                    "split_position":
                        split_position,

                    "v1_split":
                        v1_split_lookup[
                            carrier_id
                        ],

                    "same_split_as_v1":
                        bool(
                            v1_split_lookup[
                                carrier_id
                            ]
                            == split
                        ),

                    "v2_parameter_seed":
                        CARRIER_PARAMETER_SEED,

                    "v2_field_initialization_seed":
                        FIELD_INITIALIZATION_SEED,

                    "v2_physical_parameters_generated":
                        False,

                    "v2_field_generated":
                        False,
                }
            )

    rows.sort(
        key=lambda row:
            row["carrier_id"]
    )

    counts = Counter(
        row["split"]
        for row in rows
    )

    if dict(counts) != CARRIER_SPLIT_COUNTS:
        raise AssertionError(
            "Tier C v2 carrier counts changed."
        )

    return rows


def validate_transformation_split():
    rows = load_csv(
        PHASE1D_DIR
        / "transformation_split.csv"
    )

    counts = Counter(
        row["split"]
        for row in rows
    )

    if dict(counts) != TRANSFORMATION_SPLIT_COUNTS:
        raise AssertionError(
            "Transformation split changed."
        )

    if len(rows) != SEMIGROUP_SIZE:
        raise AssertionError(
            "Semigroup size changed."
        )

    return rows


def create_carrier_parameter_schema():
    return [
        {
            "parameter_index": 0,
            "parameter_name":
                "mean_phase_fraction",
            "minimum": 0.35,
            "maximum": 0.65,
            "state_dependent": False,
            "role":
                (
                    "Carrier-specific conserved phase "
                    "fraction."
                ),
        },
        {
            "parameter_index": 1,
            "parameter_name":
                "base_length_scale_pixels",
            "minimum": 2.5,
            "maximum": 7.5,
            "state_dependent": False,
            "role":
                (
                    "Carrier-specific initial spatial "
                    "correlation length."
                ),
        },
        {
            "parameter_index": 2,
            "parameter_name":
                "interface_width",
            "minimum": 0.9,
            "maximum": 1.7,
            "state_dependent": False,
            "role":
                (
                    "Carrier-specific diffuse-interface "
                    "width."
                ),
        },
        {
            "parameter_index": 3,
            "parameter_name":
                "phase_mobility",
            "minimum": 0.75,
            "maximum": 1.25,
            "state_dependent": False,
            "role":
                (
                    "Carrier-specific base phase mobility."
                ),
        },
        {
            "parameter_index": 4,
            "parameter_name":
                "orientation_correlation_length",
            "minimum": 2.0,
            "maximum": 6.0,
            "state_dependent": False,
            "role":
                (
                    "Carrier-specific orientation texture "
                    "length scale."
                ),
        },
        {
            "parameter_index": 5,
            "parameter_name":
                "mean_defect_fraction",
            "minimum": 0.08,
            "maximum": 0.22,
            "state_dependent": False,
            "role":
                (
                    "Carrier-specific defect mean preserved "
                    "across states."
                ),
        },
        {
            "parameter_index": 6,
            "parameter_name":
                "defect_diffusivity",
            "minimum": 0.65,
            "maximum": 1.35,
            "state_dependent": False,
            "role":
                (
                    "Carrier-specific base defect "
                    "diffusivity."
                ),
        },
        {
            "parameter_index": 7,
            "parameter_name":
                "bulk_driving_bias",
            "minimum": -0.18,
            "maximum": 0.18,
            "state_dependent": False,
            "role":
                (
                    "Carrier-specific bulk-driving "
                    "perturbation."
                ),
        },
    ]


def create_state_morphology_regimes():
    """
    The eight states are a complete 2 x 2 x 2 factorial design.

    State-dependent information is expressed through distributed
    morphology rather than state-dependent channel means.
    """

    rows = []

    state_id = 0

    for coarsening_level in (
        0,
        1,
    ):
        for anisotropy_level in (
            0,
            1,
        ):
            for defect_recovery_level in (
                0,
                1,
            ):
                long_anneal = bool(
                    coarsening_level == 1
                )

                strong_anisotropy = bool(
                    anisotropy_level == 1
                )

                strong_recovery = bool(
                    defect_recovery_level == 1
                )

                rows.append(
                    {
                        "state_id":
                            state_id,

                        "state_code":
                            (
                                f"{coarsening_level}"
                                f"{anisotropy_level}"
                                f"{defect_recovery_level}"
                            ),

                        "coarsening_level":
                            coarsening_level,

                        "anisotropy_level":
                            anisotropy_level,

                        "defect_recovery_level":
                            defect_recovery_level,

                        "quench_steps":
                            48,

                        "anneal_steps":
                            (
                                176
                                if long_anneal
                                else 72
                            ),

                        "phase_mobility_multiplier":
                            (
                                1.25
                                if long_anneal
                                else 0.90
                            ),

                        "orientation_anisotropy_harmonic":
                            (
                                4
                                if strong_anisotropy
                                else 2
                            ),

                        "orientation_anisotropy_strength":
                            (
                                0.34
                                if strong_anisotropy
                                else 0.08
                            ),

                        "orientation_relaxation_multiplier":
                            (
                                1.30
                                if strong_anisotropy
                                else 0.85
                            ),

                        "defect_diffusivity_multiplier":
                            (
                                1.35
                                if strong_recovery
                                else 0.65
                            ),

                        "defect_interface_coupling":
                            (
                                1.55
                                if strong_recovery
                                else 0.85
                            ),

                        "defect_localization_width":
                            (
                                0.75
                                if strong_recovery
                                else 1.35
                            ),

                        "phase_mean_state_dependent":
                            False,

                        "defect_mean_state_dependent":
                            False,

                        "distributed_morphology_signature":
                            True,
                    }
                )

                state_id += 1

    if len(rows) != STATE_COUNT:
        raise AssertionError(
            "Expected eight Tier C v2 state regimes."
        )

    if {
        row["state_id"]
        for row in rows
    } != set(
        range(STATE_COUNT)
    ):
        raise AssertionError(
            "Tier C v2 state IDs changed."
        )

    codes = {
        row["state_code"]
        for row in rows
    }

    if len(codes) != STATE_COUNT:
        raise AssertionError(
            "Tier C v2 state codes are not unique."
        )

    return rows


def create_numerical_stabilization_protocol():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "initial_time_step":
            INITIAL_TIME_STEP,

        "minimum_time_step":
            MINIMUM_TIME_STEP,

        "maximum_step_halvings":
            MAXIMUM_STEP_HALVINGS,

        "adaptive_step_rejection":
            True,

        "accepted_step_requirements": [
            (
                "All fields and energy diagnostics are finite."
            ),
            (
                "Total free energy does not increase beyond "
                "the frozen numerical tolerance."
            ),
            (
                "The bounded phase field remains within "
                "[-1.000001, 1.000001]."
            ),
            (
                "The bounded defect field remains within "
                "[-0.000001, 1.000001]."
            ),
        ],

        "phase_representation": {
            "method":
                (
                    "Bounded mass-conserving phase projection."
                ),

            "rule":
                (
                    "After each proposed phase update, apply a "
                    "bounded monotone projection and solve one scalar "
                    "offset so the carrier-specific phase mean is "
                    "restored."
                ),

            "target_mean_source":
                (
                    "Carrier-specific mean_phase_fraction; "
                    "independent of state."
                ),

            "clipping_used_as_final_repair":
                False,
        },

        "defect_representation": {
            "method":
                (
                    "Bounded logistic defect representation "
                    "with scalar mean projection."
                ),

            "rule":
                (
                    "The stored defect field is obtained from a "
                    "logistic latent field. A scalar offset preserves "
                    "the carrier-specific target defect mean."
                ),

            "target_mean_source":
                (
                    "Carrier-specific mean_defect_fraction; "
                    "independent of state."
                ),

            "clipping_used_as_final_repair":
                False,
        },

        "orientation_representation": {
            "method":
                (
                    "Two-component doubled-angle orientation "
                    "field."
                ),

            "normalization":
                (
                    "Unit-norm projection after every accepted "
                    "orientation update."
                ),
        },

        "energy_tolerance": {
            "absolute":
                1e-7,

            "relative":
                1e-5,
        },

        "simulation_failure_rule":
            (
                "If a proposed step still violates an acceptance "
                "condition after five halvings, reject the complete "
                "carrier-state simulation."
            ),

        "failed_simulations_permitted":
            0,
    }


def create_test_sealing_policy():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "test_carrier_split_frozen":
            True,

        "test_carrier_parameters_generated_before_checkpoint_freeze":
            False,

        "test_clean_fields_generated_before_checkpoint_freeze":
            False,

        "test_noisy_fields_generated_before_checkpoint_freeze":
            False,

        "test_trajectory_manifests_generated_before_checkpoint_freeze":
            False,

        "test_metrics_computed_before_checkpoint_freeze":
            False,

        "sealed_test_manifest_seed":
            SEALED_TEST_MANIFEST_SEED,

        "development_manifest_seed":
            DEVELOPMENT_MANIFEST_SEED,

        "development_cells_available_before_checkpoint_freeze": [
            "train_joint",
            "val_composition",
            "val_carrier",
            "val_joint",
        ],

        "sealed_cells": [
            "test_iid_pairing",
            "test_composition",
            "test_carrier",
            "test_joint",
        ],

        "test_generation_authorization_condition":
            (
                "All 130 final model checkpoints must be complete, "
                "numerically valid, and frozen."
            ),

        "test_generation_phase":
            "4G-R",

        "test_fields_may_be_used_for_dataset_acceptance":
            False,

        "test_fields_may_be_used_for_model_selection":
            False,
    }


def create_acceptance_gates():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "acceptance_scope":
            (
                "Training and validation carriers only. "
                "No test field or test trajectory may exist."
            ),

        "numerical_physics": {
            "required": True,

            "all_values_finite":
                True,

            "minimum_energy_nonincrease_fraction":
                0.99,

            "maximum_phase_mean_drift":
                1e-5,

            "phase_minimum":
                -1.000001,

            "phase_maximum":
                1.000001,

            "defect_minimum":
                -0.000001,

            "defect_maximum":
                1.000001,

            "maximum_failed_simulations":
                0,

            "maximum_unresolved_step_rejections":
                0,
        },

        "basic_manifold_quality": {
            "required": True,

            "all_training_channels_nonconstant":
                True,

            "every_state_has_positive_carrier_variation":
                True,

            "maximum_exact_duplicate_count":
                0,
        },

        "state_separability": {
            "required": True,

            "diagnostic_model":
                (
                    "Frozen residual convolutional state decoder."
                ),

            "diagnostic_seed_count":
                len(DIAGNOSTIC_SEEDS),

            "diagnostic_seeds":
                list(DIAGNOSTIC_SEEDS),

            "training_data":
                "clean training-carrier fields only",

            "selection_data":
                "clean validation-carrier fields only",

            "minimum_mean_validation_accuracy":
                0.80,

            "minimum_worst_seed_validation_accuracy":
                0.75,

            "test_fields_used":
                False,

            "diagnostic_checkpoints_eligible_for_predictive_use":
                False,
        },

        "global_shortcut_control": {
            "required": True,

            "diagnostic_features": [
                "per-channel mean",
                "per-channel variance",
                "per-channel skewness",
                "per-channel kurtosis",
                "per-channel 10th percentile",
                "per-channel median",
                "per-channel 90th percentile",
            ],

            "classifier":
                "regularized multinomial logistic regression",

            "maximum_validation_accuracy":
                0.95,

            "state_dependent_phase_mean_allowed":
                False,

            "state_dependent_defect_mean_allowed":
                False,

            "test_fields_used":
                False,
        },

        "affine_shortcut_control": {
            "required": True,

            "diagnostic":
                (
                    "Training-only PCA-128 plus ridge affine "
                    "primitive operators."
                ),

            "minimum_validation_continuous_mse":
                1e-8,

            "validation_exact_primitive_rate_must_be_below":
                1.0,

            "fraction_validation_carriers_all_104_exact_must_be_below":
                1.0,

            "test_fields_used":
                False,
        },

        "data_integrity": {
            "required": True,

            "carrier_splits_disjoint":
                True,

            "transformation_split_unchanged":
                True,

            "test_field_files_absent":
                True,

            "test_manifest_files_absent":
                True,

            "validation_statistics_used_for_normalization":
                False,

            "test_statistics_used_for_normalization":
                False,
        },

        "acceptance_decision": {
            "all_required_gates_must_pass":
                True,

            "thresholds_may_be_lowered_after_execution":
                False,

            "diagnostic_architectures_may_be_replaced_after_execution":
                False,

            "failed_revision_must_be_versioned_and_frozen":
                True,
        },
    }


def create_hypothesis_registry():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "registry_status":
            "frozen",

        "hypotheses": [
            {
                "hypothesis_id": "H4R-1",

                "statement":
                    (
                        "Distributed morphology controls can make "
                        "all eight process-memory states identifiable "
                        "across validation carriers without relying "
                        "on state-dependent global phase or defect "
                        "means."
                    ),

                "primary_metric":
                    (
                        "Mean validation accuracy of the frozen "
                        "three-seed convolutional diagnostic."
                    ),

                "acceptance_threshold":
                    0.80,
            },
            {
                "hypothesis_id": "H4R-2",

                "statement":
                    (
                        "The revised bounded simulator satisfies the "
                        "frozen phase, defect, conservation, and "
                        "energy diagnostics without post-hoc clipping."
                    ),

                "primary_metric":
                    (
                        "Numerical-physics acceptance gate."
                    ),
            },
            {
                "hypothesis_id": "H4R-3",

                "statement":
                    (
                        "The revised spatial manifold remains "
                        "nontrivial for global-statistics and "
                        "low-rank affine diagnostics."
                    ),

                "primary_metric":
                    (
                        "Validation global-statistics accuracy and "
                        "validation affine structural recovery."
                    ),
            },
            {
                "hypothesis_id": "H4R-4",

                "statement":
                    (
                        "Once checkpoints are frozen, unseen "
                        "carrier morphology will remain a larger "
                        "predictive challenge than unseen operation "
                        "composition for the strongest models."
                    ),

                "primary_metric":
                    (
                        "Final sealed carrier-only versus "
                        "composition-only MSE gaps."
                    ),

                "evaluated_before_checkpoint_freeze":
                    False,
            },
        ],
    }


def create_model_family_registry():
    source_rows = load_csv(
        PHASE4A_V1_DIR
        / "model_family_registry.csv"
    )

    if len(source_rows) != 7:
        raise AssertionError(
            "Tier C v1 model-family count changed."
        )

    rows = []

    for row in source_rows:
        updated = dict(row)

        updated[
            "protocol_version"
        ] = PROTOCOL_VERSION

        updated[
            "architecture_changed_from_v1"
        ] = False

        updated[
            "test_fields_available_during_development"
        ] = False

        rows.append(updated)

    return rows


def create_training_protocol():
    baseline_count = sum(
        BASELINE_TUNING_CONFIGURATION_COUNTS.values()
    )

    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "development_seed":
            DEVELOPMENT_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE_FRACTION,

        "training_cell":
            "train_joint",

        "checkpoint_selection_cell":
            "val_joint",

        "selection_metric":
            (
                "val_joint noisy-target spatial rollout MSE only"
            ),

        "development_cells": [
            "train_joint",
            "val_composition",
            "val_carrier",
            "val_joint",
        ],

        "test_cells": [
            "test_iid_pairing",
            "test_composition",
            "test_carrier",
            "test_joint",
        ],

        "test_cells_generated_during_tuning":
            False,

        "test_cells_generated_during_final_fitting":
            False,

        "clean_targets_used_for_training":
            False,

        "clean_targets_used_for_selection":
            False,

        "structural_labels_used_for_training":
            False,

        "physics_diagnostics_used_for_model_selection":
            False,

        "ocm_tuning_configuration_count":
            OCM_TUNING_CONFIGURATION_COUNT,

        "baseline_tuning_configuration_counts":
            BASELINE_TUNING_CONFIGURATION_COUNTS,

        "total_tuning_configuration_count":
            OCM_TUNING_CONFIGURATION_COUNT
            + baseline_count,

        "final_fit_counts":
            FINAL_FIT_COUNTS,

        "total_final_fit_count":
            sum(
                FINAL_FIT_COUNTS.values()
            ),

        "final_seeds":
            list(
                FROZEN_FINAL_SEEDS
            ),

        "noise_levels":
            list(
                FROZEN_NOISE_LEVELS
            ),

        "fresh_initialization_required":
            True,

        "tuning_checkpoints_eligible_for_final_use":
            False,

        "final_checkpoint_freeze_required_before_test_generation":
            True,
    }


def create_manifest_protocol():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "transformation_split_reused":
            True,

        "transformation_split_counts":
            TRANSFORMATION_SPLIT_COUNTS,

        "development_manifest_seed":
            DEVELOPMENT_MANIFEST_SEED,

        "sealed_test_manifest_seed":
            SEALED_TEST_MANIFEST_SEED,

        "training_sequence_count":
            TRAINING_SEQUENCE_COUNT,

        "validation_sequence_count":
            VALIDATION_SEQUENCE_COUNT,

        "test_sequence_count":
            TEST_SEQUENCE_COUNT,

        "trajectories_per_sequence":
            TRAJECTORIES_PER_SEQUENCE,

        "development_manifests_generated_in_phase":
            "4B-R",

        "test_manifests_generated_in_phase":
            "4G-R",

        "test_manifest_generated_before_checkpoint_freeze":
            False,

        "sequence_generator_code_frozen_before_data_generation":
            True,
    }


def create_revision_plan():
    return {
        "4A-R":
            (
                "Freeze Tier C v2 simulator, fresh carrier split, "
                "state morphology regimes, acceptance gates, and "
                "test-sealing policy."
            ),

        "4B-R":
            (
                "Generate only training and validation carrier-state "
                "fields and the four development trajectory cells."
            ),

        "4C-R":
            (
                "Audit physics, morphology, state separability, "
                "global shortcuts, affine shortcuts, and verify "
                "that no test files exist."
            ),

        "4D-R":
            (
                "Implement spatial model families and run smoke tests "
                "using train_joint and val_joint only."
            ),

        "4E-R":
            (
                "Execute the 43 frozen tuning configurations using "
                "val_joint noisy-target rollout MSE only."
            ),

        "4F-R":
            (
                "Run and freeze all 130 final fits without generating "
                "or opening test data."
            ),

        "4G-R":
            (
                "Generate the sealed test manifests and test fields "
                "from the frozen protocol, then perform final "
                "predictive and structural evaluation."
            ),

        "4H-R":
            (
                "Freeze Tier C v2 claims and synthesize Tier A, "
                "Tier B, Tier C v1 rejection, and Tier C v2 results."
            ),
    }


def write_input_hashes():
    inputs = {
        "phase3h_summary":
            PHASE3H_DIR
            / "phase3h_summary.json",

        "tier_c_v1_phase4a_summary":
            PHASE4A_V1_DIR
            / "phase4a_summary.json",

        "tier_c_v1_rejection_summary":
            PHASE4C_V1_DIR
            / "phase4c_rejection_summary.json",

        "tier_c_v1_rejection_record":
            PHASE4C_V1_DIR
            / "phase4c_rejection_record.json",

        "transformation_split":
            PHASE1D_DIR
            / "transformation_split.csv",
    }

    hashes = {}

    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(path)

        hashes[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    write_json(
        OUTPUT_DIR
        / "input_hashes.json",
        hashes,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = validate_prior_results()

    v1_split_lookup = (
        load_v1_carrier_split()
    )

    carrier_split_rows = (
        create_fresh_v2_carrier_split(
            v1_split_lookup
        )
    )

    transformation_rows = (
        validate_transformation_split()
    )

    carrier_parameter_schema = (
        create_carrier_parameter_schema()
    )

    state_regimes = (
        create_state_morphology_regimes()
    )

    numerical_protocol = (
        create_numerical_stabilization_protocol()
    )

    sealing_policy = (
        create_test_sealing_policy()
    )

    acceptance_gates = (
        create_acceptance_gates()
    )

    hypothesis_registry = (
        create_hypothesis_registry()
    )

    model_registry = (
        create_model_family_registry()
    )

    training_protocol = (
        create_training_protocol()
    )

    manifest_protocol = (
        create_manifest_protocol()
    )

    revision_plan = (
        create_revision_plan()
    )

    write_input_hashes()

    write_csv(
        OUTPUT_DIR
        / "carrier_split_v2.csv",
        carrier_split_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "carrier_parameter_schema_v2.csv",
        carrier_parameter_schema,
    )

    write_csv(
        OUTPUT_DIR
        / "state_morphology_regimes_v2.csv",
        state_regimes,
    )

    write_csv(
        OUTPUT_DIR
        / "model_family_registry_v2.csv",
        model_registry,
    )

    write_json(
        OUTPUT_DIR
        / "numerical_stabilization_protocol.json",
        numerical_protocol,
    )

    write_json(
        OUTPUT_DIR
        / "test_sealing_policy.json",
        sealing_policy,
    )

    write_json(
        OUTPUT_DIR
        / "acceptance_gates_v2.json",
        acceptance_gates,
    )

    write_json(
        OUTPUT_DIR
        / "hypothesis_registry_v2.json",
        hypothesis_registry,
    )

    write_json(
        OUTPUT_DIR
        / "training_protocol_v2.json",
        training_protocol,
    )

    write_json(
        OUTPUT_DIR
        / "trajectory_manifest_protocol_v2.json",
        manifest_protocol,
    )

    write_json(
        OUTPUT_DIR
        / "phase4_revision_plan.json",
        revision_plan,
    )

    carrier_counts = dict(
        Counter(
            row["split"]
            for row in carrier_split_rows
        )
    )

    transformation_counts = dict(
        Counter(
            row["split"]
            for row in transformation_rows
        )
    )

    v1_test_ids = {
        carrier_id
        for carrier_id, split
        in v1_split_lookup.items()
        if split == "test"
    }

    v2_test_ids = {
        row["carrier_id"]
        for row in carrier_split_rows
        if row["split"] == "test"
    }

    test_id_overlap = len(
        v1_test_ids & v2_test_ids
    )

    protocol = {
        "phase":
            "4A-R Tier C v2 revision protocol freeze",

        "protocol_version":
            PROTOCOL_VERSION,

        "revision_reason": {
            "tier_c_v1_rejected":
                True,

            "v1_failed_acceptance_gates":
                sources[
                    "rejection"
                ][
                    "failed_acceptance_gates"
                ],

            "v1_failed_physics_checks":
                sources[
                    "rejection"
                ][
                    "failed_physics_checks"
                ],

            "v1_validation_state_accuracy":
                sources[
                    "rejection"
                ][
                    "state_decoder_validation_accuracy"
                ],

            "v1_test_state_accuracy":
                sources[
                    "rejection"
                ][
                    "state_decoder_test_accuracy"
                ],
        },

        "scientific_scope":
            (
                "Fresh physics-informed synthetic spatial bridge. "
                "It is not a repair or continuation of Tier C v1, "
                "not a calibrated alloy model, and not real data."
            ),

        "carrier_split": {
            "count":
                CARRIER_COUNT,

            "counts":
                carrier_counts,

            "split_seed":
                CARRIER_SPLIT_SEED,

            "v1_v2_test_id_overlap":
                test_id_overlap,

            "fresh_physical_parameters":
                True,

            "fresh_initial_fields":
                True,
        },

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "carrier_parameter_dimension":
            PHYSICAL_CARRIER_DIMENSION,

        "transformation_split_counts":
            transformation_counts,

        "state_regimes":
            state_regimes,

        "numerical_stabilization":
            numerical_protocol,

        "test_sealing_policy":
            sealing_policy,

        "acceptance_gates":
            acceptance_gates,

        "training_protocol":
            training_protocol,

        "trajectory_manifest_protocol":
            manifest_protocol,

        "hypotheses":
            hypothesis_registry,

        "model_families": {
            row["model_id"]: row
            for row in model_registry
        },

        "fresh_seeds": {
            "carrier_split":
                CARRIER_SPLIT_SEED,

            "carrier_parameters":
                CARRIER_PARAMETER_SEED,

            "field_initialization":
                FIELD_INITIALIZATION_SEED,

            "solver":
                SOLVER_SEED,

            "measurement_noise":
                NOISE_SEED,

            "development_manifests":
                DEVELOPMENT_MANIFEST_SEED,

            "sealed_test_manifests":
                SEALED_TEST_MANIFEST_SEED,

            "diagnostic_seeds":
                list(DIAGNOSTIC_SEEDS),
        },

        "tier_c_v1_outputs_modified":
            False,

        "phase2_outputs_modified":
            False,

        "phase3_outputs_modified":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "tier_c_v2_protocol.json",
        protocol,
    )

    summary = {
        "phase":
            "4A-R Tier C v2 revision protocol freeze",

        "protocol_version":
            PROTOCOL_VERSION,

        "tier_c_v1_rejection_frozen":
            True,

        "tier_c_v1_training_authorized":
            False,

        "state_count":
            STATE_COUNT,

        "primitive_operation_count":
            PRIMITIVE_OPERATION_COUNT,

        "semigroup_size":
            SEMIGROUP_SIZE,

        "information_class_count":
            INFORMATION_CLASS_COUNT,

        "carrier_count":
            CARRIER_COUNT,

        "carrier_split_counts":
            carrier_counts,

        "v1_v2_test_carrier_id_overlap":
            test_id_overlap,

        "fresh_carrier_parameter_seed":
            CARRIER_PARAMETER_SEED,

        "fresh_field_initialization_seed":
            FIELD_INITIALIZATION_SEED,

        "fresh_solver_seed":
            SOLVER_SEED,

        "fresh_noise_seed":
            NOISE_SEED,

        "fresh_development_manifest_seed":
            DEVELOPMENT_MANIFEST_SEED,

        "sealed_test_manifest_seed":
            SEALED_TEST_MANIFEST_SEED,

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "state_regime_design":
            "2x2x2 distributed morphology factorial",

        "state_dependent_phase_mean":
            False,

        "state_dependent_defect_mean":
            False,

        "bounded_phase_representation":
            True,

        "bounded_defect_representation":
            True,

        "adaptive_step_rejection":
            True,

        "diagnostic_seed_count":
            len(DIAGNOSTIC_SEEDS),

        "minimum_mean_validation_state_accuracy":
            0.80,

        "minimum_worst_seed_validation_state_accuracy":
            0.75,

        "maximum_global_shortcut_validation_accuracy":
            0.95,

        "tuning_configuration_count":
            training_protocol[
                "total_tuning_configuration_count"
            ],

        "final_fit_count":
            training_protocol[
                "total_final_fit_count"
            ],

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_metrics_computed":
            False,

        "training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "prior_outputs_modified":
            False,

        "phase4ar_status":
            "protocol_frozen",
    }

    if test_id_overlap != 0:
        raise AssertionError(
            "Tier C v2 test IDs overlap Tier C v1 test IDs."
        )

    write_json(
        OUTPUT_DIR
        / "phase4ar_summary.json",
        summary,
    )

    print(
        "Phase 4A-R Tier C v2 revision "
        "protocol frozen successfully."
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
