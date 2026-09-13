from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PHASE1D_DIR = Path(
    "outputs/phase1d_compositional_splits"
)

PHASE3A_DIR = Path(
    "outputs/phase3a_tier_b_protocol"
)

PHASE3B_DIR = Path(
    "outputs/phase3b_tier_b_audit"
)

PHASE3H_DIR = Path(
    "outputs/phase3h_tier_b_synthesis"
)

OUTPUT_DIR = Path(
    "outputs/phase4a_tier_c_physical_protocol"
)


STATE_COUNT = 8
PRIMITIVE_OPERATION_COUNT = 6
SEMIGROUP_SIZE = 104
INFORMATION_CLASS_COUNT = 9

GRID_HEIGHT = 32
GRID_WIDTH = 32
FIELD_CHANNEL_COUNT = 4
FLATTENED_OBSERVATION_DIMENSION = (
    GRID_HEIGHT
    * GRID_WIDTH
    * FIELD_CHANNEL_COUNT
)

CARRIER_COUNT = 160
PHYSICAL_CARRIER_DIMENSION = 8

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

DEVELOPMENT_SEED = 11
DEVELOPMENT_NOISE = 0.25

CARRIER_PARAMETER_SEED = 41027
FIELD_INITIALIZATION_SEED = 41028
SOLVER_SEED = 41029
NOISE_SEED = 41030

VISIBLE_STORAGE_DTYPE = "float16"
PRIVILEGED_CLEAN_DTYPE = "float32"

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
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(path: Path, rows) -> None:
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
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
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


def validate_phase3_freeze():
    summary = load_json(
        PHASE3H_DIR
        / "phase3h_summary.json"
    )

    if (
        summary[
            "phase3h_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 3H has not passed."
        )

    if (
        summary[
            "phase3_results_frozen"
        ]
        is not True
    ):
        raise AssertionError(
            "Phase 3 results are not frozen."
        )

    if (
        summary[
            "phase2_results_frozen"
        ]
        is not True
    ):
        raise AssertionError(
            "Phase 2 results are not frozen."
        )

    if (
        summary[
            "training_performed"
        ]
        is not False
    ):
        raise AssertionError(
            "Unexpected Phase 3H training flag."
        )

    return summary


def validate_tier_b_acceptance():
    summary = load_json(
        PHASE3B_DIR
        / "phase3b_summary.json"
    )

    if (
        summary[
            "phase3b_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 3B did not pass."
        )

    if (
        summary[
            "tier_b_data_accepted"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier B data was not accepted."
        )

    if (
        summary[
            "affine_shortcut_removed"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier B affine shortcut was not removed."
        )

    return summary


def validate_and_reuse_carrier_split():
    rows = load_csv(
        PHASE3A_DIR
        / "carrier_split.csv"
    )

    if len(rows) != CARRIER_COUNT:
        raise AssertionError(
            "Carrier count changed."
        )

    counts = Counter(
        row["split"]
        for row in rows
    )

    if dict(counts) != CARRIER_SPLIT_COUNTS:
        raise AssertionError(
            "Carrier split counts changed: "
            f"{dict(counts)}"
        )

    carrier_ids = [
        int(row["carrier_id"])
        for row in rows
    ]

    if len(set(carrier_ids)) != CARRIER_COUNT:
        raise AssertionError(
            "Carrier IDs are not unique."
        )

    output = []

    for row in rows:
        output.append(
            {
                "carrier_id":
                    int(
                        row[
                            "carrier_id"
                        ]
                    ),

                "split":
                    row["split"],

                "split_position":
                    int(
                        row[
                            "split_position"
                        ]
                    ),

                "tier_b_carrier_dimension":
                    int(
                        row[
                            "carrier_dimension"
                        ]
                    ),

                "tier_c_physical_carrier_dimension":
                    PHYSICAL_CARRIER_DIMENSION,

                "carrier_identity_reused_from_tier_b":
                    True,

                "physical_parameters_generated":
                    False,
            }
        )

    output.sort(
        key=lambda item:
            item["carrier_id"]
    )

    return output


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
            "Transformation split changed: "
            f"{dict(counts)}"
        )

    if len(rows) != SEMIGROUP_SIZE:
        raise AssertionError(
            "Semigroup size changed."
        )

    return rows


def create_physical_carrier_parameter_schema():
    return [
        {
            "parameter_index": 0,
            "parameter_name":
                "mean_phase_fraction",
            "minimum": 0.35,
            "maximum": 0.65,
            "interpretation":
                (
                    "Dimensionless initial mean of the "
                    "conserved phase field."
                ),
        },
        {
            "parameter_index": 1,
            "parameter_name":
                "initial_length_scale_pixels",
            "minimum": 2.5,
            "maximum": 7.5,
            "interpretation":
                (
                    "Characteristic spatial scale of "
                    "the initialized microstructure."
                ),
        },
        {
            "parameter_index": 2,
            "parameter_name":
                "interface_width",
            "minimum": 0.8,
            "maximum": 1.8,
            "interpretation":
                (
                    "Dimensionless diffuse-interface width."
                ),
        },
        {
            "parameter_index": 3,
            "parameter_name":
                "phase_mobility",
            "minimum": 0.6,
            "maximum": 1.4,
            "interpretation":
                (
                    "Dimensionless mobility for the "
                    "conserved phase field."
                ),
        },
        {
            "parameter_index": 4,
            "parameter_name":
                "orientation_anisotropy",
            "minimum": 0.0,
            "maximum": 0.35,
            "interpretation":
                (
                    "Strength of orientation-dependent "
                    "interfacial modulation."
                ),
        },
        {
            "parameter_index": 5,
            "parameter_name":
                "initial_defect_fraction",
            "minimum": 0.02,
            "maximum": 0.18,
            "interpretation":
                (
                    "Initial dimensionless defect-density "
                    "fraction."
                ),
        },
        {
            "parameter_index": 6,
            "parameter_name":
                "defect_mobility",
            "minimum": 0.5,
            "maximum": 1.5,
            "interpretation":
                (
                    "Dimensionless defect-field relaxation "
                    "mobility."
                ),
        },
        {
            "parameter_index": 7,
            "parameter_name":
                "process_driving_bias",
            "minimum": -0.25,
            "maximum": 0.25,
            "interpretation":
                (
                    "Dimensionless bias applied to the "
                    "bulk free-energy term."
                ),
        },
    ]


def create_state_physics_regimes():
    """
    These are controlled, dimensionless simulator regimes.

    They are not calibrated alloy states and must not be given
    material-specific names in the manuscript.
    """

    regimes = [
        {
            "state_id": 0,
            "regime_label": "regime_0",
            "phase_mobility_multiplier": 1.00,
            "interfacial_energy_multiplier": 1.00,
            "bulk_driving_multiplier": 1.00,
            "anisotropy_multiplier": 1.00,
            "defect_coupling_multiplier": 1.00,
            "orientation_relaxation_multiplier": 1.00,
            "relaxation_steps": 80,
        },
        {
            "state_id": 1,
            "regime_label": "regime_1",
            "phase_mobility_multiplier": 1.20,
            "interfacial_energy_multiplier": 0.90,
            "bulk_driving_multiplier": 1.10,
            "anisotropy_multiplier": 1.15,
            "defect_coupling_multiplier": 0.90,
            "orientation_relaxation_multiplier": 1.10,
            "relaxation_steps": 96,
        },
        {
            "state_id": 2,
            "regime_label": "regime_2",
            "phase_mobility_multiplier": 0.80,
            "interfacial_energy_multiplier": 1.25,
            "bulk_driving_multiplier": 0.90,
            "anisotropy_multiplier": 1.30,
            "defect_coupling_multiplier": 1.10,
            "orientation_relaxation_multiplier": 0.90,
            "relaxation_steps": 96,
        },
        {
            "state_id": 3,
            "regime_label": "regime_3",
            "phase_mobility_multiplier": 1.35,
            "interfacial_energy_multiplier": 0.80,
            "bulk_driving_multiplier": 1.20,
            "anisotropy_multiplier": 1.50,
            "defect_coupling_multiplier": 1.20,
            "orientation_relaxation_multiplier": 1.25,
            "relaxation_steps": 112,
        },
        {
            "state_id": 4,
            "regime_label": "regime_4",
            "phase_mobility_multiplier": 0.75,
            "interfacial_energy_multiplier": 1.15,
            "bulk_driving_multiplier": 1.25,
            "anisotropy_multiplier": 1.70,
            "defect_coupling_multiplier": 1.45,
            "orientation_relaxation_multiplier": 0.85,
            "relaxation_steps": 112,
        },
        {
            "state_id": 5,
            "regime_label": "regime_5",
            "phase_mobility_multiplier": 1.10,
            "interfacial_energy_multiplier": 0.75,
            "bulk_driving_multiplier": 0.80,
            "anisotropy_multiplier": 2.00,
            "defect_coupling_multiplier": 1.10,
            "orientation_relaxation_multiplier": 1.40,
            "relaxation_steps": 128,
        },
        {
            "state_id": 6,
            "regime_label": "regime_6",
            "phase_mobility_multiplier": 0.90,
            "interfacial_energy_multiplier": 1.35,
            "bulk_driving_multiplier": 1.00,
            "anisotropy_multiplier": 1.40,
            "defect_coupling_multiplier": 1.75,
            "orientation_relaxation_multiplier": 1.00,
            "relaxation_steps": 128,
        },
        {
            "state_id": 7,
            "regime_label": "regime_7",
            "phase_mobility_multiplier": 1.45,
            "interfacial_energy_multiplier": 0.70,
            "bulk_driving_multiplier": 1.35,
            "anisotropy_multiplier": 1.20,
            "defect_coupling_multiplier": 2.00,
            "orientation_relaxation_multiplier": 1.30,
            "relaxation_steps": 144,
        },
    ]

    if len(regimes) != STATE_COUNT:
        raise AssertionError(
            "Expected eight physical regimes."
        )

    if {
        row["state_id"]
        for row in regimes
    } != set(range(STATE_COUNT)):
        raise AssertionError(
            "Physical regime state IDs changed."
        )

    return regimes


def create_field_channel_schema():
    return [
        {
            "channel_index": 0,
            "channel_name": "phase_field",
            "symbol": "phi",
            "expected_range": "[-1, 1]",
            "role":
                (
                    "Conserved composition or phase-order field."
                ),
        },
        {
            "channel_index": 1,
            "channel_name":
                "orientation_cosine",
            "symbol": "q_cos",
            "expected_range": "[-1, 1]",
            "role":
                (
                    "Cosine component of a doubled-angle "
                    "orientation representation."
                ),
        },
        {
            "channel_index": 2,
            "channel_name":
                "orientation_sine",
            "symbol": "q_sin",
            "expected_range": "[-1, 1]",
            "role":
                (
                    "Sine component of a doubled-angle "
                    "orientation representation."
                ),
        },
        {
            "channel_index": 3,
            "channel_name":
                "defect_density",
            "symbol": "d",
            "expected_range": "[0, 1]",
            "role":
                (
                    "Dimensionless diffuse defect-density field."
                ),
        },
    ]


def create_physics_protocol():
    return {
        "tier_name":
            "Tier C spatial physical-field bridge",

        "scientific_scope":
            (
                "Controlled, dimensionless, physics-informed "
                "microstructure simulator. It is not a calibrated "
                "alloy model and is not real microscopy data."
            ),

        "grid": {
            "height": GRID_HEIGHT,
            "width": GRID_WIDTH,
            "periodic_boundary_conditions": True,
            "spatial_dimension": 2,
        },

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "flattened_observation_dimension":
            FLATTENED_OBSERVATION_DIMENSION,

        "field_channels":
            create_field_channel_schema(),

        "carrier": {
            "count": CARRIER_COUNT,
            "dimension":
                PHYSICAL_CARRIER_DIMENSION,
            "split_counts":
                CARRIER_SPLIT_COUNTS,
            "identity_split_reused_from_tier_b":
                True,
            "trajectory_rule":
                (
                    "The same physical carrier and base spatial "
                    "initialization are retained throughout one "
                    "operation history."
                ),
            "parameter_generation_seed":
                CARRIER_PARAMETER_SEED,
        },

        "simulator": {
            "family":
                (
                    "Coupled conserved phase-field, "
                    "nonconserved orientation-field, and "
                    "diffuse defect-field relaxation."
                ),

            "phase_equation":
                (
                    "Dimensionless semi-implicit "
                    "Cahn-Hilliard-style update."
                ),

            "orientation_equation":
                (
                    "Two-component Allen-Cahn-style update "
                    "with doubled-angle orientation encoding "
                    "and norm regularization."
                ),

            "defect_equation":
                (
                    "Diffusion-reaction or Allen-Cahn-style "
                    "defect-density relaxation."
                ),

            "state_conditioning":
                (
                    "Each of the eight discrete history states "
                    "selects one frozen dimensionless coefficient "
                    "regime."
                ),

            "operation_rule":
                (
                    "Primitive operations update the discrete "
                    "history state exactly according to the "
                    "frozen Phase 1 transformations."
                ),

            "observation_rule":
                (
                    "For each carrier-state pair, the simulator "
                    "relaxes the same carrier-specific base "
                    "initialization under the selected state "
                    "regime. This preserves state sufficiency "
                    "for the controlled benchmark."
                ),

            "field_initialization_seed":
                FIELD_INITIALIZATION_SEED,

            "solver_seed":
                SOLVER_SEED,

            "stochastic_dynamics_after_initialization":
                False,

            "dimensionless_parameters":
                True,

            "material_calibration_claimed":
                False,
        },

        "normalization": {
            "statistics_source":
                (
                    "Clean training-carrier fields only."
                ),

            "per_channel_normalization":
                True,

            "validation_or_test_statistics_used":
                False,
        },

        "noise": {
            "levels":
                list(
                    FROZEN_NOISE_LEVELS
                ),

            "noise_seed":
                NOISE_SEED,

            "definition":
                (
                    "Channel-standardized additive Gaussian "
                    "measurement noise applied after train-only "
                    "normalization."
                ),

            "phase_channel_clipped":
                True,

            "defect_channel_clipped":
                True,

            "orientation_pair_renormalized":
                True,

            "independent_across_trajectories":
                True,

            "independent_across_time":
                True,
        },

        "storage": {
            "visible_noisy_field_dtype":
                VISIBLE_STORAGE_DTYPE,

            "privileged_clean_field_dtype":
                PRIVILEGED_CLEAN_DTYPE,

            "visible_and_privileged_separated":
                True,

            "state_ids_stored_privately":
                True,

            "trajectory_metadata_stored_privately":
                True,
        },
    }


def reuse_evaluation_cells():
    rows = load_csv(
        PHASE3A_DIR
        / "evaluation_cells.csv"
    )

    if len(rows) != 8:
        raise AssertionError(
            "Expected eight evaluation cells."
        )

    output = []

    for row in rows:
        output.append(
            {
                **row,

                "field_channels":
                    FIELD_CHANNEL_COUNT,

                "grid_height":
                    GRID_HEIGHT,

                "grid_width":
                    GRID_WIDTH,

                "visible_storage_dtype":
                    VISIBLE_STORAGE_DTYPE,

                "privileged_clean_dtype":
                    PRIVILEGED_CLEAN_DTYPE,

                "evaluation_design_reused_from_tier_b":
                    True,
            }
        )

    selection_cells = [
        row["cell_id"]
        for row in output
        if str(
            row[
                "checkpoint_selection_allowed"
            ]
        ).lower()
        == "true"
    ]

    if selection_cells != [
        "val_joint"
    ]:
        raise AssertionError(
            "Exactly val_joint must control selection."
        )

    return output


def create_model_family_registry():
    return [
        {
            "model_id": "B0",
            "family":
                "spatial persistence",
            "description":
                (
                    "Return the initial spatial field as the "
                    "final prediction."
                ),
        },
        {
            "model_id": "B1",
            "family":
                "low-rank affine field operator",
            "description":
                (
                    "Training-only PCA field representation "
                    "followed by one ridge-regularized affine "
                    "operator per primitive operation."
                ),
            "pca_dimension": 128,
        },
        {
            "model_id": "B2",
            "family":
                "operation-conditioned convolutional residual model",
            "description":
                (
                    "Convolutional encoder, operation embedding, "
                    "residual spatial transition, and convolutional "
                    "decoder."
                ),
        },
        {
            "model_id": "B3",
            "family":
                "convolutional recurrent sequence model",
            "description":
                (
                    "Convolutional field encoder with recurrent "
                    "operation-sequence processing."
                ),
        },
        {
            "model_id": "B4",
            "family":
                "patch-transformer sequence model",
            "description":
                (
                    "Patch-token spatial encoder with operation "
                    "tokens and transformer sequence processing."
                ),
        },
        {
            "model_id": "B5",
            "family":
                "continuous latent spatial operator",
            "description":
                (
                    "Continuous latent field representation with "
                    "learned operation-conditioned latent operators."
                ),
        },
        {
            "model_id": "OCM",
            "family":
                "convolutional operation channel model",
            "description":
                (
                    "Convolutional encoder and decoder surrounding "
                    "eight-state row-stochastic operation channels."
                ),
            "latent_state_count": 8,
        },
    ]


def create_acceptance_gates():
    return {
        "numerical_physics": {
            "required": True,
            "criteria": [
                (
                    "All generated fields, energies, and solver "
                    "diagnostics are finite."
                ),
                (
                    "Conserved phase-field mean drift is at most "
                    "5e-3 for every carrier-state simulation."
                ),
                (
                    "Final total free energy does not exceed initial "
                    "energy by more than numerical tolerance in at "
                    "least 95 percent of simulations."
                ),
                (
                    "The phase field remains inside [-1.05, 1.05] "
                    "before final clipping."
                ),
                (
                    "The defect-density field remains inside "
                    "[-0.05, 1.05] before final clipping."
                ),
            ],

            "maximum_phase_mean_drift":
                5e-3,

            "minimum_energy_nonincrease_fraction":
                0.95,
        },

        "manifold_quality": {
            "required": True,
            "criteria": [
                (
                    "Every state has positive within-state "
                    "carrier variation."
                ),
                (
                    "Every field channel has nonzero variance on "
                    "training carriers."
                ),
                (
                    "No two carrier-state fields are exactly equal."
                ),
                (
                    "A privileged convolutional state decoder "
                    "trained only on training carriers exceeds "
                    "80 percent accuracy on validation and test "
                    "carriers."
                ),
                (
                    "A classifier using only global channel means "
                    "and variances does not exceed 98 percent test "
                    "accuracy."
                ),
            ],

            "minimum_privileged_state_accuracy":
                0.80,

            "maximum_global_statistics_accuracy":
                0.98,

            "chance_accuracy":
                1.0 / STATE_COUNT,
        },

        "shortcut_removal": {
            "required": True,
            "criteria": [
                (
                    "The low-rank affine B1 diagnostic cannot "
                    "recover every primitive exactly on every "
                    "held-out carrier."
                ),
                (
                    "The low-rank affine diagnostic cannot recover "
                    "all 104 transformations exactly on every "
                    "held-out carrier."
                ),
                (
                    "Held-out-carrier continuous reconstruction "
                    "error exceeds numerical interpolation "
                    "tolerance."
                ),
            ],

            "numerical_interpolation_tolerance":
                1e-10,
        },

        "data_integrity": {
            "required": True,
            "criteria": [
                (
                    "Carrier train, validation, and test identities "
                    "remain disjoint."
                ),
                (
                    "The frozen 86/9/9 transformation split remains "
                    "unchanged."
                ),
                (
                    "Only val_joint may control checkpoint "
                    "selection."
                ),
                (
                    "No test cell is opened before final checkpoint "
                    "selection."
                ),
                (
                    "Clean test fields and structural labels are "
                    "used for final evaluation only."
                ),
            ],
        },
    }


def create_hypothesis_registry():
    return {
        "registry_status": "frozen",

        "hypotheses": [
            {
                "hypothesis_id": "H4-1",
                "research_question": "RQ1/RQ2",
                "statement":
                    (
                        "At least one learned model will recover "
                        "held-out transformation consensus and "
                        "informativeness relations on clean unseen-"
                        "carrier spatial fields substantially above "
                        "the majority baseline."
                    ),
                "primary_metric":
                    (
                        "test-carrier relation balanced accuracy "
                        "under the clean structural probe"
                    ),
            },
            {
                "hypothesis_id": "H4-2",
                "research_question": "RQ3/RQ4",
                "statement":
                    (
                        "For the strongest predictive models, the "
                        "unseen-carrier continuous prediction gap "
                        "will exceed the unseen-composition gap."
                    ),
                "primary_metric":
                    (
                        "carrier-only MSE gap versus "
                        "composition-only MSE gap"
                    ),
            },
            {
                "hypothesis_id": "H4-3",
                "research_question": "RQ3",
                "statement":
                    (
                        "Explicit finite channels will not be "
                        "sufficient to guarantee the lowest "
                        "joint-OOD predictive MSE."
                    ),
                "primary_metric":
                    "joint-OOD noisy-target rollout MSE",
            },
            {
                "hypothesis_id": "H4-4",
                "research_question": "RQ4",
                "statement":
                    (
                        "OCM hard-channel structural recovery will "
                        "remain sensitive to optimization seed even "
                        "when its convolutional encoder and decoder "
                        "fit the spatial observation family."
                    ),
                "primary_metric":
                    (
                        "exact hard-channel transformation recovery "
                        "across five frozen seeds"
                    ),
            },
        ],
    }


def create_training_protocol():
    baseline_tuning_count = sum(
        BASELINE_TUNING_CONFIGURATION_COUNTS.values()
    )

    return {
        "development_seed":
            DEVELOPMENT_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE,

        "training_cell":
            "train_joint",

        "checkpoint_selection_cell":
            "val_joint",

        "selection_metric":
            (
                "val_joint noisy-target spatial rollout "
                "MSE only"
            ),

        "clean_targets_used_for_training":
            False,

        "clean_targets_used_for_selection":
            False,

        "structural_labels_used_for_training":
            False,

        "physics_diagnostics_used_for_model_selection":
            False,

        "test_cells_opened_during_tuning":
            False,

        "seeds":
            list(FROZEN_SEEDS),

        "noise_levels":
            list(FROZEN_NOISE_LEVELS),

        "ocm_tuning_configuration_count":
            OCM_TUNING_CONFIGURATION_COUNT,

        "baseline_tuning_configuration_counts":
            BASELINE_TUNING_CONFIGURATION_COUNTS,

        "total_tuning_configuration_count":
            (
                OCM_TUNING_CONFIGURATION_COUNT
                + baseline_tuning_count
            ),

        "final_fit_counts":
            FINAL_FIT_COUNTS,

        "total_final_fit_count":
            sum(
                FINAL_FIT_COUNTS.values()
            ),

        "fresh_initialization_required":
            True,

        "tuning_checkpoints_eligible_for_final_use":
            False,
    }


def calculate_storage_estimate(
    total_point_count: int,
):
    values_per_field = (
        FIELD_CHANNEL_COUNT
        * GRID_HEIGHT
        * GRID_WIDTH
    )

    visible_bytes = (
        total_point_count
        * values_per_field
        * 2
        * len(FROZEN_NOISE_LEVELS)
    )

    privileged_bytes = (
        total_point_count
        * values_per_field
        * 4
    )

    total_bytes = (
        visible_bytes
        + privileged_bytes
    )

    decimal_gb = 1_000_000_000

    return {
        "total_field_point_count":
            total_point_count,

        "values_per_field":
            values_per_field,

        "visible_noise_condition_count":
            len(FROZEN_NOISE_LEVELS),

        "estimated_visible_storage_gb":
            visible_bytes / decimal_gb,

        "estimated_privileged_clean_storage_gb":
            privileged_bytes / decimal_gb,

        "estimated_total_field_storage_gb":
            total_bytes / decimal_gb,

        "metadata_and_checkpoint_storage_excluded":
            True,
    }


def create_phase_plan():
    return {
        "4A":
            (
                "Freeze the Tier C physical-field protocol, "
                "physics regimes, carrier reuse, evaluation cells, "
                "hypotheses, and acceptance gates."
            ),

        "4B":
            (
                "Implement the coupled spatial simulator, generate "
                "carrier-state fields and trajectory datasets, and "
                "record solver diagnostics."
            ),

        "4C":
            (
                "Audit numerical physics, field-manifold quality, "
                "state separability, data leakage, and low-rank "
                "affine shortcut removal."
            ),

        "4D":
            (
                "Implement spatial model families and run "
                "leakage-safe smoke tests."
            ),

        "4E":
            (
                "Freeze and execute tuning using only train_joint "
                "and val_joint."
            ),

        "4F":
            (
                "Run all frozen final spatial fits across five "
                "seeds and five noise conditions."
            ),

        "4G":
            (
                "Evaluate prediction, behavioral structure, "
                "internal OCM channels, and physics-consistency "
                "diagnostics."
            ),

        "4H":
            (
                "Freeze Tier C claims and synthesize Tier A, "
                "Tier B, and Tier C conclusions."
            ),
    }


def save_input_hashes():
    inputs = {
        "phase3h_summary":
            PHASE3H_DIR
            / "phase3h_summary.json",

        "phase3_claim_registry":
            PHASE3H_DIR
            / "phase3_claim_registry.json",

        "phase3_do_not_claim":
            PHASE3H_DIR
            / "phase3_do_not_claim.json",

        "tier_b_carrier_split":
            PHASE3A_DIR
            / "carrier_split.csv",

        "tier_b_evaluation_cells":
            PHASE3A_DIR
            / "evaluation_cells.csv",

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
            "sha256": file_sha256(path),
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

    phase3_summary = (
        validate_phase3_freeze()
    )

    tier_b_summary = (
        validate_tier_b_acceptance()
    )

    carrier_rows = (
        validate_and_reuse_carrier_split()
    )

    transformation_rows = (
        validate_transformation_split()
    )

    physical_parameter_schema = (
        create_physical_carrier_parameter_schema()
    )

    state_regimes = (
        create_state_physics_regimes()
    )

    evaluation_cells = (
        reuse_evaluation_cells()
    )

    physics_protocol = (
        create_physics_protocol()
    )

    model_registry = (
        create_model_family_registry()
    )

    acceptance_gates = (
        create_acceptance_gates()
    )

    hypotheses = (
        create_hypothesis_registry()
    )

    training_protocol = (
        create_training_protocol()
    )

    phase_plan = (
        create_phase_plan()
    )

    total_point_count = int(
        tier_b_summary[
            "total_point_count"
        ]
    )

    storage_estimate = (
        calculate_storage_estimate(
            total_point_count
        )
    )

    save_input_hashes()

    write_csv(
        OUTPUT_DIR
        / "carrier_reuse_manifest.csv",
        carrier_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "physical_carrier_parameter_schema.csv",
        physical_parameter_schema,
    )

    write_csv(
        OUTPUT_DIR
        / "state_physics_regimes.csv",
        state_regimes,
    )

    write_csv(
        OUTPUT_DIR
        / "field_channel_schema.csv",
        create_field_channel_schema(),
    )

    write_csv(
        OUTPUT_DIR
        / "evaluation_cells.csv",
        evaluation_cells,
    )

    write_csv(
        OUTPUT_DIR
        / "model_family_registry.csv",
        model_registry,
    )

    write_json(
        OUTPUT_DIR
        / "physics_protocol.json",
        physics_protocol,
    )

    write_json(
        OUTPUT_DIR
        / "training_protocol.json",
        training_protocol,
    )

    write_json(
        OUTPUT_DIR
        / "acceptance_gates.json",
        acceptance_gates,
    )

    write_json(
        OUTPUT_DIR
        / "hypothesis_registry.json",
        hypotheses,
    )

    write_json(
        OUTPUT_DIR
        / "storage_estimate.json",
        storage_estimate,
    )

    write_json(
        OUTPUT_DIR
        / "phase4_plan.json",
        phase_plan,
    )

    protocol = {
        "phase":
            "4A Tier C physical-field protocol freeze",

        "motivation":
            (
                "Test whether predictive and structural findings "
                "survive spatial, physics-informed microstructure "
                "observations rather than vector-valued synthetic "
                "manifolds."
            ),

        "scientific_scope":
            (
                "Physics-informed synthetic bridge only. "
                "No real-data or material-calibration claim."
            ),

        "phase2_results_frozen":
            phase3_summary[
                "phase2_results_frozen"
            ],

        "phase3_results_frozen":
            phase3_summary[
                "phase3_results_frozen"
            ],

        "physics_protocol":
            physics_protocol,

        "training_protocol":
            training_protocol,

        "acceptance_gates":
            acceptance_gates,

        "hypotheses":
            hypotheses,

        "storage_estimate":
            storage_estimate,

        "model_families": {
            row["model_id"]: row
            for row in model_registry
        },

        "evaluation_cells": {
            row["cell_id"]: row
            for row in evaluation_cells
        },

        "transformation_split_reused":
            True,

        "carrier_identity_split_reused":
            True,

        "prior_outputs_modified":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "tier_c_protocol.json",
        protocol,
    )

    summary = {
        "phase":
            "4A Tier C physical-field protocol freeze",

        "phase3h_status":
            phase3_summary[
                "phase3h_status"
            ],

        "phase2_results_frozen":
            phase3_summary[
                "phase2_results_frozen"
            ],

        "phase3_results_frozen":
            phase3_summary[
                "phase3_results_frozen"
            ],

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

        "physical_carrier_dimension":
            PHYSICAL_CARRIER_DIMENSION,

        "carrier_split_counts":
            CARRIER_SPLIT_COUNTS,

        "transformation_split_counts":
            dict(
                Counter(
                    row["split"]
                    for row in transformation_rows
                )
            ),

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "flattened_observation_dimension":
            FLATTENED_OBSERVATION_DIMENSION,

        "evaluation_cell_count":
            len(evaluation_cells),

        "model_family_count":
            len(model_registry),

        "tuning_configuration_count":
            training_protocol[
                "total_tuning_configuration_count"
            ],

        "final_fit_count":
            training_protocol[
                "total_final_fit_count"
            ],

        "checkpoint_selection_cell":
            "val_joint",

        "development_seed":
            DEVELOPMENT_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE,

        "estimated_total_field_storage_gb":
            storage_estimate[
                "estimated_total_field_storage_gb"
            ],

        "material_calibration_claimed":
            False,

        "real_data_used":
            False,

        "physics_simulation_performed":
            False,

        "fields_generated":
            False,

        "training_performed":
            False,

        "test_cells_opened":
            False,

        "prior_outputs_modified":
            False,

        "phase4a_status":
            "protocol_frozen",

        "sanity_checks":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase4a_summary.json",
        summary,
    )

    print(
        "Phase 4A Tier C physical-field "
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
