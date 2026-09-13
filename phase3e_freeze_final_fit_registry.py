from __future__ import annotations

import csv
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

PHASE3D_DIR = Path(
    "outputs/phase3d_tier_b_tuning"
)

OUTPUT_DIR = Path(
    "outputs/phase3e_tier_b_final_fits"
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

EXPECTED_SELECTED_CONFIGURATIONS = {
    "OCM": "OCM_c04",
    "B1": "B1_c01",
    "B2": "B2_c04",
    "B3": "B3_c04",
    "B4": "B4_c04",
    "B5": "B5_c04",
}

EXPECTED_RUN_COUNTS = {
    "OCM": 25,
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
}

FINAL_BATCH_SIZE = 128
MAXIMUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-5


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


def write_csv(path: Path, rows):
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
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def noise_slug(noise: float):
    return (
        f"{noise:g}"
        .replace(".", "p")
    )


def validate_source_phases():
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

    phase3d = load_json(
        PHASE3D_DIR
        / "phase3d_summary.json"
    )

    phase3d_audit = load_json(
        PHASE3D_DIR
        / "phase3d_audited_summary.json"
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

    if phase3d["phase3d_status"] != "passed":
        raise AssertionError(
            "Phase 3D has not passed."
        )

    expected_audit_status = (
        "passed_with_documented_"
        "execution_metadata_deviation"
    )

    if (
        phase3d_audit["phase3d_audit_status"]
        != expected_audit_status
    ):
        raise AssertionError(
            "Phase 3D execution audit has not passed."
        )

    if phase3d_audit["rerun_required"] is not False:
        raise AssertionError(
            "Phase 3D requires a rerun."
        )

    if (
        int(
            phase3d_audit[
                "final_fit_batch_size"
            ]
        )
        != FINAL_BATCH_SIZE
    ):
        raise AssertionError(
            "Frozen Phase 3E batch size changed."
        )

    return {
        "phase3a": phase3a,
        "phase3b": phase3b,
        "phase3c": phase3c,
        "phase3d": phase3d,
        "phase3d_audit": phase3d_audit,
    }


def load_selected_configurations():
    selected = load_json(
        PHASE3D_DIR
        / "selected_configurations.json"
    )

    observed_ids = {
        model_id:
            selected[model_id][
                "configuration_id"
            ]
        for model_id in (
            "OCM",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
        )
    }

    if (
        observed_ids
        != EXPECTED_SELECTED_CONFIGURATIONS
    ):
        raise AssertionError(
            "Selected configurations changed: "
            f"{observed_ids}"
        )

    for model_id, result in selected.items():
        if result["numerically_valid"] is not True:
            raise AssertionError(
                f"Selected {model_id} result is invalid."
            )

        if result["test_cells_opened"] is not False:
            raise AssertionError(
                f"Selected {model_id} opened test data."
            )

        if result["clean_targets_read"] is not False:
            raise AssertionError(
                f"Selected {model_id} read clean targets."
            )

    return selected


def create_run_registry(
    selected,
):
    rows = []

    # Noise is the outer loop so a sequential runner can reuse
    # the loaded train/validation arrays for all runs at one noise.
    for noise in FROZEN_NOISE_LEVELS:
        rows.append(
            {
                "run_id":
                    f"B1_noise_{noise_slug(noise)}",

                "model_id":
                    "B1",

                "configuration_id":
                    selected["B1"][
                        "configuration_id"
                    ],

                "seed":
                    "deterministic",

                "noise_fraction":
                    noise,

                "fit_type":
                    "deterministic_affine",

                "batch_size":
                    FINAL_BATCH_SIZE,
            }
        )

        for model_id in (
            "OCM",
            "B2",
            "B3",
            "B4",
            "B5",
        ):
            for seed in FROZEN_SEEDS:
                rows.append(
                    {
                        "run_id":
                            (
                                f"{model_id}"
                                f"_seed_{seed}"
                                f"_noise_{noise_slug(noise)}"
                            ),

                        "model_id":
                            model_id,

                        "configuration_id":
                            selected[
                                model_id
                            ][
                                "configuration_id"
                            ],

                        "seed":
                            seed,

                        "noise_fraction":
                            noise,

                        "fit_type":
                            (
                                "neural_structured"
                                if model_id == "OCM"
                                else "neural_baseline"
                            ),

                        "batch_size":
                            FINAL_BATCH_SIZE,
                    }
                )

    if len(rows) != 130:
        raise AssertionError(
            "Expected 130 final-fit runs."
        )

    counts = Counter(
        row["model_id"]
        for row in rows
    )

    if dict(counts) != EXPECTED_RUN_COUNTS:
        raise AssertionError(
            f"Run counts changed: {dict(counts)}"
        )

    run_ids = [
        row["run_id"]
        for row in rows
    ]

    if len(run_ids) != len(set(run_ids)):
        raise AssertionError(
            "Final-fit run IDs are not unique."
        )

    return rows


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_source_phases()

    selected = (
        load_selected_configurations()
    )

    run_rows = create_run_registry(
        selected
    )

    registry = {
        "phase":
            "3E frozen Tier B final-fit registry",

        "training_cell":
            "train_joint",

        "checkpoint_selection_cell":
            "val_joint",

        "selection_metric":
            "val_joint noisy-target rollout MSE only",

        "batch_size":
            FINAL_BATCH_SIZE,

        "batch_size_source":
            (
                "Actual uniformly executed Phase 3D "
                "tuning batch size"
            ),

        "maximum_epochs":
            MAXIMUM_EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "early_stopping_min_delta":
            EARLY_STOPPING_MIN_DELTA,

        "seeds":
            list(FROZEN_SEEDS),

        "noise_levels":
            list(FROZEN_NOISE_LEVELS),

        "selected_configurations":
            selected,

        "run_count":
            len(run_rows),

        "run_counts":
            EXPECTED_RUN_COUNTS,

        "runs":
            run_rows,

        "fresh_initialization_required":
            True,

        "tuning_checkpoints_reused":
            False,

        "test_cells_opened":
            False,

        "privileged_arrays_opened":
            False,

        "clean_targets_read":
            False,

        "structural_labels_read":
            False,

        "registry_status":
            "frozen",
    }

    write_json(
        OUTPUT_DIR
        / "final_fit_registry.json",
        registry,
    )

    write_csv(
        OUTPUT_DIR
        / "final_fit_registry.csv",
        run_rows,
    )

    summary = {
        "phase":
            "3E final-fit registry freeze",

        "run_count":
            len(run_rows),

        "run_counts":
            EXPECTED_RUN_COUNTS,

        "selected_configuration_ids":
            EXPECTED_SELECTED_CONFIGURATIONS,

        "seed_count":
            len(FROZEN_SEEDS),

        "noise_condition_count":
            len(FROZEN_NOISE_LEVELS),

        "batch_size":
            FINAL_BATCH_SIZE,

        "maximum_epochs":
            MAXIMUM_EPOCHS,

        "training_cell":
            "train_joint",

        "checkpoint_selection_cell":
            "val_joint",

        "tuning_checkpoints_reused":
            False,

        "training_performed":
            False,

        "test_cells_opened":
            False,

        "privileged_arrays_opened":
            False,

        "phase3e_registry_status":
            "frozen",
    }

    write_json(
        OUTPUT_DIR
        / "phase3e_registry_summary.json",
        summary,
    )

    print(
        "Phase 3E final-fit registry frozen."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
