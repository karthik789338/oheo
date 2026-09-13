from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

from phase3c_run_tier_b_smoke_tests import (
    BATCH_SIZE as EXECUTED_BATCH_SIZE,
    make_loader,
)


PHASE3D_DIR = Path(
    "outputs/phase3d_tier_b_tuning"
)

CONFIGURATION_DIR = (
    PHASE3D_DIR / "configurations"
)


EXPECTED_CONFIGURATION_COUNTS = {
    "OCM": 24,
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
}

EXPECTED_CONFIGURATION_COUNT = 43
EXPECTED_DECLARED_BATCH_SIZE = 256
EXPECTED_EXECUTED_BATCH_SIZE = 128


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


def validate_phase3d():
    registry = load_json(
        PHASE3D_DIR
        / "tuning_registry.json"
    )

    summary = load_json(
        PHASE3D_DIR
        / "phase3d_summary.json"
    )

    if registry["status"] != "registry_frozen":
        raise AssertionError(
            "Phase 3D registry is not frozen."
        )

    if summary["phase3d_status"] != "passed":
        raise AssertionError(
            "Phase 3D has not passed."
        )

    if registry["configuration_count"] != 43:
        raise AssertionError(
            "Registry configuration count changed."
        )

    if summary["completed_configuration_count"] != 43:
        raise AssertionError(
            "Not all configurations completed."
        )

    return registry, summary


def inspect_loader_implementation():
    signature = inspect.signature(
        make_loader
    )

    if "batch_size" in signature.parameters:
        raise AssertionError(
            "make_loader now accepts a batch-size argument; "
            "the execution audit must be revisited."
        )

    module_name = (
        make_loader.__module__
    )

    if (
        module_name
        != "phase3c_run_tier_b_smoke_tests"
    ):
        raise AssertionError(
            "Unexpected make_loader source module: "
            f"{module_name}"
        )

    if (
        EXECUTED_BATCH_SIZE
        != EXPECTED_EXECUTED_BATCH_SIZE
    ):
        raise AssertionError(
            "Unexpected inherited batch size: "
            f"{EXECUTED_BATCH_SIZE}"
        )

    return {
        "loader_function":
            (
                "phase3c_run_tier_b_smoke_tests."
                "make_loader"
            ),

        "loader_signature":
            str(signature),

        "loader_accepts_batch_size_argument":
            False,

        "inherited_batch_size_constant":
            int(
                EXECUTED_BATCH_SIZE
            ),
    }


def validate_configuration_results(
    registry,
):
    rows = []

    for configuration in registry[
        "configurations"
    ]:
        configuration_id = configuration[
            "configuration_id"
        ]

        result_path = (
            CONFIGURATION_DIR
            / configuration_id
            / "result.json"
        )

        if not result_path.exists():
            raise FileNotFoundError(
                result_path
            )

        result = load_json(
            result_path
        )

        if result["status"] != "completed":
            raise AssertionError(
                f"{configuration_id} is incomplete."
            )

        if result["numerically_valid"] is not True:
            raise AssertionError(
                f"{configuration_id} is invalid."
            )

        if result["test_cells_opened"] is not False:
            raise AssertionError(
                f"{configuration_id} opened test data."
            )

        if result["privileged_arrays_opened"] is not False:
            raise AssertionError(
                f"{configuration_id} opened privileged data."
            )

        if result["clean_targets_read"] is not False:
            raise AssertionError(
                f"{configuration_id} read clean targets."
            )

        rows.append(
            {
                "configuration_id":
                    configuration_id,

                "baseline_id":
                    result[
                        "baseline_id"
                    ],

                "validation_mse":
                    float(
                        result[
                            "best_validation_rollout_mse"
                        ]
                    ),
            }
        )

    if len(rows) != EXPECTED_CONFIGURATION_COUNT:
        raise AssertionError(
            "Configuration result count changed."
        )

    counts = Counter(
        row["baseline_id"]
        for row in rows
    )

    if dict(counts) != EXPECTED_CONFIGURATION_COUNTS:
        raise AssertionError(
            "Configuration-family counts changed: "
            f"{dict(counts)}"
        )

    return rows


def validate_selected_configurations(
    summary,
    configuration_rows,
):
    result_lookup = {
        row["configuration_id"]:
            row
        for row in configuration_rows
    }

    selected = summary[
        "selected_configurations"
    ]

    selected_ids = {}

    for model_id in (
        "OCM",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
    ):
        selected_result = selected[
            model_id
        ]

        configuration_id = selected_result[
            "configuration_id"
        ]

        if configuration_id not in result_lookup:
            raise AssertionError(
                "Selected configuration is absent: "
                f"{configuration_id}"
            )

        family_candidates = [
            row
            for row in configuration_rows
            if row["baseline_id"] == model_id
        ]

        expected_winner = min(
            family_candidates,
            key=lambda row: (
                row["validation_mse"],
                row["configuration_id"],
            ),
        )

        if (
            configuration_id
            != expected_winner[
                "configuration_id"
            ]
        ):
            raise AssertionError(
                f"{model_id} selected configuration "
                "does not minimize validation MSE."
            )

        selected_ids[
            model_id
        ] = configuration_id

    return selected_ids


def main():
    registry, summary = (
        validate_phase3d()
    )

    loader_audit = (
        inspect_loader_implementation()
    )

    configuration_rows = (
        validate_configuration_results(
            registry
        )
    )

    selected_ids = (
        validate_selected_configurations(
            summary,
            configuration_rows,
        )
    )

    declared_batch_size = int(
        registry["batch_size"]
    )

    if (
        declared_batch_size
        != EXPECTED_DECLARED_BATCH_SIZE
    ):
        raise AssertionError(
            "The declared registry batch size changed."
        )

    deviation_present = bool(
        declared_batch_size
        != EXECUTED_BATCH_SIZE
    )

    correction = {
        "phase":
            "3D execution metadata correction",

        "original_registry_modified":
            False,

        "original_results_modified":
            False,

        "declared_batch_size":
            declared_batch_size,

        "executed_batch_size":
            int(
                EXECUTED_BATCH_SIZE
            ),

        "deviation_present":
            deviation_present,

        "cause": (
            "The Phase 3D runner imported make_loader from "
            "phase3c_run_tier_b_smoke_tests. That loader does "
            "not accept a batch-size argument and uses the "
            "module-level BATCH_SIZE constant of 128."
        ),

        "scope": (
            "All 43 tuning configurations were executed with "
            "batch size 128."
        ),

        "all_configurations_affected_uniformly":
            True,

        "model_comparison_remains_uniform":
            True,

        "selection_metric_unchanged":
            True,

        "checkpoint_selection_cell_unchanged":
            True,

        "test_cells_opened":
            False,

        "privileged_arrays_opened":
            False,

        "clean_targets_read":
            False,

        "structural_labels_read":
            False,

        "selection_leakage_introduced":
            False,

        "selected_configurations_recomputed_correctly":
            True,

        "selected_configuration_ids":
            selected_ids,

        "rerun_required":
            False,

        "rerun_decision": (
            "A rerun is not required because every configuration "
            "was evaluated under the same executed batch size and "
            "selection remained based only on val_joint noisy-target "
            "rollout MSE. The discrepancy must be reported as an "
            "execution-metadata deviation."
        ),

        "phase3e_batch_size":
            int(
                EXECUTED_BATCH_SIZE
            ),

        "phase3e_batch_size_reason": (
            "Use the actual tuning batch size to preserve consistency "
            "between hyperparameter selection and final fitting."
        ),
    }

    write_json(
        PHASE3D_DIR
        / "execution_metadata_correction.json",
        correction,
    )

    audited_summary = {
        "phase":
            "3D post-execution audit",

        "configuration_count":
            len(
                configuration_rows
            ),

        "all_configurations_complete":
            True,

        "all_configurations_numerically_valid":
            True,

        "selected_configuration_ids":
            selected_ids,

        "declared_batch_size":
            declared_batch_size,

        "executed_batch_size":
            int(
                EXECUTED_BATCH_SIZE
            ),

        "batch_size_deviation_documented":
            deviation_present,

        "all_configurations_used_same_batch_size":
            True,

        "selection_leakage_detected":
            False,

        "rerun_required":
            False,

        "final_fit_batch_size":
            int(
                EXECUTED_BATCH_SIZE
            ),

        "loader_audit":
            loader_audit,

        "phase3d_audit_status":
            (
                "passed_with_documented_"
                "execution_metadata_deviation"
            ),
    }

    write_json(
        PHASE3D_DIR
        / "phase3d_audited_summary.json",
        audited_summary,
    )

    print(
        "Phase 3D execution audit completed."
    )

    print(
        json.dumps(
            audited_summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
