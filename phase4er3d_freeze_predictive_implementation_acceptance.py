from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "tier_c_v4"

MODEL_IDS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

EXPECTED_MODEL_COUNT = 6
EXPECTED_TOTAL_OPTIMIZER_STEPS = 6
GRADIENT_CLIP_NORM = 5.0
GRADIENT_TOLERANCE = 1e-4


SMOKE_DIR = Path(
    "outputs/"
    "phase4er3c_tier_c_v4_predictive_smoke"
)

SMOKE_SUMMARY_PATH = (
    SMOKE_DIR
    / "phase4er3c_predictive_smoke_summary.json"
)

EXECUTION_CONTRACT_SUMMARY_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "phase4dr32_execution_contract_summary.json"
)

CODEC_FREEZE_SUMMARY_PATH = Path(
    "outputs/"
    "phase4dr33_tier_c_v4_codec_microarchitecture/"
    "phase4dr33_codec_microarchitecture_summary.json"
)

SMOKE_BATCH_PROVENANCE_PATH = Path(
    "outputs/"
    "phase4er3br1_tier_c_v4_operation_complete_smoke_batch/"
    "operation_complete_smoke_batch_provenance.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4er3d_tier_c_v4_implementation_acceptance"
)


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(
    path: Path,
    value: Any,
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
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    fieldnames: list[str] = []

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
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def validate_phase_sources() -> dict[str, Any]:
    required_paths = (
        SMOKE_SUMMARY_PATH,
        EXECUTION_CONTRACT_SUMMARY_PATH,
        CODEC_FREEZE_SUMMARY_PATH,
        SMOKE_BATCH_PROVENANCE_PATH,
    )

    for model_id in MODEL_IDS:
        required_paths += (
            SMOKE_DIR
            / f"{model_id}_smoke_result.json",
        )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    smoke_summary = load_json(
        SMOKE_SUMMARY_PATH
    )

    execution_summary = load_json(
        EXECUTION_CONTRACT_SUMMARY_PATH
    )

    codec_summary = load_json(
        CODEC_FREEZE_SUMMARY_PATH
    )

    batch_provenance = load_json(
        SMOKE_BATCH_PROVENANCE_PATH
    )

    require(
        smoke_summary[
            "phase4er3c_status"
        ] == "passed",
        "Phase 4E-R3C did not pass.",
    )

    require(
        smoke_summary[
            "implementation_smoke_test_passed"
        ] is True,
        "Implementation smoke acceptance is false.",
    )

    require(
        smoke_summary[
            "model_family_count"
        ] == EXPECTED_MODEL_COUNT,
        "Unexpected smoke model-family count.",
    )

    require(
        smoke_summary[
            "model_family_pass_count"
        ] == EXPECTED_MODEL_COUNT,
        "Not every model family passed.",
    )

    require(
        set(
            smoke_summary[
                "model_ids_passed"
            ]
        ) == set(MODEL_IDS),
        "Passed model-family registry is incomplete.",
    )

    require(
        smoke_summary[
            "total_optimizer_steps"
        ] == EXPECTED_TOTAL_OPTIMIZER_STEPS,
        "Unexpected smoke optimizer-step count.",
    )

    require(
        smoke_summary[
            "all_gradients_finite"
        ] is True,
        "Smoke summary records non-finite gradients.",
    )

    require(
        smoke_summary[
            "all_model_families_updated_parameters"
        ] is True,
        "Not every model family updated parameters.",
    )

    require(
        smoke_summary[
            "validation_data_read"
        ] is False,
        "Validation data was read by the smoke phase.",
    )

    require(
        smoke_summary[
            "validation_selection_performed"
        ] is False,
        "Validation selection was performed.",
    )

    require(
        smoke_summary[
            "privileged_data_read"
        ] is False,
        "Privileged data was read.",
    )

    require(
        smoke_summary[
            "test_data_generated"
        ] is False,
        "Test data was generated.",
    )

    require(
        smoke_summary[
            "test_data_read"
        ] is False,
        "Test data was read.",
    )

    require(
        smoke_summary[
            "test_metrics_computed"
        ] is False,
        "Test metrics were computed.",
    )

    require(
        smoke_summary[
            "test_open_count"
        ] == 0,
        "Test data was opened.",
    )

    require(
        smoke_summary[
            "scientific_comparison_performed"
        ] is False,
        "A scientific comparison was performed.",
    )

    require(
        smoke_summary[
            "scientific_metrics_reported"
        ] is False,
        "Scientific metrics were reported.",
    )

    require(
        execution_summary[
            "phase4dr32_status"
        ] == "execution_contract_frozen",
        "Execution contract is not frozen.",
    )

    require(
        codec_summary[
            "phase4dr33_status"
        ] == "codec_microarchitecture_frozen",
        "Codec microarchitecture is not frozen.",
    )

    require(
        batch_provenance[
            "operation_complete_smoke_batch_status"
        ] == "prepared_and_verified",
        "Smoke batch is not verified.",
    )

    require(
        batch_provenance[
            "all_six_primitives_covered"
        ] is True,
        "Smoke batch does not cover all primitives.",
    )

    require(
        batch_provenance[
            "coverage"
        ][
            "all_required_sequence_lengths_present"
        ] is True,
        "Smoke batch does not cover sequence lengths 14.",
    )

    return {
        "required_paths":
            required_paths,

        "smoke_summary":
            smoke_summary,

        "execution_summary":
            execution_summary,

        "codec_summary":
            codec_summary,

        "batch_provenance":
            batch_provenance,
    }


def validate_model_result(
    model_id: str,
) -> dict[str, Any]:
    path = (
        SMOKE_DIR
        / f"{model_id}_smoke_result.json"
    )

    result = load_json(path)

    require(
        result["model_id"] == model_id,
        f"Model ID changed in {path}.",
    )

    require(
        result["smoke_status"] == "passed",
        f"{model_id} smoke status is not passed.",
    )

    require(
        result[
            "backward_pass_completed"
        ] is True,
        f"{model_id} backward pass did not complete.",
    )

    require(
        result[
            "optimizer_step_count"
        ] == 1,
        f"{model_id} did not execute exactly one optimizer step.",
    )

    require(
        result[
            "physical_microbatch_count"
        ] == 3,
        f"{model_id} did not execute three physical microbatches.",
    )

    before = result[
        "gradient_before_clipping"
    ]

    after = result[
        "gradient_after_clipping"
    ]

    require(
        before[
            "none_gradient_parameter_count"
        ] == 0,
        f"{model_id} has missing gradients.",
    )

    require(
        before[
            "nonfinite_gradient_parameter_count"
        ] == 0,
        f"{model_id} has non-finite gradients.",
    )

    require(
        before[
            "zero_gradient_parameter_count"
        ] == 0,
        f"{model_id} has zero-gradient parameters.",
    )

    require(
        after[
            "nonfinite_gradient_parameter_count"
        ] == 0,
        f"{model_id} has non-finite gradients after clipping.",
    )

    require(
        math.isfinite(
            before[
                "global_gradient_norm"
            ]
        ),
        f"{model_id} pre-clipping gradient norm is non-finite.",
    )

    require(
        math.isfinite(
            after[
                "global_gradient_norm"
            ]
        ),
        f"{model_id} post-clipping gradient norm is non-finite.",
    )

    require(
        after[
            "global_gradient_norm"
        ]
        <= (
            GRADIENT_CLIP_NORM
            + GRADIENT_TOLERANCE
        ),
        (
            f"{model_id} post-clipping norm exceeds "
            f"{GRADIENT_CLIP_NORM}."
        ),
    )

    require(
        math.isfinite(
            result[
                "clip_grad_norm_returned_preclip_norm"
            ]
        ),
        f"{model_id} clipping return value is non-finite.",
    )

    parameter_changes = result[
        "parameter_changes"
    ]

    require(
        parameter_changes[
            "parameter_tensor_count"
        ] > 0,
        f"{model_id} has no trainable parameter tensors.",
    )

    require(
        parameter_changes[
            "changed_parameter_tensor_count"
        ] > 0,
        f"{model_id} optimizer step changed no parameters.",
    )

    require(
        parameter_changes[
            "maximum_absolute_parameter_change"
        ] > 0.0,
        f"{model_id} maximum parameter change is zero.",
    )

    require(
        math.isfinite(
            parameter_changes[
                "maximum_absolute_parameter_change"
            ]
        ),
        f"{model_id} parameter change is non-finite.",
    )

    require(
        result["checkpoint_saved"] is False,
        f"{model_id} saved a smoke checkpoint.",
    )

    require(
        result["scientific_metric"] is False,
        f"{model_id} labeled a smoke loss as scientific.",
    )

    component_values = result[
        "weighted_component_means"
    ]

    for name, value in (
        component_values.items()
    ):
        require(
            math.isfinite(value),
            (
                f"{model_id} component {name} "
                "is non-finite."
            ),
        )

    return {
        "model_id":
            model_id,

        "configuration_id":
            result[
                "configuration_id"
            ],

        "model_class":
            result[
                "model_class"
            ],

        "codec_parameter_count":
            result[
                "parameter_counts"
            ][
                "codec"
            ],

        "transition_model_parameter_count":
            result[
                "parameter_counts"
            ][
                "transition_model"
            ],

        "total_parameter_count":
            result[
                "parameter_counts"
            ][
                "total"
            ],

        "physical_microbatch_count":
            result[
                "physical_microbatch_count"
            ],

        "preclip_gradient_norm":
            before[
                "global_gradient_norm"
            ],

        "postclip_gradient_norm":
            after[
                "global_gradient_norm"
            ],

        "optimizer_step_count":
            result[
                "optimizer_step_count"
            ],

        "parameter_tensor_count":
            parameter_changes[
                "parameter_tensor_count"
            ],

        "changed_parameter_tensor_count":
            parameter_changes[
                "changed_parameter_tensor_count"
            ],

        "unchanged_parameter_tensor_count":
            parameter_changes[
                "unchanged_parameter_tensor_count"
            ],

        "maximum_absolute_parameter_change":
            parameter_changes[
                "maximum_absolute_parameter_change"
            ],

        "finite_loss_components":
            True,

        "finite_gradients":
            True,

        "gradient_clipping_passed":
            True,

        "parameter_update_passed":
            True,

        "model_smoke_passed":
            True,

        "source_result_path":
            str(path),

        "source_result_sha256":
            sha256_file(path),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "phase4er3d_implementation_acceptance_summary.json"
    )

    if summary_path.exists():
        existing = load_json(
            summary_path
        )

        if existing.get(
            "phase4er3d_status"
        ) == "implementation_accepted":
            print(
                "Phase 4E-R3D implementation acceptance "
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
        "[1/4] Validating protocol, codec, and smoke sources"
    )

    sources = validate_phase_sources()

    print(
        "[2/4] Validating all six model-family smoke results"
    )

    model_rows = [
        validate_model_result(
            model_id
        )
        for model_id in MODEL_IDS
    ]

    write_csv(
        OUTPUT_DIR
        / "model_family_implementation_acceptance.csv",
        model_rows,
    )

    print(
        "[3/4] Freezing source hashes"
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
        / "implementation_acceptance_source_hashes.json",
        source_hashes,
    )

    print(
        "[4/4] Freezing implementation acceptance"
    )

    summary = {
        "phase":
            (
                "4E-R3D Tier C v4 predictive "
                "implementation acceptance freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_phase4er3c_status":
            sources[
                "smoke_summary"
            ][
                "phase4er3c_status"
            ],

        "source_execution_contract_frozen":
            True,

        "source_codec_microarchitecture_frozen":
            True,

        "source_operation_complete_smoke_batch_verified":
            True,

        "model_ids":
            list(MODEL_IDS),

        "model_family_count":
            len(model_rows),

        "model_family_acceptance_count":
            sum(
                int(
                    row[
                        "model_smoke_passed"
                    ]
                )
                for row in model_rows
            ),

        "all_forward_paths_passed":
            True,

        "all_backward_paths_passed":
            True,

        "all_gradients_finite":
            True,

        "all_gradient_clipping_checks_passed":
            True,

        "all_parameter_update_checks_passed":
            True,

        "optimizer_steps_observed":
            sum(
                row[
                    "optimizer_step_count"
                ]
                for row in model_rows
            ),

        "expected_optimizer_steps":
            EXPECTED_TOTAL_OPTIMIZER_STEPS,

        "all_six_primitives_covered":
            True,

        "sequence_lengths_covered":
            [1, 2, 3, 4],

        "identity_single_step_available":
            False,

        "identity_coverage_mode":
            "contained_in_length_2_sequence",

        "validation_data_read":
            False,

        "validation_selection_performed":
            False,

        "privileged_data_read":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "scientific_comparison_performed":
            False,

        "scientific_metrics_reported":
            False,

        "smoke_losses_are_scientific_results":
            False,

        "checkpoints_saved":
            False,

        "implementation_accepted":
            True,

        "tuning_execution_authorized":
            True,

        "final_fit_execution_authorized":
            False,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "tier_c_v5_authorized":
            False,

        "prior_protocol_files_modified":
            False,

        "phase4er3d_status":
            "implementation_accepted",

        "next_phase":
            (
                "4F-R3 Tier C v4 predictive "
                "development tuning execution"
            ),
    }

    write_json(
        summary_path,
        summary,
    )

    print(
        "Phase 4E-R3D predictive implementation "
        "acceptance frozen."
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
