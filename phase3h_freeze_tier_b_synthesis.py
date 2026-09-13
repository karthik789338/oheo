from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PHASE2L_DIR = Path(
    "outputs/phase2l_phase2_synthesis"
)

PHASE3A_DIR = Path(
    "outputs/phase3a_tier_b_protocol"
)

PHASE3B_DIR = Path(
    "outputs/phase3b_tier_b_audit"
)

PHASE3D_DIR = Path(
    "outputs/phase3d_tier_b_tuning"
)

PHASE3E_DIR = Path(
    "outputs/phase3e_tier_b_final_fits"
)

PHASE3F_DIR = Path(
    "outputs/phase3f_tier_b_prediction"
)

PHASE3G_DIR = Path(
    "outputs/phase3g_tier_b_structure"
)

OUTPUT_DIR = Path(
    "outputs/phase3h_tier_b_synthesis"
)


NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

MODEL_IDS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

GENERIC_STRUCTURAL_MODELS = (
    "B2",
    "B3",
    "B5",
)

EXPECTED_PREDICTIVE_SUMMARY_ROWS = 140
EXPECTED_GAP_ROWS = 35
EXPECTED_BEHAVIORAL_SUMMARY_ROWS = 105
EXPECTED_BEHAVIORAL_RUN_ROWS = 405
EXPECTED_OCM_HARD_ROWS = 25


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

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

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
        writer.writerows(
            rows
        )


def write_text(
    path: Path,
    text: str,
):
    path.write_text(
        text,
        encoding="utf-8",
    )


def as_float(
    row,
    key,
):
    return float(
        row[key]
    )


def as_int(
    row,
    key,
):
    return int(
        float(
            row[key]
        )
    )


def approximately_equal(
    first,
    second,
    tolerance=1e-10,
):
    return bool(
        abs(
            float(first)
            - float(second)
        )
        <= tolerance
    )


def validate_source_phases():
    phase2l = load_json(
        PHASE2L_DIR
        / "phase2l_summary.json"
    )

    phase3a = load_json(
        PHASE3A_DIR
        / "phase3a_summary.json"
    )

    phase3b = load_json(
        PHASE3B_DIR
        / "phase3b_summary.json"
    )

    phase3d = load_json(
        PHASE3D_DIR
        / "phase3d_audited_summary.json"
    )

    phase3e = load_json(
        PHASE3E_DIR
        / "phase3e_summary.json"
    )

    phase3f = load_json(
        PHASE3F_DIR
        / "phase3f_summary.json"
    )

    phase3g = load_json(
        PHASE3G_DIR
        / "phase3g_summary.json"
    )

    checks = {
        "phase2l": (
            phase2l[
                "phase2l_status"
            ]
            == "passed"
            and phase2l[
                "phase2_results_frozen"
            ]
            is True
        ),

        "phase3a": (
            phase3a[
                "status"
            ]
            == "protocol_frozen"
        ),

        "phase3b": (
            phase3b[
                "phase3b_status"
            ]
            == "passed"
            and phase3b[
                "tier_b_data_accepted"
            ]
            is True
            and phase3b[
                "affine_shortcut_removed"
            ]
            is True
        ),

        "phase3d": (
            phase3d[
                "phase3d_audit_status"
            ]
            == (
                "passed_with_documented_"
                "execution_metadata_deviation"
            )
            and phase3d[
                "rerun_required"
            ]
            is False
        ),

        "phase3e": (
            phase3e[
                "phase3e_status"
            ]
            == "passed"
            and phase3e[
                "completed_run_count"
            ]
            == 130
        ),

        "phase3f": (
            phase3f[
                "phase3f_status"
            ]
            == "passed"
            and phase3f[
                "evaluation_run_count"
            ]
            == 135
            and phase3f[
                "cell_result_count"
            ]
            == 540
        ),

        "phase3g": (
            phase3g[
                "phase3g_status"
            ]
            == "passed"
            and phase3g[
                "behavioral_evaluation_run_count"
            ]
            == 135
            and phase3g[
                "ocm_internal_hard_run_count"
            ]
            == 25
        ),
    }

    failed = [
        phase
        for phase, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise AssertionError(
            "Incomplete source phases: "
            + ", ".join(
                failed
            )
        )

    return {
        "phase2l": phase2l,
        "phase3a": phase3a,
        "phase3b": phase3b,
        "phase3d": phase3d,
        "phase3e": phase3e,
        "phase3f": phase3f,
        "phase3g": phase3g,
        "checks": checks,
    }


def load_phase3_results():
    predictive_rows = load_csv(
        PHASE3F_DIR
        / "model_cell_noise_summary.csv"
    )

    gap_rows = load_csv(
        PHASE3F_DIR
        / "generalization_gap_summary.csv"
    )

    validation_rows = load_csv(
        PHASE3E_DIR
        / "validation_summary.csv"
    )

    structural_summary_rows = load_csv(
        PHASE3G_DIR
        / "behavioral_model_noise_summary.csv"
    )

    behavioral_run_rows = load_csv(
        PHASE3G_DIR
        / "all_behavioral_run_metrics.csv"
    )

    hard_run_rows = load_csv(
        PHASE3G_DIR
        / "ocm_internal_hard_run_metrics.csv"
    )

    if (
        len(predictive_rows)
        != EXPECTED_PREDICTIVE_SUMMARY_ROWS
    ):
        raise AssertionError(
            "Unexpected predictive-summary size: "
            f"{len(predictive_rows)}"
        )

    if len(gap_rows) != EXPECTED_GAP_ROWS:
        raise AssertionError(
            "Unexpected gap-summary size: "
            f"{len(gap_rows)}"
        )

    if (
        len(structural_summary_rows)
        != EXPECTED_BEHAVIORAL_SUMMARY_ROWS
    ):
        raise AssertionError(
            "Unexpected structural-summary size: "
            f"{len(structural_summary_rows)}"
        )

    if (
        len(behavioral_run_rows)
        != EXPECTED_BEHAVIORAL_RUN_ROWS
    ):
        raise AssertionError(
            "Unexpected behavioral-run size: "
            f"{len(behavioral_run_rows)}"
        )

    if (
        len(hard_run_rows)
        != EXPECTED_OCM_HARD_ROWS
    ):
        raise AssertionError(
            "Unexpected OCM-hard-run size: "
            f"{len(hard_run_rows)}"
        )

    return {
        "predictive": predictive_rows,
        "gaps": gap_rows,
        "validation": validation_rows,
        "structural_summary":
            structural_summary_rows,
        "behavioral_runs":
            behavioral_run_rows,
        "hard_runs":
            hard_run_rows,
    }


def find_unique_row(
    rows,
    **conditions,
):
    matches = []

    for row in rows:
        matched = True

        for key, expected in (
            conditions.items()
        ):
            value = row[key]

            if isinstance(
                expected,
                float,
            ):
                matched = matched and (
                    approximately_equal(
                        value,
                        expected,
                    )
                )
            else:
                matched = matched and (
                    str(value)
                    == str(expected)
                )

        if matched:
            matches.append(row)

    if len(matches) != 1:
        raise AssertionError(
            "Expected one matching row for "
            f"{conditions}, found {len(matches)}."
        )

    return matches[0]


def build_predictive_key_results(
    results,
):
    predictive_rows = results[
        "predictive"
    ]

    validation_rows = results[
        "validation"
    ]

    output = []

    for noise in NOISE_LEVELS:
        joint_rows = [
            row
            for row in predictive_rows
            if (
                row["cell_id"]
                == "test_joint"
                and approximately_equal(
                    row[
                        "noise_fraction"
                    ],
                    noise,
                )
            )
        ]

        if len(joint_rows) != len(
            MODEL_IDS
        ):
            raise AssertionError(
                f"Noise {noise} has an invalid "
                "joint-OOD model count."
            )

        joint_rows.sort(
            key=lambda row: (
                as_float(
                    row,
                    "noisy_mse_mean",
                ),
                row["model_id"],
            )
        )

        best_test = joint_rows[0]

        validation_candidates = [
            row
            for row in validation_rows
            if approximately_equal(
                row[
                    "noise_fraction"
                ],
                noise,
            )
        ]

        validation_candidates.sort(
            key=lambda row: (
                as_float(
                    row,
                    "validation_mse_mean",
                ),
                row["model_id"],
            )
        )

        best_validation = (
            validation_candidates[0]
        )

        b2 = find_unique_row(
            joint_rows,
            model_id="B2",
        )

        b3 = find_unique_row(
            joint_rows,
            model_id="B3",
        )

        b4 = find_unique_row(
            joint_rows,
            model_id="B4",
        )

        ocm = find_unique_row(
            joint_rows,
            model_id="OCM",
        )

        output.append(
            {
                "noise_fraction": noise,

                "best_validation_model":
                    best_validation[
                        "model_id"
                    ],

                "best_validation_mse":
                    as_float(
                        best_validation,
                        "validation_mse_mean",
                    ),

                "best_joint_test_model":
                    best_test[
                        "model_id"
                    ],

                "best_joint_test_noisy_mse":
                    as_float(
                        best_test,
                        "noisy_mse_mean",
                    ),

                "best_joint_test_clean_mse":
                    as_float(
                        best_test,
                        "clean_mse_mean",
                    ),

                "b2_joint_noisy_mse":
                    as_float(
                        b2,
                        "noisy_mse_mean",
                    ),

                "b3_joint_noisy_mse":
                    as_float(
                        b3,
                        "noisy_mse_mean",
                    ),

                "b4_joint_noisy_mse":
                    as_float(
                        b4,
                        "noisy_mse_mean",
                    ),

                "ocm_joint_noisy_mse":
                    as_float(
                        ocm,
                        "noisy_mse_mean",
                    ),

                "ocm_joint_clean_mse":
                    as_float(
                        ocm,
                        "clean_mse_mean",
                    ),

                "b3_parameter_count":
                    as_int(
                        b3,
                        "parameter_count",
                    ),

                "ocm_parameter_count":
                    as_int(
                        ocm,
                        "parameter_count",
                    ),

                "b3_to_ocm_parameter_ratio":
                    (
                        as_int(
                            b3,
                            "parameter_count",
                        )
                        / as_int(
                            ocm,
                            "parameter_count",
                        )
                    ),

                "ocm_relative_noisy_disadvantage":
                    (
                        as_float(
                            ocm,
                            "noisy_mse_mean",
                        )
                        / as_float(
                            b3,
                            "noisy_mse_mean",
                        )
                        - 1.0
                    ),

                "validation_to_test_winner_changed":
                    (
                        best_validation[
                            "model_id"
                        ]
                        != best_test[
                            "model_id"
                        ]
                    ),
            }
        )

    return output


def build_gap_key_results(
    results,
):
    output = []

    for model_id in (
        "B2",
        "B3",
        "OCM",
    ):
        for noise in NOISE_LEVELS:
            row = find_unique_row(
                results["gaps"],
                model_id=model_id,
                noise_fraction=noise,
            )

            output.append(
                {
                    "model_id": model_id,
                    "noise_fraction": noise,

                    "composition_noisy_gap":
                        as_float(
                            row,
                            "composition_noisy_gap",
                        ),

                    "carrier_noisy_gap":
                        as_float(
                            row,
                            "carrier_noisy_gap",
                        ),

                    "joint_noisy_gap":
                        as_float(
                            row,
                            "joint_noisy_gap",
                        ),

                    "composition_carrier_noisy_interaction":
                        as_float(
                            row,
                            (
                                "composition_carrier_"
                                "noisy_interaction"
                            ),
                        ),

                    "carrier_gap_exceeds_composition_gap":
                        (
                            as_float(
                                row,
                                "carrier_noisy_gap",
                            )
                            >
                            as_float(
                                row,
                                "composition_noisy_gap",
                            )
                        ),
                }
            )

    return output


def build_structural_key_results(
    results,
):
    output = []

    for noise in NOISE_LEVELS:
        for model_id in MODEL_IDS:
            row = find_unique_row(
                results[
                    "structural_summary"
                ],
                model_id=model_id,
                training_noise_fraction=noise,
                carrier_split="test",
            )

            output.append(
                {
                    "noise_fraction": noise,
                    "model_id": model_id,

                    "fit_count":
                        as_int(
                            row,
                            "fit_count",
                        ),

                    "parameter_count":
                        as_int(
                            row,
                            "parameter_count",
                        ),

                    "test_mapping_accuracy":
                        as_float(
                            row,
                            "test_mapping_accuracy_mean",
                        ),

                    "test_exact_transformation_rate":
                        as_float(
                            row,
                            (
                                "test_exact_"
                                "transformation_rate_mean"
                            ),
                        ),

                    "test_consensus_exact_transformation_rate":
                        as_float(
                            row,
                            (
                                "test_consensus_exact_"
                                "transformation_rate_mean"
                            ),
                        ),

                    "test_carrier_consistency":
                        as_float(
                            row,
                            "test_carrier_consistency_mean",
                        ),

                    "test_relation_accuracy":
                        as_float(
                            row,
                            (
                                "test_involved_relation_"
                                "accuracy_mean"
                            ),
                        ),

                    "test_relation_balanced_accuracy":
                        as_float(
                            row,
                            (
                                "test_involved_relation_"
                                "balanced_accuracy_mean"
                            ),
                        ),

                    "test_relation_majority_baseline":
                        as_float(
                            row,
                            (
                                "test_involved_relation_"
                                "majority_baseline"
                            ),
                        ),
                }
            )

    return output


def build_ocm_seed_stability(
    results,
):
    behavioral_lookup = {
        row["run_id"]: row
        for row in results[
            "behavioral_runs"
        ]
        if (
            row["model_id"] == "OCM"
            and row[
                "carrier_split"
            ] == "test"
        )
    }

    if len(behavioral_lookup) != 25:
        raise AssertionError(
            "Expected 25 OCM test-carrier rows."
        )

    hard_lookup = {
        row["run_id"]: row
        for row in results[
            "hard_runs"
        ]
    }

    if set(
        behavioral_lookup
    ) != set(
        hard_lookup
    ):
        raise AssertionError(
            "OCM behavioral and hard run IDs differ."
        )

    output = []

    for run_id in sorted(
        behavioral_lookup,
        key=lambda value: (
            as_float(
                behavioral_lookup[value],
                "training_noise_fraction",
            ),
            as_int(
                behavioral_lookup[value],
                "seed",
            ),
        ),
    ):
        behavioral = behavioral_lookup[
            run_id
        ]

        hard = hard_lookup[
            run_id
        ]

        behavioral_exact = (
            as_float(
                behavioral,
                (
                    "test_consensus_exact_"
                    "transformation_rate"
                ),
            )
        )

        hard_exact = as_float(
            hard,
            (
                "hard_test_exact_"
                "transformation_rate"
            ),
        )

        hard_relation = as_float(
            hard,
            (
                "hard_test_involved_relation_"
                "balanced_accuracy"
            ),
        )

        behavioral_relation = as_float(
            behavioral,
            (
                "test_involved_relation_"
                "balanced_accuracy"
            ),
        )

        alignment = as_float(
            hard,
            "test_carrier_alignment_accuracy",
        )

        output.append(
            {
                "run_id": run_id,

                "seed":
                    as_int(
                        behavioral,
                        "seed",
                    ),

                "training_noise_fraction":
                    as_float(
                        behavioral,
                        "training_noise_fraction",
                    ),

                "test_alignment_accuracy":
                    alignment,

                "hard_test_exact_transformation_rate":
                    hard_exact,

                "behavioral_consensus_exact_transformation_rate":
                    behavioral_exact,

                "hard_test_relation_balanced_accuracy":
                    hard_relation,

                "behavioral_test_relation_balanced_accuracy":
                    behavioral_relation,

                "hard_relation_advantage":
                    (
                        hard_relation
                        - behavioral_relation
                    ),

                "hard_structure_exact":
                    approximately_equal(
                        hard_exact,
                        1.0,
                    ),

                "behavioral_consensus_exact":
                    approximately_equal(
                        behavioral_exact,
                        1.0,
                    ),

                "alignment_exact":
                    approximately_equal(
                        alignment,
                        1.0,
                    ),
            }
        )

    return output


def build_ocm_noise_summary(
    seed_rows,
):
    output = []

    for noise in NOISE_LEVELS:
        rows = [
            row
            for row in seed_rows
            if approximately_equal(
                row[
                    "training_noise_fraction"
                ],
                noise,
            )
        ]

        if len(rows) != 5:
            raise AssertionError(
                f"Noise {noise} does not contain "
                "five OCM seeds."
            )

        hard_exact_count = sum(
            row[
                "hard_structure_exact"
            ]
            for row in rows
        )

        behavioral_exact_count = sum(
            row[
                "behavioral_consensus_exact"
            ]
            for row in rows
        )

        failed_rows = [
            row
            for row in rows
            if not row[
                "hard_structure_exact"
            ]
        ]

        output.append(
            {
                "noise_fraction": noise,

                "seed_count": 5,

                "hard_exact_seed_count":
                    hard_exact_count,

                "hard_nonexact_seed_count":
                    5 - hard_exact_count,

                "behavioral_consensus_exact_seed_count":
                    behavioral_exact_count,

                "behavioral_consensus_nonexact_seed_count":
                    5
                    - behavioral_exact_count,

                "mean_test_alignment_accuracy":
                    float(
                        np.mean(
                            [
                                row[
                                    "test_alignment_accuracy"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "failed_run_count":
                    len(
                        failed_rows
                    ),

                "failed_runs_with_hard_relation_advantage":
                    sum(
                        row[
                            "hard_relation_advantage"
                        ] > 0.0
                        for row in failed_rows
                    ),
            }
        )

    return output


def derive_frozen_findings(
    predictive_rows,
    gap_rows,
    structural_rows,
    ocm_seed_rows,
    ocm_noise_rows,
):
    b3_wins_all_joint_conditions = all(
        row[
            "best_joint_test_model"
        ] == "B3"
        for row in predictive_rows
    )

    validation_test_reversal_count = sum(
        row[
            "validation_to_test_winner_changed"
        ]
        for row in predictive_rows
    )

    carrier_dominates_for_b2_b3 = all(
        row[
            "carrier_gap_exceeds_composition_gap"
        ]
        for row in gap_rows
        if row["model_id"] in {
            "B2",
            "B3",
        }
    )

    generic_exact_consensus = all(
        approximately_equal(
            row[
                "test_consensus_exact_transformation_rate"
            ],
            1.0,
        )
        and approximately_equal(
            row[
                "test_relation_balanced_accuracy"
            ],
            1.0,
        )
        for row in structural_rows
        if row["model_id"]
        in GENERIC_STRUCTURAL_MODELS
    )

    ocm_hard_exact_count = sum(
        row["hard_structure_exact"]
        for row in ocm_seed_rows
    )

    ocm_behavioral_exact_count = sum(
        row[
            "behavioral_consensus_exact"
        ]
        for row in ocm_seed_rows
    )

    ocm_failed_rows = [
        row
        for row in ocm_seed_rows
        if not row[
            "hard_structure_exact"
        ]
    ]

    hard_advantage_on_all_failed_runs = all(
        row[
            "hard_relation_advantage"
        ] > 0.0
        for row in ocm_failed_rows
    )

    exact_alignment_association = all(
        row[
            "hard_structure_exact"
        ]
        == row[
            "alignment_exact"
        ]
        for row in ocm_seed_rows
    )

    success_counts = [
        row[
            "hard_exact_seed_count"
        ]
        for row in ocm_noise_rows
    ]

    monotonic_noise_degradation = all(
        success_counts[index]
        >= success_counts[index + 1]
        for index in range(
            len(success_counts) - 1
        )
    )

    b1_noise_zero = find_unique_row(
        structural_rows,
        model_id="B1",
        noise_fraction=0.0,
    )

    consensus_masks_local_error = bool(
        approximately_equal(
            b1_noise_zero[
                "test_consensus_exact_transformation_rate"
            ],
            1.0,
        )
        and b1_noise_zero[
            "test_exact_transformation_rate"
        ] < 1.0
    )

    ocm_is_never_best = all(
        row[
            "best_joint_test_model"
        ] != "OCM"
        for row in predictive_rows
    )

    parameter_ratio = predictive_rows[
        0
    ][
        "b3_to_ocm_parameter_ratio"
    ]

    return {
        "b3_wins_all_joint_conditions":
            b3_wins_all_joint_conditions,

        "validation_test_reversal_count":
            validation_test_reversal_count,

        "carrier_dominates_for_b2_b3":
            carrier_dominates_for_b2_b3,

        "generic_exact_consensus":
            generic_exact_consensus,

        "ocm_hard_exact_count":
            ocm_hard_exact_count,

        "ocm_hard_nonexact_count":
            25 - ocm_hard_exact_count,

        "ocm_behavioral_exact_count":
            ocm_behavioral_exact_count,

        "ocm_behavioral_nonexact_count":
            25
            - ocm_behavioral_exact_count,

        "hard_advantage_on_all_failed_runs":
            hard_advantage_on_all_failed_runs,

        "exact_alignment_association":
            exact_alignment_association,

        "ocm_success_counts_by_noise":
            success_counts,

        "monotonic_noise_degradation":
            monotonic_noise_degradation,

        "consensus_masks_local_error":
            consensus_masks_local_error,

        "ocm_is_never_best":
            ocm_is_never_best,

        "b3_to_ocm_parameter_ratio":
            parameter_ratio,
    }


def build_claim_registry(
    findings,
):
    claims = [
        {
            "claim_id": "TBC01",
            "status": "supported",
            "claim": (
                "Tier B removes the exact affine interpolation "
                "shortcut present in Tier A."
            ),
            "supported": True,
            "qualification": (
                "Affine models remain behaviorally competitive; "
                "shortcut removal means exact held-out-manifold "
                "interpolation no longer holds."
            ),
        },

        {
            "claim_id": "TBC02",
            "status": "supported",
            "claim": (
                "B3 provides the lowest joint-OOD predictive MSE "
                "at every tested noise level."
            ),
            "supported": findings[
                "b3_wins_all_joint_conditions"
            ],
        },

        {
            "claim_id": "TBC03",
            "status": "supported",
            "claim": (
                "Unseen carrier variation is a larger continuous "
                "prediction challenge than unseen operation "
                "composition for B2 and B3."
            ),
            "supported": findings[
                "carrier_dominates_for_b2_b3"
            ],
        },

        {
            "claim_id": "TBC04",
            "status": "supported_with_scope",
            "claim": (
                "B2, B3, and B5 recover the held-out "
                "transformation consensus and informativeness "
                "relations exactly on unseen carriers across all "
                "training-noise conditions."
            ),
            "supported": findings[
                "generic_exact_consensus"
            ],
            "qualification": (
                "The structural probe begins from clean manifold "
                "observations and uses nearest clean same-carrier "
                "state decoding."
            ),
        },

        {
            "claim_id": "TBC05",
            "status": "rejected",
            "claim": (
                "Explicit finite stochastic channels are necessary "
                "for recovery of the observable operation algebra."
            ),
            "supported": not findings[
                "generic_exact_consensus"
            ],
            "finding": (
                "Rejected because several generic baselines recover "
                "the observable algebra exactly."
            ),
        },

        {
            "claim_id": "TBC06",
            "status": "rejected",
            "claim": (
                "OCM improves joint-OOD predictive accuracy over "
                "the strongest generic baselines."
            ),
            "supported": not findings[
                "ocm_is_never_best"
            ],
            "finding": (
                "Rejected. OCM is not the predictive winner in any "
                "tested joint-OOD condition."
            ),
        },

        {
            "claim_id": "TBC07",
            "status": "supported_with_qualification",
            "claim": (
                "OCM provides a compact explicit representation."
            ),
            "supported": (
                findings[
                    "b3_to_ocm_parameter_ratio"
                ] > 30.0
            ),
            "qualification": (
                "OCM uses over 35 times fewer parameters than B3, "
                "but this compactness does not produce superior "
                "predictive or structural accuracy."
            ),
        },

        {
            "claim_id": "TBC08",
            "status": "supported",
            "claim": (
                "OCM exhibits seed-dependent structural bifurcation."
            ),
            "supported": (
                findings[
                    "ocm_hard_exact_count"
                ] == 18
                and findings[
                    "ocm_hard_nonexact_count"
                ] == 7
            ),
            "evidence": {
                "exact_runs":
                    findings[
                        "ocm_hard_exact_count"
                    ],
                "nonexact_runs":
                    findings[
                        "ocm_hard_nonexact_count"
                    ],
            },
        },

        {
            "claim_id": "TBC09",
            "status": "supported_with_qualification",
            "claim": (
                "Hard OCM channels expose more relation structure "
                "than behavioral rollout in each nonexact OCM run."
            ),
            "supported": findings[
                "hard_advantage_on_all_failed_runs"
            ],
            "qualification": (
                "Hard projection improves relation recovery but "
                "does not restore exact transformations in failed "
                "runs."
            ),
        },

        {
            "claim_id": "TBC10",
            "status": "supported",
            "claim": (
                "Continuous predictive accuracy and discrete "
                "structural correctness are empirically distinct."
            ),
            "supported": (
                findings[
                    "carrier_dominates_for_b2_b3"
                ]
                and findings[
                    "generic_exact_consensus"
                ]
            ),
        },

        {
            "claim_id": "TBC11",
            "status": "rejected",
            "claim": (
                "OCM structural recovery degrades monotonically "
                "as training noise increases."
            ),
            "supported": findings[
                "monotonic_noise_degradation"
            ],
            "finding": (
                "Rejected. Exact-seed counts are non-monotonic, "
                "supporting an optimization-sensitive rather than "
                "simple noise-threshold interpretation."
            ),
        },

        {
            "claim_id": "TBC12",
            "status": "supported_with_scope",
            "claim": (
                "Consensus structural recovery can be exact even "
                "when carrier-specific recovery is imperfect."
            ),
            "supported": findings[
                "consensus_masks_local_error"
            ],
            "qualification": (
                "Consensus metrics must therefore be reported "
                "together with carrier-level exact recovery."
            ),
        },
    ]

    for claim in claims:
        if claim["status"].startswith(
            "supported"
        ):
            if claim["supported"] is not True:
                raise AssertionError(
                    "Supported claim failed: "
                    f"{claim['claim_id']}"
                )

        if claim["status"] == "rejected":
            if claim["supported"] is not False:
                raise AssertionError(
                    "Rejected claim was supported: "
                    f"{claim['claim_id']}"
                )

    return claims


def build_research_question_status(
    findings,
):
    return {
        "RQ1": {
            "question": (
                "Can relative history-informativeness be recovered "
                "from noisy continuous observations?"
            ),
            "status": (
                "supported_under_clean_structural_probe_"
                "with_qualification"
            ),
            "finding": (
                "Models trained under every tested noise condition "
                "can recover held-out informativeness relations on "
                "clean unseen-carrier manifold points."
            ),
            "qualification": (
                "Phase 3G does not evaluate structural decoding "
                "from noisy probe inputs."
            ),
        },

        "RQ2": {
            "question": (
                "Can transformation structure and unseen "
                "compositions be recovered?"
            ),
            "status": (
                "strongly_supported_behaviorally"
            ),
            "finding": (
                "B2, B3, and B5 recover the held-out "
                "transformation consensus and relations exactly "
                "across all training-noise conditions."
            ),
            "internal_representation_caveat": (
                "Behavioral equivalence does not establish that "
                "generic models internally represent the true "
                "finite semigroup."
            ),
        },

        "RQ3": {
            "question": (
                "Does explicit finite operation structure improve "
                "unseen process-sequence prediction?"
            ),
            "strong_version_status": "rejected",
            "finding": (
                "B3 is the joint-OOD predictive winner at all five "
                "noise levels, while OCM is never the winner."
            ),
            "revised_version_status": (
                "supported_with_qualification"
            ),
            "revised_statement": (
                "OCM provides a compact and auditable explicit "
                "representation, but no predictive or behavioral "
                "structural superiority."
            ),
        },

        "RQ4": {
            "question": (
                "What determines structural identifiability?"
            ),
            "status": (
                "supported_with_revised_characterization"
            ),
            "finding": (
                "Identifiability is model- and optimization-dependent "
                "rather than governed by a simple noise threshold. "
                "OCM shows seed-specific exact and failed solutions, "
                "while several generic models remain behaviorally "
                "exact."
            ),
            "ocm_exact_run_count":
                findings[
                    "ocm_hard_exact_count"
                ],
            "ocm_nonexact_run_count":
                findings[
                    "ocm_hard_nonexact_count"
                ],
        },
    }


def build_tier_comparison(
    sources,
    findings,
):
    return {
        "comparison_scope": (
            "Descriptive comparison only. Tier A and Tier B differ "
            "in observation generation, data volume, carrier "
            "variation, and selected model sizes."
        ),

        "tier_a": {
            "observation_regime": (
                "Eight fixed prototypes in 12 dimensions."
            ),
            "affine_interpolation_shortcut":
                sources[
                    "phase2l"
                ][
                    "affine_shortcut_confirmed"
                ],
            "main_predictive_conclusion": (
                "Explicit OCM structure did not improve prediction."
            ),
            "main_structural_conclusion": (
                "Generic behavioral recovery was possible and often "
                "stronger than OCM."
            ),
        },

        "tier_b": {
            "observation_regime": (
                "Thirty-two-dimensional nonlinear distributed "
                "state-carrier manifolds."
            ),
            "affine_interpolation_shortcut":
                False,
            "carrier_variation_present":
                True,
            "best_joint_predictive_model":
                "B3",
            "generic_consensus_structure_exact":
                findings[
                    "generic_exact_consensus"
                ],
            "ocm_exact_run_count":
                findings[
                    "ocm_hard_exact_count"
                ],
            "ocm_nonexact_run_count":
                findings[
                    "ocm_hard_nonexact_count"
                ],
        },

        "shared_conclusion": (
            "Explicit finite channels are useful for direct inspection "
            "but are neither necessary for behavioral structural "
            "recovery nor sufficient for predictive superiority."
        ),

        "new_tier_b_conclusion": (
            "Continuous carrier-level prediction and discrete "
            "transformation recovery can separate sharply."
        ),

        "causal_cross_tier_claim_allowed":
            False,
    }


def build_limitations():
    return {
        "data_scope": [
            (
                "Tier B is a synthetic nonlinear manifold, not a "
                "phase-field, kinetic Monte Carlo, experimental, or "
                "real microstructure dataset."
            ),
            (
                "The continuous carrier is fixed throughout each "
                "trajectory."
            ),
            (
                "Only one frozen observation-generator realization "
                "is evaluated."
            ),
        ],

        "operation_scope": [
            (
                "All primitive operation identities are known during "
                "training."
            ),
            (
                "OOD evaluation holds out compositions of known "
                "primitives, not unseen primitive operations."
            ),
            (
                "The latent semigroup remains the same controlled "
                "eight-state system used in Tier A."
            ),
        ],

        "structural_probe_scope": [
            (
                "The Phase 3G structural probe starts from clean "
                "carrier-state manifold observations."
            ),
            (
                "Nearest-state decoding uses the clean state set on "
                "the same carrier."
            ),
            (
                "The audit does not establish structural recovery "
                "from noisy, off-manifold, or real observations."
            ),
            (
                "Consensus metrics can conceal carrier-specific "
                "transformation errors."
            ),
        ],

        "internal_representation_scope": [
            (
                "Behavioral recovery by B2, B3, B4, or B5 does not "
                "prove that their internal representations are "
                "isomorphic to the true semigroup."
            ),
            (
                "OCM latent-state alignment uses clean training "
                "carriers and true state identities for final "
                "evaluation only."
            ),
            (
                "The observed association between alignment failure "
                "and structural failure is diagnostic, not a causal "
                "proof of the failure mechanism."
            ),
        ],

        "model_selection_scope": [
            (
                "Hyperparameters were selected at one development "
                "seed and noise level."
            ),
            (
                "Model sizes differ substantially across families."
            ),
            (
                "The documented Phase 3D batch-size deviation was "
                "uniform across all configurations and was retained "
                "for final fitting."
            ),
        ],
    }


def build_do_not_claim():
    return [
        (
            "Do not claim that OCM is the best predictive model."
        ),
        (
            "Do not claim that OCM is necessary for recovering the "
            "observable transformation algebra."
        ),
        (
            "Do not claim that behavioral equivalence proves an "
            "internally equivalent algebraic representation."
        ),
        (
            "Do not claim structural recovery from noisy test inputs; "
            "the Phase 3G probe inputs are clean."
        ),
        (
            "Do not claim that exact consensus recovery means every "
            "carrier is recovered exactly."
        ),
        (
            "Do not claim universal robustness to arbitrary noise."
        ),
        (
            "Do not claim that OCM failures are caused by latent-state "
            "alignment; only an empirical association is shown."
        ),
        (
            "Do not attribute Tier A versus Tier B differences solely "
            "to shortcut removal because architecture size and data "
            "volume also changed."
        ),
        (
            "Do not describe the Tier B carrier as a validated physical "
            "microstructure variable."
        ),
        (
            "Do not claim results on unseen primitive operations."
        ),
        (
            "Do not claim results on real metallurgy data."
        ),
        (
            "Do not claim an information-theoretic impossibility or "
            "identifiability theorem from these experiments."
        ),
    ]


def build_paper_summary(
    predictive_rows,
    gap_rows,
    findings,
):
    noise_025 = find_unique_row(
        predictive_rows,
        noise_fraction=0.25,
    )

    b3_gap_025 = find_unique_row(
        gap_rows,
        model_id="B3",
        noise_fraction=0.25,
    )

    success_counts = (
        findings[
            "ocm_success_counts_by_noise"
        ]
    )

    return f"""PHASE 3 TIER B PAPER-SAFE SUMMARY

DATA REGIME

Tier B replaces the fixed Tier A prototypes with nonlinear,
32-dimensional state-carrier manifolds. The exact affine interpolation
shortcut is absent, while the underlying eight-state transformation
semigroup and held-out composition split remain controlled.

PREDICTION

B3 achieves the lowest joint-OOD predictive MSE at all five tested
training-noise conditions. At noise 0.25, its joint noisy-target MSE is
{noise_025['b3_joint_noisy_mse']:.6f}, compared with
{noise_025['ocm_joint_noisy_mse']:.6f} for OCM.

OCM uses {noise_025['ocm_parameter_count']} parameters, whereas B3 uses
{noise_025['b3_parameter_count']}, a ratio of
{noise_025['b3_to_ocm_parameter_ratio']:.2f}. This supports a compactness
claim but not predictive superiority.

GENERALIZATION DECOMPOSITION

For B3 at noise 0.25, the composition-only noisy-MSE gap is
{b3_gap_025['composition_noisy_gap']:.6f}, whereas the carrier-only gap is
{b3_gap_025['carrier_noisy_gap']:.6f}. Unseen continuous carrier variation
is therefore the dominant predictive difficulty.

STRUCTURAL RECOVERY

Under the clean-manifold structural probe, B2, B3, and B5 recover the
consensus mappings of all nine held-out transformations and all associated
informativeness relations exactly across every training-noise condition.

This does not imply that these models internally represent the true finite
semigroup. It establishes behavioral equivalence on the tested manifold
points.

OCM STABILITY

OCM's exact hard-channel recovery counts across noise levels
0.0, 0.1, 0.25, 0.5, and 1.0 are respectively:

{success_counts}

Across all conditions, {findings['ocm_hard_exact_count']} of 25 OCM runs
recover the hard transformation structure exactly, while
{findings['ocm_hard_nonexact_count']} do not.

In every nonexact OCM run, hard-channel projection retains more
informativeness-relation structure than observable behavioral rollout.
However, hard projection does not restore exact transformations.

MAIN CONCLUSION

Continuous predictive accuracy, discrete behavioral transformation
recovery, and explicit internal channel recovery are distinct properties.

Explicit finite channels provide a compact, directly auditable
representation, but they are not necessary for observable compositional
recovery and do not improve prediction over the strongest generic models.
"""


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = validate_source_phases()

    results = load_phase3_results()

    predictive_key = (
        build_predictive_key_results(
            results
        )
    )

    gap_key = build_gap_key_results(
        results
    )

    structural_key = (
        build_structural_key_results(
            results
        )
    )

    ocm_seed_rows = (
        build_ocm_seed_stability(
            results
        )
    )

    ocm_noise_rows = (
        build_ocm_noise_summary(
            ocm_seed_rows
        )
    )

    findings = derive_frozen_findings(
        predictive_rows=predictive_key,
        gap_rows=gap_key,
        structural_rows=structural_key,
        ocm_seed_rows=ocm_seed_rows,
        ocm_noise_rows=ocm_noise_rows,
    )

    claims = build_claim_registry(
        findings
    )

    rq_status = (
        build_research_question_status(
            findings
        )
    )

    tier_comparison = (
        build_tier_comparison(
            sources=sources,
            findings=findings,
        )
    )

    limitations = build_limitations()

    do_not_claim = build_do_not_claim()

    paper_summary = build_paper_summary(
        predictive_rows=predictive_key,
        gap_rows=gap_key,
        findings=findings,
    )

    write_csv(
        OUTPUT_DIR
        / "phase3_predictive_key_results.csv",
        predictive_key,
    )

    write_csv(
        OUTPUT_DIR
        / "phase3_generalization_key_results.csv",
        gap_key,
    )

    write_csv(
        OUTPUT_DIR
        / "phase3_structural_key_results.csv",
        structural_key,
    )

    write_csv(
        OUTPUT_DIR
        / "ocm_seed_stability.csv",
        ocm_seed_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "ocm_noise_stability_summary.csv",
        ocm_noise_rows,
    )

    write_json(
        OUTPUT_DIR
        / "phase3_frozen_findings.json",
        findings,
    )

    write_json(
        OUTPUT_DIR
        / "phase3_claim_registry.json",
        {
            "claim_count":
                len(claims),

            "claims":
                claims,
        },
    )

    write_json(
        OUTPUT_DIR
        / "phase3_research_question_status.json",
        rq_status,
    )

    write_json(
        OUTPUT_DIR
        / "tier_a_vs_tier_b_comparison.json",
        tier_comparison,
    )

    write_json(
        OUTPUT_DIR
        / "phase3_limitations.json",
        limitations,
    )

    write_json(
        OUTPUT_DIR
        / "phase3_do_not_claim.json",
        {
            "do_not_claim":
                do_not_claim
        },
    )

    write_text(
        OUTPUT_DIR
        / "phase3_paper_safe_summary.txt",
        paper_summary,
    )

    supported_count = sum(
        claim["status"].startswith(
            "supported"
        )
        for claim in claims
    )

    rejected_count = sum(
        claim["status"] == "rejected"
        for claim in claims
    )

    final_overview = {
        "phase":
            "Phase 3 Tier B final synthesis",

        "status":
            (
                "completed_with_positive_negative_"
                "and_qualified_results"
            ),

        "primary_predictive_finding": (
            "B3 is the joint-OOD predictive winner "
            "at every tested noise level."
        ),

        "primary_generalization_finding": (
            "Unseen carrier variation dominates unseen "
            "composition as a source of continuous prediction error."
        ),

        "primary_structural_finding": (
            "Generic models can recover the observable operation "
            "algebra exactly without explicit finite channels."
        ),

        "primary_ocm_finding": (
            "OCM is compact and directly auditable but exhibits "
            "seed-dependent exact and failed structural solutions."
        ),

        "primary_conceptual_finding": (
            "Continuous prediction, discrete behavioral recovery, "
            "and explicit internal channel recovery are distinct."
        ),

        "findings":
            findings,

        "research_question_status":
            rq_status,

        "phase2_results_remain_frozen":
            True,

        "phase3_results_frozen":
            True,
    }

    write_json(
        OUTPUT_DIR
        / "phase3_final_overview.json",
        final_overview,
    )

    summary = {
        "phase":
            "3H final Tier B synthesis and claim freeze",

        "source_phase_validation":
            sources["checks"],

        "predictive_key_row_count":
            len(predictive_key),

        "generalization_key_row_count":
            len(gap_key),

        "structural_key_row_count":
            len(structural_key),

        "ocm_seed_run_count":
            len(ocm_seed_rows),

        "ocm_exact_hard_structure_run_count":
            findings[
                "ocm_hard_exact_count"
            ],

        "ocm_nonexact_hard_structure_run_count":
            findings[
                "ocm_hard_nonexact_count"
            ],

        "claim_count":
            len(claims),

        "supported_or_qualified_claim_count":
            supported_count,

        "rejected_claim_count":
            rejected_count,

        "rq1_status":
            rq_status["RQ1"]["status"],

        "rq2_status":
            rq_status["RQ2"]["status"],

        "rq3_strong_version_status":
            rq_status[
                "RQ3"
            ][
                "strong_version_status"
            ],

        "rq3_revised_version_status":
            rq_status[
                "RQ3"
            ][
                "revised_version_status"
            ],

        "rq4_status":
            rq_status["RQ4"]["status"],

        "training_performed":
            False,

        "new_model_evaluation_performed":
            False,

        "checkpoint_selection_reopened":
            False,

        "checkpoints_modified":
            False,

        "prior_outputs_modified":
            False,

        "phase2_results_frozen":
            True,

        "phase3_results_frozen":
            True,

        "phase3h_status":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase3h_summary.json",
        summary,
    )

    print(
        "Phase 3H Tier B synthesis and "
        "claim freeze completed."
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
