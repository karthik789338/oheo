from __future__ import annotations

import csv
import json
from pathlib import Path


PHASE2D_DIR = Path(
    "outputs/phase2d_final_ocm"
)

PHASE2F_DIR = Path(
    "outputs/phase2f_ood_structural_audit"
)

PHASE2G_DIR = Path(
    "outputs/phase2g_baseline_protocol"
)

PHASE2H_DIR = Path(
    "outputs/phase2h_baseline_tuning"
)

PHASE2I_DIR = Path(
    "outputs/phase2i_final_baselines"
)

PHASE2J_DIR = Path(
    "outputs/phase2j_predictive_audit"
)

PHASE2K_DIR = Path(
    "outputs/phase2k_behavioral_structure"
)

OUTPUT_DIR = Path(
    "outputs/phase2l_phase2_synthesis"
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

EXPECTED_PREDICTIVE_SUMMARY_ROWS = 35
EXPECTED_BEHAVIORAL_SUMMARY_ROWS = 35
EXPECTED_HARD_COMPARISON_ROWS = 5


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


def validate_source_phases():
    phase2d = load_json(
        PHASE2D_DIR
        / "phase2d_summary.json"
    )

    phase2f = load_json(
        PHASE2F_DIR
        / "phase2f_summary.json"
    )

    phase2g = load_json(
        PHASE2G_DIR
        / "phase2g_summary.json"
    )

    phase2h = load_json(
        PHASE2H_DIR
        / "phase2h_summary.json"
    )

    phase2i = load_json(
        PHASE2I_DIR
        / "phase2i_summary.json"
    )

    phase2j = load_json(
        PHASE2J_DIR
        / "phase2j_summary.json"
    )

    phase2k = load_json(
        PHASE2K_DIR
        / "phase2k_summary.json"
    )

    checks = {
        "phase2d":
            (
                phase2d[
                    "completion_status"
                ]
                == "passed"
                and phase2d[
                    "total_run_count"
                ]
                == 25
            ),

        "phase2f":
            (
                phase2f[
                    "phase2f_status"
                ]
                == "passed"
                and phase2f[
                    "evaluated_run_estimator_groups"
                ]
                == 50
            ),

        "phase2g":
            (
                phase2g[
                    "status"
                ]
                == "protocol_frozen"
                and phase2g[
                    "sanity_checks"
                ]
                == "passed"
            ),

        "phase2h":
            (
                phase2h[
                    "phase2h_status"
                ]
                == "passed"
                and phase2h[
                    "completed_configuration_count"
                ]
                == 19
            ),

        "phase2i":
            (
                phase2i[
                    "completion_status"
                ]
                == "passed"
                and phase2i[
                    "baseline_final_fit_count"
                ]
                == 105
                and phase2i[
                    "ocm_run_count"
                ]
                == 25
            ),

        "phase2j":
            (
                phase2j[
                    "phase2j_status"
                ]
                == "passed"
                and phase2j[
                    "source_predictive_row_count"
                ]
                == 135
            ),

        "phase2k":
            (
                phase2k[
                    "phase2k_status"
                ]
                == "passed"
                and phase2k[
                    "behavioral_run_group_count"
                ]
                == 135
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
            "Required source phases are incomplete: "
            + ", ".join(
                failed
            )
        )

    return {
        "phase2d":
            phase2d,

        "phase2f":
            phase2f,

        "phase2g":
            phase2g,

        "phase2h":
            phase2h,

        "phase2i":
            phase2i,

        "phase2j":
            phase2j,

        "phase2k":
            phase2k,

        "validation_checks":
            checks,
    }


def load_predictive_summary():
    rows = load_csv(
        PHASE2I_DIR
        / "model_noise_summary.csv"
    )

    if (
        len(rows)
        != EXPECTED_PREDICTIVE_SUMMARY_ROWS
    ):
        raise AssertionError(
            "Unexpected predictive summary size: "
            f"{len(rows)}"
        )

    parsed = []

    for row in rows:
        parsed.append(
            {
                "model_id":
                    row[
                        "model_id"
                    ],

                "noise_fraction":
                    float(
                        row[
                            "noise_fraction"
                        ]
                    ),

                "seed_count":
                    int(
                        row[
                            "seed_count"
                        ]
                    ),

                "parameter_count":
                    int(
                        row[
                            "parameter_count"
                        ]
                    ),

                "noisy_mse_mean":
                    float(
                        row[
                            "noisy_mse_mean"
                        ]
                    ),

                "noisy_mse_std":
                    float(
                        row[
                            "noisy_mse_std"
                        ]
                    ),

                "clean_mse_mean":
                    float(
                        row[
                            "clean_mse_mean"
                        ]
                    ),

                "clean_mse_std":
                    float(
                        row[
                            "clean_mse_std"
                        ]
                    ),
            }
        )

    return parsed


def load_behavioral_summary():
    rows = load_csv(
        PHASE2K_DIR
        / "model_noise_behavioral_summary.csv"
    )

    if (
        len(rows)
        != EXPECTED_BEHAVIORAL_SUMMARY_ROWS
    ):
        raise AssertionError(
            "Unexpected behavioral summary size: "
            f"{len(rows)}"
        )

    numeric_fields = [
        "primitive_mapping_accuracy_mean",
        "overall_exact_transformation_rate_mean",
        "overall_partition_accuracy_mean",
        "overall_relation_accuracy_mean",
        "overall_relation_balanced_accuracy_mean",
        "test_mapping_accuracy_mean",
        "test_exact_transformation_rate_mean",
        "test_partition_accuracy_mean",
        "test_rank_accuracy_mean",
        "test_involved_relation_accuracy_mean",
        "test_involved_relation_balanced_accuracy_mean",
        "test_involved_relation_macro_f1_mean",
        "test_involved_relation_majority_baseline",
        "test_test_relation_accuracy_mean",
        "test_test_relation_balanced_accuracy_mean",
    ]

    parsed = []

    for row in rows:
        item = {
            "model_id":
                row[
                    "model_id"
                ],

            "noise_fraction":
                float(
                    row[
                        "noise_fraction"
                    ]
                ),

            "seed_or_fit_count":
                int(
                    row[
                        "seed_or_fit_count"
                    ]
                ),
        }

        for field in numeric_fields:
            item[
                field
            ] = float(
                row[
                    field
                ]
            )

        parsed.append(
            item
        )

    return parsed


def load_ocm_hard_comparison():
    rows = load_csv(
        PHASE2K_DIR
        / "ocm_hard_vs_behavioral.csv"
    )

    if (
        len(rows)
        != EXPECTED_HARD_COMPARISON_ROWS
    ):
        raise AssertionError(
            "Unexpected OCM hard comparison size."
        )

    numeric_fields = [
        "ocm_hard_test_exact_transformation_rate",
        "ocm_behavioral_test_exact_transformation_rate",
        "ocm_hard_test_relation_accuracy",
        "ocm_hard_test_relation_balanced_accuracy",
        "ocm_behavioral_test_relation_accuracy",
        "ocm_behavioral_test_relation_balanced_accuracy",
        "best_baseline_test_exact_transformation_rate",
        "best_baseline_test_relation_accuracy",
        "best_baseline_test_relation_balanced_accuracy",
    ]

    parsed = []

    for row in rows:
        item = {
            "noise_fraction":
                float(
                    row[
                        "noise_fraction"
                    ]
                ),

            "best_baseline_behavioral_model":
                row[
                    "best_baseline_behavioral_model"
                ],
        }

        for field in numeric_fields:
            item[
                field
            ] = float(
                row[
                    field
                ]
            )

        parsed.append(
            item
        )

    return parsed


def load_phase2j_audits():
    affine = load_json(
        PHASE2J_DIR
        / "affine_shortcut_audit.json"
    )

    metadata = load_json(
        PHASE2J_DIR
        / "metadata_correction.json"
    )

    if (
        affine[
            "affine_interpolation_shortcut_confirmed"
        ]
        is not True
    ):
        raise AssertionError(
            "Expected the affine shortcut audit "
            "to be confirmed."
        )

    required_metadata = {
        "clean_targets_used_for_training":
            False,

        "clean_targets_used_for_checkpoint_selection":
            False,

        "clean_validation_targets_accessed":
            True,

        "test_used_for_checkpoint_selection":
            False,
    }

    for field, expected in (
        required_metadata.items()
    ):
        if metadata[
            field
        ] is not expected:
            raise AssertionError(
                "Metadata correction changed for "
                f"{field}."
            )

    return (
        affine,
        metadata,
    )


def find_row(
    rows,
    model_id,
    noise,
):
    matches = [
        row
        for row in rows
        if (
            row[
                "model_id"
            ]
            == model_id
            and abs(
                row[
                    "noise_fraction"
                ]
                - noise
            ) < 1e-12
        )
    ]

    if len(
        matches
    ) != 1:
        raise AssertionError(
            f"Could not uniquely locate "
            f"{model_id} at noise {noise}."
        )

    return matches[
        0
    ]


def find_hard_row(
    rows,
    noise,
):
    matches = [
        row
        for row in rows
        if abs(
            row[
                "noise_fraction"
            ]
            - noise
        ) < 1e-12
    ]

    if len(
        matches
    ) != 1:
        raise AssertionError(
            "Could not uniquely locate OCM "
            f"hard comparison at noise {noise}."
        )

    return matches[
        0
    ]


def best_predictive_model(
    predictive_rows,
    noise,
    metric,
):
    candidates = [
        row
        for row in predictive_rows
        if abs(
            row[
                "noise_fraction"
            ]
            - noise
        ) < 1e-12
    ]

    return min(
        candidates,
        key=lambda row: (
            row[
                metric
            ],
            row[
                "model_id"
            ],
        ),
    )


def best_behavioral_model(
    behavioral_rows,
    noise,
):
    candidates = [
        row
        for row in behavioral_rows
        if abs(
            row[
                "noise_fraction"
            ]
            - noise
        ) < 1e-12
    ]

    return max(
        candidates,
        key=lambda row: (
            row[
                "test_involved_relation_balanced_accuracy_mean"
            ],
            row[
                "test_exact_transformation_rate_mean"
            ],
            row[
                "test_involved_relation_accuracy_mean"
            ],
        ),
    )


def build_noise_summary(
    predictive_rows,
    behavioral_rows,
    hard_rows,
):
    output = []

    for noise in NOISE_LEVELS:
        best_noisy = best_predictive_model(
            predictive_rows,
            noise,
            "noisy_mse_mean",
        )

        best_clean = best_predictive_model(
            predictive_rows,
            noise,
            "clean_mse_mean",
        )

        best_behavior = best_behavioral_model(
            behavioral_rows,
            noise,
        )

        ocm_predictive = find_row(
            predictive_rows,
            "OCM",
            noise,
        )

        b2_predictive = find_row(
            predictive_rows,
            "B2",
            noise,
        )

        b3_predictive = find_row(
            predictive_rows,
            "B3",
            noise,
        )

        ocm_behavior = find_row(
            behavioral_rows,
            "OCM",
            noise,
        )

        b2_behavior = find_row(
            behavioral_rows,
            "B2",
            noise,
        )

        hard = find_hard_row(
            hard_rows,
            noise,
        )

        output.append(
            {
                "noise_fraction":
                    noise,

                "best_noisy_predictive_model":
                    best_noisy[
                        "model_id"
                    ],

                "best_noisy_predictive_mse":
                    best_noisy[
                        "noisy_mse_mean"
                    ],

                "best_clean_predictive_model":
                    best_clean[
                        "model_id"
                    ],

                "best_clean_predictive_mse":
                    best_clean[
                        "clean_mse_mean"
                    ],

                "ocm_noisy_mse":
                    ocm_predictive[
                        "noisy_mse_mean"
                    ],

                "ocm_clean_mse":
                    ocm_predictive[
                        "clean_mse_mean"
                    ],

                "b2_noisy_mse":
                    b2_predictive[
                        "noisy_mse_mean"
                    ],

                "b3_noisy_mse":
                    b3_predictive[
                        "noisy_mse_mean"
                    ],

                "best_behavioral_structure_model":
                    best_behavior[
                        "model_id"
                    ],

                "best_behavioral_test_exact_rate":
                    best_behavior[
                        "test_exact_transformation_rate_mean"
                    ],

                "best_behavioral_test_relation_accuracy":
                    best_behavior[
                        "test_involved_relation_accuracy_mean"
                    ],

                "best_behavioral_test_relation_balanced_accuracy":
                    best_behavior[
                        "test_involved_relation_balanced_accuracy_mean"
                    ],

                "b2_behavioral_test_exact_rate":
                    b2_behavior[
                        "test_exact_transformation_rate_mean"
                    ],

                "b2_behavioral_test_relation_accuracy":
                    b2_behavior[
                        "test_involved_relation_accuracy_mean"
                    ],

                "b2_behavioral_test_relation_balanced_accuracy":
                    b2_behavior[
                        "test_involved_relation_balanced_accuracy_mean"
                    ],

                "ocm_behavioral_test_exact_rate":
                    ocm_behavior[
                        "test_exact_transformation_rate_mean"
                    ],

                "ocm_behavioral_test_relation_accuracy":
                    ocm_behavior[
                        "test_involved_relation_accuracy_mean"
                    ],

                "ocm_behavioral_test_relation_balanced_accuracy":
                    ocm_behavior[
                        "test_involved_relation_balanced_accuracy_mean"
                    ],

                "ocm_hard_test_exact_rate":
                    hard[
                        "ocm_hard_test_exact_transformation_rate"
                    ],

                "ocm_hard_test_relation_accuracy":
                    hard[
                        "ocm_hard_test_relation_accuracy"
                    ],

                "ocm_hard_test_relation_balanced_accuracy":
                    hard[
                        "ocm_hard_test_relation_balanced_accuracy"
                    ],

                "test_relation_majority_baseline":
                    ocm_behavior[
                        "test_involved_relation_majority_baseline"
                    ],
            }
        )

    return output


def build_claim_registry(
    noise_summary,
    affine_audit,
):
    by_noise = {
        row[
            "noise_fraction"
        ]:
            row
        for row in noise_summary
    }

    exact_through_025 = all(
        by_noise[
            noise
        ][
            "best_behavioral_test_exact_rate"
        ]
        == 1.0
        for noise in (
            0.0,
            0.1,
            0.25,
        )
    )

    b2_near_exact_at_05 = (
        by_noise[
            0.5
        ][
            "b2_behavioral_test_exact_rate"
        ]
        > 0.95
        and by_noise[
            0.5
        ][
            "b2_behavioral_test_relation_balanced_accuracy"
        ]
        > 0.99
    )

    severe_noise_failure = (
        by_noise[
            1.0
        ][
            "best_behavioral_test_relation_accuracy"
        ]
        <
        by_noise[
            1.0
        ][
            "test_relation_majority_baseline"
        ]
    )

    ocm_not_best_predictor = all(
        row[
            "best_noisy_predictive_model"
        ]
        != "OCM"
        for row in noise_summary
    )

    b2_beats_ocm_behavior_05 = (
        by_noise[
            0.5
        ][
            "b2_behavioral_test_relation_balanced_accuracy"
        ]
        >
        by_noise[
            0.5
        ][
            "ocm_behavioral_test_relation_balanced_accuracy"
        ]
    )

    hardening_helped_at_05 = (
        by_noise[
            0.5
        ][
            "ocm_hard_test_relation_balanced_accuracy"
        ]
        >
        by_noise[
            0.5
        ][
            "ocm_behavioral_test_relation_balanced_accuracy"
        ]
    )

    claims = [
        {
            "claim_id":
                "C01",

            "status":
                "supported",

            "claim": (
                "Held-out composite transformations and their "
                "informativeness relations are exactly recoverable "
                "in Tier A through noise 0.25."
            ),

            "supported":
                exact_through_025,

            "qualification": (
                "This is demonstrated for unseen compositions of "
                "known primitive operations, not unseen primitive "
                "operation identities."
            ),
        },

        {
            "claim_id":
                "C02",

            "status":
                "supported",

            "claim": (
                "Structural recovery at moderate noise is strongly "
                "model dependent."
            ),

            "supported":
                b2_near_exact_at_05,

            "evidence": {
                "noise_fraction":
                    0.5,

                "b2_exact_test_transformation_rate":
                    by_noise[
                        0.5
                    ][
                        "b2_behavioral_test_exact_rate"
                    ],

                "b2_test_relation_accuracy":
                    by_noise[
                        0.5
                    ][
                        "b2_behavioral_test_relation_accuracy"
                    ],

                "b2_test_relation_balanced_accuracy":
                    by_noise[
                        0.5
                    ][
                        "b2_behavioral_test_relation_balanced_accuracy"
                    ],
            },
        },

        {
            "claim_id":
                "C03",

            "status":
                "supported_with_scope",

            "claim": (
                "Reliable structural recovery is not achieved at "
                "noise 1.0 under the tested Tier A setting."
            ),

            "supported":
                severe_noise_failure,

            "qualification": (
                "This is an empirical failure under the tested models, "
                "training data, observation protocol, and noise level. "
                "It is not a universal impossibility theorem."
            ),
        },

        {
            "claim_id":
                "C04",

            "status":
                "rejected",

            "claim": (
                "OCM provides the best unseen-composition predictive "
                "accuracy."
            ),

            "supported":
                not ocm_not_best_predictor,

            "finding": (
                "Rejected. OCM is not the best noisy-target predictor "
                "at any tested noise level."
            ),
        },

        {
            "claim_id":
                "C05",

            "status":
                "rejected",

            "claim": (
                "Explicit finite stochastic channels are necessary "
                "for behavioral recovery of the transformation algebra."
            ),

            "supported":
                not b2_beats_ocm_behavior_05,

            "finding": (
                "Rejected. B2 recovers held-out transformation behavior "
                "and informativeness relations more robustly than OCM "
                "at noise 0.5."
            ),
        },

        {
            "claim_id":
                "C06",

            "status":
                "supported",

            "claim": (
                "Hard projection of OCM channels can expose partial "
                "informativeness structure that is hidden by repeated "
                "soft-channel composition."
            ),

            "supported":
                hardening_helped_at_05,

            "evidence": {
                "noise_fraction":
                    0.5,

                "ocm_hard_balanced_accuracy":
                    by_noise[
                        0.5
                    ][
                        "ocm_hard_test_relation_balanced_accuracy"
                    ],

                "ocm_behavioral_balanced_accuracy":
                    by_noise[
                        0.5
                    ][
                        "ocm_behavioral_test_relation_balanced_accuracy"
                    ],
            },

            "qualification": (
                "This is an internal discrete-projection advantage, "
                "not a predictive or behavioral-superiority result."
            ),
        },

        {
            "claim_id":
                "C07",

            "status":
                "rejected",

            "claim": (
                "Low prediction error is sufficient evidence that the "
                "underlying transformation and informativeness "
                "structure has been identified."
            ),

            "supported":
                False,

            "finding": (
                "Rejected. Prediction remains accurate at noise 1.0 "
                "while reliable structural recovery is not achieved."
            ),
        },

        {
            "claim_id":
                "C08",

            "status":
                "supported",

            "claim": (
                "Tier A contains an observation-space affine "
                "interpolation shortcut."
            ),

            "supported":
                affine_audit[
                    "affine_interpolation_shortcut_confirmed"
                ],

            "qualification": (
                "The shortcut results from eight prototype states "
                "embedded in a 12-dimensional observation space and "
                "does not establish that real material dynamics are "
                "affine."
            ),
        },

        {
            "claim_id":
                "C09",

            "status":
                "supported_with_qualification",

            "claim": (
                "OCM offers a directly inspectable finite-channel "
                "representation with competitive prediction."
            ),

            "supported":
                True,

            "qualification": (
                "No predictive or behavioral structural advantage over "
                "B2 or B3 is demonstrated. The benefit is representation "
                "and auditability."
            ),
        },

        {
            "claim_id":
                "C10",

            "status":
                "supported",

            "claim": (
                "Predictive accuracy, behavioral compositional "
                "recovery, and internal structural recovery are "
                "empirically distinct properties."
            ),

            "supported":
                True,

            "qualification": (
                "The distinction is established within the controlled "
                "Tier A benchmark."
            ),
        },
    ]

    for claim in claims:
        if (
            claim[
                "status"
            ]
            in {
                "supported",
                "supported_with_scope",
                "supported_with_qualification",
            }
            and claim[
                "supported"
            ]
            is not True
        ):
            raise AssertionError(
                "A declared supported claim failed: "
                f"{claim['claim_id']}"
            )

        if (
            claim[
                "status"
            ]
            == "rejected"
            and claim[
                "supported"
            ]
            is not False
        ):
            raise AssertionError(
                "A declared rejected claim was supported: "
                f"{claim['claim_id']}"
            )

    return claims


def build_rq_status(
    noise_summary,
):
    by_noise = {
        row[
            "noise_fraction"
        ]:
            row
        for row in noise_summary
    }

    return {
        "RQ1": {
            "question": (
                "Can relative history-informativeness be recovered "
                "from noisy continuous observations?"
            ),

            "status":
                "supported_with_qualifications",

            "finding": (
                "Held-out informativeness relations are recovered "
                "exactly through noise 0.25. B2 remains nearly exact "
                "at noise 0.5. Reliable recovery is not achieved at "
                "noise 1.0."
            ),

            "method_specificity": (
                "The result is not unique to OCM."
            ),
        },

        "RQ2": {
            "question": (
                "Can preserving and collapsing transformation "
                "structure and compositions be recovered?"
            ),

            "status":
                "supported_behaviorally",

            "finding": (
                "Multiple model families recover unseen compositions "
                "exactly at low noise. B2 provides the strongest "
                "behavioral recovery at moderate noise."
            ),

            "qualification": (
                "Behavioral recovery does not imply that a model "
                "internally represents the same finite algebra."
            ),
        },

        "RQ3": {
            "question": (
                "Does explicit operation structure improve unseen "
                "process-sequence prediction?"
            ),

            "strong_version_status":
                "rejected",

            "strong_version_finding": (
                "B2 and B3 produce lower predictive error than OCM."
            ),

            "revised_version_status":
                "supported_with_qualification",

            "revised_statement": (
                "OCM provides competitive unseen-composition prediction "
                "and a directly inspectable finite-channel representation, "
                "but no predictive or behavioral-recovery superiority."
            ),
        },

        "RQ4": {
            "question": (
                "When is the structure identifiable under observation "
                "noise and limited process coverage?"
            ),

            "status":
                "strongly_supported_in_tier_a",

            "finding": {
                "low_noise_regime": (
                    "Exact recovery through noise 0.25."
                ),

                "moderate_noise_regime": (
                    "Strongly model-dependent recovery at noise 0.5."
                ),

                "severe_noise_regime": (
                    "No reliable structural recovery under the tested "
                    "setting at noise 1.0."
                ),
            },

            "key_conclusion": (
                "Predictive performance alone is not a sufficient "
                "identifiability diagnostic."
            ),

            "supporting_values": {
                "b2_noise_0p5_exact_test_rate":
                    by_noise[
                        0.5
                    ][
                        "b2_behavioral_test_exact_rate"
                    ],

                "b2_noise_0p5_relation_balanced_accuracy":
                    by_noise[
                        0.5
                    ][
                        "b2_behavioral_test_relation_balanced_accuracy"
                    ],

                "noise_1p0_best_relation_accuracy":
                    by_noise[
                        1.0
                    ][
                        "best_behavioral_test_relation_accuracy"
                    ],

                "noise_1p0_majority_baseline":
                    by_noise[
                        1.0
                    ][
                        "test_relation_majority_baseline"
                    ],
            },
        },
    }


def build_limitations():
    return {
        "benchmark_scope": [
            (
                "Tier A contains eight discrete latent states and six "
                "known primitive operation labels."
            ),
            (
                "OOD evaluation holds out composite transformations, "
                "not primitive operation identities."
            ),
            (
                "The experiment is controlled and synthetic; causal "
                "language should remain limited to the interventionally "
                "defined simulator."
            ),
        ],

        "observation_scope": [
            (
                "The continuous observation space is generated from "
                "fixed state prototypes with additive noise."
            ),
            (
                "Tier A admits an exact affine interpolation shortcut "
                "on the eight prototypes."
            ),
            (
                "The present results do not establish recovery from "
                "real microstructure images, spectra, or process sensor "
                "streams."
            ),
        ],

        "identifiability_scope": [
            (
                "Failure at noise 1.0 is empirical under the tested "
                "models and protocol, not a universal impossibility "
                "result."
            ),
            (
                "Behavioral recovery does not prove that a model's "
                "internal representation is isomorphic to the true "
                "transformation semigroup."
            ),
            (
                "OCM hard-channel alignment uses the known Tier A latent "
                "cardinality and an evaluation-only state permutation."
            ),
        ],

        "model_comparison_scope": [
            (
                "B2 and B3 outperform OCM predictively, so OCM should "
                "not be presented as a state-of-the-art predictive model."
            ),
            (
                "B2 also outperforms OCM behaviorally at moderate noise."
            ),
            (
                "OCM's demonstrated advantage is explicit channel "
                "inspection and discrete projection, not superior MSE "
                "or behavioral recovery."
            ),
        ],

        "protocol_note": [
            (
                "Clean validation targets were accessed to compute "
                "diagnostic metrics during final-run validation."
            ),
            (
                "They were not used in training, early stopping, "
                "hyperparameter selection, or checkpoint selection."
            ),
            (
                "Test data remained excluded from checkpoint selection."
            ),
        ],
    }


def build_do_not_claim():
    return [
        (
            "Do not claim that OCM outperforms all generic sequence "
            "or operation-conditioned models."
        ),
        (
            "Do not claim that explicit stochastic channels are "
            "necessary for compositional transformation recovery."
        ),
        (
            "Do not claim that strong prediction proves recovery of "
            "the true process-memory structure."
        ),
        (
            "Do not claim that noise 1.0 creates a universal "
            "information-theoretic impossibility."
        ),
        (
            "Do not claim generalization to unseen primitive "
            "operations."
        ),
        (
            "Do not claim that the Tier A affine result represents "
            "real material-process dynamics."
        ),
        (
            "Do not claim that behavioral equivalence proves internal "
            "algebraic equivalence."
        ),
        (
            "Do not claim that every irreversible operation is a reset."
        ),
        (
            "Do not claim that mutual-information ranking is equivalent "
            "to the Blackwell informativeness order."
        ),
        (
            "Do not claim that Phase 2 establishes results on real "
            "materials datasets."
        ),
    ]


def build_paper_safe_summary(
    noise_summary,
):
    by_noise = {
        row[
            "noise_fraction"
        ]:
            row
        for row in noise_summary
    }

    return f"""PHASE 2 PAPER-SAFE SUMMARY

We evaluated an Operation Channel Model (OCM) and six predictive
comparators on a controlled process-history benchmark containing
104 composite transformations and nine held-out OOD transformations.

PREDICTION

OCM did not provide the best predictive accuracy. The operation-conditioned
MLP and GRU produced lower OOD prediction error across the principal noisy
conditions. OCM should therefore not be positioned as a superior predictive
architecture.

BEHAVIORAL STRUCTURAL RECOVERY

Several model families recovered all held-out transformations and
informativeness relations exactly through noise 0.25.

At noise 0.5, the operation-conditioned MLP recovered
{by_noise[0.5]['b2_behavioral_test_exact_rate']:.4f} of held-out
transformations exactly and obtained relation balanced accuracy
{by_noise[0.5]['b2_behavioral_test_relation_balanced_accuracy']:.4f}.

At the same noise level, OCM behavioral rollout recovered
{by_noise[0.5]['ocm_behavioral_test_exact_rate']:.4f} of held-out
transformations exactly and obtained relation balanced accuracy
{by_noise[0.5]['ocm_behavioral_test_relation_balanced_accuracy']:.4f}.

OCM HARD CHANNEL PROJECTION

Hard projection of OCM's learned channels improved held-out relation
balanced accuracy at noise 0.5 to
{by_noise[0.5]['ocm_hard_test_relation_balanced_accuracy']:.4f}.

This demonstrates an internal discrete-projection benefit, but not
predictive or behavioral superiority over the strongest baseline.

SEVERE-NOISE REGIME

At noise 1.0, the best behavioral relation accuracy was
{by_noise[1.0]['best_behavioral_test_relation_accuracy']:.4f}, compared
with a majority baseline of
{by_noise[1.0]['test_relation_majority_baseline']:.4f}.

Reliable structural recovery was therefore not achieved under the tested
single-observation Tier A setting.

AFFINE SHORTCUT

The eight continuous prototypes are affinely interpolable in the
12-dimensional observation space. Consequently, a per-operation affine
model can exactly fit arbitrary primitive mappings on the prototype set.
Tier A should therefore be treated primarily as a controlled structural
identifiability benchmark rather than a difficult predictive benchmark.

MAIN CONCLUSION

Predictive accuracy, behavioral compositional recovery, and internal
structural recovery are empirically distinct properties. Explicit finite
channels provide an auditable representation, but the current experiments
do not show that they are necessary for accurate prediction or behavioral
recovery.
"""


def build_phase2_overview(
    sources,
    claim_registry,
    rq_status,
    affine_audit,
    metadata_correction,
):
    supported_claims = [
        claim[
            "claim_id"
        ]
        for claim in claim_registry
        if claim[
            "status"
        ].startswith(
            "supported"
        )
    ]

    rejected_claims = [
        claim[
            "claim_id"
        ]
        for claim in claim_registry
        if claim[
            "status"
        ]
        == "rejected"
    ]

    return {
        "phase":
            "Phase 2 complete synthesis",

        "phase_status":
            "completed_with_positive_negative_and_qualified_results",

        "completed_subphases": [
            "2A protocol freeze",
            "2B OCM smoke test",
            "2C OCM tuning",
            "2D final OCM prediction",
            "2E structural recovery",
            "2F OOD structural audit",
            "2G baseline protocol freeze",
            "2H baseline tuning",
            "2I final predictive comparison",
            "2J predictive audit",
            "2K behavioral structural comparison",
            "2L claim freeze",
        ],

        "execution_totals": {
            "ocm_final_runs":
                sources[
                    "phase2d"
                ][
                    "total_run_count"
                ],

            "baseline_tuning_configurations":
                sources[
                    "phase2h"
                ][
                    "completed_configuration_count"
                ],

            "baseline_final_fits":
                sources[
                    "phase2i"
                ][
                    "baseline_final_fit_count"
                ],

            "behavioral_evaluation_groups":
                sources[
                    "phase2k"
                ][
                    "behavioral_run_group_count"
                ],
        },

        "primary_positive_finding": (
            "Held-out transformation and informativeness structure can "
            "be recovered exactly through moderate noise, but robustness "
            "depends strongly on model class."
        ),

        "primary_negative_finding": (
            "Explicit finite OCM structure does not improve predictive "
            "accuracy and is not necessary for behavioral compositional "
            "recovery in Tier A."
        ),

        "primary_identifiability_finding": (
            "Prediction, behavioral recovery, and internal structural "
            "recovery are empirically distinct."
        ),

        "affine_shortcut_confirmed":
            affine_audit[
                "affine_interpolation_shortcut_confirmed"
            ],

        "metadata_correction": {
            "clean_targets_used_for_training":
                metadata_correction[
                    "clean_targets_used_for_training"
                ],

            "clean_targets_used_for_selection":
                metadata_correction[
                    "clean_targets_used_for_checkpoint_selection"
                ],

            "clean_validation_targets_accessed":
                metadata_correction[
                    "clean_validation_targets_accessed"
                ],

            "test_used_for_selection":
                metadata_correction[
                    "test_used_for_checkpoint_selection"
                ],
        },

        "supported_claim_ids":
            supported_claims,

        "rejected_claim_ids":
            rejected_claims,

        "research_question_status":
            rq_status,

        "phase2_results_frozen":
            True,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = validate_source_phases()

    predictive_rows = (
        load_predictive_summary()
    )

    behavioral_rows = (
        load_behavioral_summary()
    )

    hard_rows = (
        load_ocm_hard_comparison()
    )

    (
        affine_audit,
        metadata_correction,
    ) = load_phase2j_audits()

    noise_summary = build_noise_summary(
        predictive_rows=(
            predictive_rows
        ),
        behavioral_rows=(
            behavioral_rows
        ),
        hard_rows=(
            hard_rows
        ),
    )

    write_csv(
        OUTPUT_DIR
        / "phase2_key_results_by_noise.csv",
        noise_summary,
    )

    claim_registry = build_claim_registry(
        noise_summary=(
            noise_summary
        ),
        affine_audit=(
            affine_audit
        ),
    )

    write_json(
        OUTPUT_DIR
        / "phase2_claim_registry.json",
        {
            "claim_count":
                len(
                    claim_registry
                ),

            "claims":
                claim_registry,
        },
    )

    rq_status = build_rq_status(
        noise_summary
    )

    write_json(
        OUTPUT_DIR
        / "phase2_research_question_status.json",
        rq_status,
    )

    limitations = build_limitations()

    write_json(
        OUTPUT_DIR
        / "phase2_limitations.json",
        limitations,
    )

    do_not_claim = build_do_not_claim()

    write_json(
        OUTPUT_DIR
        / "phase2_do_not_claim.json",
        {
            "do_not_claim":
                do_not_claim
        },
    )

    paper_summary = build_paper_safe_summary(
        noise_summary
    )

    write_text(
        OUTPUT_DIR
        / "phase2_paper_safe_summary.txt",
        paper_summary,
    )

    phase2_overview = build_phase2_overview(
        sources=sources,
        claim_registry=(
            claim_registry
        ),
        rq_status=rq_status,
        affine_audit=(
            affine_audit
        ),
        metadata_correction=(
            metadata_correction
        ),
    )

    write_json(
        OUTPUT_DIR
        / "phase2_final_overview.json",
        phase2_overview,
    )

    summary = {
        "phase":
            "2L final Phase 2 synthesis and claim freeze",

        "source_phase_validation":
            sources[
                "validation_checks"
            ],

        "predictive_summary_row_count":
            len(
                predictive_rows
            ),

        "behavioral_summary_row_count":
            len(
                behavioral_rows
            ),

        "noise_summary_row_count":
            len(
                noise_summary
            ),

        "claim_count":
            len(
                claim_registry
            ),

        "supported_or_qualified_claim_count":
            sum(
                claim[
                    "status"
                ].startswith(
                    "supported"
                )
                for claim in claim_registry
            ),

        "rejected_claim_count":
            sum(
                claim[
                    "status"
                ]
                == "rejected"
                for claim in claim_registry
            ),

        "rq1_status":
            rq_status[
                "RQ1"
            ][
                "status"
            ],

        "rq2_status":
            rq_status[
                "RQ2"
            ][
                "status"
            ],

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
            rq_status[
                "RQ4"
            ][
                "status"
            ],

        "affine_shortcut_confirmed":
            affine_audit[
                "affine_interpolation_shortcut_confirmed"
            ],

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

        "phase2l_status":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase2l_summary.json",
        summary,
    )

    print(
        "Phase 2L synthesis and claim freeze completed."
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
