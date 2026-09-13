from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


PROTOCOL_VERSION = "tier_c_v4"

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3g2e_tier_c_v4_final_freeze"
)

HYPOTHESIS_REGISTRY = Path(
    "outputs/"
    "phase4ar3_tier_c_v4_protocol/"
    "hypothesis_registry_v4.json"
)

TEST_POLICY = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_test_opening_policy.json"
)

METRIC_REGISTRY = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_metric_registry.csv"
)

G2B_RESULT = Path(
    "outputs/"
    "phase4gr3g2b_tier_c_v4_primary_confirmatory/"
    "primary_confirmatory_result.json"
)

G2C_SUMMARY = Path(
    "outputs/"
    "phase4gr3g2c_tier_c_v4_full_predictive/"
    "phase4gr3g2c_full_predictive_summary.json"
)

G2D_SUMMARY = Path(
    "outputs/"
    "phase4gr3g2dr1_tier_c_v4_exact_phase3g_structural/"
    "phase4gr3g2d_structural_summary.json"
)

G2D_NOISE = Path(
    "outputs/"
    "phase4gr3g2dr1_tier_c_v4_exact_phase3g_structural/"
    "ocm_noise_structural_summary.csv"
)

G2D_STATE_PATH = Path(
    "outputs/"
    "phase4gr3g2dr1_tier_c_v4_exact_phase3g_structural/"
    "state_path_condition_summary.csv"
)

STATE_DECODER_SUMMARY = Path(
    "outputs/"
    "phase4cr3_tier_c_v4_audit/"
    "state_decoder_summary.json"
)

PREDECODER_SUMMARY = Path(
    "outputs/"
    "phase4cr3_tier_c_v4_audit/"
    "phase4cr3_predecoder_summary.json"
)

PHASE4D_ROOT = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol"
)


EXPECTED_HASHES = {
    str(G2B_RESULT):
        "1e64424c6771a748271fdec3dd921b527b399a99aea2b8a15d9c865ffd343862",

    str(G2C_SUMMARY):
        "337c39a55825ec12d5ff6f8afd4dc49d6fc20662ea79bd84eae2098eb733d823",

    str(G2D_SUMMARY):
        "a8929c43fc430ef4dcf99c37c720bd6ea1dec1b0854efa925803fb678f3dc0d2",

    str(TEST_POLICY):
        "5a3a600b442093ddc50d0b1db5fdc15832cd654f9fccd3a3660e2fc341c597e9",

    str(METRIC_REGISTRY):
        "2cb5883ee2b34a07189256ec6399c72328f43d517eef0b144f4ed5044f1c5296",
}


# Frozen primary result.
PRIMARY_OCM_MEAN = (
    1.0393291007520424
)

PRIMARY_B4_MEAN = (
    1.0001117385509941
)

PRIMARY_DIFFERENCE = (
    0.03921736220104826
)

PRIMARY_CI_LOWER = (
    0.03416843388979841
)

PRIMARY_CI_UPPER = (
    0.04445381079101935
)

PRIMARY_OCM_BETTER_SEQUENCE_COUNT = 12
PRIMARY_SEQUENCE_COUNT = 144

STATE_CHANCE_ACCURACY = 1.0 / 8.0


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(
            message
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


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


def load_csv(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def write_csv(
    path: Path,
    rows,
):
    require(
        bool(rows),
        f"No rows for {path}.",
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
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


def recursive_values_for_key(
    value,
    target_key,
):
    results = []

    if isinstance(
        value,
        dict,
    ):
        for key, item in (
            value.items()
        ):
            if key == target_key:
                results.append(
                    item
                )

            results.extend(
                recursive_values_for_key(
                    item,
                    target_key,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for item in value:
            results.extend(
                recursive_values_for_key(
                    item,
                    target_key,
                )
            )

    return results


def verify_source_hashes():
    print(
        "[1/8] Verifying frozen result hashes"
    )

    observed = {}

    for filename, expected in (
        EXPECTED_HASHES.items()
    ):
        path = Path(
            filename
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        value = sha256_file(
            path
        )

        observed[
            filename
        ] = value

        require(
            value == expected,
            (
                "Frozen source hash changed:\n"
                f"{filename}\n"
                f"expected={expected}\n"
                f"observed={value}"
            ),
        )

    return observed


def verify_test_policy():
    print(
        "[2/8] Verifying frozen confirmatory specification"
    )

    policy = load_json(
        TEST_POLICY
    )

    primary = policy[
        "primary_confirmatory_condition"
    ]

    require(
        primary[
            "cell"
        ] == "test_joint",
        "Primary test cell changed.",
    )

    require(
        float(
            primary[
                "noise_fraction"
            ]
        ) == 0.25,
        "Primary noise changed.",
    )

    require(
        primary[
            "metric"
        ]
        == "noisy_target_rollout_mse",
        "Primary metric changed.",
    )

    require(
        primary[
            "primary_model"
        ] == "OCM",
        "Primary model changed.",
    )

    require(
        int(
            primary[
                "bootstrap_replicates"
            ]
        ) == 10000,
        "Bootstrap replicate count changed.",
    )

    require(
        int(
            primary[
                "bootstrap_seed"
            ]
        ) == 73011,
        "Bootstrap seed changed.",
    )

    secondary = policy[
        "secondary_conditions"
    ]

    require(
        secondary[
            "confirmatory_status"
        ] is False,
        (
            "Secondary conditions unexpectedly "
            "became confirmatory."
        ),
    )

    return policy


def verify_hypothesis_registry():
    print(
        "[3/8] Verifying frozen H4R3 hypothesis registry"
    )

    registry = load_json(
        HYPOTHESIS_REGISTRY
    )

    require(
        registry[
            "protocol_version"
        ]
        == PROTOCOL_VERSION,
        "Hypothesis protocol changed.",
    )

    require(
        registry[
            "registry_status"
        ] == "frozen",
        "Hypothesis registry is not frozen.",
    )

    hypotheses = (
        registry[
            "hypotheses"
        ]
    )

    ids = [
        row[
            "hypothesis_id"
        ]
        for row in hypotheses
    ]

    require(
        ids
        == [
            "H4R3-1",
            "H4R3-2",
            "H4R3-3",
            "H4R3-4",
            "H4R3-5",
        ],
        (
            "Frozen hypothesis IDs "
            "changed."
        ),
    )

    return registry


def verify_development_acceptance():
    print(
        "[4/8] Binding frozen development evidence"
    )

    predecoder = load_json(
        PREDECODER_SUMMARY
    )

    decoder = load_json(
        STATE_DECODER_SUMMARY
    )

    require(
        predecoder[
            "predecoder_acceptance_gate_passed"
        ] is True,
        (
            "Tier-C-v4 predecoder "
            "acceptance did not pass."
        ),
    )

    require(
        predecoder[
            "independent_morphology_gate_passed"
        ] is True,
        (
            "Independent morphology "
            "gate did not pass."
        ),
    )

    require(
        decoder[
            "state_separability_gate_passed"
        ] is True,
        (
            "State-separability "
            "gate did not pass."
        ),
    )

    require(
        decoder[
            "mean_state_accuracy_gate_passed"
        ] is True,
        (
            "Mean state accuracy "
            "gate did not pass."
        ),
    )

    require(
        decoder[
            "worst_state_accuracy_gate_passed"
        ] is True,
        (
            "Worst state accuracy "
            "gate did not pass."
        ),
    )

    require(
        decoder[
            "mean_coarsening_accuracy_gate_passed"
        ] is True,
        (
            "Mean coarsening accuracy "
            "gate did not pass."
        ),
    )

    require(
        decoder[
            "worst_coarsening_accuracy_gate_passed"
        ] is True,
        (
            "Worst coarsening accuracy "
            "gate did not pass."
        ),
    )

    # Locate the already-frozen Phase-4D statement that
    # development acceptance passed before predictive work.
    development_acceptance_found = False
    development_acceptance_source = None

    for path in sorted(
        PHASE4D_ROOT.glob(
            "*.json"
        )
    ):
        try:
            data = load_json(
                path
            )
        except Exception:
            continue

        values = (
            recursive_values_for_key(
                data,
                "source_development_acceptance_passed",
            )
        )

        if True in values:
            development_acceptance_found = True
            development_acceptance_source = (
                str(
                    path
                )
            )
            break

    require(
        development_acceptance_found,
        (
            "Could not verify frozen "
            "development-acceptance pass "
            "from the Phase-4D protocol."
        ),
    )

    return {
        "predecoder":
            predecoder,

        "decoder":
            decoder,

        "development_acceptance_source":
            development_acceptance_source,
    }


def freeze_registered_hypotheses(
    registry,
    development,
):
    """
    H4R3-1 through H4R3-5 are the only objects in the
    frozen hypothesis registry.

    They were development/audit hypotheses and Tier-C-v4
    predictive training was authorized only after the
    terminal development acceptance passed.

    These are therefore kept separate from the later
    predictive confirmatory test and structural diagnostics.
    """

    predecoder = (
        development[
            "predecoder"
        ]
    )

    decoder = (
        development[
            "decoder"
        ]
    )

    evidence = {
        "H4R3-1": (
            "Independent morphology gate passed. "
            f"Mean characteristic-length ratio="
            f"{predecoder['independent_mean_length_ratio']}; "
            f"minimum ratio="
            f"{predecoder['independent_minimum_length_ratio']}; "
            f"mean interface-density ratio="
            f"{predecoder['independent_mean_interface_ratio']}."
        ),

        "H4R3-2": (
            "Tier-C-v4 terminal development acceptance "
            "passed before predictive protocol execution; "
            "the final coarsening protocol therefore cleared "
            "the frozen carrier-scale-dependence acceptance "
            "workflow."
        ),

        "H4R3-3": (
            "State-separability gate passed. "
            f"Mean validation state accuracy="
            f"{decoder['mean_validation_state_accuracy']}; "
            f"worst-seed validation state accuracy="
            f"{decoder['worst_seed_validation_state_accuracy']}; "
            f"mean validation coarsening accuracy="
            f"{decoder['mean_validation_coarsening_accuracy']}."
        ),

        "H4R3-4": (
            "Tier-C-v4 terminal development acceptance "
            "passed before predictive training, including "
            "the frozen shortcut-diagnostic stage."
        ),

        "H4R3-5": (
            "Tier-C-v4 terminal development acceptance "
            "passed before predictive training, including "
            "the frozen PCA-128 affine-shortcut diagnostic."
        ),
    }

    rows = []

    for hypothesis in (
        registry[
            "hypotheses"
        ]
    ):
        hypothesis_id = (
            hypothesis[
                "hypothesis_id"
            ]
        )

        rows.append(
            {
                "hypothesis_id":
                    hypothesis_id,

                "registered_before_test":
                    True,

                "statement":
                    hypothesis[
                        "statement"
                    ],

                "verdict":
                    "SUPPORTED",

                "evidence":
                    evidence[
                        hypothesis_id
                    ],

                "classification":
                    (
                        "frozen development/audit "
                        "hypothesis"
                    ),
            }
        )

    return rows


def verify_primary_result():
    print(
        "[5/8] Freezing primary confirmatory verdict"
    )

    result = load_json(
        G2B_RESULT
    )

    # These exact values were already frozen in G2B.
    # G2E does not recompute the test.
    expected_values = {
        "ocm_mean":
            PRIMARY_OCM_MEAN,

        "b4_mean":
            PRIMARY_B4_MEAN,

        "ocm_minus_b4":
            PRIMARY_DIFFERENCE,

        "bootstrap_ci_lower":
            PRIMARY_CI_LOWER,

        "bootstrap_ci_upper":
            PRIMARY_CI_UPPER,

        "ocm_better_sequence_count":
            PRIMARY_OCM_BETTER_SEQUENCE_COUNT,

        "sequence_count":
            PRIMARY_SEQUENCE_COUNT,
    }

    # Confirm that the frozen G2B JSON contains the key
    # numerical primary values somewhere in its structure.
    def collect_numbers(value):
        output = []

        if isinstance(
            value,
            bool,
        ):
            return output

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            output.append(
                float(
                    value
                )
            )

        elif isinstance(
            value,
            dict,
        ):
            for item in (
                value.values()
            ):
                output.extend(
                    collect_numbers(
                        item
                    )
                )

        elif isinstance(
            value,
            list,
        ):
            for item in value:
                output.extend(
                    collect_numbers(
                        item
                    )
                )

        return output

    numbers = collect_numbers(
        result
    )

    for value in (
        PRIMARY_OCM_MEAN,
        PRIMARY_B4_MEAN,
        PRIMARY_DIFFERENCE,
        PRIMARY_CI_LOWER,
        PRIMARY_CI_UPPER,
    ):
        require(
            any(
                abs(
                    candidate
                    - value
                )
                < 1e-12
                for candidate in numbers
            ),
            (
                "Expected frozen G2B "
                f"value not found: {value}"
            ),
        )

    superiority_supported = (
        PRIMARY_DIFFERENCE < 0.0
        and PRIMARY_CI_UPPER < 0.0
    )

    require(
        superiority_supported
        is False,
        (
            "Primary conclusion unexpectedly "
            "changed."
        ),
    )

    return {
        **expected_values,

        "relative_difference":
            float(
                PRIMARY_DIFFERENCE
                / PRIMARY_B4_MEAN
            ),

        "ocm_better_sequence_fraction":
            float(
                PRIMARY_OCM_BETTER_SEQUENCE_COUNT
                / PRIMARY_SEQUENCE_COUNT
            ),

        "predictive_superiority_supported":
            False,

        "verdict":
            "NOT_SUPPORTED",

        "interpretation":
            (
                "The frozen primary comparison does not "
                "support OCM predictive superiority. "
                "The entire 95% paired-bootstrap interval "
                "for OCM minus B4 MSE is above zero."
            ),
    }


def freeze_structural_diagnostics():
    print(
        "[6/8] Freezing secondary structural diagnostics"
    )

    summary = load_json(
        G2D_SUMMARY
    )

    require(
        summary[
            "phase4gr3g2d_status"
        ]
        == "exact_phase3g_structural_results_frozen",
        (
            "Corrected G2D-R1 structural "
            "result is not frozen."
        ),
    )

    noise_rows = load_csv(
        G2D_NOISE
    )

    state_rows = load_csv(
        G2D_STATE_PATH
    )

    require(
        len(
            noise_rows
        ) == 5,
        "Expected five structural noise rows.",
    )

    require(
        len(
            state_rows
        ) == 20,
        (
            "Expected 20 state-path "
            "cell/noise rows."
        ),
    )

    exact = [
        float(
            row[
                "exact_transformation_rate_mean"
            ]
        )
        for row in noise_rows
    ]

    test_exact = [
        float(
            row[
                "test_exact_transformation_rate_mean"
            ]
        )
        for row in noise_rows
    ]

    information = [
        float(
            row[
                "information_class_accuracy_mean"
            ]
        )
        for row in noise_rows
    ]

    blackwell = [
        float(
            row[
                "blackwell_relation_balanced_accuracy_mean"
            ]
        )
        for row in noise_rows
    ]

    alignment = [
        float(
            row[
                "alignment_mean_probability_mean"
            ]
        )
        for row in noise_rows
    ]

    state_path = [
        float(
            row[
                "state_path_accuracy_mean"
            ]
        )
        for row in state_rows
    ]

    require(
        all(
            abs(
                value
                - (
                    1.0
                    / 104.0
                )
            )
            < 1e-12
            for value in exact
        ),
        (
            "Exact transformation result "
            "changed."
        ),
    )

    require(
        all(
            abs(
                value
            )
            < 1e-12
            for value in test_exact
        ),
        (
            "Held-out exact transformation "
            "result changed."
        ),
    )

    return {
        "exact_transformation_rate":
            {
                "noise_level_means":
                    exact,

                "grand_mean":
                    mean(
                        exact
                    ),

                "verdict":
                    "NOT_SUPPORTED",
            },

        "heldout_exact_transformation_rate":
            {
                "noise_level_means":
                    test_exact,

                "grand_mean":
                    mean(
                        test_exact
                    ),

                "verdict":
                    "NOT_SUPPORTED",
            },

        "information_class_accuracy":
            {
                "noise_level_means":
                    information,

                "grand_mean":
                    mean(
                        information
                    ),

                "verdict":
                    "NOT_SUPPORTED",
            },

        "blackwell_relation_balanced_accuracy":
            {
                "noise_level_means":
                    blackwell,

                "grand_mean":
                    mean(
                        blackwell
                    ),

                "minimum":
                    min(
                        blackwell
                    ),

                "maximum":
                    max(
                        blackwell
                    ),

                "verdict":
                    "NOT_SUPPORTED",
            },

        "training_carrier_state_alignment_accuracy":
            {
                "noise_level_means":
                    alignment,

                "grand_mean":
                    mean(
                        alignment
                    ),

                "chance_reference":
                    STATE_CHANCE_ACCURACY,

                "verdict":
                    "WEAK_NEAR_CHANCE",
            },

        "sealed_state_path_accuracy":
            {
                "condition_count":
                    len(
                        state_path
                    ),

                "grand_mean":
                    mean(
                        state_path
                    ),

                "minimum":
                    min(
                        state_path
                    ),

                "maximum":
                    max(
                        state_path
                    ),

                "chance_reference":
                    STATE_CHANCE_ACCURACY,

                "verdict":
                    "NOT_SUPPORTED",
            },
    }


def make_empirical_findings(
    development,
    primary,
    structural,
):
    decoder = (
        development[
            "decoder"
        ]
    )

    return [
        {
            "finding_id":
                "F1",

            "status":
                "SUPPORTED_EMPIRICAL_FINDING",

            "confirmatory":
                False,

            "finding":
                (
                    "The physical eight-state observation "
                    "manifold is strongly identifiable."
                ),

            "evidence":
                (
                    "Mean clean validation state accuracy="
                    f"{decoder['mean_validation_state_accuracy']:.6f}; "
                    "worst seed="
                    f"{decoder['worst_seed_validation_state_accuracy']:.6f}."
                ),
        },

        {
            "finding_id":
                "F2",

            "status":
                "SUPPORTED_CONFIRMATORY_FINDING",

            "confirmatory":
                True,

            "finding":
                (
                    "OCM does not provide predictive "
                    "superiority over the frozen strongest "
                    "validation-selected baseline B4 under "
                    "the primary joint-OOD condition."
                ),

            "evidence":
                (
                    f"OCM-B4 MSE="
                    f"{primary['ocm_minus_b4']:.6f}; "
                    "95% paired-bootstrap CI=["
                    f"{primary['bootstrap_ci_lower']:.6f}, "
                    f"{primary['bootstrap_ci_upper']:.6f}]."
                ),
        },

        {
            "finding_id":
                "F3",

            "status":
                "SUPPORTED_EMPIRICAL_FINDING",

            "confirmatory":
                False,

            "finding":
                (
                    "Strong physical-state identifiability "
                    "does not guarantee emergence of the "
                    "corresponding symbolic state representation "
                    "under predictive OCM training."
                ),

            "evidence":
                (
                    "Physical-state decoder validation accuracy "
                    f"is {decoder['mean_validation_state_accuracy']:.6f}, "
                    "whereas mean frozen OCM state alignment is "
                    f"{structural['training_carrier_state_alignment_accuracy']['grand_mean']:.6f} "
                    f"against an eight-state chance reference "
                    f"of {STATE_CHANCE_ACCURACY:.3f}."
                ),
        },

        {
            "finding_id":
                "F4",

            "status":
                "SUPPORTED_EMPIRICAL_FINDING",

            "confirmatory":
                False,

            "finding":
                (
                    "Predictive training does not automatically "
                    "recover the underlying transformation algebra."
                ),

            "evidence":
                (
                    "Exact 104-element transformation recovery="
                    f"{structural['exact_transformation_rate']['grand_mean']:.6f}; "
                    "held-out transformation exact recovery=0."
                ),
        },

        {
            "finding_id":
                "F5",

            "status":
                "SUPPORTED_EMPIRICAL_FINDING",

            "confirmatory":
                False,

            "finding":
                (
                    "The learned OCM representation does not "
                    "recover reliable symbolic state trajectories."
                ),

            "evidence":
                (
                    "Grand mean state-path accuracy="
                    f"{structural['sealed_state_path_accuracy']['grand_mean']:.6f}; "
                    f"range="
                    f"[{structural['sealed_state_path_accuracy']['minimum']:.6f}, "
                    f"{structural['sealed_state_path_accuracy']['maximum']:.6f}], "
                    "with chance reference=0.125."
                ),
        },

        {
            "finding_id":
                "F6",

            "status":
                "SUPPORTED_EMPIRICAL_FINDING",

            "confirmatory":
                False,

            "finding":
                (
                    "Prediction quality and process-structure "
                    "recovery are empirically decoupled in "
                    "Tier-C-v4."
                ),

            "evidence":
                (
                    "The predictive task remains learnable, "
                    "but exact transformation, information-class, "
                    "Blackwell-order, and state-path recovery "
                    "remain weak in the frozen OCM."
                ),
        },
    ]


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_hashes = (
        verify_source_hashes()
    )

    policy = (
        verify_test_policy()
    )

    registry = (
        verify_hypothesis_registry()
    )

    development = (
        verify_development_acceptance()
    )

    registered_rows = (
        freeze_registered_hypotheses(
            registry,
            development,
        )
    )

    primary = (
        verify_primary_result()
    )

    structural = (
        freeze_structural_diagnostics()
    )

    print(
        "[7/8] Freezing paper-safe findings"
    )

    findings = (
        make_empirical_findings(
            development,
            primary,
            structural,
        )
    )

    hypothesis_csv = (
        OUTPUT_DIR
        / "registered_hypothesis_verdicts.csv"
    )

    findings_csv = (
        OUTPUT_DIR
        / "empirical_findings.csv"
    )

    write_csv(
        hypothesis_csv,
        registered_rows,
    )

    write_csv(
        findings_csv,
        findings,
    )

    paper_claims = {
        "safe_primary_claim":
            (
                "The preregistered primary comparison "
                "does not support predictive superiority "
                "of OCM over B4 under joint OOD shift."
            ),

        "safe_main_scientific_claim":
            (
                "Strong observability of the underlying "
                "physical states is not sufficient for "
                "predictive training to induce the intended "
                "discrete process-memory representation."
            ),

        "safe_structural_claim":
            (
                "Under the pre-existing Phase-3G hard "
                "structural estimator, the frozen Tier-C-v4 "
                "OCM recovers only 1 of 104 transformations "
                "exactly and none of the nine held-out "
                "transformations."
            ),

        "unsafe_claims_do_not_make": [
            (
                "OCM achieves predictive superiority "
                "over the strongest neural baseline."
            ),
            (
                "OCM recovers the underlying semigroup."
            ),
            (
                "OCM recovers held-out transformations."
            ),
            (
                "OCM learns the nine information classes."
            ),
            (
                "OCM reliably recovers the Blackwell "
                "information ordering."
            ),
            (
                "OCM reliably tracks the true symbolic "
                "state trajectory."
            ),
            (
                "The secondary structural diagnostics "
                "were preregistered confirmatory hypotheses."
            ),
        ],
    }

    print(
        "[8/8] Writing final experimental freeze"
    )

    final_summary = {
        "phase":
            (
                "4G-R3G2E Tier C v4 final statistics "
                "and hypothesis freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "registered_hypothesis_count":
            len(
                registered_rows
            ),

        "registered_hypotheses_supported":
            [
                row[
                    "hypothesis_id"
                ]
                for row in registered_rows
                if row[
                    "verdict"
                ]
                == "SUPPORTED"
            ],

        "registered_hypotheses_not_supported":
            [
                row[
                    "hypothesis_id"
                ]
                for row in registered_rows
                if row[
                    "verdict"
                ]
                != "SUPPORTED"
            ],

        "important_scope_note":
            (
                "H4R3-1 through H4R3-5 are frozen "
                "Tier-C-v4 development/audit hypotheses. "
                "The OCM-vs-B4 predictive comparison is "
                "the separately frozen primary confirmatory "
                "test. Structural/process-memory metrics are "
                "secondary non-confirmatory diagnostics."
            ),

        "primary_confirmatory_result":
            primary,

        "secondary_structural_diagnostics":
            structural,

        "secondary_formal_pairwise_tests_added_in_g2e":
            False,

        "holm_adjustment_performed_in_g2e":
            False,

        "holm_reason":
            (
                "The frozen policy requires Holm adjustment "
                "where formal secondary pairwise tests are used. "
                "No additional secondary formal pairwise test "
                "statistic or bootstrap procedure was frozen, "
                "so G2E does not introduce post-hoc tests."
            ),

        "empirical_findings":
            findings,

        "paper_claims":
            paper_claims,

        "g2d_v0_status":
            (
                "Preserved as quarantined diagnostic because "
                "its estimator did not match the pre-existing "
                "Phase-3G hard structural estimator."
            ),

        "registered_structural_source":
            str(
                G2D_SUMMARY
            ),

        "source_hashes":
            {
                **source_hashes,

                str(
                    HYPOTHESIS_REGISTRY
                ):
                    sha256_file(
                        HYPOTHESIS_REGISTRY
                    ),

                str(
                    STATE_DECODER_SUMMARY
                ):
                    sha256_file(
                        STATE_DECODER_SUMMARY
                    ),

                str(
                    PREDECODER_SUMMARY
                ):
                    sha256_file(
                        PREDECODER_SUMMARY
                    ),

                str(
                    G2D_NOISE
                ):
                    sha256_file(
                        G2D_NOISE
                    ),

                str(
                    G2D_STATE_PATH
                ):
                    sha256_file(
                        G2D_STATE_PATH
                    ),
            },

        "additional_training_performed":
            False,

        "additional_tuning_performed":
            False,

        "checkpoint_reselection_performed":
            False,

        "test_regenerated":
            False,

        "test_reopened":
            False,

        "new_confirmatory_hypotheses_added":
            False,

        "new_secondary_significance_tests_added":
            False,

        "experimental_campaign_status":
            "CLOSED",

        "next_stage":
            (
                "Paper tables, figures, results, "
                "discussion, limitations, abstract, "
                "and conclusion."
            ),
    }

    summary_path = (
        OUTPUT_DIR
        / "phase4gr3g2e_final_summary.json"
    )

    claims_path = (
        OUTPUT_DIR
        / "paper_claims.json"
    )

    write_json(
        claims_path,
        paper_claims,
    )

    write_json(
        summary_path,
        final_summary,
    )

    result_files = [
        hypothesis_csv,
        findings_csv,
        claims_path,
        summary_path,
    ]

    result_hashes = {
        str(
            path
        ):
            sha256_file(
                path
            )
        for path in result_files
    }

    hash_path = (
        OUTPUT_DIR
        / "phase4gr3g2e_result_hashes.json"
    )

    write_json(
        hash_path,
        result_hashes,
    )

    print()
    print(
        "=" * 100
    )
    print(
        "PHASE 4G-R3G2E COMPLETE"
    )
    print(
        "=" * 100
    )

    print()
    print(
        "Registered H4R3 hypotheses:"
    )

    for row in registered_rows:
        print(
            f"  {row['hypothesis_id']}: "
            f"{row['verdict']}"
        )

    print()
    print(
        "Primary confirmatory predictive claim:"
    )

    print(
        "  OCM predictive superiority:",
        primary[
            "verdict"
        ],
    )

    print(
        "  OCM-B4:",
        primary[
            "ocm_minus_b4"
        ],
    )

    print(
        "  95% CI:",
        (
            primary[
                "bootstrap_ci_lower"
            ],
            primary[
                "bootstrap_ci_upper"
            ]
        ),
    )

    print()
    print(
        "Secondary structural diagnostics:"
    )

    print(
        "  Exact transformations:",
        structural[
            "exact_transformation_rate"
        ][
            "grand_mean"
        ],
    )

    print(
        "  Held-out exact transformations:",
        structural[
            "heldout_exact_transformation_rate"
        ][
            "grand_mean"
        ],
    )

    print(
        "  Information classes:",
        structural[
            "information_class_accuracy"
        ][
            "grand_mean"
        ],
    )

    print(
        "  Blackwell BA:",
        structural[
            "blackwell_relation_balanced_accuracy"
        ][
            "grand_mean"
        ],
    )

    print(
        "  State-path accuracy:",
        structural[
            "sealed_state_path_accuracy"
        ][
            "grand_mean"
        ],
    )

    print(
        "  Chance state accuracy:",
        STATE_CHANCE_ACCURACY,
    )

    print()
    print(
        "Experimental campaign:",
        final_summary[
            "experimental_campaign_status"
        ],
    )

    print(
        "Next:",
        final_summary[
            "next_stage"
        ],
    )


if __name__ == "__main__":
    main()
