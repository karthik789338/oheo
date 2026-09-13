from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PROTOCOL_DIR = Path(
    "outputs/phase4ar3_tier_c_v4_protocol"
)

FIELD_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_fields"
)

DATA_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_data"
)

AUDIT_DIR = Path(
    "outputs/phase4cr3_tier_c_v4_audit"
)

OUTPUT_DIR = Path(
    "outputs/phase4cr3_tier_c_v4_acceptance_freeze"
)


PROTOCOL_VERSION = "tier_c_v4"

DEVELOPMENT_CELLS = (
    "train_joint",
    "val_composition",
    "val_carrier",
    "val_joint",
)

SEALED_TEST_FRAGMENTS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
    "test_carrier_parameters",
    "test_carrier_state_fields",
    "sealed_test_fields",
    "sealed_test_manifest",
)


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


def scan_for_test_artifacts():
    found = []

    for root in (
        FIELD_DIR,
        DATA_DIR,
        AUDIT_DIR,
    ):
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            lower_name = path.name.lower()

            if any(
                fragment in lower_name
                for fragment
                in SEALED_TEST_FRAGMENTS
            ):
                found.append(
                    str(path)
                )

    return sorted(set(found))


def load_and_validate_sources():
    paths = {
        "protocol_summary":
            PROTOCOL_DIR
            / "phase4ar3_summary.json",

        "field_summary":
            FIELD_DIR
            / "phase4br3_field_summary.json",

        "morphology_summary":
            FIELD_DIR
            / "independent_morphology_summary.json",

        "dataset_summary":
            DATA_DIR
            / "phase4br3_summary.json",

        "predecoder_summary":
            AUDIT_DIR
            / "phase4cr3_predecoder_summary.json",

        "state_decoder_summary":
            AUDIT_DIR
            / "state_decoder_summary.json",

        "global_statistics_summary":
            AUDIT_DIR
            / "global_statistics_summary.json",

        "affine_shortcut_summary":
            AUDIT_DIR
            / "affine_shortcut_summary.json",

        "transition_provenance":
            AUDIT_DIR
            / "recovered_transition_table"
            / "transition_recovery_provenance.json",

        "normalized_transition_table":
            AUDIT_DIR
            / "recovered_transition_table"
            / "primitive_transition_table_long.csv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    values = {
        name:
            (
                load_json(path)
                if path.suffix == ".json"
                else path
            )
        for name, path in paths.items()
    }

    protocol = values[
        "protocol_summary"
    ]

    fields = values[
        "field_summary"
    ]

    morphology = values[
        "morphology_summary"
    ]

    dataset = values[
        "dataset_summary"
    ]

    predecoder = values[
        "predecoder_summary"
    ]

    state_decoder = values[
        "state_decoder_summary"
    ]

    global_statistics = values[
        "global_statistics_summary"
    ]

    affine = values[
        "affine_shortcut_summary"
    ]

    transition = values[
        "transition_provenance"
    ]

    require(
        protocol[
            "protocol_version"
        ] == PROTOCOL_VERSION,
        "Protocol version changed.",
    )

    require(
        protocol[
            "phase4ar3_status"
        ] == "protocol_frozen",
        "Phase 4A-R3 is not frozen.",
    )

    require(
        protocol[
            "tier_c_v4_is_final_revision"
        ] is True,
        "Tier C v4 is not recorded as final.",
    )

    require(
        protocol[
            "tier_c_v5_authorized"
        ] is False,
        "Tier C v5 was unexpectedly authorized.",
    )

    require(
        fields[
            "phase4br3_status"
        ] == "completed_and_verified",
        "Development field construction did not pass.",
    )

    require(
        fields[
            "independent_morphology_gate_passed"
        ] is True,
        "Field morphology gate failed.",
    )

    require(
        morphology[
            "independent_morphology_gate_passed"
        ] is True,
        "Independent morphology audit failed.",
    )

    require(
        morphology["failed_checks"] == [],
        "Independent morphology has failed checks.",
    )

    require(
        dataset[
            "phase4br3_status"
        ] == "completed",
        "Development data materialization is incomplete.",
    )

    require(
        dataset[
            "development_cell_count"
        ] == 4,
        "Development-cell count changed.",
    )

    require(
        set(
            dataset[
                "development_cells"
            ]
        ) == set(DEVELOPMENT_CELLS),
        "Development-cell registry changed.",
    )

    require(
        predecoder[
            "phase4cr3_predecoder_status"
        ] == "passed",
        "Pre-decoder audit did not pass.",
    )

    require(
        predecoder[
            "failed_predecoder_gate_count"
        ] == 0,
        "Pre-decoder audit has failed gates.",
    )

    require(
        state_decoder[
            "state_decoder_diagnostic_status"
        ] == "passed",
        "State-separability diagnostic did not pass.",
    )

    require(
        state_decoder[
            "state_separability_gate_passed"
        ] is True,
        "State-separability gate failed.",
    )

    require(
        global_statistics[
            "global_statistics_diagnostic_status"
        ] == "passed",
        "Global-statistics diagnostic did not pass.",
    )

    require(
        global_statistics[
            "global_statistics_gate_passed"
        ] is True,
        "Global-statistics gate failed.",
    )

    require(
        global_statistics[
            "global_statistics_shortcut_present"
        ] is False,
        "Global-statistics shortcut was detected.",
    )

    require(
        affine[
            "affine_shortcut_diagnostic_status"
        ] == "passed",
        "Affine-shortcut diagnostic did not pass.",
    )

    require(
        affine[
            "affine_shortcut_gate_passed"
        ] is True,
        "Affine-shortcut gate failed.",
    )

    require(
        affine[
            "exact_affine_shortcut_present"
        ] is False,
        "An exact affine shortcut was detected.",
    )

    require(
        affine[
            "semigroup_size"
        ] == 104,
        "Affine diagnostic semigroup size changed.",
    )

    require(
        transition[
            "status"
        ] == "verified_and_normalized",
        "Primitive-transition normalization did not pass.",
    )

    require(
        transition[
            "generator_count"
        ] == 6,
        "Primitive generator count changed.",
    )

    require(
        transition[
            "reconstructed_semigroup_size"
        ] == 104,
        "Recovered primitive system did not produce 104 elements.",
    )

    require(
        transition[
            "reconstructed_semigroup_matches_frozen_registry"
        ] is True,
        "Recovered semigroup differs from Phase 1.",
    )

    require(
        transition[
            "source_artifacts_modified"
        ] is False,
        "Transition source artifacts were modified.",
    )

    require(
        transition[
            "transition_mappings_modified"
        ] is False,
        "Frozen transition mappings were modified.",
    )

    require(
        transition[
            "acceptance_thresholds_modified"
        ] is False,
        "Acceptance thresholds were modified.",
    )

    require(
        transition[
            "test_artifacts_accessed"
        ] is False,
        "Test artifacts were accessed during transition recovery.",
    )

    for source_name, source in (
        ("field summary", fields),
        ("dataset summary", dataset),
        ("predecoder summary", predecoder),
        ("state decoder", state_decoder),
        ("global statistics", global_statistics),
        ("affine shortcut", affine),
    ):
        for key in (
            "test_carrier_parameters_generated",
            "test_fields_generated",
            "test_manifests_generated",
            "test_metrics_computed",
        ):
            if key in source:
                require(
                    source[key] is False,
                    f"{source_name}: {key} is not false.",
                )

    sealed_test_files = (
        scan_for_test_artifacts()
    )

    require(
        not sealed_test_files,
        (
            "Sealed Tier C v4 test artifacts were found: "
            + ", ".join(sealed_test_files)
        ),
    )

    hashes = {
        name: {
            "path":
                str(path),

            "sha256":
                sha256_file(path),
        }
        for name, path in paths.items()
    }

    return {
        "paths":
            paths,

        "hashes":
            hashes,

        "protocol":
            protocol,

        "fields":
            fields,

        "morphology":
            morphology,

        "dataset":
            dataset,

        "predecoder":
            predecoder,

        "state_decoder":
            state_decoder,

        "global_statistics":
            global_statistics,

        "affine":
            affine,

        "transition":
            transition,

        "sealed_test_files":
            sealed_test_files,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = load_and_validate_sources()

    morphology = sources[
        "morphology"
    ]

    state_decoder = sources[
        "state_decoder"
    ]

    global_statistics = sources[
        "global_statistics"
    ]

    affine = sources[
        "affine"
    ]

    gate_rows = [
        {
            "gate":
                "tier_c_v4_protocol_frozen",

            "passed":
                True,
        },
        {
            "gate":
                "development_field_construction",

            "passed":
                True,
        },
        {
            "gate":
                "independent_morphology",

            "passed":
                True,
        },
        {
            "gate":
                "development_data_integrity",

            "passed":
                True,
        },
        {
            "gate":
                "state_separability",

            "passed":
                True,
        },
        {
            "gate":
                "global_statistics_shortcut_control",

            "passed":
                True,
        },
        {
            "gate":
                "pca128_ridge_affine_shortcut_control",

            "passed":
                True,
        },
        {
            "gate":
                "frozen_primitive_transition_provenance",

            "passed":
                True,
        },
        {
            "gate":
                "sealed_test_absence",

            "passed":
                True,
        },
    ]

    write_csv(
        OUTPUT_DIR
        / "tier_c_v4_acceptance_gates.csv",
        gate_rows,
    )

    interpretation = {
        "supported_development_claims": [
            (
                "Tier C v4 constructs distinct fine and "
                "coarse spatial morphologies while exactly "
                "preserving the frozen phase, defect, and "
                "orientation value multisets."
            ),
            (
                "The eight latent states are strongly "
                "separable on held-out development carriers "
                "using the frozen convolutional diagnostic."
            ),
            (
                "The frozen 28 global marginal statistics "
                "contain no state information beyond "
                "balanced chance."
            ),
            (
                "The frozen PCA-128 ridge-affine diagnostic "
                "does not exactly recover the primitive "
                "operations or the full 104-element "
                "semigroup on held-out development carriers."
            ),
        ],

        "claims_not_supported_yet": [
            (
                "No claim about sealed test-set "
                "generalization is supported."
            ),
            (
                "No predictive advantage over frozen "
                "baselines is supported."
            ),
            (
                "No Process-Memory Poset recovery claim "
                "is supported for Tier C v4 yet."
            ),
            (
                "The diagnostics do not rule out every "
                "possible linear, nonlinear, or "
                "representation-specific shortcut."
            ),
            (
                "The affine result should not be described "
                "as absence of partial affine signal."
            ),
        ],

        "affine_interpretation": {
            "validation_continuous_latent_mse":
                affine[
                    "validation_continuous_latent_mse"
                ],

            "validation_exact_primitive_recovery_rate":
                affine[
                    "validation_exact_primitive_recovery_rate"
                ],

            "semigroup_overall_exact_state_recovery_rate":
                affine[
                    "semigroup_overall_exact_state_recovery_rate"
                ],

            "fraction_validation_carriers_all_104_exact":
                affine[
                    "fraction_validation_carriers_all_104_exact"
                ],

            "pca_explained_variance_ratio_sum":
                affine[
                    "pca_explained_variance_ratio_sum"
                ],

            "interpretation":
                (
                    "Partial affine structure is present, "
                    "but the frozen PCA-128 ridge model is "
                    "not an exact shortcut."
                ),
        },
    }

    write_json(
        OUTPUT_DIR
        / "tier_c_v4_interpretation_freeze.json",
        interpretation,
    )

    write_json(
        OUTPUT_DIR
        / "tier_c_v4_acceptance_input_hashes.json",
        sources["hashes"],
    )

    summary = {
        "phase":
            (
                "4C-R3D Tier C v4 final development "
                "acceptance freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "development_acceptance_gate_count":
            len(gate_rows),

        "development_acceptance_pass_count":
            sum(
                int(row["passed"])
                for row in gate_rows
            ),

        "development_acceptance_failed_gates":
            [
                row["gate"]
                for row in gate_rows
                if not row["passed"]
            ],

        "independent_mean_length_ratio":
            morphology[
                "mean_characteristic_length_ratio"
            ],

        "independent_minimum_length_ratio":
            morphology[
                "minimum_characteristic_length_ratio"
            ],

        "independent_mean_interface_ratio":
            morphology[
                "mean_interface_density_ratio"
            ],

        "independent_maximum_interface_ratio":
            morphology[
                "maximum_interface_density_ratio"
            ],

        "interface_pair_pass_fraction":
            morphology[
                "fraction_pairs_at_or_below_mean_interface_threshold"
            ],

        "mean_validation_state_accuracy":
            state_decoder[
                "mean_validation_state_accuracy"
            ],

        "worst_seed_validation_state_accuracy":
            state_decoder[
                "worst_seed_validation_state_accuracy"
            ],

        "mean_validation_coarsening_accuracy":
            state_decoder[
                "mean_validation_coarsening_accuracy"
            ],

        "global_statistics_validation_accuracy":
            global_statistics[
                "validation_accuracy"
            ],

        "global_statistics_balanced_chance":
            1.0 / 8.0,

        "global_statistics_shortcut_present":
            False,

        "affine_validation_continuous_latent_mse":
            affine[
                "validation_continuous_latent_mse"
            ],

        "affine_exact_primitive_recovery_rate":
            affine[
                "validation_exact_primitive_recovery_rate"
            ],

        "affine_semigroup_exact_state_recovery_rate":
            affine[
                "semigroup_overall_exact_state_recovery_rate"
            ],

        "validation_carriers_all_104_exact_count":
            affine[
                "validation_carriers_all_104_exact_count"
            ],

        "exact_affine_shortcut_present":
            False,

        "primitive_count":
            6,

        "semigroup_size":
            104,

        "primitive_transition_provenance_verified":
            True,

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_fields_read":
            False,

        "test_metrics_computed":
            False,

        "sealed_test_files_absent":
            True,

        "predictive_training_performed":
            False,

        "predictive_checkpoint_selection_performed":
            False,

        "tier_c_v4_development_acceptance_passed":
            True,

        "phase4d_predictive_protocol_freeze_authorized":
            True,

        "predictive_training_authorized":
            False,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "tier_c_v5_authorized":
            False,

        "prior_tier_c_outputs_modified":
            False,

        "freeze_status":
            "frozen_development_acceptance",

        "next_phase":
            (
                "4D-R3 Tier C v4 predictive-learning "
                "protocol freeze"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4cr3d_acceptance_summary.json",
        summary,
    )

    print(
        "Phase 4C-R3D Tier C v4 development "
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
