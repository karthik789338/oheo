from __future__ import annotations

import csv
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

import phase4fr3b_run_tier_c_v4_tuning as tuning_runtime


PROTOCOL_VERSION = "tier_c_v4"

G1_ROOT = Path(
    "outputs/phase4gr3g1_tier_c_v4_sealed_test"
)

G1_SUMMARY = (
    G1_ROOT
    / "phase4gr3g1_sealed_test_generation_summary.json"
)

G1_LEDGER = (
    G1_ROOT
    / "single_test_opening_ledger.json"
)

G1_ARTIFACT_MANIFEST = (
    G1_ROOT
    / "sealed_test_artifact_manifest.csv"
)

FINAL_FIT_REGISTRY = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "frozen_final_fit_registry.csv"
)

FINAL_FIT_FREEZE = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "phase4gr3c_final_fit_freeze_summary.json"
)

COMPARATOR = Path(
    "outputs/"
    "phase4gr3d1_tier_c_v4_primary_comparator/"
    "primary_validation_comparator.json"
)

TEST_POLICY = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_test_opening_policy.json"
)

G0_HASHES = Path(
    "outputs/"
    "phase4gr3g0_tier_c_v4_test_execution_interfaces/"
    "phase4gr3g0_source_hashes.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3g2a_tier_c_v4_evaluation_interfaces"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
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


def write_text(
    path: Path,
    value: str,
) -> None:
    path.write_text(
        value,
        encoding="utf-8",
    )


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


def locate_column(
    columns: list[str],
    candidates: list[str],
) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise KeyError(
        "Could not locate any of: "
        + ", ".join(candidates)
    )


def serialize_value(value: Any) -> Any:
    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            serialize_value(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key):
                serialize_value(child)
            for key, child in value.items()
        }

    return repr(value)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1/8] Verifying frozen G1 sealed-test state"
    )

    for path in (
        G1_SUMMARY,
        G1_LEDGER,
        G1_ARTIFACT_MANIFEST,
        FINAL_FIT_REGISTRY,
        FINAL_FIT_FREEZE,
        COMPARATOR,
        TEST_POLICY,
        G0_HASHES,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    g1 = load_json(
        G1_SUMMARY
    )

    ledger = load_json(
        G1_LEDGER
    )

    require(
        g1[
            "phase4gr3g1_status"
        ] == "sealed_test_generation_frozen",
        "G1 is not frozen.",
    )

    require(
        g1[
            "test_open_count"
        ] == 1,
        "G1 test-open count is not one.",
    )

    require(
        g1[
            "maximum_test_open_count"
        ] == 1,
        "Maximum test-open count changed.",
    )

    require(
        g1[
            "single_opening_consumed"
        ] is True,
        "Single opening is not marked consumed.",
    )

    require(
        g1[
            "test_artifacts_hashed"
        ] is True,
        "G1 artifacts were not frozen by hash.",
    )

    require(
        g1[
            "test_metrics_computed"
        ] is False,
        (
            "Test metrics were already computed "
            "before G2."
        ),
    )

    require(
        g1[
            "test_regeneration_authorized"
        ] is False,
        "Test regeneration is still authorized.",
    )

    require(
        ledger[
            "test_open_count"
        ] == 1,
        "Opening ledger count changed.",
    )

    require(
        ledger[
            "single_opening_consumed"
        ] is True,
        "Opening ledger is not consumed.",
    )

    print(
        "[2/8] Re-verifying all 36 G1 artifacts"
    )

    artifact_rows = load_csv(
        G1_ARTIFACT_MANIFEST
    )

    require(
        len(
            artifact_rows
        ) == 36,
        (
            "Expected 36 G1 frozen artifacts; "
            f"found {len(artifact_rows)}."
        ),
    )

    artifact_columns = list(
        artifact_rows[0].keys()
    )

    path_column = locate_column(
        artifact_columns,
        [
            "relative_path",
            "path",
            "artifact_path",
            "file_path",
        ],
    )

    sha_column = locate_column(
        artifact_columns,
        [
            "sha256",
            "artifact_sha256",
            "file_sha256",
        ],
    )

    for row in artifact_rows:
        manifest_path = Path(
            row[
                path_column
            ]
        )

        if path_column == "relative_path":
            path = (
                G1_ROOT
                / manifest_path
            )
        else:
            path = manifest_path

        if not path.exists():
            raise FileNotFoundError(path)

        observed = sha256_file(
            path
        )

        require(
            observed
            == row[
                sha_column
            ],
            (
                "Frozen G1 artifact changed: "
                f"{path}"
            ),
        )

    print(
        "[3/8] Verifying frozen final-fit registry"
    )

    final_rows = load_csv(
        FINAL_FIT_REGISTRY
    )

    require(
        len(final_rows) == 130,
        (
            "Expected 130 frozen final fits; "
            f"found {len(final_rows)}."
        ),
    )

    columns = list(
        final_rows[0].keys()
    )

    print(
        "Final-fit registry columns:"
    )
    print(
        json.dumps(
            columns,
            indent=2,
        )
    )

    model_column = locate_column(
        columns,
        [
            "model_id",
        ],
    )

    noise_column = locate_column(
        columns,
        [
            "noise_fraction",
        ],
    )

    seed_column = locate_column(
        columns,
        [
            "effective_seed",
            "seed",
        ],
    )

    checkpoint_path_column = (
        locate_column(
            columns,
            [
                "best_checkpoint_path",
                "checkpoint_path",
            ],
        )
    )

    checkpoint_sha_column = (
        locate_column(
            columns,
            [
                "best_checkpoint_sha256",
                "checkpoint_sha256",
            ],
        )
    )

    model_counts = Counter(
        row[
            model_column
        ]
        for row in final_rows
    )

    require(
        dict(model_counts)
        == {
            "B1": 5,
            "B2": 25,
            "B3": 25,
            "B4": 25,
            "B5": 25,
            "OCM": 25,
        },
        (
            "Frozen final-fit model counts changed: "
            f"{dict(model_counts)}"
        ),
    )

    print(
        "[4/8] Re-verifying 130 checkpoint hashes"
    )

    for row in final_rows:
        checkpoint_path = Path(
            row[
                checkpoint_path_column
            ]
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )

        require(
            sha256_file(
                checkpoint_path
            )
            == row[
                checkpoint_sha_column
            ],
            (
                "Checkpoint hash changed: "
                f"{checkpoint_path}"
            ),
        )

    print(
        "[5/8] Inspecting one checkpoint per model family"
    )

    checkpoint_reports = {}

    for model_id in (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "OCM",
    ):
        candidates = [
            row
            for row in final_rows
            if (
                row[
                    model_column
                ] == model_id
                and abs(
                    float(
                        row[
                            noise_column
                        ]
                    )
                    - 0.25
                ) < 1e-12
            )
        ]

        require(
            bool(candidates),
            (
                f"No noise=0.25 checkpoint "
                f"for {model_id}."
            ),
        )

        candidates.sort(
            key=lambda row: int(
                row[
                    seed_column
                ]
            )
        )

        row = candidates[0]

        checkpoint_path = Path(
            row[
                checkpoint_path_column
            ]
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        top_keys = sorted(
            checkpoint.keys()
        )

        source_signature = (
            checkpoint.get(
                "source_signature",
                {},
            )
        )

        configuration = (
            source_signature.get(
                "configuration"
            )
            if isinstance(
                source_signature,
                dict,
            )
            else None
        )

        if configuration is None:
            configuration = checkpoint.get(
                "configuration"
            )

        codec_state = checkpoint.get(
            "codec_state_dict",
            {},
        )

        model_state = checkpoint.get(
            "model_state_dict",
            {},
        )

        checkpoint_reports[
            model_id
        ] = {
            "checkpoint_path":
                str(
                    checkpoint_path
                ),

            "checkpoint_sha256":
                sha256_file(
                    checkpoint_path
                ),

            "registry_noise_fraction":
                float(
                    row[
                        noise_column
                    ]
                ),

            "registry_seed":
                int(
                    row[
                        seed_column
                    ]
                ),

            "top_level_keys":
                top_keys,

            "source_signature_keys":
                (
                    sorted(
                        source_signature.keys()
                    )
                    if isinstance(
                        source_signature,
                        dict,
                    )
                    else []
                ),

            "configuration":
                serialize_value(
                    configuration
                ),

            "codec_state_key_count":
                len(
                    codec_state
                ),

            "model_state_key_count":
                len(
                    model_state
                ),

            "codec_first_keys":
                list(
                    codec_state.keys()
                )[:12],

            "model_first_keys":
                list(
                    model_state.keys()
                )[:12],
        }

        del checkpoint

    print(
        "[6/8] Freezing exact inference source"
    )

    accepted_impl_path = Path(
        inspect.getsourcefile(
            tuning_runtime.accepted_impl
        )
    )

    tier_b_path = Path(
        inspect.getsourcefile(
            tuning_runtime.tier_b
        )
    )

    tuning_path = Path(
        inspect.getsourcefile(
            tuning_runtime
        )
    )

    runtime_sources = {
        str(tuning_path):
            sha256_file(
                tuning_path
            ),

        str(accepted_impl_path):
            sha256_file(
                accepted_impl_path
            ),

        str(tier_b_path):
            sha256_file(
                tier_b_path
            ),
    }

    source_extracts = {
        "TrajectoryCell.__init__":
            inspect.getsource(
                tuning_runtime
                .TrajectoryCell
                .__init__
            ),

        "TrajectoryCell._load_manifest":
            inspect.getsource(
                tuning_runtime
                .TrajectoryCell
                ._load_manifest
            ),

        "TrajectoryCell._load_index":
            inspect.getsource(
                tuning_runtime
                .TrajectoryCell
                ._load_index
            ),

        "TrajectoryCell.make_batch":
            inspect.getsource(
                tuning_runtime
                .TrajectoryCell
                .make_batch
            ),

        "build_model_bundle":
            inspect.getsource(
                tuning_runtime
                .build_model_bundle
            ),

        "compute_losses":
            inspect.getsource(
                tuning_runtime
                .compute_losses
            ),

        "evaluate_validation":
            inspect.getsource(
                tuning_runtime
                .evaluate_validation
            ),

        "baseline_forward_losses":
            inspect.getsource(
                tuning_runtime
                .accepted_impl
                .baseline_forward_losses
            ),

        "ocm_forward_losses":
            inspect.getsource(
                tuning_runtime
                .accepted_impl
                .ocm_forward_losses
            ),
    }

    source_extract_path = (
        OUTPUT_DIR
        / "frozen_evaluation_source_extracts.txt"
    )

    blocks = []

    for name, source in (
        source_extracts.items()
    ):
        blocks.append(
            "\n"
            + "=" * 100
            + "\n"
            + name
            + "\n"
            + "=" * 100
            + "\n"
            + source
        )

    write_text(
        source_extract_path,
        "\n".join(blocks),
    )

    print(
        "[7/8] Inspecting sealed test schemas"
    )

    test_schema = {}

    for cell_id in (
        "test_iid_pairing",
        "test_composition",
        "test_carrier",
        "test_joint",
    ):
        visible_index = (
            G1_ROOT
            / "visible"
            / f"{cell_id}_trajectory_index.csv"
        )

        visible_manifest = (
            G1_ROOT
            / "visible"
            / f"{cell_id}_sequence_manifest.csv"
        )

        privileged_metadata = (
            G1_ROOT
            / "privileged"
            / f"{cell_id}_trajectory_metadata.csv"
        )

        fields = (
            G1_ROOT
            / "visible"
            / f"{cell_id}_fields.npy"
        )

        index_rows = load_csv(
            visible_index
        )

        manifest_rows = load_csv(
            visible_manifest
        )

        metadata_rows = load_csv(
            privileged_metadata
        )

        import numpy as np

        field_array = np.load(
            fields,
            mmap_mode="r",
        )

        test_schema[
            cell_id
        ] = {
            "trajectory_index_columns":
                list(
                    index_rows[0].keys()
                ),

            "sequence_manifest_columns":
                list(
                    manifest_rows[0].keys()
                ),

            "trajectory_metadata_columns":
                list(
                    metadata_rows[0].keys()
                ),

            "trajectory_count":
                len(
                    index_rows
                ),

            "sequence_count":
                len(
                    manifest_rows
                ),

            "metadata_count":
                len(
                    metadata_rows
                ),

            "field_shape":
                list(
                    field_array.shape
                ),

            "field_dtype":
                str(
                    field_array.dtype
                ),

            "first_visible_index_row":
                index_rows[0],

            "first_sequence_manifest_row":
                manifest_rows[0],

            "first_privileged_metadata_row":
                metadata_rows[0],
        }

        del field_array

    print(
        "[8/8] Writing G2A interface freeze"
    )

    comparator = load_json(
        COMPARATOR
    )

    policy = load_json(
        TEST_POLICY
    )

    require(
        comparator[
            "selected_comparator_model_id"
        ] == "B4",
        "Frozen comparator is no longer B4.",
    )

    primary = policy[
        "primary_confirmatory_condition"
    ]

    require(
        primary[
            "primary_model"
        ] == "OCM",
        "Primary model changed.",
    )

    require(
        primary[
            "cell"
        ] == "test_joint",
        "Primary cell changed.",
    )

    require(
        float(
            primary[
                "noise_fraction"
            ]
        ) == 0.25,
        "Primary noise changed.",
    )

    report = {
        "phase":
            (
                "4G-R3G2A Tier C v4 frozen "
                "sealed-test evaluation interface"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "g1_summary_sha256":
            sha256_file(
                G1_SUMMARY
            ),

        "g1_artifact_manifest_sha256":
            sha256_file(
                G1_ARTIFACT_MANIFEST
            ),

        "g1_artifacts_reverified":
            True,

        "g1_artifact_count":
            len(
                artifact_rows
            ),

        "test_open_count":
            1,

        "maximum_test_open_count":
            1,

        "test_regeneration_authorized":
            False,

        "final_fit_count":
            len(
                final_rows
            ),

        "final_fit_registry_columns":
            columns,

        "model_counts":
            dict(
                model_counts
            ),

        "checkpoint_reports":
            checkpoint_reports,

        "runtime_source_hashes":
            runtime_sources,

        "source_extract_path":
            str(
                source_extract_path
            ),

        "source_extract_sha256":
            sha256_file(
                source_extract_path
            ),

        "test_schema":
            test_schema,

        "primary_model":
            "OCM",

        "primary_comparator":
            "B4",

        "primary_cell":
            "test_joint",

        "primary_noise_fraction":
            0.25,

        "primary_metric":
            "noisy_target_rollout_mse",

        "statistical_unit":
            "sequence",

        "bootstrap_replicates":
            10000,

        "bootstrap_seed":
            73011,

        "test_metrics_computed":
            False,

        "phase4gr3g2a_status":
            "evaluation_interfaces_frozen",
    }

    report_path = (
        OUTPUT_DIR
        / "phase4gr3g2a_evaluation_interface_report.json"
    )

    write_json(
        report_path,
        report,
    )

    print()
    print(
        "Phase 4G-R3G2A evaluation "
        "interface freeze passed."
    )

    print(
        json.dumps(
            {
                "g1_artifacts_reverified":
                    True,

                "g1_artifact_count":
                    len(
                        artifact_rows
                    ),

                "test_open_count":
                    1,

                "final_fit_count":
                    len(
                        final_rows
                    ),

                "model_counts":
                    dict(
                        model_counts
                    ),

                "runtime_source_hashes":
                    runtime_sources,

                "test_schema":
                    {
                        key: {
                            "trajectory_count":
                                value[
                                    "trajectory_count"
                                ],

                            "sequence_count":
                                value[
                                    "sequence_count"
                                ],

                            "field_shape":
                                value[
                                    "field_shape"
                                ],
                        }
                        for key, value
                        in test_schema.items()
                    },

                "test_metrics_computed":
                    False,

                "phase4gr3g2a_status":
                    (
                        "evaluation_interfaces_frozen"
                    ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
