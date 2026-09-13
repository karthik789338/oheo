from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_VERSION = "tier_c_v4"

CELLS = (
    "train_joint",
    "val_joint",
)

SAFE_COLUMNS = (
    "trajectory_index",
    "cell_sequence_index",
    "source_sequence_id",
    "point_start",
    "point_count",
    "sequence_length",
)

SOURCE_REQUIRED_COLUMNS = (
    "trajectory_index",
    "cell_sequence_index",
    "source_sequence_id",
)

DATA_ROOT = Path(
    "outputs/"
    "phase4br3_tier_c_v4_development_data"
)

ORIGINAL_PREFLIGHT_PATH = Path(
    "outputs/"
    "phase4fr3_tier_c_v4_tuning_preflight/"
    "phase4fr3_tuning_preflight_summary.json"
)

IMPLEMENTATION_ACCEPTANCE_PATH = Path(
    "outputs/"
    "phase4er3d_tier_c_v4_implementation_acceptance/"
    "phase4er3d_implementation_acceptance_summary.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4fr3ar1_tier_c_v4_visible_trajectory_indexes"
)

VISIBLE_INDEX_DIR = (
    OUTPUT_DIR
    / "visible"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


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


def load_visible_sequence_manifest(
    path: Path,
) -> tuple[
    list[dict[str, str]],
    dict[int, dict[str, Any]],
]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        fieldnames = list(
            reader.fieldnames or []
        )

        required = {
            "cell_sequence_index",
            "source_sequence_id",
            "sequence_length",
        }

        missing = sorted(
            required
            - set(fieldnames)
        )

        require(
            not missing,
            (
                f"Visible sequence manifest {path} "
                f"lacks columns {missing}."
            ),
        )

        rows = list(reader)

    manifest_by_index = {}

    for row in rows:
        cell_sequence_index = int(
            row[
                "cell_sequence_index"
            ]
        )

        require(
            cell_sequence_index
            not in manifest_by_index,
            (
                "Duplicate cell_sequence_index "
                f"{cell_sequence_index} in {path}."
            ),
        )

        manifest_by_index[
            cell_sequence_index
        ] = {
            "source_sequence_id":
                str(
                    row[
                        "source_sequence_id"
                    ]
                ),

            "sequence_length":
                int(
                    row[
                        "sequence_length"
                    ]
                ),
        }

    return rows, manifest_by_index


def load_and_validate_structural_rows(
    source_path: Path,
    manifest_by_index: dict[int, dict[str, Any]],
    point_count_expected: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    with source_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        source_columns = tuple(
            reader.fieldnames or []
        )

        missing_source_columns = sorted(
            set(SOURCE_REQUIRED_COLUMNS)
            - set(source_columns)
        )

        require(
            not missing_source_columns,
            (
                f"{source_path} lacks required "
                "structural columns: "
                f"{missing_source_columns}.\n"
                f"Observed: {source_columns}"
            ),
        )

        unused_source_columns = sorted(
            set(source_columns)
            - set(SOURCE_REQUIRED_COLUMNS)
        )

        raw_rows = list(reader)

    rows = []

    for raw in raw_rows:
        row = {
            "trajectory_index":
                int(
                    raw[
                        "trajectory_index"
                    ]
                ),

            "cell_sequence_index":
                int(
                    raw[
                        "cell_sequence_index"
                    ]
                ),

            "source_sequence_id":
                str(
                    raw[
                        "source_sequence_id"
                    ]
                ),
        }

        rows.append(row)

    rows.sort(
        key=lambda row:
            row[
                "trajectory_index"
            ]
    )

    require(
        rows,
        f"No trajectory rows found in {source_path}.",
    )

    expected_indices = list(
        range(len(rows))
    )

    observed_indices = [
        row[
            "trajectory_index"
        ]
        for row in rows
    ]

    require(
        observed_indices
        == expected_indices,
        (
            f"Trajectory indexes in {source_path} "
            "are not contiguous from zero."
        ),
    )

    expected_point_start = 0

    sequence_trajectory_counts: Counter[int] = (
        Counter()
    )

    for row in rows:
        sequence_index = row[
            "cell_sequence_index"
        ]

        require(
            sequence_index
            in manifest_by_index,
            (
                f"Trajectory references unknown "
                f"cell_sequence_index {sequence_index}."
            ),
        )

        manifest_row = manifest_by_index[
            sequence_index
        ]

        # Derive all released indexing fields from the visible
        # sequence manifest and the globally ordered trajectory ID.
        # No carrier, state, path, or test-carrier metadata is
        # copied into the visible release.
        row["sequence_length"] = int(
            manifest_row[
                "sequence_length"
            ]
        )

        row["point_count"] = (
            row[
                "sequence_length"
            ]
            + 1
        )

        row["point_start"] = (
            expected_point_start
        )

        require(
            row[
                "source_sequence_id"
            ]
            == manifest_row[
                "source_sequence_id"
            ],
            (
                "source_sequence_id mismatch for "
                f"cell_sequence_index {sequence_index}."
            ),
        )

        require(
            row[
                "sequence_length"
            ]
            == manifest_row[
                "sequence_length"
            ],
            (
                "sequence_length mismatch for "
                f"cell_sequence_index {sequence_index}."
            ),
        )

        require(
            row[
                "point_count"
            ]
            == (
                row[
                    "sequence_length"
                ]
                + 1
            ),
            (
                "point_count must equal "
                "sequence_length + 1."
            ),
        )

        require(
            row[
                "point_start"
            ]
            == expected_point_start,
            (
                "Trajectory points are not contiguous. "
                f"Expected point_start "
                f"{expected_point_start}; found "
                f"{row['point_start']}."
            ),
        )

        expected_point_start += row[
            "point_count"
        ]

        sequence_trajectory_counts[
            sequence_index
        ] += 1

    require(
        expected_point_start
        == point_count_expected,
        (
            f"Trajectory index covers "
            f"{expected_point_start} points, but "
            f"the visible field array contains "
            f"{point_count_expected} points."
        ),
    )

    require(
        set(
            sequence_trajectory_counts
        )
        == set(
            manifest_by_index
        ),
        (
            "Not every visible sequence has "
            "trajectory metadata."
        ),
    )

    multiplicities = sorted(
        set(
            sequence_trajectory_counts.values()
        )
    )

    require(
        len(multiplicities) == 1,
        (
            "Trajectory multiplicity differs "
            f"across sequences: {multiplicities}"
        ),
    )

    audit = {
        "source_path":
            str(source_path),

        "source_schema":
            list(source_columns),

        "required_source_structural_columns":
            list(
                SOURCE_REQUIRED_COLUMNS
            ),

        "unused_source_columns":
            unused_source_columns,

        "unused_source_column_count":
            len(
                unused_source_columns
            ),

        "source_schema_is_structural_superset":
            True,

        "released_schema":
            list(SAFE_COLUMNS),

        "released_extra_columns_present":
            False,

        "trajectory_count":
            len(rows),

        "sequence_count":
            len(
                sequence_trajectory_counts
            ),

        "trajectories_per_sequence":
            multiplicities[0],

        "covered_point_count":
            expected_point_start,

        "expected_point_count":
            point_count_expected,

        "contiguous_point_coverage":
            True,

        "manifest_consistency":
            True,

        "scientific_target_columns_present":
            False,

        "clean_target_columns_present":
            False,

        "privileged_state_columns_present":
            False,
    }

    return rows, audit


def write_visible_index(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                SAFE_COLUMNS
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column:
                        row[column]
                    for column
                    in SAFE_COLUMNS
                }
            )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VISIBLE_INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = (
        ORIGINAL_PREFLIGHT_PATH,
        IMPLEMENTATION_ACCEPTANCE_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    original_preflight = load_json(
        ORIGINAL_PREFLIGHT_PATH
    )

    acceptance = load_json(
        IMPLEMENTATION_ACCEPTANCE_PATH
    )

    require(
        original_preflight[
            "phase4fr3_preflight_status"
        ]
        == "ready_for_development_tuning",
        "Original preflight did not otherwise pass.",
    )

    require(
        "/privileged/"
        in original_preflight[
            "train_trajectory_index_path"
        ],
        (
            "Original preflight did not exhibit "
            "the expected privileged-path issue."
        ),
    )

    require(
        acceptance[
            "phase4er3d_status"
        ] == "implementation_accepted",
        "Implementation acceptance is not frozen.",
    )

    require(
        acceptance[
            "tuning_execution_authorized"
        ] is True,
        "Tuning is not authorized.",
    )

    require(
        acceptance[
            "final_fit_execution_authorized"
        ] is False,
        "Final fitting is already authorized.",
    )

    cell_records = {}

    for cell_id in CELLS:
        print(
            f"Processing {cell_id}"
        )

        field_path = (
            DATA_ROOT
            / "visible"
            / f"{cell_id}_fields.npy"
        )

        manifest_path = (
            DATA_ROOT
            / "visible"
            / f"{cell_id}_sequence_manifest.csv"
        )

        structural_source_path = (
            DATA_ROOT
            / "privileged"
            / f"{cell_id}_trajectory_metadata.csv"
        )

        visible_index_path = (
            VISIBLE_INDEX_DIR
            / f"{cell_id}_trajectory_index.csv"
        )

        for path in (
            field_path,
            manifest_path,
            structural_source_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        fields = np.load(
            field_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        require(
            len(fields.shape) == 5,
            (
                f"Unexpected field-array shape "
                f"for {cell_id}: {fields.shape}"
            ),
        )

        require(
            list(
                fields.shape[-3:]
            ) == [4, 32, 32],
            (
                f"Unexpected spatial shape "
                f"for {cell_id}."
            ),
        )

        point_count_expected = int(
            fields.shape[1]
        )

        manifest_rows, manifest_by_index = (
            load_visible_sequence_manifest(
                manifest_path
            )
        )

        structural_rows, audit = (
            load_and_validate_structural_rows(
                source_path=(
                    structural_source_path
                ),
                manifest_by_index=(
                    manifest_by_index
                ),
                point_count_expected=(
                    point_count_expected
                ),
            )
        )

        write_visible_index(
            visible_index_path,
            structural_rows,
        )

        cell_records[cell_id] = {
            "field_path":
                str(field_path),

            "visible_sequence_manifest_path":
                str(manifest_path),

            "source_structural_metadata_path":
                str(
                    structural_source_path
                ),

            "released_visible_trajectory_index_path":
                str(
                    visible_index_path
                ),

            "released_visible_trajectory_index_sha256":
                sha256_file(
                    visible_index_path
                ),

            "field_point_count":
                point_count_expected,

            "visible_sequence_count":
                len(manifest_rows),

            **audit,
        }

    write_json(
        OUTPUT_DIR
        / "structural_index_release_audit.json",
        cell_records,
    )

    summary = {
        "phase":
            (
                "4F-R3A-R1 Tier C v4 controlled "
                "structural-index derivation and "
                "tuning-preflight repair"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "original_preflight_path":
            str(
                ORIGINAL_PREFLIGHT_PATH
            ),

        "original_preflight_status":
            (
                "superseded_due_to_incorrect_"
                "privileged_data_read_claim"
            ),

        "original_preflight_training_executed":
            False,

        "original_preflight_optimizer_steps":
            0,

        "original_preflight_scientific_metrics_computed":
            False,

        "controlled_structural_release_performed":
            True,

        "released_cells":
            list(CELLS),

        "approved_structural_columns":
            list(SAFE_COLUMNS),

        "privileged_container_files_accessed":
            [
                cell_records[cell][
                    "source_structural_metadata_path"
                ]
                for cell in CELLS
            ],

        "privileged_container_access_acknowledged":
            True,

        "privileged_metadata_tables_read":
            True,

        "privileged_scientific_columns_present_in_source":
            True,

        "privileged_scientific_columns_exported":
            False,

        "privileged_scientific_columns_used_for_training":
            False,

        "clean_targets_exported":
            False,

        "privileged_state_ids_exported":
            False,

        "scientific_outcomes_exported":
            False,

        "released_indexes_contain_only_structural_fields":
            True,

        "training_must_use_released_visible_indexes":
            True,

        "training_may_not_read_privileged_directory":
            True,

        "train_visible_trajectory_index":
            cell_records[
                "train_joint"
            ][
                "released_visible_trajectory_index_path"
            ],

        "val_joint_visible_trajectory_index":
            cell_records[
                "val_joint"
            ][
                "released_visible_trajectory_index_path"
            ],

        "train_visible_trajectory_index_sha256":
            cell_records[
                "train_joint"
            ][
                "released_visible_trajectory_index_sha256"
            ],

        "val_joint_visible_trajectory_index_sha256":
            cell_records[
                "val_joint"
            ][
                "released_visible_trajectory_index_sha256"
            ],

        "train_point_coverage_valid":
            True,

        "val_joint_point_coverage_valid":
            True,

        "train_manifest_consistency_valid":
            True,

        "val_joint_manifest_consistency_valid":
            True,

        "configuration_count":
            original_preflight[
                "configuration_count"
            ],

        "configuration_counts":
            original_preflight[
                "configuration_counts"
            ],

        "physical_microbatch_size":
            original_preflight[
                "physical_microbatch_size"
            ],

        "gradient_accumulation_steps":
            original_preflight[
                "gradient_accumulation_steps"
            ],

        "effective_batch_size":
            original_preflight[
                "effective_batch_size"
            ],

        "model_parameters_initialized":
            False,

        "training_batches_read":
            False,

        "validation_batches_read":
            False,

        "optimizer_steps_performed":
            0,

        "checkpoints_created":
            False,

        "scientific_metrics_computed":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_open_count":
            0,

        "tuning_execution_authorized":
            True,

        "final_fit_execution_authorized":
            False,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "phase4fr3ar1_status":
            (
                "ready_for_development_tuning_"
                "with_derived_visible_structural_indexes"
            ),

        "next_phase":
            (
                "4F-R3B Tier C v4 resumable "
                "43-configuration development tuning"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4fr3ar1_repaired_tuning_preflight_summary.json",
        summary,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "Phase 4F-R3A-R1 derived structural-index "
        "release passed."
    )


if __name__ == "__main__":
    main()
