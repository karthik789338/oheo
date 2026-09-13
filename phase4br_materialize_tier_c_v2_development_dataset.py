from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PHASE3B_DIR = Path(
    "outputs/phase3b_tier_b_data"
)

PHASE4AR_DIR = Path(
    "outputs/phase4ar_tier_c_v2_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase4br_tier_c_v2_development_data"
)

VISIBLE_DIR = OUTPUT_DIR / "visible"
PRIVILEGED_DIR = OUTPUT_DIR / "privileged"
MANIFEST_DIR = OUTPUT_DIR / "cell_manifests"


PROTOCOL_VERSION = "tier_c_v2"

FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32
STATE_COUNT = 8

NOISE_SEED = 51030
DEVELOPMENT_MANIFEST_SEED = 51032

NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

DEVELOPMENT_CELLS = (
    "train_joint",
    "val_composition",
    "val_carrier",
    "val_joint",
)

SEALED_TEST_CELLS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)

EXPECTED_SEQUENCE_COUNTS = {
    "train_joint": 1376,
    "val_composition": 144,
    "val_carrier": 144,
    "val_joint": 144,
}

EXPECTED_TRAJECTORIES_PER_SEQUENCE = 8

SEQUENCE_ID_ALIASES = (
    "sequence_id",
    "sequence_index",
    "balanced_sequence_id",
    "sequence_uid",
)

TRAJECTORY_ID_ALIASES = (
    "trajectory_index",
    "trajectory_id",
)

INITIAL_STATE_ALIASES = (
    "initial_state",
    "start_state",
    "state_0",
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cells",
        default="all",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(path, rows):
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with Path(path).open(
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


def stable_seed(
    label,
):
    payload = (
        f"{DEVELOPMENT_MANIFEST_SEED}|"
        f"{label}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def stable_noise_seed(
    cell_id,
    noise,
):
    payload = (
        f"{NOISE_SEED}|"
        f"{cell_id}|"
        f"{noise:g}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def find_column(
    rows,
    aliases,
    description,
):
    if not rows:
        raise ValueError(
            f"No rows available for {description}."
        )

    available = set(
        rows[0]
    )

    for alias in aliases:
        if alias in available:
            return alias

    raise KeyError(
        f"Could not find {description}. "
        f"Available columns: {sorted(available)}"
    )


def parse_json_integer_list(value):
    parsed = json.loads(
        str(value)
    )

    return tuple(
        int(item)
        for item in parsed
    )


def validate_sources():
    phase4ar = load_json(
        PHASE4AR_DIR
        / "phase4ar_summary.json"
    )

    field_bank = load_json(
        OUTPUT_DIR
        / "phase4br_field_bank_summary.json"
    )

    sealing_policy = load_json(
        PHASE4AR_DIR
        / "test_sealing_policy.json"
    )

    if (
        phase4ar["phase4ar_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A-R is not frozen."
        )

    if (
        field_bank[
            "field_bank_generation_status"
        ]
        != "completed"
    ):
        raise AssertionError(
            "Tier C v2 development field bank "
            "is incomplete."
        )

    if (
        field_bank[
            "test_fields_generated"
        ]
        is not False
    ):
        raise AssertionError(
            "Test fields were generated prematurely."
        )

    if (
        sealing_policy[
            "test_trajectory_manifests_generated_before_checkpoint_freeze"
        ]
        is not False
    ):
        raise AssertionError(
            "Test-manifest sealing policy changed."
        )

    return phase4ar, field_bank


def parse_requested_cells(value):
    if value == "all":
        return list(
            DEVELOPMENT_CELLS
        )

    requested = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    unknown = [
        item
        for item in requested
        if item not in DEVELOPMENT_CELLS
    ]

    if unknown:
        raise ValueError(
            "Unknown development cells: "
            + ", ".join(unknown)
        )

    if not requested:
        raise ValueError(
            "No development cells requested."
        )

    return requested


def load_field_bank():
    carrier_ids = np.load(
        PRIVILEGED_DIR
        / "development_carrier_ids.npy"
    )

    carrier_splits = np.load(
        PRIVILEGED_DIR
        / "development_carrier_splits.npy"
    ).astype(str)

    raw_bank = np.load(
        PRIVILEGED_DIR
        / "development_carrier_state_fields_raw.npy",
        mmap_mode="r",
    )

    normalized_bank = np.load(
        PRIVILEGED_DIR
        / "development_carrier_state_fields_normalized.npy",
        mmap_mode="r",
    )

    expected_shape = (
        128,
        STATE_COUNT,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    if raw_bank.shape != expected_shape:
        raise AssertionError(
            f"Raw bank shape changed: {raw_bank.shape}"
        )

    if normalized_bank.shape != expected_shape:
        raise AssertionError(
            "Normalized bank shape changed."
        )

    if len(carrier_ids) != 128:
        raise AssertionError(
            "Development carrier-ID count changed."
        )

    lookup = {
        int(carrier_id):
            local_index
        for local_index, carrier_id
        in enumerate(carrier_ids)
    }

    split_ids = {
        "train":
            np.asarray(
                [
                    int(carrier_ids[index])
                    for index in np.where(
                        carrier_splits
                        == "train"
                    )[0]
                ],
                dtype=np.int64,
            ),

        "val":
            np.asarray(
                [
                    int(carrier_ids[index])
                    for index in np.where(
                        carrier_splits
                        == "val"
                    )[0]
                ],
                dtype=np.int64,
            ),
    }

    if len(split_ids["train"]) != 96:
        raise AssertionError(
            "Training carrier count changed."
        )

    if len(split_ids["val"]) != 32:
        raise AssertionError(
            "Validation carrier count changed."
        )

    return {
        "carrier_ids":
            carrier_ids,

        "carrier_splits":
            carrier_splits,

        "carrier_lookup":
            lookup,

        "split_ids":
            split_ids,

        "raw_bank":
            raw_bank,

        "normalized_bank":
            normalized_bank,
    }


def load_normalization():
    value = load_json(
        OUTPUT_DIR
        / "normalization_v2.json"
    )

    mean = np.asarray(
        value["channel_mean"],
        dtype=np.float32,
    )

    standard_deviation = np.asarray(
        value[
            "channel_standard_deviation"
        ],
        dtype=np.float32,
    )

    if mean.shape != (
        FIELD_CHANNEL_COUNT,
    ):
        raise AssertionError(
            "Normalization mean shape changed."
        )

    if standard_deviation.shape != (
        FIELD_CHANNEL_COUNT,
    ):
        raise AssertionError(
            "Normalization scale shape changed."
        )

    return mean, standard_deviation


def load_source_cell(
    cell_id,
):
    sequence_rows = load_csv(
        PHASE3B_DIR
        / "visible"
        / f"{cell_id}_sequence_manifest.csv"
    )

    trajectory_rows = load_csv(
        PHASE3B_DIR
        / "visible"
        / f"{cell_id}_trajectory_index.csv"
    )

    privileged_rows = load_csv(
        PHASE3B_DIR
        / "privileged"
        / f"{cell_id}_trajectory_metadata.csv"
    )

    sequence_id_column = find_column(
        sequence_rows,
        SEQUENCE_ID_ALIASES,
        "sequence ID in sequence manifest",
    )

    trajectory_id_column = find_column(
        trajectory_rows,
        TRAJECTORY_ID_ALIASES,
        "trajectory ID in trajectory index",
    )

    privileged_trajectory_id_column = (
        find_column(
            privileged_rows,
            TRAJECTORY_ID_ALIASES,
            "trajectory ID in privileged metadata",
        )
    )

    trajectory_sequence_column = find_column(
        trajectory_rows,
        SEQUENCE_ID_ALIASES,
        "sequence ID in trajectory index",
    )

    privileged_lookup = {
        int(
            row[
                privileged_trajectory_id_column
            ]
        ):
            row
        for row in privileged_rows
    }

    grouped = defaultdict(list)

    for trajectory_row in trajectory_rows:
        trajectory_id = int(
            trajectory_row[
                trajectory_id_column
            ]
        )

        sequence_id = str(
            trajectory_row[
                trajectory_sequence_column
            ]
        )

        privileged = privileged_lookup[
            trajectory_id
        ]

        state_path = (
            parse_json_integer_list(
                privileged["state_path"]
            )
        )

        initial_state = int(
            state_path[0]
        )

        grouped[
            sequence_id
        ].append(
            {
                "source_trajectory_id":
                    trajectory_id,

                "initial_state":
                    initial_state,

                "state_path":
                    state_path,
            }
        )

    sequence_lookup = {
        str(
            row[
                sequence_id_column
            ]
        ):
            row
        for row in sequence_rows
    }

    if set(sequence_lookup) != set(grouped):
        raise AssertionError(
            f"{cell_id} source sequence IDs differ "
            "between manifests."
        )

    for sequence_id, trajectories in (
        grouped.items()
    ):
        trajectories.sort(
            key=lambda row:
                row["initial_state"]
        )

        if len(trajectories) != 8:
            raise AssertionError(
                f"Source sequence {sequence_id} "
                "does not contain eight trajectories."
            )

        if [
            row["initial_state"]
            for row in trajectories
        ] != list(range(8)):
            raise AssertionError(
                f"Source sequence {sequence_id} "
                "does not cover initial states 0-7."
            )

    ordered_sequence_ids = [
        str(
            row[
                sequence_id_column
            ]
        )
        for row in sequence_rows
    ]

    return {
        "sequence_rows":
            sequence_rows,

        "sequence_lookup":
            sequence_lookup,

        "sequence_id_column":
            sequence_id_column,

        "ordered_sequence_ids":
            ordered_sequence_ids,

        "trajectory_groups":
            grouped,
    }


def choose_source_sequences():
    train_source = load_source_cell(
        "train_joint"
    )

    validation_source = load_source_cell(
        "val_composition"
    )

    if (
        len(
            train_source[
                "ordered_sequence_ids"
            ]
        )
        != 1376
    ):
        raise AssertionError(
            "Training sequence count changed."
        )

    if (
        len(
            validation_source[
                "ordered_sequence_ids"
            ]
        )
        != 144
    ):
        raise AssertionError(
            "Validation-composition sequence "
            "count changed."
        )

    generator = np.random.default_rng(
        stable_seed(
            "val_carrier_sequence_subset"
        )
    )

    val_carrier_ids = generator.choice(
        np.asarray(
            train_source[
                "ordered_sequence_ids"
            ],
            dtype=object,
        ),
        size=144,
        replace=False,
    ).tolist()

    return {
        "train_joint": {
            "source":
                train_source,

            "sequence_ids":
                train_source[
                    "ordered_sequence_ids"
                ],

            "carrier_split":
                "train",
        },

        "val_composition": {
            "source":
                validation_source,

            "sequence_ids":
                validation_source[
                    "ordered_sequence_ids"
                ],

            "carrier_split":
                "train",
        },

        "val_carrier": {
            "source":
                train_source,

            "sequence_ids":
                val_carrier_ids,

            "carrier_split":
                "val",
        },

        "val_joint": {
            "source":
                validation_source,

            "sequence_ids":
                validation_source[
                    "ordered_sequence_ids"
                ],

            "carrier_split":
                "val",
        },
    }


def build_fresh_manifests(
    cell_id,
    cell_specification,
    split_carrier_ids,
):
    source = cell_specification[
        "source"
    ]

    sequence_ids = cell_specification[
        "sequence_ids"
    ]

    expected_count = (
        EXPECTED_SEQUENCE_COUNTS[
            cell_id
        ]
    )

    if len(sequence_ids) != expected_count:
        raise AssertionError(
            f"{cell_id} sequence count changed."
        )

    allowed_carriers = split_carrier_ids[
        cell_specification[
            "carrier_split"
        ]
    ]

    generator = np.random.default_rng(
        stable_seed(
            f"carrier_pairing|{cell_id}"
        )
    )

    sequence_manifest_rows = []
    visible_trajectory_rows = []
    privileged_trajectory_rows = []

    point_bank_indices = []

    trajectory_index = 0
    point_start = 0

    for cell_sequence_index, sequence_id in enumerate(
        sequence_ids
    ):
        source_sequence_row = dict(
            source[
                "sequence_lookup"
            ][str(sequence_id)]
        )

        source_sequence_row[
            "cell_sequence_index"
        ] = cell_sequence_index

        source_sequence_row[
            "source_sequence_id"
        ] = str(sequence_id)

        source_sequence_row[
            "development_manifest_seed"
        ] = DEVELOPMENT_MANIFEST_SEED

        source_sequence_row[
            "tier_c_v2_cell_id"
        ] = cell_id

        sequence_manifest_rows.append(
            source_sequence_row
        )

        selected_carriers = (
            generator.choice(
                allowed_carriers,
                size=8,
                replace=False,
            )
        )

        canonical_trajectories = (
            source[
                "trajectory_groups"
            ][str(sequence_id)]
        )

        for trajectory_offset, canonical in enumerate(
            canonical_trajectories
        ):
            carrier_id = int(
                selected_carriers[
                    trajectory_offset
                ]
            )

            state_path = tuple(
                canonical[
                    "state_path"
                ]
            )

            point_count = len(
                state_path
            )

            visible_trajectory_rows.append(
                {
                    "trajectory_index":
                        trajectory_index,

                    "cell_sequence_index":
                        cell_sequence_index,

                    "source_sequence_id":
                        str(sequence_id),

                    "point_start":
                        point_start,

                    "point_count":
                        point_count,

                    "sequence_length":
                        point_count - 1,
                }
            )

            privileged_trajectory_rows.append(
                {
                    "trajectory_index":
                        trajectory_index,

                    "cell_sequence_index":
                        cell_sequence_index,

                    "source_sequence_id":
                        str(sequence_id),

                    "source_trajectory_id":
                        canonical[
                            "source_trajectory_id"
                        ],

                    "carrier_id":
                        carrier_id,

                    "carrier_split":
                        cell_specification[
                            "carrier_split"
                        ],

                    "initial_state":
                        canonical[
                            "initial_state"
                        ],

                    "state_path":
                        json.dumps(
                            list(state_path)
                        ),
                }
            )

            for state in state_path:
                point_bank_indices.append(
                    (
                        carrier_id,
                        int(state),
                    )
                )

            point_start += point_count
            trajectory_index += 1

    expected_trajectory_count = (
        expected_count
        * EXPECTED_TRAJECTORIES_PER_SEQUENCE
    )

    if (
        len(visible_trajectory_rows)
        != expected_trajectory_count
    ):
        raise AssertionError(
            f"{cell_id} trajectory count changed."
        )

    return {
        "sequence_manifest_rows":
            sequence_manifest_rows,

        "visible_trajectory_rows":
            visible_trajectory_rows,

        "privileged_trajectory_rows":
            privileged_trajectory_rows,

        "carrier_state_pairs":
            point_bank_indices,

        "point_count":
            point_start,
    }


def convert_pairs_to_bank_indices(
    carrier_state_pairs,
    carrier_lookup,
):
    indices = np.empty(
        len(carrier_state_pairs),
        dtype=np.int32,
    )

    for index, (
        carrier_id,
        state_id,
    ) in enumerate(
        carrier_state_pairs
    ):
        local_carrier_index = (
            carrier_lookup[
                int(carrier_id)
            ]
        )

        indices[index] = (
            local_carrier_index
            * STATE_COUNT
            + int(state_id)
        )

    return indices


def postprocess_measurement_noise(
    raw_fields,
):
    raw_fields[:, 0] = np.clip(
        raw_fields[:, 0],
        -1.0,
        1.0,
    )

    orientation_norm = np.sqrt(
        raw_fields[:, 1] ** 2
        + raw_fields[:, 2] ** 2
    )

    orientation_norm = np.maximum(
        orientation_norm,
        1e-6,
    )

    raw_fields[:, 1] /= (
        orientation_norm
    )

    raw_fields[:, 2] /= (
        orientation_norm
    )

    raw_fields[:, 3] = np.clip(
        raw_fields[:, 3],
        0.0,
        1.0,
    )

    return raw_fields


def materialize_cell(
    cell_id,
    cell_specification,
    field_data,
    normalization_mean,
    normalization_standard_deviation,
    chunk_size,
    force,
):
    marker_path = (
        MANIFEST_DIR
        / f"{cell_id}_complete.json"
    )

    field_path = (
        VISIBLE_DIR
        / f"{cell_id}_fields.npy"
    )

    if marker_path.exists() and not force:
        marker = load_json(
            marker_path
        )

        if (
            marker.get("status")
            == "completed"
            and field_path.exists()
        ):
            print(
                f"{cell_id} already completed."
            )

            return marker

    manifests = build_fresh_manifests(
        cell_id=cell_id,
        cell_specification=(
            cell_specification
        ),
        split_carrier_ids=(
            field_data[
                "split_ids"
            ]
        ),
    )

    point_bank_indices = (
        convert_pairs_to_bank_indices(
            carrier_state_pairs=(
                manifests[
                    "carrier_state_pairs"
                ]
            ),
            carrier_lookup=(
                field_data[
                    "carrier_lookup"
                ]
            ),
        )
    )

    visible_trajectory_path = (
        VISIBLE_DIR
        / f"{cell_id}_trajectory_index.csv"
    )

    visible_sequence_path = (
        VISIBLE_DIR
        / f"{cell_id}_sequence_manifest.csv"
    )

    privileged_trajectory_path = (
        PRIVILEGED_DIR
        / f"{cell_id}_trajectory_metadata.csv"
    )

    point_index_path = (
        PRIVILEGED_DIR
        / f"{cell_id}_point_bank_indices.npy"
    )

    write_csv(
        visible_trajectory_path,
        manifests[
            "visible_trajectory_rows"
        ],
    )

    write_csv(
        visible_sequence_path,
        manifests[
            "sequence_manifest_rows"
        ],
    )

    write_csv(
        privileged_trajectory_path,
        manifests[
            "privileged_trajectory_rows"
        ],
    )

    np.save(
        point_index_path,
        point_bank_indices,
    )

    point_count = int(
        manifests["point_count"]
    )

    stored_shape = (
        len(NOISE_LEVELS),
        point_count,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    target = np.lib.format.open_memmap(
        field_path,
        mode="w+",
        dtype=np.float16,
        shape=stored_shape,
    )

    raw_flat = (
        field_data[
            "raw_bank"
        ].reshape(
            -1,
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        )
    )

    normalized_flat = (
        field_data[
            "normalized_bank"
        ].reshape(
            -1,
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        )
    )

    mean = normalization_mean[
        None,
        :,
        None,
        None,
    ]

    standard_deviation = (
        normalization_standard_deviation[
            None,
            :,
            None,
            None,
        ]
    )

    generators = {
        noise:
            (
                None
                if noise == 0.0
                else np.random.default_rng(
                    stable_noise_seed(
                        cell_id,
                        noise,
                    )
                )
            )
        for noise in NOISE_LEVELS
    }

    for start in range(
        0,
        point_count,
        chunk_size,
    ):
        end = min(
            point_count,
            start + chunk_size,
        )

        selected_indices = (
            point_bank_indices[
                start:end
            ]
        )

        raw_clean = np.asarray(
            raw_flat[
                selected_indices
            ],
            dtype=np.float32,
        )

        normalized_clean = np.asarray(
            normalized_flat[
                selected_indices
            ],
            dtype=np.float32,
        )

        for noise_index, noise in enumerate(
            NOISE_LEVELS
        ):
            if noise == 0.0:
                values = normalized_clean

            else:
                epsilon = generators[
                    noise
                ].standard_normal(
                    size=raw_clean.shape
                ).astype(
                    np.float32
                )

                noisy_raw = (
                    raw_clean
                    + float(noise)
                    * standard_deviation
                    * epsilon
                )

                noisy_raw = (
                    postprocess_measurement_noise(
                        noisy_raw
                    )
                )

                values = (
                    noisy_raw - mean
                ) / standard_deviation

            if not np.isfinite(
                values
            ).all():
                raise FloatingPointError(
                    f"{cell_id} contains non-finite "
                    f"values at noise {noise}."
                )

            target[
                noise_index,
                start:end,
            ] = values.astype(
                np.float16
            )

        if (
            start == 0
            or end == point_count
            or end % 4096 == 0
        ):
            print(
                f"  {cell_id}: "
                f"{end}/{point_count} points"
            )

    target.flush()
    del target

    verification = np.load(
        field_path,
        mmap_mode="r",
    )

    if verification.shape != stored_shape:
        raise AssertionError(
            f"{cell_id} stored shape changed."
        )

    if verification.dtype != np.float16:
        raise AssertionError(
            f"{cell_id} dtype changed."
        )

    del verification

    marker = {
        "phase":
            "4B-R Tier C v2 development cell",

        "protocol_version":
            PROTOCOL_VERSION,

        "cell_id":
            cell_id,

        "sequence_count":
            len(
                manifests[
                    "sequence_manifest_rows"
                ]
            ),

        "trajectory_count":
            len(
                manifests[
                    "visible_trajectory_rows"
                ]
            ),

        "point_count":
            point_count,

        "carrier_split":
            cell_specification[
                "carrier_split"
            ],

        "noise_levels":
            list(NOISE_LEVELS),

        "stored_shape":
            list(stored_shape),

        "stored_dtype":
            "float16",

        "field_path":
            str(field_path),

        "file_size_bytes":
            os.path.getsize(
                field_path
            ),

        "fresh_carrier_pairing":
            True,

        "development_manifest_seed":
            DEVELOPMENT_MANIFEST_SEED,

        "test_manifest_seed_used":
            False,

        "test_carriers_used":
            False,

        "test_metrics_computed":
            False,

        "status":
            "completed",
    }

    write_json(
        marker_path,
        marker,
    )

    return marker


def verify_test_files_absent():
    forbidden_names = []

    for directory in (
        VISIBLE_DIR,
        PRIVILEGED_DIR,
        MANIFEST_DIR,
    ):
        if not directory.exists():
            continue

        for path in directory.iterdir():
            if any(
                cell_id in path.name
                for cell_id in SEALED_TEST_CELLS
            ):
                forbidden_names.append(
                    str(path)
                )

    return forbidden_names


def finalize_phase():
    markers = []

    for cell_id in DEVELOPMENT_CELLS:
        marker_path = (
            MANIFEST_DIR
            / f"{cell_id}_complete.json"
        )

        if not marker_path.exists():
            return None

        marker = load_json(
            marker_path
        )

        if marker["status"] != "completed":
            return None

        markers.append(marker)

    forbidden_test_files = (
        verify_test_files_absent()
    )

    if forbidden_test_files:
        raise AssertionError(
            "Sealed test files were generated: "
            + ", ".join(
                forbidden_test_files
            )
        )

    total_visible_storage_bytes = sum(
        int(marker["file_size_bytes"])
        for marker in markers
    )

    summary = {
        "phase":
            "4B-R Tier C v2 development dataset generation",

        "protocol_version":
            PROTOCOL_VERSION,

        "development_cell_count":
            len(markers),

        "development_cells": [
            marker["cell_id"]
            for marker in markers
        ],

        "training_sequence_count":
            next(
                marker["sequence_count"]
                for marker in markers
                if marker["cell_id"]
                == "train_joint"
            ),

        "validation_sequence_count_per_cell":
            144,

        "total_development_trajectory_count":
            sum(
                int(
                    marker[
                        "trajectory_count"
                    ]
                )
                for marker in markers
            ),

        "total_development_point_count":
            sum(
                int(
                    marker[
                        "point_count"
                    ]
                )
                for marker in markers
            ),

        "noise_levels":
            list(NOISE_LEVELS),

        "visible_storage_dtype":
            "float16",

        "total_visible_storage_bytes":
            total_visible_storage_bytes,

        "total_visible_storage_gb":
            total_visible_storage_bytes
            / 1e9,

        "fresh_development_manifests_generated":
            True,

        "fresh_carrier_pairings_generated":
            True,

        "test_cell_count_generated":
            0,

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "sealed_test_files_absent":
            True,

        "test_metrics_computed":
            False,

        "training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "tier_c_v1_outputs_modified":
            False,

        "phase2_outputs_modified":
            False,

        "phase3_outputs_modified":
            False,

        "phase4br_status":
            "completed",
    }

    write_json(
        OUTPUT_DIR
        / "phase4br_summary.json",
        summary,
    )

    return summary


def main():
    arguments = parse_arguments()

    if arguments.chunk_size <= 0:
        raise ValueError(
            "Chunk size must be positive."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VISIBLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PRIVILEGED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_sources()

    requested_cells = (
        parse_requested_cells(
            arguments.cells
        )
    )

    field_data = load_field_bank()

    (
        normalization_mean,
        normalization_standard_deviation,
    ) = load_normalization()

    specifications = (
        choose_source_sequences()
    )

    np.save(
        VISIBLE_DIR
        / "noise_levels.npy",
        np.asarray(
            NOISE_LEVELS,
            dtype=np.float32,
        ),
    )

    for index, cell_id in enumerate(
        requested_cells,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(requested_cells)}] "
            f"Materializing {cell_id}"
        )

        materialize_cell(
            cell_id=cell_id,
            cell_specification=(
                specifications[
                    cell_id
                ]
            ),
            field_data=field_data,
            normalization_mean=(
                normalization_mean
            ),
            normalization_standard_deviation=(
                normalization_standard_deviation
            ),
            chunk_size=(
                arguments.chunk_size
            ),
            force=arguments.force,
        )

    summary = finalize_phase()

    if summary is None:
        completed_count = sum(
            (
                MANIFEST_DIR
                / f"{cell_id}_complete.json"
            ).exists()
            for cell_id in DEVELOPMENT_CELLS
        )

        print()
        print(
            "Phase 4B-R progress: "
            f"{completed_count}/4 development "
            "cells completed."
        )

    else:
        print()
        print(
            "Phase 4B-R Tier C v2 development "
            "dataset generation completed."
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
