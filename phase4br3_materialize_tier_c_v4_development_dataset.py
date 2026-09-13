from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


PHASE3B_DIR = Path(
    "outputs/phase3b_tier_b_data"
)

PHASE4AR3_DIR = Path(
    "outputs/phase4ar3_tier_c_v4_protocol"
)

FIELD_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_fields"
)

OUTPUT_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_data"
)

VISIBLE_DIR = OUTPUT_DIR / "visible"
PRIVILEGED_DIR = OUTPUT_DIR / "privileged"
MANIFEST_DIR = OUTPUT_DIR / "cell_manifests"


PROTOCOL_VERSION = "tier_c_v4"

FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32
STATE_COUNT = 8

DEVELOPMENT_CARRIER_COUNT = 128
TRAIN_CARRIER_COUNT = 96
VALIDATION_CARRIER_COUNT = 32

NOISE_SEED = 71030
DEVELOPMENT_MANIFEST_SEED = 71032

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

TRAJECTORIES_PER_SEQUENCE = 8

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

STATE_PATH_ALIASES = (
    "state_path",
    "state_path_json",
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cells",
        default="all",
        help=(
            "Comma-separated development cells, "
            "or 'all'."
        ),
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


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def stable_seed(label):
    digest = hashlib.sha256(
        (
            f"{DEVELOPMENT_MANIFEST_SEED}|"
            f"{label}"
        ).encode("utf-8")
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
    digest = hashlib.sha256(
        (
            f"{NOISE_SEED}|"
            f"{cell_id}|"
            f"{noise:g}"
        ).encode("utf-8")
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

    available = set(rows[0])

    for alias in aliases:
        if alias in available:
            return alias

    raise KeyError(
        f"Could not find {description}. "
        f"Available columns: {sorted(available)}"
    )


def parse_json_integer_list(value):
    parsed = json.loads(str(value))

    return tuple(
        int(item)
        for item in parsed
    )


def parse_requested_cells(value):
    if value == "all":
        return list(DEVELOPMENT_CELLS)

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


def validate_sources():
    protocol_summary = load_json(
        PHASE4AR3_DIR
        / "phase4ar3_summary.json"
    )

    field_summary = load_json(
        FIELD_DIR
        / "phase4br3_field_summary.json"
    )

    independent_morphology = load_json(
        FIELD_DIR
        / "independent_morphology_summary.json"
    )

    sealing_policy = load_json(
        PHASE4AR3_DIR
        / "test_sealing_policy.json"
    )

    terminal_policy = load_json(
        PHASE4AR3_DIR
        / "terminal_revision_policy.json"
    )

    if (
        protocol_summary["phase4ar3_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A-R3 is not frozen."
        )

    if (
        protocol_summary["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Tier C v4 protocol version changed."
        )

    if (
        field_summary["phase4br3_status"]
        != "completed_and_verified"
    ):
        raise AssertionError(
            "Tier C v4 field construction did not "
            "complete successfully."
        )

    if (
        field_summary[
            "trajectory_materialization_authorized"
        ]
        is not True
    ):
        raise AssertionError(
            "Trajectory materialization is not authorized."
        )

    if (
        field_summary[
            "all_pairs_found_valid_iteration"
        ]
        is not True
    ):
        raise AssertionError(
            "Not every Tier C v4 pair found a "
            "valid construction iteration."
        )

    if (
        field_summary[
            "all_internal_targets_passed"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v4 internal construction targets failed."
        )

    if (
        field_summary["physics_gate_passed"]
        is not True
    ):
        raise AssertionError(
            "Tier C v4 numerical physics gate failed."
        )

    if (
        field_summary[
            "independent_morphology_gate_passed"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v4 independent morphology gate failed."
        )

    if (
        independent_morphology[
            "independent_morphology_gate_passed"
        ]
        is not True
    ):
        raise AssertionError(
            "Independent morphology summary did not pass."
        )

    if independent_morphology["failed_checks"]:
        raise AssertionError(
            "Independent morphology has failed checks: "
            f"{independent_morphology['failed_checks']}"
        )

    if (
        field_summary[
            "test_carrier_count_generated"
        ]
        != 0
    ):
        raise AssertionError(
            "Tier C v4 test carriers were generated."
        )

    if (
        field_summary[
            "test_carrier_parameters_generated"
        ]
        is not False
        or field_summary[
            "test_fields_generated"
        ]
        is not False
        or field_summary[
            "test_manifests_generated"
        ]
        is not False
        or field_summary[
            "test_metrics_computed"
        ]
        is not False
    ):
        raise AssertionError(
            "Tier C v4 test-sealing policy was violated."
        )

    if (
        field_summary[
            "diagnostic_decoder_fitted"
        ]
        is not False
    ):
        raise AssertionError(
            "A diagnostic decoder was fitted before "
            "trajectory materialization."
        )

    if (
        field_summary[
            "predictive_training_performed"
        ]
        is not False
        or field_summary[
            "checkpoint_selection_performed"
        ]
        is not False
    ):
        raise AssertionError(
            "Predictive training occurred before "
            "trajectory materialization."
        )

    if (
        sealing_policy[
            "test_trajectory_manifests_generated_before_checkpoint_freeze"
        ]
        is not False
    ):
        raise AssertionError(
            "Tier C v4 test-sealing policy changed."
        )

    if (
        terminal_policy["tier_c_v5_authorized"]
        is not False
    ):
        raise AssertionError(
            "Tier C v5 was unexpectedly authorized."
        )

    return {
        "protocol_summary":
            protocol_summary,

        "field_summary":
            field_summary,

        "independent_morphology":
            independent_morphology,

        "sealing_policy":
            sealing_policy,

        "terminal_policy":
            terminal_policy,
    }


def load_field_bank():
    carrier_ids = np.load(
        FIELD_DIR
        / "privileged"
        / "development_carrier_ids.npy"
    ).astype(np.int64)

    carrier_splits = np.load(
        FIELD_DIR
        / "privileged"
        / "development_carrier_splits.npy"
    ).astype(str)

    raw_bank = np.load(
        FIELD_DIR
        / "privileged"
        / "development_carrier_state_fields_raw.npy",
        mmap_mode="r",
    )

    normalized_bank = np.load(
        FIELD_DIR
        / "privileged"
        / "development_carrier_state_fields_normalized.npy",
        mmap_mode="r",
    )

    expected_shape = (
        DEVELOPMENT_CARRIER_COUNT,
        STATE_COUNT,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    if raw_bank.shape != expected_shape:
        raise AssertionError(
            f"Raw Tier C v4 field shape changed: "
            f"{raw_bank.shape}"
        )

    if normalized_bank.shape != expected_shape:
        raise AssertionError(
            "Normalized Tier C v4 field shape changed."
        )

    if raw_bank.dtype != np.float32:
        raise AssertionError(
            "Raw Tier C v4 field bank is not float32."
        )

    if normalized_bank.dtype != np.float32:
        raise AssertionError(
            "Normalized Tier C v4 field bank is not float32."
        )

    if len(set(carrier_ids.tolist())) != (
        DEVELOPMENT_CARRIER_COUNT
    ):
        raise AssertionError(
            "Tier C v4 development carrier IDs "
            "are not unique."
        )

    carrier_lookup = {
        int(carrier_id):
            int(local_index)
        for local_index, carrier_id
        in enumerate(carrier_ids)
    }

    split_ids = {
        "train":
            np.asarray(
                [
                    int(carrier_ids[index])
                    for index in np.where(
                        carrier_splits == "train"
                    )[0]
                ],
                dtype=np.int64,
            ),

        "val":
            np.asarray(
                [
                    int(carrier_ids[index])
                    for index in np.where(
                        carrier_splits == "val"
                    )[0]
                ],
                dtype=np.int64,
            ),
    }

    if len(split_ids["train"]) != (
        TRAIN_CARRIER_COUNT
    ):
        raise AssertionError(
            "Tier C v4 training carrier count changed."
        )

    if len(split_ids["val"]) != (
        VALIDATION_CARRIER_COUNT
    ):
        raise AssertionError(
            "Tier C v4 validation carrier count changed."
        )

    return {
        "carrier_ids":
            carrier_ids,

        "carrier_splits":
            carrier_splits,

        "carrier_lookup":
            carrier_lookup,

        "split_ids":
            split_ids,

        "raw_bank":
            raw_bank,

        "normalized_bank":
            normalized_bank,
    }


def load_normalization():
    value = load_json(
        FIELD_DIR
        / "normalization_v4.json"
    )

    if (
        value["statistics_source"]
        != "clean training-carrier fields only"
    ):
        raise AssertionError(
            "Tier C v4 normalization source changed."
        )

    if (
        value["validation_statistics_used"]
        is not False
        or value["test_statistics_used"]
        is not False
    ):
        raise AssertionError(
            "Validation or test statistics were used "
            "for normalization."
        )

    mean = np.asarray(
        value["channel_mean"],
        dtype=np.float32,
    )

    standard_deviation = np.asarray(
        value["channel_standard_deviation"],
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

    if np.any(
        standard_deviation <= 0.0
    ):
        raise AssertionError(
            "Invalid Tier C v4 normalization scale."
        )

    return mean, standard_deviation


def load_source_cell(cell_id):
    sequence_path = (
        PHASE3B_DIR
        / "visible"
        / f"{cell_id}_sequence_manifest.csv"
    )

    trajectory_path = (
        PHASE3B_DIR
        / "visible"
        / f"{cell_id}_trajectory_index.csv"
    )

    privileged_path = (
        PHASE3B_DIR
        / "privileged"
        / f"{cell_id}_trajectory_metadata.csv"
    )

    for path in (
        sequence_path,
        trajectory_path,
        privileged_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    sequence_rows = load_csv(
        sequence_path
    )

    trajectory_rows = load_csv(
        trajectory_path
    )

    privileged_rows = load_csv(
        privileged_path
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
            (
                "trajectory ID in privileged "
                "metadata"
            ),
        )
    )

    trajectory_sequence_column = find_column(
        trajectory_rows,
        SEQUENCE_ID_ALIASES,
        "sequence ID in trajectory index",
    )

    state_path_column = find_column(
        privileged_rows,
        STATE_PATH_ALIASES,
        "state path in privileged metadata",
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

        state_path = parse_json_integer_list(
            privileged[
                state_path_column
            ]
        )

        if not state_path:
            raise AssertionError(
                "Encountered an empty state path."
            )

        if any(
            state_id < 0
            or state_id >= STATE_COUNT
            for state_id in state_path
        ):
            raise AssertionError(
                "Source trajectory contains an "
                "invalid state ID."
            )

        grouped[
            sequence_id
        ].append(
            {
                "source_trajectory_id":
                    trajectory_id,

                "initial_state":
                    int(
                        state_path[0]
                    ),

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

        if len(trajectories) != (
            TRAJECTORIES_PER_SEQUENCE
        ):
            raise AssertionError(
                f"Sequence {sequence_id} does not "
                "contain eight trajectories."
            )

        initial_states = [
            row["initial_state"]
            for row in trajectories
        ]

        if initial_states != list(
            range(STATE_COUNT)
        ):
            raise AssertionError(
                f"Sequence {sequence_id} does not "
                "cover initial states 0-7."
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

        "ordered_sequence_ids":
            ordered_sequence_ids,

        "trajectory_groups":
            grouped,

        "source_paths": {
            "sequence_manifest":
                str(sequence_path),

            "trajectory_index":
                str(trajectory_path),

            "trajectory_metadata":
                str(privileged_path),
        },
    }


def choose_source_sequences():
    training_source = load_source_cell(
        "train_joint"
    )

    composition_source = load_source_cell(
        "val_composition"
    )

    if len(
        training_source[
            "ordered_sequence_ids"
        ]
    ) != 1376:
        raise AssertionError(
            "Tier B training sequence count changed."
        )

    if len(
        composition_source[
            "ordered_sequence_ids"
        ]
    ) != 144:
        raise AssertionError(
            "Tier B validation-composition "
            "sequence count changed."
        )

    generator = np.random.default_rng(
        stable_seed(
            "tier_c_v4_val_carrier_sequence_subset"
        )
    )

    validation_carrier_sequence_ids = (
        generator.choice(
            np.asarray(
                training_source[
                    "ordered_sequence_ids"
                ],
                dtype=object,
            ),
            size=144,
            replace=False,
        ).tolist()
    )

    return {
        "train_joint": {
            "source":
                training_source,

            "sequence_ids":
                training_source[
                    "ordered_sequence_ids"
                ],

            "carrier_split":
                "train",
        },

        "val_composition": {
            "source":
                composition_source,

            "sequence_ids":
                composition_source[
                    "ordered_sequence_ids"
                ],

            "carrier_split":
                "train",
        },

        "val_carrier": {
            "source":
                training_source,

            "sequence_ids":
                validation_carrier_sequence_ids,

            "carrier_split":
                "val",
        },

        "val_joint": {
            "source":
                composition_source,

            "sequence_ids":
                composition_source[
                    "ordered_sequence_ids"
                ],

            "carrier_split":
                "val",
        },
    }


def build_fresh_manifests(
    cell_id,
    specification,
    split_carrier_ids,
):
    source = specification["source"]

    sequence_ids = specification[
        "sequence_ids"
    ]

    expected_sequence_count = (
        EXPECTED_SEQUENCE_COUNTS[
            cell_id
        ]
    )

    if len(sequence_ids) != (
        expected_sequence_count
    ):
        raise AssertionError(
            f"{cell_id} sequence count changed."
        )

    carrier_split = specification[
        "carrier_split"
    ]

    allowed_carriers = split_carrier_ids[
        carrier_split
    ]

    if len(allowed_carriers) < (
        TRAJECTORIES_PER_SEQUENCE
    ):
        raise AssertionError(
            "Insufficient carriers for a fresh "
            "eight-trajectory pairing."
        )

    generator = np.random.default_rng(
        stable_seed(
            f"tier_c_v4_carrier_pairing|{cell_id}"
        )
    )

    sequence_manifest_rows = []
    visible_trajectory_rows = []
    privileged_trajectory_rows = []
    carrier_state_pairs = []

    trajectory_index = 0
    point_start = 0

    for cell_sequence_index, sequence_id in (
        enumerate(sequence_ids)
    ):
        source_sequence_row = dict(
            source[
                "sequence_lookup"
            ][str(sequence_id)]
        )

        source_sequence_row.update(
            {
                "cell_sequence_index":
                    cell_sequence_index,

                "source_sequence_id":
                    str(sequence_id),

                "development_manifest_seed":
                    DEVELOPMENT_MANIFEST_SEED,

                "tier_c_v4_cell_id":
                    cell_id,

                "tier_c_v4_protocol_version":
                    PROTOCOL_VERSION,

                "fresh_carrier_pairing":
                    True,
            }
        )

        sequence_manifest_rows.append(
            source_sequence_row
        )

        selected_carriers = generator.choice(
            allowed_carriers,
            size=TRAJECTORIES_PER_SEQUENCE,
            replace=False,
        )

        canonical_trajectories = (
            source[
                "trajectory_groups"
            ][str(sequence_id)]
        )

        for trajectory_offset, canonical in (
            enumerate(canonical_trajectories)
        ):
            carrier_id = int(
                selected_carriers[
                    trajectory_offset
                ]
            )

            state_path = tuple(
                canonical["state_path"]
            )

            point_count = len(state_path)

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
                        carrier_split,

                    "initial_state":
                        canonical[
                            "initial_state"
                        ],

                    "state_path":
                        json.dumps(
                            list(state_path)
                        ),

                    "tier_c_v4_test_carrier":
                        False,
                }
            )

            for state_id in state_path:
                carrier_state_pairs.append(
                    (
                        carrier_id,
                        int(state_id),
                    )
                )

            point_start += point_count
            trajectory_index += 1

    expected_trajectory_count = (
        expected_sequence_count
        * TRAJECTORIES_PER_SEQUENCE
    )

    if len(
        visible_trajectory_rows
    ) != expected_trajectory_count:
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
            carrier_state_pairs,

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
        if carrier_id not in carrier_lookup:
            raise KeyError(
                f"Carrier {carrier_id} is absent "
                "from the development field bank."
            )

        local_carrier_index = (
            carrier_lookup[
                int(carrier_id)
            ]
        )

        if (
            state_id < 0
            or state_id >= STATE_COUNT
        ):
            raise AssertionError(
                f"Invalid state ID: {state_id}"
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


def calculate_point_counts(
    requested_cells,
    specifications,
):
    point_counts = {}

    for cell_id in requested_cells:
        specification = specifications[
            cell_id
        ]

        point_count = 0

        for sequence_id in specification[
            "sequence_ids"
        ]:
            trajectories = specification[
                "source"
            ][
                "trajectory_groups"
            ][str(sequence_id)]

            point_count += sum(
                len(
                    trajectory[
                        "state_path"
                    ]
                )
                for trajectory in trajectories
            )

        point_counts[cell_id] = point_count

    return point_counts


def check_disk_space(
    requested_cells,
    specifications,
):
    point_counts = calculate_point_counts(
        requested_cells=requested_cells,
        specifications=specifications,
    )

    required_bytes = sum(
        point_counts.values()
    ) * (
        len(NOISE_LEVELS)
        * FIELD_CHANNEL_COUNT
        * GRID_HEIGHT
        * GRID_WIDTH
        * np.dtype(np.float16).itemsize
    )

    available_bytes = shutil.disk_usage(
        OUTPUT_DIR
    ).free

    required_with_margin = int(
        required_bytes * 1.10
    )

    if available_bytes < (
        required_with_margin
    ):
        raise RuntimeError(
            "Insufficient free disk space. "
            f"Required with margin: "
            f"{required_with_margin / 1e9:.2f} GB; "
            f"available: "
            f"{available_bytes / 1e9:.2f} GB."
        )

    return {
        "point_counts":
            point_counts,

        "required_visible_bytes":
            required_bytes,

        "required_visible_gb":
            required_bytes / 1e9,

        "available_bytes_before_generation":
            available_bytes,

        "available_gb_before_generation":
            available_bytes / 1e9,

        "safety_margin_fraction":
            0.10,
    }


def materialize_cell(
    cell_id,
    specification,
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
        specification=specification,
        split_carrier_ids=(
            field_data["split_ids"]
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

    write_csv(
        VISIBLE_DIR
        / f"{cell_id}_trajectory_index.csv",
        manifests[
            "visible_trajectory_rows"
        ],
    )

    write_csv(
        VISIBLE_DIR
        / f"{cell_id}_sequence_manifest.csv",
        manifests[
            "sequence_manifest_rows"
        ],
    )

    write_csv(
        PRIVILEGED_DIR
        / f"{cell_id}_trajectory_metadata.csv",
        manifests[
            "privileged_trajectory_rows"
        ],
    )

    np.save(
        PRIVILEGED_DIR
        / f"{cell_id}_point_bank_indices.npy",
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

    raw_flat = field_data[
        "raw_bank"
    ].reshape(
        -1,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    normalized_flat = field_data[
        "normalized_bank"
    ].reshape(
        -1,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
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
                ).astype(np.float32)

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
            f"{cell_id} stored dtype changed."
        )

    if not np.isfinite(
        np.asarray(
            verification[
                :,
                :min(
                    16,
                    point_count,
                ),
            ],
            dtype=np.float32,
        )
    ).all():
        raise FloatingPointError(
            f"{cell_id} stored sample contains "
            "non-finite values."
        )

    del verification

    marker = {
        "phase":
            (
                "4B-R3 Tier C v4 development "
                "cell materialization"
            ),

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
            specification[
                "carrier_split"
            ],

        "unique_carrier_count":
            len(
                {
                    int(row["carrier_id"])
                    for row in manifests[
                        "privileged_trajectory_rows"
                    ]
                }
            ),

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

        "measurement_noise_seed":
            NOISE_SEED,

        "test_manifest_seed_used":
            False,

        "test_carriers_used":
            False,

        "test_fields_used":
            False,

        "test_metrics_computed":
            False,

        "diagnostic_decoder_fitted":
            False,

        "predictive_training_performed":
            False,

        "status":
            "completed",
    }

    write_json(
        marker_path,
        marker,
    )

    return marker


def find_sealed_test_files():
    found = []

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
                found.append(
                    str(path)
                )

    return found


def write_input_hashes():
    paths = {
        "phase4ar3_summary":
            PHASE4AR3_DIR
            / "phase4ar3_summary.json",

        "test_sealing_policy":
            PHASE4AR3_DIR
            / "test_sealing_policy.json",

        "terminal_revision_policy":
            PHASE4AR3_DIR
            / "terminal_revision_policy.json",

        "field_summary":
            FIELD_DIR
            / "phase4br3_field_summary.json",

        "independent_morphology_summary":
            FIELD_DIR
            / "independent_morphology_summary.json",

        "carrier_ids":
            FIELD_DIR
            / "privileged"
            / "development_carrier_ids.npy",

        "carrier_splits":
            FIELD_DIR
            / "privileged"
            / "development_carrier_splits.npy",

        "raw_field_bank":
            FIELD_DIR
            / "privileged"
            / "development_carrier_state_fields_raw.npy",

        "normalized_field_bank":
            FIELD_DIR
            / "privileged"
            / "development_carrier_state_fields_normalized.npy",

        "normalization":
            FIELD_DIR
            / "normalization_v4.json",
    }

    hashes = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        hashes[name] = {
            "path":
                str(path),

            "sha256":
                sha256_file(path),
        }

    write_json(
        OUTPUT_DIR
        / "input_hashes.json",
        hashes,
    )


def finalize_phase(disk_preflight):
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

    sealed_test_files = (
        find_sealed_test_files()
    )

    if sealed_test_files:
        raise AssertionError(
            "Sealed Tier C v4 test files were generated: "
            + ", ".join(sealed_test_files)
        )

    observed_cells = {
        marker["cell_id"]
        for marker in markers
    }

    if observed_cells != set(
        DEVELOPMENT_CELLS
    ):
        raise AssertionError(
            "Tier C v4 development-cell set changed."
        )

    for marker in markers:
        expected_sequences = (
            EXPECTED_SEQUENCE_COUNTS[
                marker["cell_id"]
            ]
        )

        if (
            marker["sequence_count"]
            != expected_sequences
        ):
            raise AssertionError(
                f"{marker['cell_id']} sequence "
                "count changed."
            )

        if (
            marker["trajectory_count"]
            != expected_sequences
            * TRAJECTORIES_PER_SEQUENCE
        ):
            raise AssertionError(
                f"{marker['cell_id']} trajectory "
                "count changed."
            )

        if (
            marker["test_carriers_used"]
            is not False
            or marker["test_fields_used"]
            is not False
        ):
            raise AssertionError(
                f"{marker['cell_id']} used test data."
            )

    total_visible_storage_bytes = sum(
        int(marker["file_size_bytes"])
        for marker in markers
    )

    total_trajectory_count = sum(
        int(marker["trajectory_count"])
        for marker in markers
    )

    total_point_count = sum(
        int(marker["point_count"])
        for marker in markers
    )

    summary = {
        "phase":
            (
                "4B-R3 Tier C v4 development "
                "dataset materialization"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_field_phase_status":
            "completed_and_verified",

        "trajectory_materialization_authorized":
            True,

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
            total_trajectory_count,

        "total_development_point_count":
            total_point_count,

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

        "normalization_source":
            "clean training-carrier fields only",

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

        "diagnostic_decoder_fitted":
            False,

        "diagnostic_decoder_fitting_authorized":
            False,

        "predictive_training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "tier_c_v1_outputs_modified":
            False,

        "tier_c_v2_outputs_modified":
            False,

        "tier_c_v3_outputs_modified":
            False,

        "tier_c_v5_authorized":
            False,

        "disk_preflight":
            disk_preflight,

        "phase4br3_status":
            "completed",
    }

    write_json(
        OUTPUT_DIR
        / "phase4br3_summary.json",
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

    disk_preflight = check_disk_space(
        requested_cells=requested_cells,
        specifications=specifications,
    )

    write_input_hashes()

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
            specification=(
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
            chunk_size=arguments.chunk_size,
            force=arguments.force,
        )

    summary = finalize_phase(
        disk_preflight
    )

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
            "Phase 4B-R3 progress: "
            f"{completed_count}/4 development "
            "cells completed."
        )

    else:
        print()
        print(
            "Phase 4B-R3 Tier C v4 development "
            "dataset materialization completed."
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
