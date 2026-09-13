from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PHASE4AR3_DIR = Path(
    "outputs/phase4ar3_tier_c_v4_protocol"
)

FIELD_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_fields"
)

DATA_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_data"
)

VISIBLE_DIR = DATA_DIR / "visible"
DATA_PRIVILEGED_DIR = DATA_DIR / "privileged"
FIELD_PRIVILEGED_DIR = FIELD_DIR / "privileged"
MANIFEST_DIR = DATA_DIR / "cell_manifests"

OUTPUT_DIR = Path(
    "outputs/phase4cr3_tier_c_v4_audit"
)


PROTOCOL_VERSION = "tier_c_v4"

STATE_COUNT = 8
FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32

DEVELOPMENT_CARRIER_COUNT = 128
TRAIN_CARRIER_COUNT = 96
VALIDATION_CARRIER_COUNT = 32
TEST_CARRIER_COUNT = 32

EXPECTED_FIELD_COUNT = (
    DEVELOPMENT_CARRIER_COUNT
    * STATE_COUNT
)

EXPECTED_PAIR_COUNT = 512

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

EXPECTED_CARRIER_SPLITS = {
    "train_joint": "train",
    "val_composition": "train",
    "val_carrier": "val",
    "val_joint": "val",
}

EXPECTED_NOISE_LEVELS = np.asarray(
    [
        0.0,
        0.1,
        0.25,
        0.5,
        1.0,
    ],
    dtype=np.float32,
)

MINIMUM_ENERGY_NONINCREASE_FRACTION = 0.99
MAXIMUM_PHASE_MEAN_DRIFT = 1e-5
MAXIMUM_DEFECT_MEAN_DRIFT = 1e-5

PHASE_MINIMUM = -1.000001
PHASE_MAXIMUM = 1.000001
DEFECT_MINIMUM = -0.000001
DEFECT_MAXIMUM = 1.000001

MINIMUM_MEAN_LENGTH_RATIO = 2.50
MINIMUM_WORST_LENGTH_RATIO = 1.80

MAXIMUM_MEAN_INTERFACE_RATIO = 0.70
MAXIMUM_PAIR_INTERFACE_RATIO = 0.85
MINIMUM_INTERFACE_PAIR_PASS_FRACTION = 0.95

MAXIMUM_SORTED_PHASE_DIFFERENCE = 1e-7
MAXIMUM_SORTED_DEFECT_DIFFERENCE = 1e-7
MAXIMUM_SORTED_ORIENTATION_DIFFERENCE = 1e-7


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--visible-chunk-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--repeated-noise-pairs-per-cell",
        type=int,
        default=32,
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


def parse_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def parse_state_path(value):
    parsed = json.loads(
        str(value)
    )

    state_path = tuple(
        int(item)
        for item in parsed
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
            f"Invalid state path: {state_path}"
        )

    return state_path


def validate_frozen_sources():
    protocol_summary = load_json(
        PHASE4AR3_DIR
        / "phase4ar3_summary.json"
    )

    terminal_policy = load_json(
        PHASE4AR3_DIR
        / "terminal_revision_policy.json"
    )

    field_summary = load_json(
        FIELD_DIR
        / "phase4br3_field_summary.json"
    )

    morphology_summary = load_json(
        FIELD_DIR
        / "independent_morphology_summary.json"
    )

    dataset_summary = load_json(
        DATA_DIR
        / "phase4br3_summary.json"
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
        protocol_summary[
            "tier_c_v4_is_final_revision"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v4 is not frozen as final."
        )

    if (
        protocol_summary[
            "tier_c_v5_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "Tier C v5 was unexpectedly authorized."
        )

    if (
        terminal_policy[
            "tier_c_v5_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "Terminal revision policy changed."
        )

    if (
        field_summary["phase4br3_status"]
        != "completed_and_verified"
    ):
        raise AssertionError(
            "Tier C v4 field construction did not pass."
        )

    if (
        field_summary[
            "trajectory_materialization_authorized"
        ]
        is not True
    ):
        raise AssertionError(
            "Trajectory materialization was not authorized."
        )

    if (
        field_summary[
            "independent_morphology_gate_passed"
        ]
        is not True
    ):
        raise AssertionError(
            "Independent morphology did not pass."
        )

    if (
        morphology_summary[
            "independent_morphology_gate_passed"
        ]
        is not True
    ):
        raise AssertionError(
            "Independent morphology summary did not pass."
        )

    if morphology_summary["failed_checks"]:
        raise AssertionError(
            "Independent morphology has failed checks."
        )

    if (
        dataset_summary["phase4br3_status"]
        != "completed"
    ):
        raise AssertionError(
            "Tier C v4 development data is incomplete."
        )

    if (
        dataset_summary[
            "development_cell_count"
        ]
        != len(DEVELOPMENT_CELLS)
    ):
        raise AssertionError(
            "Tier C v4 development-cell count changed."
        )

    if set(
        dataset_summary["development_cells"]
    ) != set(DEVELOPMENT_CELLS):
        raise AssertionError(
            "Tier C v4 development-cell registry changed."
        )

    if (
        dataset_summary[
            "diagnostic_decoder_fitted"
        ]
        is not False
    ):
        raise AssertionError(
            "A decoder was fitted before this audit."
        )

    if (
        dataset_summary[
            "predictive_training_performed"
        ]
        is not False
        or dataset_summary[
            "checkpoint_selection_performed"
        ]
        is not False
    ):
        raise AssertionError(
            "Predictive training occurred before "
            "the acceptance audit."
        )

    if (
        dataset_summary[
            "sealed_test_files_absent"
        ]
        is not True
    ):
        raise AssertionError(
            "Sealed test files are not recorded as absent."
        )

    return {
        "protocol_summary":
            protocol_summary,

        "terminal_policy":
            terminal_policy,

        "field_summary":
            field_summary,

        "morphology_summary":
            morphology_summary,

        "dataset_summary":
            dataset_summary,
    }


def audit_carrier_split():
    split_rows = load_csv(
        PHASE4AR3_DIR
        / "carrier_split_v4.csv"
    )

    if len(split_rows) != 160:
        raise AssertionError(
            "Tier C v4 carrier registry must contain "
            "160 carriers."
        )

    registry = {
        int(row["carrier_id"]):
            row["split"]
        for row in split_rows
    }

    if set(registry) != set(range(160)):
        raise AssertionError(
            "Tier C v4 carrier IDs changed."
        )

    counts = dict(
        Counter(registry.values())
    )

    expected_counts = {
        "train": 96,
        "val": 32,
        "test": 32,
    }

    if counts != expected_counts:
        raise AssertionError(
            f"Carrier split counts changed: {counts}"
        )

    development_ids = np.load(
        FIELD_PRIVILEGED_DIR
        / "development_carrier_ids.npy"
    ).astype(np.int64)

    development_splits = np.load(
        FIELD_PRIVILEGED_DIR
        / "development_carrier_splits.npy"
    ).astype(str)

    if len(development_ids) != (
        DEVELOPMENT_CARRIER_COUNT
    ):
        raise AssertionError(
            "Development carrier count changed."
        )

    if len(set(
        development_ids.tolist()
    )) != DEVELOPMENT_CARRIER_COUNT:
        raise AssertionError(
            "Development carrier IDs are not unique."
        )

    observed_split_counts = dict(
        Counter(
            development_splits.tolist()
        )
    )

    if observed_split_counts != {
        "train": 96,
        "val": 32,
    }:
        raise AssertionError(
            "Development carrier split counts changed."
        )

    test_ids = {
        carrier_id
        for carrier_id, split
        in registry.items()
        if split == "test"
    }

    development_id_set = set(
        int(value)
        for value in development_ids
    )

    overlap = (
        development_id_set
        & test_ids
    )

    if overlap:
        raise AssertionError(
            "Tier C v4 test carriers entered "
            "development data."
        )

    for carrier_id, observed_split in zip(
        development_ids,
        development_splits,
    ):
        expected_split = registry[
            int(carrier_id)
        ]

        if observed_split != expected_split:
            raise AssertionError(
                f"Carrier {carrier_id} split mismatch: "
                f"{observed_split} vs {expected_split}."
            )

    lookup = {
        int(carrier_id):
            int(index)
        for index, carrier_id in enumerate(
            development_ids
        )
    }

    return {
        "registry":
            registry,

        "development_ids":
            development_ids,

        "development_splits":
            development_splits,

        "development_lookup":
            lookup,

        "test_ids":
            test_ids,

        "development_test_overlap_count":
            len(overlap),

        "carrier_split_gate_passed":
            True,
    }


def audit_physics():
    rows = load_csv(
        FIELD_PRIVILEGED_DIR
        / "development_physics_diagnostics.csv"
    )

    if len(rows) != EXPECTED_FIELD_COUNT:
        raise AssertionError(
            "Tier C v4 physics diagnostic count changed."
        )

    finite_flags = np.asarray(
        [
            parse_bool(
                row["all_values_finite"]
            )
            for row in rows
        ],
        dtype=bool,
    )

    energy_flags = np.asarray(
        [
            parse_bool(
                row["energy_nonincrease"]
            )
            for row in rows
        ],
        dtype=bool,
    )

    phase_drifts = np.asarray(
        [
            float(row["phase_mean_drift"])
            for row in rows
        ],
        dtype=np.float64,
    )

    defect_drifts = np.asarray(
        [
            float(row["defect_mean_drift"])
            for row in rows
        ],
        dtype=np.float64,
    )

    phase_minima = np.asarray(
        [
            float(row["phase_minimum"])
            for row in rows
        ],
        dtype=np.float64,
    )

    phase_maxima = np.asarray(
        [
            float(row["phase_maximum"])
            for row in rows
        ],
        dtype=np.float64,
    )

    defect_minima = np.asarray(
        [
            float(row["defect_minimum"])
            for row in rows
        ],
        dtype=np.float64,
    )

    defect_maxima = np.asarray(
        [
            float(row["defect_maximum"])
            for row in rows
        ],
        dtype=np.float64,
    )

    checks = {
        "all_values_finite":
            bool(finite_flags.all()),

        "energy_nonincrease_fraction":
            bool(
                energy_flags.mean()
                >= MINIMUM_ENERGY_NONINCREASE_FRACTION
            ),

        "phase_mean_drift":
            bool(
                phase_drifts.max()
                <= MAXIMUM_PHASE_MEAN_DRIFT
            ),

        "defect_mean_drift":
            bool(
                defect_drifts.max()
                <= MAXIMUM_DEFECT_MEAN_DRIFT
            ),

        "phase_minimum":
            bool(
                phase_minima.min()
                >= PHASE_MINIMUM
            ),

        "phase_maximum":
            bool(
                phase_maxima.max()
                <= PHASE_MAXIMUM
            ),

        "defect_minimum":
            bool(
                defect_minima.min()
                >= DEFECT_MINIMUM
            ),

        "defect_maximum":
            bool(
                defect_maxima.max()
                <= DEFECT_MAXIMUM
            ),
    }

    result = {
        "diagnostic_count":
            len(rows),

        "energy_nonincrease_fraction":
            float(energy_flags.mean()),

        "maximum_phase_mean_drift":
            float(phase_drifts.max()),

        "maximum_defect_mean_drift":
            float(defect_drifts.max()),

        "phase_minimum":
            float(phase_minima.min()),

        "phase_maximum":
            float(phase_maxima.max()),

        "defect_minimum":
            float(defect_minima.min()),

        "defect_maximum":
            float(defect_maxima.max()),

        "checks":
            checks,

        "failed_checks": [
            name
            for name, passed
            in checks.items()
            if not passed
        ],

        "physics_gate_passed":
            bool(all(checks.values())),
    }

    write_json(
        OUTPUT_DIR
        / "physics_predecoder_audit.json",
        result,
    )

    return result


def audit_independent_morphology():
    rows = load_csv(
        FIELD_PRIVILEGED_DIR
        / "independent_pair_verification.csv"
    )

    if len(rows) != EXPECTED_PAIR_COUNT:
        raise AssertionError(
            "Tier C v4 independent pair count changed."
        )

    length_ratios = np.asarray(
        [
            float(
                row[
                    "independent_coarse_to_fine_length_ratio"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    interface_ratios = np.asarray(
        [
            float(
                row[
                    "independent_coarse_to_fine_interface_ratio"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    phase_differences = np.asarray(
        [
            float(
                row[
                    "maximum_sorted_phase_difference"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    defect_differences = np.asarray(
        [
            float(
                row[
                    "maximum_sorted_defect_difference"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    orientation_differences = np.asarray(
        [
            float(
                row[
                    "maximum_sorted_orientation_pair_difference"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    interface_pass_fraction = float(
        np.mean(
            interface_ratios
            <= MAXIMUM_MEAN_INTERFACE_RATIO
        )
    )

    checks = {
        "mean_characteristic_length_ratio":
            bool(
                length_ratios.mean()
                >= MINIMUM_MEAN_LENGTH_RATIO
            ),

        "worst_characteristic_length_ratio":
            bool(
                length_ratios.min()
                >= MINIMUM_WORST_LENGTH_RATIO
            ),

        "mean_interface_density_ratio":
            bool(
                interface_ratios.mean()
                <= MAXIMUM_MEAN_INTERFACE_RATIO
            ),

        "maximum_pair_interface_density_ratio":
            bool(
                interface_ratios.max()
                <= MAXIMUM_PAIR_INTERFACE_RATIO
            ),

        "interface_pair_pass_fraction":
            bool(
                interface_pass_fraction
                >= MINIMUM_INTERFACE_PAIR_PASS_FRACTION
            ),

        "phase_multiset_preserved":
            bool(
                phase_differences.max()
                <= MAXIMUM_SORTED_PHASE_DIFFERENCE
            ),

        "defect_multiset_preserved":
            bool(
                defect_differences.max()
                <= MAXIMUM_SORTED_DEFECT_DIFFERENCE
            ),

        "orientation_pair_multiset_preserved":
            bool(
                orientation_differences.max()
                <= MAXIMUM_SORTED_ORIENTATION_DIFFERENCE
            ),
    }

    result = {
        "pair_count":
            len(rows),

        "mean_characteristic_length_ratio":
            float(length_ratios.mean()),

        "minimum_characteristic_length_ratio":
            float(length_ratios.min()),

        "maximum_characteristic_length_ratio":
            float(length_ratios.max()),

        "mean_interface_density_ratio":
            float(interface_ratios.mean()),

        "minimum_interface_density_ratio":
            float(interface_ratios.min()),

        "maximum_interface_density_ratio":
            float(interface_ratios.max()),

        "interface_pair_pass_fraction":
            interface_pass_fraction,

        "maximum_sorted_phase_difference":
            float(phase_differences.max()),

        "maximum_sorted_defect_difference":
            float(defect_differences.max()),

        "maximum_sorted_orientation_pair_difference":
            float(orientation_differences.max()),

        "checks":
            checks,

        "failed_checks": [
            name
            for name, passed
            in checks.items()
            if not passed
        ],

        "morphology_reproduction_gate_passed":
            bool(all(checks.values())),
    }

    write_json(
        OUTPUT_DIR
        / "morphology_predecoder_audit.json",
        result,
    )

    return result


def audit_clean_field_bank(
    carrier_audit,
):
    raw_bank = np.load(
        FIELD_PRIVILEGED_DIR
        / "development_carrier_state_fields_raw.npy",
        mmap_mode="r",
    )

    normalized_bank = np.load(
        FIELD_PRIVILEGED_DIR
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
            f"Raw field shape changed: {raw_bank.shape}"
        )

    if normalized_bank.shape != expected_shape:
        raise AssertionError(
            "Normalized field shape changed."
        )

    if raw_bank.dtype != np.float32:
        raise AssertionError(
            "Raw development fields are not float32."
        )

    if normalized_bank.dtype != np.float32:
        raise AssertionError(
            "Normalized development fields are not float32."
        )

    if not np.isfinite(
        np.asarray(raw_bank)
    ).all():
        raise FloatingPointError(
            "Raw development field bank contains "
            "NaN or Inf."
        )

    if not np.isfinite(
        np.asarray(normalized_bank)
    ).all():
        raise FloatingPointError(
            "Normalized development field bank "
            "contains NaN or Inf."
        )

    train_indices = np.where(
        carrier_audit[
            "development_splits"
        ] == "train"
    )[0]

    training_fields = np.asarray(
        normalized_bank[
            train_indices
        ],
        dtype=np.float32,
    )

    channel_variances = training_fields.var(
        axis=(0, 1, 3, 4),
        dtype=np.float64,
    )

    state_carrier_variances = np.asarray(
        [
            np.asarray(
                normalized_bank[:, state_id],
                dtype=np.float32,
            ).var(
                axis=0,
                dtype=np.float64,
            ).mean()
            for state_id in range(
                STATE_COUNT
            )
        ],
        dtype=np.float64,
    )

    digests = []

    for carrier_index in range(
        DEVELOPMENT_CARRIER_COUNT
    ):
        for state_id in range(
            STATE_COUNT
        ):
            values = np.asarray(
                raw_bank[
                    carrier_index,
                    state_id,
                ],
                dtype=np.float32,
            )

            digest = hashlib.sha256(
                values.tobytes(
                    order="C"
                )
            ).hexdigest()

            digests.append(digest)

    duplicate_count = (
        len(digests)
        - len(set(digests))
    )

    checks = {
        "raw_shape":
            raw_bank.shape == expected_shape,

        "normalized_shape":
            normalized_bank.shape
            == expected_shape,

        "all_training_channels_nonconstant":
            bool(
                np.all(
                    channel_variances > 1e-12
                )
            ),

        "every_state_has_carrier_variation":
            bool(
                np.all(
                    state_carrier_variances
                    > 1e-12
                )
            ),

        "no_exact_duplicate_fields":
            duplicate_count == 0,
    }

    result = {
        "field_count":
            EXPECTED_FIELD_COUNT,

        "raw_shape":
            list(raw_bank.shape),

        "normalized_shape":
            list(normalized_bank.shape),

        "channel_variances":
            channel_variances.tolist(),

        "state_carrier_variances":
            state_carrier_variances.tolist(),

        "exact_duplicate_field_count":
            duplicate_count,

        "checks":
            checks,

        "failed_checks": [
            name
            for name, passed
            in checks.items()
            if not passed
        ],

        "basic_manifold_gate_passed":
            bool(all(checks.values())),
    }

    write_json(
        OUTPUT_DIR
        / "basic_manifold_predecoder_audit.json",
        result,
    )

    return result


def find_duplicate_index_pairs(
    indices,
    maximum_pair_count,
):
    first_position = {}
    pairs = []

    for position, value in enumerate(
        indices.tolist()
    ):
        value = int(value)

        if value in first_position:
            pairs.append(
                (
                    first_position[value],
                    position,
                    value,
                )
            )

            if len(pairs) >= maximum_pair_count:
                break

        else:
            first_position[value] = position

    return pairs


def audit_cell(
    cell_id,
    carrier_audit,
    normalized_flat,
    visible_chunk_size,
    repeated_noise_pairs_per_cell,
):
    marker = load_json(
        MANIFEST_DIR
        / f"{cell_id}_complete.json"
    )

    trajectory_rows = load_csv(
        VISIBLE_DIR
        / f"{cell_id}_trajectory_index.csv"
    )

    sequence_rows = load_csv(
        VISIBLE_DIR
        / f"{cell_id}_sequence_manifest.csv"
    )

    metadata_rows = load_csv(
        DATA_PRIVILEGED_DIR
        / f"{cell_id}_trajectory_metadata.csv"
    )

    point_indices = np.load(
        DATA_PRIVILEGED_DIR
        / f"{cell_id}_point_bank_indices.npy"
    ).astype(np.int64)

    visible_fields = np.load(
        VISIBLE_DIR
        / f"{cell_id}_fields.npy",
        mmap_mode="r",
    )

    expected_sequence_count = (
        EXPECTED_SEQUENCE_COUNTS[
            cell_id
        ]
    )

    expected_trajectory_count = (
        expected_sequence_count
        * STATE_COUNT
    )

    if marker["status"] != "completed":
        raise AssertionError(
            f"{cell_id} marker is incomplete."
        )

    if (
        marker["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            f"{cell_id} protocol version changed."
        )

    if (
        marker["sequence_count"]
        != expected_sequence_count
    ):
        raise AssertionError(
            f"{cell_id} sequence count changed."
        )

    if (
        marker["trajectory_count"]
        != expected_trajectory_count
    ):
        raise AssertionError(
            f"{cell_id} trajectory count changed."
        )

    if (
        marker["carrier_split"]
        != EXPECTED_CARRIER_SPLITS[
            cell_id
        ]
    ):
        raise AssertionError(
            f"{cell_id} carrier split changed."
        )

    if len(sequence_rows) != (
        expected_sequence_count
    ):
        raise AssertionError(
            f"{cell_id} sequence-manifest row "
            "count changed."
        )

    if len(trajectory_rows) != (
        expected_trajectory_count
    ):
        raise AssertionError(
            f"{cell_id} trajectory-index row "
            "count changed."
        )

    if len(metadata_rows) != (
        expected_trajectory_count
    ):
        raise AssertionError(
            f"{cell_id} trajectory-metadata row "
            "count changed."
        )

    trajectory_rows.sort(
        key=lambda row:
            int(row["trajectory_index"])
    )

    metadata_rows.sort(
        key=lambda row:
            int(row["trajectory_index"])
    )

    expected_indices = []
    observed_carriers = set()
    state_counts = Counter()
    initial_states_by_sequence = defaultdict(list)

    expected_point_start = 0

    development_lookup = carrier_audit[
        "development_lookup"
    ]

    registry = carrier_audit[
        "registry"
    ]

    for trajectory_row, metadata_row in zip(
        trajectory_rows,
        metadata_rows,
    ):
        trajectory_index = int(
            trajectory_row[
                "trajectory_index"
            ]
        )

        metadata_trajectory_index = int(
            metadata_row[
                "trajectory_index"
            ]
        )

        if (
            trajectory_index
            != metadata_trajectory_index
        ):
            raise AssertionError(
                f"{cell_id} trajectory ordering changed."
            )

        point_start = int(
            trajectory_row["point_start"]
        )

        point_count = int(
            trajectory_row["point_count"]
        )

        if point_start != expected_point_start:
            raise AssertionError(
                f"{cell_id} has a noncontiguous "
                "point index at trajectory "
                f"{trajectory_index}."
            )

        state_path = parse_state_path(
            metadata_row["state_path"]
        )

        if point_count != len(state_path):
            raise AssertionError(
                f"{cell_id} trajectory "
                f"{trajectory_index} point count "
                "does not match its state path."
            )

        initial_state = int(
            metadata_row["initial_state"]
        )

        if initial_state != state_path[0]:
            raise AssertionError(
                f"{cell_id} trajectory "
                f"{trajectory_index} initial state "
                "does not match its state path."
            )

        carrier_id = int(
            metadata_row["carrier_id"]
        )

        carrier_split = metadata_row[
            "carrier_split"
        ]

        if carrier_id not in development_lookup:
            raise AssertionError(
                f"{cell_id} uses carrier {carrier_id}, "
                "which is absent from the development bank."
            )

        if carrier_id in carrier_audit[
            "test_ids"
        ]:
            raise AssertionError(
                f"{cell_id} uses sealed test carrier "
                f"{carrier_id}."
            )

        if registry[carrier_id] != (
            carrier_split
        ):
            raise AssertionError(
                f"{cell_id} carrier {carrier_id} "
                "split metadata changed."
            )

        if carrier_split != (
            EXPECTED_CARRIER_SPLITS[
                cell_id
            ]
        ):
            raise AssertionError(
                f"{cell_id} uses an incorrect "
                "carrier split."
            )

        local_carrier_index = (
            development_lookup[
                carrier_id
            ]
        )

        for state_id in state_path:
            expected_indices.append(
                local_carrier_index
                * STATE_COUNT
                + state_id
            )

            state_counts[state_id] += 1

        observed_carriers.add(
            carrier_id
        )

        sequence_index = int(
            metadata_row[
                "cell_sequence_index"
            ]
        )

        initial_states_by_sequence[
            sequence_index
        ].append(initial_state)

        expected_point_start += (
            point_count
        )

    expected_indices = np.asarray(
        expected_indices,
        dtype=np.int64,
    )

    if len(expected_indices) != (
        marker["point_count"]
    ):
        raise AssertionError(
            f"{cell_id} reconstructed point count changed."
        )

    if point_indices.shape != (
        expected_indices.shape
    ):
        raise AssertionError(
            f"{cell_id} point-index shape changed."
        )

    if not np.array_equal(
        point_indices,
        expected_indices,
    ):
        mismatch_count = int(
            np.sum(
                point_indices
                != expected_indices
            )
        )

        raise AssertionError(
            f"{cell_id} has {mismatch_count} "
            "carrier-state bank-index mismatches."
        )

    for sequence_index, initial_states in (
        initial_states_by_sequence.items()
    ):
        if sorted(initial_states) != list(
            range(STATE_COUNT)
        ):
            raise AssertionError(
                f"{cell_id} sequence "
                f"{sequence_index} does not contain "
                "initial states 0-7."
            )

    if set(state_counts) != set(
        range(STATE_COUNT)
    ):
        raise AssertionError(
            f"{cell_id} does not cover all states."
        )

    stored_shape = tuple(
        int(value)
        for value in marker[
            "stored_shape"
        ]
    )

    if visible_fields.shape != stored_shape:
        raise AssertionError(
            f"{cell_id} visible-field shape changed."
        )

    if visible_fields.dtype != np.float16:
        raise AssertionError(
            f"{cell_id} visible fields are not float16."
        )

    if stored_shape != (
        len(EXPECTED_NOISE_LEVELS),
        marker["point_count"],
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    ):
        raise AssertionError(
            f"{cell_id} marker shape is inconsistent."
        )

    zero_noise_index = 0

    zero_noise_mismatch_count = 0
    nonfinite_value_count = 0

    for start in range(
        0,
        marker["point_count"],
        visible_chunk_size,
    ):
        end = min(
            marker["point_count"],
            start + visible_chunk_size,
        )

        selected_indices = (
            point_indices[start:end]
        )

        expected_zero_noise = np.asarray(
            normalized_flat[
                selected_indices
            ],
            dtype=np.float32,
        ).astype(np.float16)

        observed_zero_noise = np.asarray(
            visible_fields[
                zero_noise_index,
                start:end,
            ],
            dtype=np.float16,
        )

        zero_noise_mismatch_count += int(
            np.sum(
                observed_zero_noise
                != expected_zero_noise
            )
        )

        visible_chunk = np.asarray(
            visible_fields[
                :,
                start:end,
            ],
            dtype=np.float32,
        )

        nonfinite_value_count += int(
            np.size(visible_chunk)
            - np.isfinite(
                visible_chunk
            ).sum()
        )

    repeated_pairs = find_duplicate_index_pairs(
        point_indices,
        repeated_noise_pairs_per_cell,
    )

    repeated_observation_checks = 0
    repeated_observation_difference_count = 0

    for first, second, _ in repeated_pairs:
        for noise_index in range(
            1,
            len(EXPECTED_NOISE_LEVELS),
        ):
            first_observation = np.asarray(
                visible_fields[
                    noise_index,
                    first,
                ],
                dtype=np.float16,
            )

            second_observation = np.asarray(
                visible_fields[
                    noise_index,
                    second,
                ],
                dtype=np.float16,
            )

            repeated_observation_checks += 1

            if not np.array_equal(
                first_observation,
                second_observation,
            ):
                repeated_observation_difference_count += 1

    noisy_slice_difference_checks = 0
    noisy_slice_difference_passes = 0

    sample_count = min(
        64,
        marker["point_count"],
    )

    sample_positions = np.linspace(
        0,
        marker["point_count"] - 1,
        num=sample_count,
        dtype=np.int64,
    )

    clean_sample = np.asarray(
        visible_fields[
            zero_noise_index,
            sample_positions,
        ],
        dtype=np.float16,
    )

    for noise_index in range(
        1,
        len(EXPECTED_NOISE_LEVELS),
    ):
        noisy_sample = np.asarray(
            visible_fields[
                noise_index,
                sample_positions,
            ],
            dtype=np.float16,
        )

        noisy_slice_difference_checks += 1

        if not np.array_equal(
            clean_sample,
            noisy_sample,
        ):
            noisy_slice_difference_passes += 1

    for first_noise_index in range(
        1,
        len(EXPECTED_NOISE_LEVELS),
    ):
        for second_noise_index in range(
            first_noise_index + 1,
            len(EXPECTED_NOISE_LEVELS),
        ):
            first_noisy_sample = np.asarray(
                visible_fields[
                    first_noise_index,
                    sample_positions,
                ],
                dtype=np.float16,
            )

            second_noisy_sample = np.asarray(
                visible_fields[
                    second_noise_index,
                    sample_positions,
                ],
                dtype=np.float16,
            )

            noisy_slice_difference_checks += 1

            if not np.array_equal(
                first_noisy_sample,
                second_noisy_sample,
            ):
                noisy_slice_difference_passes += 1

    expected_unique_carrier_count = (
        TRAIN_CARRIER_COUNT
        if EXPECTED_CARRIER_SPLITS[
            cell_id
        ] == "train"
        else VALIDATION_CARRIER_COUNT
    )

    checks = {
        "sequence_count":
            len(sequence_rows)
            == expected_sequence_count,

        "trajectory_count":
            len(trajectory_rows)
            == expected_trajectory_count,

        "point_count":
            len(point_indices)
            == marker["point_count"],

        "carrier_state_bank_indices":
            np.array_equal(
                point_indices,
                expected_indices,
            ),

        "carrier_split":
            marker["carrier_split"]
            == EXPECTED_CARRIER_SPLITS[
                cell_id
            ],

        "full_carrier_coverage":
            len(observed_carriers)
            == expected_unique_carrier_count,

        "all_states_present":
            set(state_counts)
            == set(range(STATE_COUNT)),

        "zero_noise_exact_after_float16_storage":
            zero_noise_mismatch_count == 0,

        "all_visible_values_finite":
            nonfinite_value_count == 0,

        "noisy_slices_are_distinct":
            noisy_slice_difference_passes
            == noisy_slice_difference_checks,

        "repeated_observations_have_independent_noise":
            (
                repeated_observation_checks > 0
                and repeated_observation_difference_count
                == repeated_observation_checks
            ),

        "no_test_carriers":
            bool(
                observed_carriers.isdisjoint(
                    carrier_audit["test_ids"]
                )
            ),

        "marker_reports_no_test_use":
            (
                marker["test_carriers_used"]
                is False
                and marker["test_fields_used"]
                is False
                and marker[
                    "test_manifest_seed_used"
                ]
                is False
            ),

        "marker_reports_no_training":
            (
                marker[
                    "diagnostic_decoder_fitted"
                ]
                is False
                and marker[
                    "predictive_training_performed"
                ]
                is False
            ),
    }

    result = {
        "cell_id":
            cell_id,

        "sequence_count":
            len(sequence_rows),

        "trajectory_count":
            len(trajectory_rows),

        "point_count":
            len(point_indices),

        "carrier_split":
            marker["carrier_split"],

        "unique_carrier_count":
            len(observed_carriers),

        "state_point_counts":
            {
                str(state_id):
                    int(state_counts[state_id])
                for state_id in range(
                    STATE_COUNT
                )
            },

        "zero_noise_mismatch_count":
            zero_noise_mismatch_count,

        "nonfinite_value_count":
            nonfinite_value_count,

        "repeated_point_pair_count":
            len(repeated_pairs),

        "repeated_observation_checks":
            repeated_observation_checks,

        "repeated_observation_difference_count":
            repeated_observation_difference_count,

        "noisy_slice_difference_checks":
            noisy_slice_difference_checks,

        "noisy_slice_difference_passes":
            noisy_slice_difference_passes,

        "checks":
            checks,

        "failed_checks": [
            name
            for name, passed
            in checks.items()
            if not passed
        ],

        "cell_integrity_passed":
            bool(all(checks.values())),
    }

    write_json(
        OUTPUT_DIR
        / f"{cell_id}_integrity_audit.json",
        result,
    )

    return result


def audit_all_cells(
    carrier_audit,
    visible_chunk_size,
    repeated_noise_pairs_per_cell,
):
    noise_levels = np.load(
        VISIBLE_DIR
        / "noise_levels.npy"
    ).astype(np.float32)

    if not np.array_equal(
        noise_levels,
        EXPECTED_NOISE_LEVELS,
    ):
        raise AssertionError(
            "Tier C v4 noise-level registry changed."
        )

    normalized_bank = np.load(
        FIELD_PRIVILEGED_DIR
        / "development_carrier_state_fields_normalized.npy",
        mmap_mode="r",
    )

    normalized_flat = normalized_bank.reshape(
        -1,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    results = []

    for index, cell_id in enumerate(
        DEVELOPMENT_CELLS,
        start=1,
    ):
        print(
            f"[{index}/{len(DEVELOPMENT_CELLS)}] "
            f"Auditing {cell_id}"
        )

        result = audit_cell(
            cell_id=cell_id,
            carrier_audit=carrier_audit,
            normalized_flat=normalized_flat,
            visible_chunk_size=visible_chunk_size,
            repeated_noise_pairs_per_cell=(
                repeated_noise_pairs_per_cell
            ),
        )

        results.append(result)

    gate_passed = bool(
        all(
            result[
                "cell_integrity_passed"
            ]
            for result in results
        )
    )

    summary = {
        "noise_levels":
            noise_levels.tolist(),

        "cell_count":
            len(results),

        "cell_results":
            {
                result["cell_id"]:
                    {
                        "sequence_count":
                            result[
                                "sequence_count"
                            ],

                        "trajectory_count":
                            result[
                                "trajectory_count"
                            ],

                        "point_count":
                            result[
                                "point_count"
                            ],

                        "unique_carrier_count":
                            result[
                                "unique_carrier_count"
                            ],

                        "cell_integrity_passed":
                            result[
                                "cell_integrity_passed"
                            ],

                        "failed_checks":
                            result[
                                "failed_checks"
                            ],
                    }
                for result in results
            },

        "development_cells_gate_passed":
            gate_passed,
    }

    write_json(
        OUTPUT_DIR
        / "development_cells_predecoder_audit.json",
        summary,
    )

    return summary


def audit_test_sealing():
    found_test_files = []

    for directory in (
        FIELD_DIR,
        DATA_DIR,
    ):
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            name = path.name.lower()

            if any(
                cell_id in name
                for cell_id in SEALED_TEST_CELLS
            ):
                found_test_files.append(
                    str(path)
                )

    checkpoint_files = [
        str(path)
        for path in OUTPUT_DIR.glob("*.pt")
    ]

    checks = {
        "sealed_test_files_absent":
            len(found_test_files) == 0,

        "diagnostic_checkpoints_absent":
            len(checkpoint_files) == 0,
    }

    result = {
        "sealed_test_files_found":
            found_test_files,

        "diagnostic_checkpoints_found":
            checkpoint_files,

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

        "checks":
            checks,

        "failed_checks": [
            name
            for name, passed
            in checks.items()
            if not passed
        ],

        "test_sealing_gate_passed":
            bool(all(checks.values())),
    }

    write_json(
        OUTPUT_DIR
        / "test_sealing_predecoder_audit.json",
        result,
    )

    return result


def write_input_hashes():
    paths = {
        "phase4ar3_summary":
            PHASE4AR3_DIR
            / "phase4ar3_summary.json",

        "acceptance_gates_v4":
            PHASE4AR3_DIR
            / "acceptance_gates_v4.json",

        "terminal_revision_policy":
            PHASE4AR3_DIR
            / "terminal_revision_policy.json",

        "carrier_split_v4":
            PHASE4AR3_DIR
            / "carrier_split_v4.csv",

        "field_summary":
            FIELD_DIR
            / "phase4br3_field_summary.json",

        "independent_morphology_summary":
            FIELD_DIR
            / "independent_morphology_summary.json",

        "independent_pair_verification":
            FIELD_PRIVILEGED_DIR
            / "independent_pair_verification.csv",

        "physics_diagnostics":
            FIELD_PRIVILEGED_DIR
            / "development_physics_diagnostics.csv",

        "raw_field_bank":
            FIELD_PRIVILEGED_DIR
            / "development_carrier_state_fields_raw.npy",

        "normalized_field_bank":
            FIELD_PRIVILEGED_DIR
            / "development_carrier_state_fields_normalized.npy",

        "dataset_summary":
            DATA_DIR
            / "phase4br3_summary.json",
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
        / "predecoder_input_hashes.json",
        hashes,
    )


def main():
    arguments = parse_arguments()

    if arguments.visible_chunk_size <= 0:
        raise ValueError(
            "Visible chunk size must be positive."
        )

    if (
        arguments.repeated_noise_pairs_per_cell
        <= 0
    ):
        raise ValueError(
            "Repeated-noise pair count must be positive."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1/7] Validating frozen phase sources"
    )

    sources = validate_frozen_sources()

    print(
        "[2/7] Auditing carrier splits"
    )

    carrier_audit = audit_carrier_split()

    write_json(
        OUTPUT_DIR
        / "carrier_split_predecoder_audit.json",
        {
            "development_carrier_count":
                len(
                    carrier_audit[
                        "development_ids"
                    ]
                ),

            "training_carrier_count":
                int(
                    np.sum(
                        carrier_audit[
                            "development_splits"
                        ] == "train"
                    )
                ),

            "validation_carrier_count":
                int(
                    np.sum(
                        carrier_audit[
                            "development_splits"
                        ] == "val"
                    )
                ),

            "frozen_test_carrier_count":
                len(
                    carrier_audit[
                        "test_ids"
                    ]
                ),

            "development_test_overlap_count":
                carrier_audit[
                    "development_test_overlap_count"
                ],

            "carrier_split_gate_passed":
                carrier_audit[
                    "carrier_split_gate_passed"
                ],
        },
    )

    print(
        "[3/7] Reproducing numerical-physics audit"
    )

    physics_audit = audit_physics()

    print(
        "[4/7] Reproducing independent morphology audit"
    )

    morphology_audit = (
        audit_independent_morphology()
    )

    print(
        "[5/7] Auditing clean field manifold"
    )

    manifold_audit = (
        audit_clean_field_bank(
            carrier_audit
        )
    )

    print(
        "[6/7] Auditing all visible development cells"
    )

    cells_audit = audit_all_cells(
        carrier_audit=carrier_audit,
        visible_chunk_size=(
            arguments.visible_chunk_size
        ),
        repeated_noise_pairs_per_cell=(
            arguments.repeated_noise_pairs_per_cell
        ),
    )

    print(
        "[7/7] Auditing sealed-test absence"
    )

    sealing_audit = audit_test_sealing()

    write_input_hashes()

    gate_rows = [
        {
            "gate":
                "carrier_split_integrity",

            "passed":
                carrier_audit[
                    "carrier_split_gate_passed"
                ],
        },
        {
            "gate":
                "numerical_physics",

            "passed":
                physics_audit[
                    "physics_gate_passed"
                ],
        },
        {
            "gate":
                "independent_coarsening_morphology",

            "passed":
                morphology_audit[
                    "morphology_reproduction_gate_passed"
                ],
        },
        {
            "gate":
                "basic_field_manifold",

            "passed":
                manifold_audit[
                    "basic_manifold_gate_passed"
                ],
        },
        {
            "gate":
                "development_cell_integrity",

            "passed":
                cells_audit[
                    "development_cells_gate_passed"
                ],
        },
        {
            "gate":
                "sealed_test_absence",

            "passed":
                sealing_audit[
                    "test_sealing_gate_passed"
                ],
        },
    ]

    write_csv(
        OUTPUT_DIR
        / "predecoder_gate_results.csv",
        gate_rows,
    )

    failed_gates = [
        row["gate"]
        for row in gate_rows
        if not row["passed"]
    ]

    predecoder_gate_passed = (
        len(failed_gates) == 0
    )

    summary = {
        "phase":
            (
                "4C-R3 Tier C v4 "
                "pre-decoder acceptance audit"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_phase_validation": {
            "phase4ar3":
                True,

            "phase4br3_fields":
                True,

            "phase4br3_dataset":
                True,
        },

        "training_carrier_count":
            TRAIN_CARRIER_COUNT,

        "validation_carrier_count":
            VALIDATION_CARRIER_COUNT,

        "test_carrier_count_generated":
            0,

        "development_field_count":
            EXPECTED_FIELD_COUNT,

        "development_cell_count":
            len(DEVELOPMENT_CELLS),

        "total_development_trajectory_count":
            sources[
                "dataset_summary"
            ][
                "total_development_trajectory_count"
            ],

        "total_development_point_count":
            sources[
                "dataset_summary"
            ][
                "total_development_point_count"
            ],

        "carrier_split_gate_passed":
            carrier_audit[
                "carrier_split_gate_passed"
            ],

        "development_test_carrier_overlap_count":
            carrier_audit[
                "development_test_overlap_count"
            ],

        "physics_gate_passed":
            physics_audit[
                "physics_gate_passed"
            ],

        "energy_nonincrease_fraction":
            physics_audit[
                "energy_nonincrease_fraction"
            ],

        "maximum_phase_mean_drift":
            physics_audit[
                "maximum_phase_mean_drift"
            ],

        "maximum_defect_mean_drift":
            physics_audit[
                "maximum_defect_mean_drift"
            ],

        "independent_morphology_gate_passed":
            morphology_audit[
                "morphology_reproduction_gate_passed"
            ],

        "independent_mean_length_ratio":
            morphology_audit[
                "mean_characteristic_length_ratio"
            ],

        "independent_minimum_length_ratio":
            morphology_audit[
                "minimum_characteristic_length_ratio"
            ],

        "independent_mean_interface_ratio":
            morphology_audit[
                "mean_interface_density_ratio"
            ],

        "independent_maximum_interface_ratio":
            morphology_audit[
                "maximum_interface_density_ratio"
            ],

        "independent_interface_pair_pass_fraction":
            morphology_audit[
                "interface_pair_pass_fraction"
            ],

        "basic_manifold_gate_passed":
            manifold_audit[
                "basic_manifold_gate_passed"
            ],

        "exact_duplicate_field_count":
            manifold_audit[
                "exact_duplicate_field_count"
            ],

        "development_cells_gate_passed":
            cells_audit[
                "development_cells_gate_passed"
            ],

        "test_sealing_gate_passed":
            sealing_audit[
                "test_sealing_gate_passed"
            ],

        "sealed_test_files_absent":
            (
                len(
                    sealing_audit[
                        "sealed_test_files_found"
                    ]
                )
                == 0
            ),

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

        "diagnostic_decoder_fitted":
            False,

        "global_statistics_classifier_fitted":
            False,

        "affine_diagnostic_fitted":
            False,

        "predictive_training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "failed_predecoder_gate_count":
            len(failed_gates),

        "failed_predecoder_gates":
            failed_gates,

        "predecoder_acceptance_gate_passed":
            predecoder_gate_passed,

        "diagnostic_decoder_fitting_authorized":
            predecoder_gate_passed,

        "predictive_training_authorized":
            False,

        "tier_c_v5_authorized":
            False,

        "tier_c_v1_outputs_modified":
            False,

        "tier_c_v2_outputs_modified":
            False,

        "tier_c_v3_outputs_modified":
            False,

        "phase4ar3_outputs_modified":
            False,

        "phase4br3_outputs_modified":
            False,

        "phase4cr3_predecoder_status":
            (
                "passed"
                if predecoder_gate_passed
                else "failed_terminal_gate"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4cr3_predecoder_summary.json",
        summary,
    )

    print()
    print(
        "Phase 4C-R3 pre-decoder audit completed."
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

    if not predecoder_gate_passed:
        raise SystemExit(
            "Tier C v4 failed a terminal pre-decoder "
            "gate. Do not fit diagnostic decoders, do "
            "not train predictive models, and do not "
            "create Tier C v5."
        )


if __name__ == "__main__":
    main()
