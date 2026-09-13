from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


PHASE3B_DIR = Path(
    "outputs/phase3b_tier_b_data"
)

PHASE4A_DIR = Path(
    "outputs/phase4a_tier_c_physical_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase4b_tier_c_data"
)

VISIBLE_DIR = OUTPUT_DIR / "visible"
PRIVILEGED_DIR = OUTPUT_DIR / "privileged"
MANIFEST_DIR = OUTPUT_DIR / "cell_manifests"


STATE_COUNT = 8
CARRIER_COUNT = 160
FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32

NOISE_SEED = 41030

NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

EXPECTED_CELLS = (
    "train_joint",
    "val_composition",
    "val_carrier",
    "val_joint",
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)

EXPECTED_TOTAL_POINT_COUNT = 280880


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cells",
        default="all",
        help=(
            "Comma-separated cell IDs or 'all'."
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


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value):
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(path: Path, rows):
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
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def stable_seed(
    cell_id: str,
    noise: float,
):
    payload = (
        f"{NOISE_SEED}|{cell_id}|{noise:g}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def validate_sources():
    phase4a = load_json(
        PHASE4A_DIR
        / "phase4a_summary.json"
    )

    field_bank = load_json(
        OUTPUT_DIR
        / "phase4b_field_bank_summary.json"
    )

    phase3b = load_json(
        PHASE3B_DIR
        / "phase3b_generation_summary.json"
    )

    if (
        phase4a[
            "phase4a_status"
        ]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A is not frozen."
        )

    if (
        field_bank[
            "field_bank_generation_status"
        ]
        != "completed"
    ):
        raise AssertionError(
            "Tier C field bank is incomplete."
        )

    if (
        field_bank[
            "field_count"
        ]
        != CARRIER_COUNT
        * STATE_COUNT
    ):
        raise AssertionError(
            "Field-bank count changed."
        )

    if (
        field_bank[
            "all_values_finite"
        ]
        is not True
    ):
        raise AssertionError(
            "Field bank contains invalid values."
        )

    if (
        phase3b[
            "generation_status"
        ]
        != "completed"
    ):
        raise AssertionError(
            "Tier B trajectory metadata is incomplete."
        )

    if (
        phase3b[
            "total_point_count"
        ]
        != EXPECTED_TOTAL_POINT_COUNT
    ):
        raise AssertionError(
            "Trajectory point count changed."
        )

    return phase4a, field_bank, phase3b


def parse_requested_cells(
    value: str,
):
    if value == "all":
        return list(
            EXPECTED_CELLS
        )

    requested = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    unknown = [
        item
        for item in requested
        if item not in EXPECTED_CELLS
    ]

    if unknown:
        raise ValueError(
            "Unknown cells: "
            + ", ".join(unknown)
        )

    if not requested:
        raise ValueError(
            "No cells were requested."
        )

    return requested


def load_normalization():
    value = load_json(
        OUTPUT_DIR
        / "normalization.json"
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
            "Normalization standard-deviation "
            "shape changed."
        )

    if np.any(
        standard_deviation <= 0.0
    ):
        raise AssertionError(
            "Invalid normalization scale."
        )

    return mean, standard_deviation


def load_field_banks():
    raw = np.load(
        PRIVILEGED_DIR
        / "carrier_state_fields_raw.npy",
        mmap_mode="r",
    )

    normalized = np.load(
        PRIVILEGED_DIR
        / "carrier_state_fields_normalized.npy",
        mmap_mode="r",
    )

    expected_shape = (
        CARRIER_COUNT,
        STATE_COUNT,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    if raw.shape != expected_shape:
        raise AssertionError(
            f"Raw field-bank shape changed: {raw.shape}"
        )

    if normalized.shape != expected_shape:
        raise AssertionError(
            "Normalized field-bank shape changed."
        )

    return raw, normalized


def load_cell_metadata(
    cell_id: str,
):
    visible_trajectory_path = (
        PHASE3B_DIR
        / "visible"
        / f"{cell_id}_trajectory_index.csv"
    )

    visible_sequence_path = (
        PHASE3B_DIR
        / "visible"
        / f"{cell_id}_sequence_manifest.csv"
    )

    privileged_trajectory_path = (
        PHASE3B_DIR
        / "privileged"
        / f"{cell_id}_trajectory_metadata.csv"
    )

    trajectory_rows = load_csv(
        visible_trajectory_path
    )

    sequence_rows = load_csv(
        visible_sequence_path
    )

    privileged_rows = load_csv(
        privileged_trajectory_path
    )

    if (
        len(trajectory_rows)
        != len(privileged_rows)
    ):
        raise AssertionError(
            f"{cell_id} visible and privileged "
            "trajectory counts differ."
        )

    visible_lookup = {
        int(row["trajectory_index"]):
            row
        for row in trajectory_rows
    }

    privileged_lookup = {
        int(row["trajectory_index"]):
            row
        for row in privileged_rows
    }

    if set(visible_lookup) != set(
        privileged_lookup
    ):
        raise AssertionError(
            f"{cell_id} trajectory IDs differ."
        )

    point_count = sum(
        int(row["point_count"])
        for row in trajectory_rows
    )

    return {
        "trajectory_rows":
            trajectory_rows,

        "sequence_rows":
            sequence_rows,

        "privileged_rows":
            privileged_rows,

        "visible_lookup":
            visible_lookup,

        "privileged_lookup":
            privileged_lookup,

        "point_count":
            point_count,

        "visible_trajectory_path":
            visible_trajectory_path,

        "visible_sequence_path":
            visible_sequence_path,

        "privileged_trajectory_path":
            privileged_trajectory_path,
    }


def create_point_bank_indices(
    cell_id: str,
    metadata,
):
    point_count = metadata[
        "point_count"
    ]

    indices = np.empty(
        point_count,
        dtype=np.int32,
    )

    point_coverage = np.zeros(
        point_count,
        dtype=np.uint8,
    )

    for trajectory_id in sorted(
        metadata[
            "visible_lookup"
        ]
    ):
        visible = metadata[
            "visible_lookup"
        ][trajectory_id]

        privileged = metadata[
            "privileged_lookup"
        ][trajectory_id]

        point_start = int(
            visible["point_start"]
        )

        trajectory_point_count = int(
            visible["point_count"]
        )

        carrier_id = int(
            privileged["carrier_id"]
        )

        state_path = tuple(
            int(value)
            for value in json.loads(
                privileged["state_path"]
            )
        )

        if (
            len(state_path)
            != trajectory_point_count
        ):
            raise AssertionError(
                f"{cell_id} trajectory {trajectory_id} "
                "has an invalid state path."
            )

        if not (
            0 <= carrier_id
            < CARRIER_COUNT
        ):
            raise AssertionError(
                "Carrier ID is outside the frozen range."
            )

        if any(
            state < 0
            or state >= STATE_COUNT
            for state in state_path
        ):
            raise AssertionError(
                "State path is outside the frozen range."
            )

        point_end = (
            point_start
            + trajectory_point_count
        )

        bank_indices = (
            carrier_id
            * STATE_COUNT
            + np.asarray(
                state_path,
                dtype=np.int32,
            )
        )

        indices[
            point_start:point_end
        ] = bank_indices

        point_coverage[
            point_start:point_end
        ] += 1

    if not np.all(
        point_coverage == 1
    ):
        raise AssertionError(
            f"{cell_id} points are missing or duplicated."
        )

    if indices.min() < 0:
        raise AssertionError(
            "Negative field-bank index."
        )

    if indices.max() >= (
        CARRIER_COUNT
        * STATE_COUNT
    ):
        raise AssertionError(
            "Field-bank index exceeds the bank."
        )

    return indices


def postprocess_noisy_fields(
    noisy_raw,
):
    noisy_raw[:, 0] = np.clip(
        noisy_raw[:, 0],
        -1.0,
        1.0,
    )

    orientation_norm = np.sqrt(
        noisy_raw[:, 1] ** 2
        + noisy_raw[:, 2] ** 2
    )

    valid = (
        orientation_norm
        > 1e-6
    )

    noisy_raw[:, 1] = np.where(
        valid,
        noisy_raw[:, 1]
        / np.maximum(
            orientation_norm,
            1e-6,
        ),
        1.0,
    )

    noisy_raw[:, 2] = np.where(
        valid,
        noisy_raw[:, 2]
        / np.maximum(
            orientation_norm,
            1e-6,
        ),
        0.0,
    )

    noisy_raw[:, 3] = np.clip(
        noisy_raw[:, 3],
        0.0,
        1.0,
    )

    return noisy_raw


def initialize_statistics():
    result = {}

    for noise in NOISE_LEVELS:
        result[noise] = {
            "count":
                np.zeros(
                    FIELD_CHANNEL_COUNT,
                    dtype=np.int64,
                ),

            "sum":
                np.zeros(
                    FIELD_CHANNEL_COUNT,
                    dtype=np.float64,
                ),

            "sum_of_squares":
                np.zeros(
                    FIELD_CHANNEL_COUNT,
                    dtype=np.float64,
                ),

            "minimum":
                np.full(
                    FIELD_CHANNEL_COUNT,
                    np.inf,
                    dtype=np.float64,
                ),

            "maximum":
                np.full(
                    FIELD_CHANNEL_COUNT,
                    -np.inf,
                    dtype=np.float64,
                ),
        }

    return result


def update_statistics(
    statistics,
    noise,
    values,
):
    for channel in range(
        FIELD_CHANNEL_COUNT
    ):
        selected = values[
            :,
            channel,
        ].astype(
            np.float64,
            copy=False,
        )

        statistics[
            noise
        ]["count"][channel] += (
            selected.size
        )

        statistics[
            noise
        ]["sum"][channel] += (
            selected.sum()
        )

        statistics[
            noise
        ][
            "sum_of_squares"
        ][channel] += (
            selected**2
        ).sum()

        statistics[
            noise
        ]["minimum"][channel] = min(
            statistics[
                noise
            ]["minimum"][channel],
            float(
                selected.min()
            ),
        )

        statistics[
            noise
        ]["maximum"][channel] = max(
            statistics[
                noise
            ]["maximum"][channel],
            float(
                selected.max()
            ),
        )


def finalize_statistics(
    cell_id,
    statistics,
):
    rows = []

    channel_names = (
        "phase_field",
        "orientation_cosine",
        "orientation_sine",
        "defect_density",
    )

    for noise in NOISE_LEVELS:
        item = statistics[
            noise
        ]

        for channel, name in enumerate(
            channel_names
        ):
            count = int(
                item["count"][channel]
            )

            mean = (
                item["sum"][channel]
                / count
            )

            variance = max(
                0.0,
                item[
                    "sum_of_squares"
                ][channel]
                / count
                - mean**2,
            )

            rows.append(
                {
                    "cell_id":
                        cell_id,

                    "noise_fraction":
                        noise,

                    "channel_index":
                        channel,

                    "channel_name":
                        name,

                    "value_count":
                        count,

                    "mean":
                        float(mean),

                    "standard_deviation":
                        float(
                            np.sqrt(
                                variance
                            )
                        ),

                    "minimum":
                        float(
                            item[
                                "minimum"
                            ][channel]
                        ),

                    "maximum":
                        float(
                            item[
                                "maximum"
                            ][channel]
                        ),
                }
            )

    return rows


def check_disk_space(
    point_counts,
):
    values_per_point = (
        len(NOISE_LEVELS)
        * FIELD_CHANNEL_COUNT
        * GRID_HEIGHT
        * GRID_WIDTH
    )

    required_bytes = sum(
        point_counts.values()
    ) * values_per_point * 2

    available_bytes = shutil.disk_usage(
        OUTPUT_DIR
    ).free

    required_with_margin = int(
        required_bytes
        * 1.10
    )

    if available_bytes < required_with_margin:
        raise RuntimeError(
            "Insufficient free disk space. "
            f"Required with margin: "
            f"{required_with_margin / 1e9:.2f} GB; "
            f"available: "
            f"{available_bytes / 1e9:.2f} GB."
        )

    return {
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
    raw_bank,
    normalized_bank,
    normalization_mean,
    normalization_std,
    chunk_size,
    force,
):
    marker_path = (
        MANIFEST_DIR
        / f"{cell_id}_complete.json"
    )

    output_path = (
        VISIBLE_DIR
        / f"{cell_id}_fields.npy"
    )

    index_path = (
        PRIVILEGED_DIR
        / f"{cell_id}_point_bank_indices.npy"
    )

    if marker_path.exists() and not force:
        marker = load_json(
            marker_path
        )

        if (
            marker.get("status")
            == "completed"
            and output_path.exists()
            and index_path.exists()
        ):
            print(
                f"{cell_id} already completed."
            )

            return marker

    metadata = load_cell_metadata(
        cell_id
    )

    point_indices = (
        create_point_bank_indices(
            cell_id=cell_id,
            metadata=metadata,
        )
    )

    np.save(
        index_path,
        point_indices,
    )

    shutil.copy2(
        metadata[
            "visible_trajectory_path"
        ],
        VISIBLE_DIR
        / f"{cell_id}_trajectory_index.csv",
    )

    shutil.copy2(
        metadata[
            "visible_sequence_path"
        ],
        VISIBLE_DIR
        / f"{cell_id}_sequence_manifest.csv",
    )

    shutil.copy2(
        metadata[
            "privileged_trajectory_path"
        ],
        PRIVILEGED_DIR
        / f"{cell_id}_trajectory_metadata.csv",
    )

    point_count = len(
        point_indices
    )

    shape = (
        len(NOISE_LEVELS),
        point_count,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    target = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=shape,
    )

    flat_raw_bank = raw_bank.reshape(
        CARRIER_COUNT
        * STATE_COUNT,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    flat_normalized_bank = (
        normalized_bank.reshape(
            CARRIER_COUNT
            * STATE_COUNT,
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        )
    )

    generators = {
        noise:
            (
                None
                if noise == 0.0
                else np.random.default_rng(
                    stable_seed(
                        cell_id,
                        noise,
                    )
                )
            )
        for noise in NOISE_LEVELS
    }

    statistics = (
        initialize_statistics()
    )

    mean = normalization_mean[
        None,
        :,
        None,
        None,
    ]

    standard_deviation = (
        normalization_std[
            None,
            :,
            None,
            None,
        ]
    )

    for start in range(
        0,
        point_count,
        chunk_size,
    ):
        end = min(
            point_count,
            start + chunk_size,
        )

        selected_indices = point_indices[
            start:end
        ]

        raw_clean = np.asarray(
            flat_raw_bank[
                selected_indices
            ],
            dtype=np.float32,
        )

        normalized_clean = np.asarray(
            flat_normalized_bank[
                selected_indices
            ],
            dtype=np.float32,
        )

        for noise_index, noise in enumerate(
            NOISE_LEVELS
        ):
            if noise == 0.0:
                normalized_noisy = (
                    normalized_clean
                )

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
                    postprocess_noisy_fields(
                        noisy_raw
                    )
                )

                normalized_noisy = (
                    noisy_raw
                    - mean
                ) / standard_deviation

            if not np.isfinite(
                normalized_noisy
            ).all():
                raise FloatingPointError(
                    f"{cell_id} generated non-finite "
                    f"values at noise {noise}."
                )

            target[
                noise_index,
                start:end,
            ] = normalized_noisy.astype(
                np.float16
            )

            update_statistics(
                statistics=statistics,
                noise=noise,
                values=normalized_noisy,
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
        output_path,
        mmap_mode="r",
    )

    if verification.shape != shape:
        raise AssertionError(
            f"{cell_id} stored shape changed."
        )

    if verification.dtype != np.float16:
        raise AssertionError(
            f"{cell_id} stored dtype changed."
        )

    del verification

    statistics_rows = (
        finalize_statistics(
            cell_id=cell_id,
            statistics=statistics,
        )
    )

    write_csv(
        MANIFEST_DIR
        / f"{cell_id}_statistics.csv",
        statistics_rows,
    )

    marker = {
        "cell_id":
            cell_id,

        "trajectory_count":
            len(
                metadata[
                    "trajectory_rows"
                ]
            ),

        "sequence_count":
            len(
                metadata[
                    "sequence_rows"
                ]
            ),

        "point_count":
            point_count,

        "noise_levels":
            list(NOISE_LEVELS),

        "stored_shape":
            list(shape),

        "stored_dtype":
            "float16",

        "file_path":
            str(output_path),

        "file_size_bytes":
            os.path.getsize(
                output_path
            ),

        "point_bank_index_path":
            str(index_path),

        "clean_fields_duplicated":
            False,

        "independent_noise_per_point":
            True,

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


def finalize_phase(
    disk_check,
):
    markers = []

    statistics_rows = []

    for cell_id in EXPECTED_CELLS:
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

        statistics_rows.extend(
            load_csv(
                MANIFEST_DIR
                / f"{cell_id}_statistics.csv"
            )
        )

    total_point_count = sum(
        int(marker["point_count"])
        for marker in markers
    )

    if (
        total_point_count
        != EXPECTED_TOTAL_POINT_COUNT
    ):
        raise AssertionError(
            "Total Tier C point count changed."
        )

    total_visible_bytes = sum(
        int(marker["file_size_bytes"])
        for marker in markers
    )

    write_csv(
        OUTPUT_DIR
        / "all_cell_field_statistics.csv",
        statistics_rows,
    )

    write_json(
        OUTPUT_DIR
        / "cell_generation_summary.json",
        {
            marker["cell_id"]:
                marker
            for marker in markers
        },
    )

    summary = {
        "phase":
            "4B Tier C physical-field dataset generation",

        "carrier_count":
            CARRIER_COUNT,

        "state_count":
            STATE_COUNT,

        "unique_clean_field_count":
            CARRIER_COUNT
            * STATE_COUNT,

        "clean_field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "evaluation_cell_count":
            len(
                markers
            ),

        "total_trajectory_count":
            sum(
                int(
                    marker[
                        "trajectory_count"
                    ]
                )
                for marker in markers
            ),

        "total_point_count":
            total_point_count,

        "noise_levels":
            list(
                NOISE_LEVELS
            ),

        "visible_storage_dtype":
            "float16",

        "total_visible_storage_bytes":
            total_visible_bytes,

        "total_visible_storage_gb":
            total_visible_bytes / 1e9,

        "phase4a_conservative_storage_estimate_gb":
            16.10678272,

        "clean_field_bank_stored_once":
            True,

        "duplicated_clean_trajectory_fields_stored":
            False,

        "storage_optimization_changes_protocol":
            False,

        "visible_files_contain_clean_float32_bank":
            False,

        "privileged_point_to_field_indices_stored":
            True,

        "independent_measurement_noise_per_point":
            True,

        "test_cells_materialized":
            True,

        "test_cells_evaluated":
            False,

        "test_metrics_computed":
            False,

        "training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "structural_labels_used_for_training":
            False,

        "phase2_outputs_modified":
            False,

        "phase3_outputs_modified":
            False,

        "disk_preflight":
            disk_check,

        "phase4b_status":
            "completed",
    }

    write_json(
        OUTPUT_DIR
        / "phase4b_summary.json",
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

    mean, standard_deviation = (
        load_normalization()
    )

    raw_bank, normalized_bank = (
        load_field_banks()
    )

    all_point_counts = {}

    for cell_id in EXPECTED_CELLS:
        metadata = load_cell_metadata(
            cell_id
        )

        all_point_counts[
            cell_id
        ] = metadata[
            "point_count"
        ]

    disk_check = check_disk_space(
        {
            cell_id:
                all_point_counts[
                    cell_id
                ]
            for cell_id in requested_cells
        }
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
            raw_bank=raw_bank,
            normalized_bank=normalized_bank,
            normalization_mean=mean,
            normalization_std=(
                standard_deviation
            ),
            chunk_size=(
                arguments.chunk_size
            ),
            force=arguments.force,
        )

    summary = finalize_phase(
        disk_check
    )

    if summary is None:
        completed_count = sum(
            (
                MANIFEST_DIR
                / f"{cell_id}_complete.json"
            ).exists()
            for cell_id in EXPECTED_CELLS
        )

        print()
        print(
            "Phase 4B progress: "
            f"{completed_count}/8 cells completed."
        )

    else:
        print()
        print(
            "Phase 4B Tier C dataset generation completed."
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
