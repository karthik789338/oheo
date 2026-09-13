from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROTOCOL_VERSION = "tier_c_v4"
SMOKE_NOISE_FRACTION = 0.25
REQUESTED_SEQUENCE_LENGTHS = (1, 2, 3, 4)

DATA_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_data"
)

VISIBLE_DIR = DATA_DIR / "visible"

CONTRACT_DIR = Path(
    "outputs/phase4dr32_tier_c_v4_execution_contract"
)

INTERFACE_DIR = Path(
    "outputs/phase4er3a_tier_c_v4_interface_snapshot"
)

TRANSITION_DIR = Path(
    "outputs/phase4cr3_tier_c_v4_audit"
    "/recovered_transition_table"
)

OUTPUT_DIR = Path(
    "outputs/phase4er3b_tier_c_v4_smoke_batch"
)


CONTRACT_SUMMARY_PATH = (
    CONTRACT_DIR
    / "phase4dr32_execution_contract_summary.json"
)

INTERFACE_SUMMARY_PATH = (
    INTERFACE_DIR
    / "phase4er3a_interface_snapshot_summary.json"
)

CELL_MANIFEST_PATH = (
    DATA_DIR
    / "cell_manifests"
    / "train_joint_complete.json"
)

SEQUENCE_MANIFEST_PATH = (
    VISIBLE_DIR
    / "train_joint_sequence_manifest.csv"
)

TRAJECTORY_INDEX_PATH = (
    VISIBLE_DIR
    / "train_joint_trajectory_index.csv"
)

FIELD_PATH = (
    VISIBLE_DIR
    / "train_joint_fields.npy"
)

NOISE_LEVEL_PATH = (
    VISIBLE_DIR
    / "noise_levels.npy"
)

TRANSITION_TABLE_PATH = (
    TRANSITION_DIR
    / "primitive_transition_table_long.csv"
)


FORBIDDEN_TERMS = (
    "test_iid",
    "test_composition",
    "test_carrier",
    "test_joint",
    "sealed_test",
    "privileged",
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


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
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


def require(
    condition,
    message,
):
    if not condition:
        raise AssertionError(message)


def assert_development_only(path: Path):
    lower = str(path).lower()

    require(
        not any(
            term in lower
            for term in FORBIDDEN_TERMS
        ),
        f"Forbidden artifact path: {path}",
    )


def load_operation_registry():
    rows = read_csv(
        TRANSITION_TABLE_PATH
    )

    operation_to_index = {}

    element_ids = {}

    for row in rows:
        name = row["primitive_id"]
        index = int(
            row["primitive_index"]
        )

        element_id = row["element_id"]

        if name in operation_to_index:
            require(
                operation_to_index[name] == index,
                f"Inconsistent index for {name}.",
            )

            require(
                element_ids[name] == element_id,
                f"Inconsistent element ID for {name}.",
            )

        operation_to_index[name] = index
        element_ids[name] = element_id

    require(
        len(operation_to_index) == 6,
        "Expected six primitive operations.",
    )

    require(
        sorted(
            operation_to_index.values()
        ) == list(range(6)),
        "Primitive indices must be 0 through 5.",
    )

    return operation_to_index, element_ids


def load_sequence_registry():
    rows = read_csv(
        SEQUENCE_MANIFEST_PATH
    )

    registry = {}

    for row in rows:
        cell_sequence_index = int(
            row["cell_sequence_index"]
        )

        require(
            cell_sequence_index not in registry,
            (
                "Duplicate cell_sequence_index: "
                f"{cell_sequence_index}"
            ),
        )

        operations = json.loads(
            row["operation_sequence"]
        )

        sequence_length = int(
            row["sequence_length"]
        )

        require(
            len(operations) == sequence_length,
            (
                "Manifest operation count differs from "
                f"sequence_length for {cell_sequence_index}."
            ),
        )

        require(
            row["cell_id"] == "train_joint",
            "Unexpected cell in train manifest.",
        )

        require(
            row["tier_c_v4_protocol_version"]
            == PROTOCOL_VERSION,
            "Protocol version changed.",
        )

        require(
            row["fresh_carrier_pairing"]
            == "True",
            "Carrier pairing is not marked fresh.",
        )

        registry[cell_sequence_index] = {
            "cell_sequence_index":
                cell_sequence_index,

            "source_sequence_id":
                int(
                    row["source_sequence_id"]
                ),

            "sequence_index":
                int(
                    row["sequence_index"]
                ),

            "sequence_length":
                sequence_length,

            "operations":
                operations,
        }

    return registry


def validate_complete_cell(
    sequence_registry,
    trajectory_rows,
    cell_manifest,
):
    require(
        cell_manifest["status"] == "completed",
        "Development cell is incomplete.",
    )

    require(
        cell_manifest["cell_id"]
        == "train_joint",
        "Unexpected development cell.",
    )

    require(
        cell_manifest["protocol_version"]
        == PROTOCOL_VERSION,
        "Development-cell protocol changed.",
    )

    require(
        cell_manifest["sequence_count"]
        == len(sequence_registry),
        "Sequence count differs from manifest.",
    )

    require(
        cell_manifest["trajectory_count"]
        == len(trajectory_rows),
        "Trajectory count differs from manifest.",
    )

    counts = Counter(
        int(row["cell_sequence_index"])
        for row in trajectory_rows
    )

    require(
        set(counts)
        == set(sequence_registry),
        "Trajectory and sequence registries differ.",
    )

    unique_counts = set(
        counts.values()
    )

    require(
        unique_counts == {8},
        (
            "Expected exactly eight trajectories per "
            f"sequence, observed {sorted(unique_counts)}."
        ),
    )

    maximum_point_end = 0

    for row in trajectory_rows:
        cell_sequence_index = int(
            row["cell_sequence_index"]
        )

        point_start = int(
            row["point_start"]
        )

        point_count = int(
            row["point_count"]
        )

        sequence_length = int(
            row["sequence_length"]
        )

        source = sequence_registry[
            cell_sequence_index
        ]

        require(
            sequence_length
            == source["sequence_length"],
            "Trajectory sequence length changed.",
        )

        require(
            point_count
            == sequence_length + 1,
            (
                "point_count must equal "
                "sequence_length + 1."
            ),
        )

        require(
            point_start >= 0,
            "Negative point_start.",
        )

        maximum_point_end = max(
            maximum_point_end,
            point_start + point_count,
        )

    require(
        maximum_point_end
        == cell_manifest["point_count"],
        (
            "Trajectory index does not terminate at "
            "the frozen point count."
        ),
    )


def select_smoke_trajectories(
    sequence_registry,
    trajectory_rows,
):
    selected = []

    used_sequence_indices = set()

    for requested_length in (
        REQUESTED_SEQUENCE_LENGTHS
    ):
        candidate = None

        for row in trajectory_rows:
            sequence_length = int(
                row["sequence_length"]
            )

            cell_sequence_index = int(
                row["cell_sequence_index"]
            )

            if sequence_length != requested_length:
                continue

            if (
                cell_sequence_index
                in used_sequence_indices
            ):
                continue

            candidate = row
            break

        require(
            candidate is not None,
            (
                "Could not find smoke trajectory with "
                f"length {requested_length}."
            ),
        )

        cell_sequence_index = int(
            candidate["cell_sequence_index"]
        )

        used_sequence_indices.add(
            cell_sequence_index
        )

        source = sequence_registry[
            cell_sequence_index
        ]

        selected.append(
            {
                "trajectory_index":
                    int(
                        candidate[
                            "trajectory_index"
                        ]
                    ),

                "cell_sequence_index":
                    cell_sequence_index,

                "source_sequence_id":
                    int(
                        candidate[
                            "source_sequence_id"
                        ]
                    ),

                "point_start":
                    int(
                        candidate["point_start"]
                    ),

                "point_count":
                    int(
                        candidate["point_count"]
                    ),

                "sequence_length":
                    requested_length,

                "operations":
                    source["operations"],
            }
        )

    return selected


def materialize_smoke_batch(
    selected,
    operation_to_index,
):
    noise_levels = np.load(
        NOISE_LEVEL_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    matching_noise_indices = np.flatnonzero(
        np.isclose(
            noise_levels,
            SMOKE_NOISE_FRACTION,
            rtol=0.0,
            atol=1e-7,
        )
    )

    require(
        matching_noise_indices.size == 1,
        (
            "Could not uniquely identify noise "
            f"fraction {SMOKE_NOISE_FRACTION}."
        ),
    )

    noise_index = int(
        matching_noise_indices[0]
    )

    fields = np.load(
        FIELD_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    require(
        tuple(fields.shape[2:])
        == (4, 32, 32),
        f"Unexpected field shape: {fields.shape}",
    )

    require(
        fields.shape[0]
        == len(noise_levels),
        "Noise and field axes differ.",
    )

    batch_size = len(selected)

    maximum_sequence_length = max(
        item["sequence_length"]
        for item in selected
    )

    maximum_point_count = (
        maximum_sequence_length + 1
    )

    observations = np.zeros(
        (
            batch_size,
            maximum_point_count,
            4,
            32,
            32,
        ),
        dtype=np.float32,
    )

    operation_ids = np.zeros(
        (
            batch_size,
            maximum_sequence_length,
        ),
        dtype=np.int64,
    )

    point_mask = np.zeros(
        (
            batch_size,
            maximum_point_count,
        ),
        dtype=np.bool_,
    )

    step_mask = np.zeros(
        (
            batch_size,
            maximum_sequence_length,
        ),
        dtype=np.bool_,
    )

    sequence_lengths = np.zeros(
        batch_size,
        dtype=np.int64,
    )

    trajectory_indices = np.zeros(
        batch_size,
        dtype=np.int64,
    )

    for batch_index, item in enumerate(
        selected
    ):
        start = item["point_start"]
        count = item["point_count"]
        length = item["sequence_length"]

        end = start + count

        require(
            end <= fields.shape[1],
            "Smoke trajectory exceeds field array.",
        )

        trajectory_fields = np.asarray(
            fields[
                noise_index,
                start:end,
            ],
            dtype=np.float32,
        )

        require(
            trajectory_fields.shape
            == (
                count,
                4,
                32,
                32,
            ),
            "Unexpected trajectory field shape.",
        )

        require(
            np.isfinite(
                trajectory_fields
            ).all(),
            "Non-finite smoke field detected.",
        )

        encoded_operations = []

        for operation_name in item[
            "operations"
        ]:
            require(
                operation_name
                in operation_to_index,
                (
                    "Unknown operation in sequence: "
                    f"{operation_name}"
                ),
            )

            encoded_operations.append(
                operation_to_index[
                    operation_name
                ]
            )

        require(
            len(encoded_operations) == length,
            "Encoded operation length changed.",
        )

        observations[
            batch_index,
            :count,
        ] = trajectory_fields

        operation_ids[
            batch_index,
            :length,
        ] = np.asarray(
            encoded_operations,
            dtype=np.int64,
        )

        point_mask[
            batch_index,
            :count,
        ] = True

        step_mask[
            batch_index,
            :length,
        ] = True

        sequence_lengths[
            batch_index
        ] = length

        trajectory_indices[
            batch_index
        ] = item[
            "trajectory_index"
        ]

    return {
        "noise_index":
            noise_index,

        "noise_levels":
            np.asarray(
                noise_levels,
                dtype=np.float32,
            ),

        "observations":
            observations,

        "operation_ids":
            operation_ids,

        "point_mask":
            point_mask,

        "step_mask":
            step_mask,

        "sequence_lengths":
            sequence_lengths,

        "trajectory_indices":
            trajectory_indices,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = (
        CONTRACT_SUMMARY_PATH,
        INTERFACE_SUMMARY_PATH,
        CELL_MANIFEST_PATH,
        SEQUENCE_MANIFEST_PATH,
        TRAJECTORY_INDEX_PATH,
        FIELD_PATH,
        NOISE_LEVEL_PATH,
        TRANSITION_TABLE_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

        assert_development_only(path)

    print(
        "[1/5] Validating smoke-test authorization"
    )

    contract_summary = load_json(
        CONTRACT_SUMMARY_PATH
    )

    interface_summary = load_json(
        INTERFACE_SUMMARY_PATH
    )

    cell_manifest = load_json(
        CELL_MANIFEST_PATH
    )

    require(
        contract_summary[
            "phase4dr32_status"
        ] == "execution_contract_frozen",
        "Execution contract is not frozen.",
    )

    require(
        contract_summary[
            "implementation_smoke_test_authorized"
        ] is True,
        "Implementation smoke test is not authorized.",
    )

    require(
        contract_summary[
            "tuning_execution_authorized"
        ] is False,
        "Tuning is already authorized.",
    )

    require(
        contract_summary[
            "test_open_count"
        ] == 0,
        "Test artifacts were accessed.",
    )

    require(
        interface_summary[
            "interface_snapshot_status"
        ] == "complete",
        "Interface snapshot is incomplete.",
    )

    require(
        interface_summary[
            "test_files_inspected"
        ] is False,
        "Interface snapshot inspected test files.",
    )

    print(
        "[2/5] Validating development manifests"
    )

    operation_to_index, element_ids = (
        load_operation_registry()
    )

    sequence_registry = (
        load_sequence_registry()
    )

    trajectory_rows = read_csv(
        TRAJECTORY_INDEX_PATH
    )

    validate_complete_cell(
        sequence_registry=sequence_registry,
        trajectory_rows=trajectory_rows,
        cell_manifest=cell_manifest,
    )

    print(
        "[3/5] Selecting four nonzero-length trajectories"
    )

    selected = select_smoke_trajectories(
        sequence_registry=sequence_registry,
        trajectory_rows=trajectory_rows,
    )

    print(
        "[4/5] Materializing memory-mapped smoke batch"
    )

    batch = materialize_smoke_batch(
        selected=selected,
        operation_to_index=(
            operation_to_index
        ),
    )

    batch_path = (
        OUTPUT_DIR
        / "tier_c_v4_smoke_batch.npz"
    )

    np.savez(
        batch_path,
        observations=batch[
            "observations"
        ],

        operation_ids=batch[
            "operation_ids"
        ],

        point_mask=batch[
            "point_mask"
        ],

        step_mask=batch[
            "step_mask"
        ],

        sequence_lengths=batch[
            "sequence_lengths"
        ],

        trajectory_indices=batch[
            "trajectory_indices"
        ],

        noise_levels=batch[
            "noise_levels"
        ],

        selected_noise_index=np.asarray(
            batch["noise_index"],
            dtype=np.int64,
        ),

        selected_noise_fraction=np.asarray(
            SMOKE_NOISE_FRACTION,
            dtype=np.float32,
        ),
    )

    print(
        "[5/5] Freezing smoke-batch provenance"
    )

    provenance = {
        "phase":
            (
                "4E-R3B Tier C v4 predictive "
                "smoke-batch preparation"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_cell":
            "train_joint",

        "source_noise_fraction":
            SMOKE_NOISE_FRACTION,

        "source_noise_index":
            batch["noise_index"],

        "requested_sequence_lengths":
            list(
                REQUESTED_SEQUENCE_LENGTHS
            ),

        "selected_trajectories":
            selected,

        "operation_to_index":
            operation_to_index,

        "operation_to_element_id":
            element_ids,

        "batch_shapes": {
            "observations":
                list(
                    batch[
                        "observations"
                    ].shape
                ),

            "operation_ids":
                list(
                    batch[
                        "operation_ids"
                    ].shape
                ),

            "point_mask":
                list(
                    batch[
                        "point_mask"
                    ].shape
                ),

            "step_mask":
                list(
                    batch[
                        "step_mask"
                    ].shape
                ),
        },

        "batch_dtypes": {
            "observations":
                str(
                    batch[
                        "observations"
                    ].dtype
                ),

            "operation_ids":
                str(
                    batch[
                        "operation_ids"
                    ].dtype
                ),

            "point_mask":
                str(
                    batch[
                        "point_mask"
                    ].dtype
                ),

            "step_mask":
                str(
                    batch[
                        "step_mask"
                    ].dtype
                ),
        },

        "source_hashes": {
            str(path):
                sha256_file(path)
            for path in required_paths
        },

        "smoke_batch_path":
            str(batch_path),

        "smoke_batch_sha256":
            sha256_file(batch_path),

        "development_data_only":
            True,

        "zero_length_sequences_excluded":
            True,

        "privileged_fields_read":
            False,

        "test_fields_read":
            False,

        "test_fields_generated":
            False,

        "test_metrics_computed":
            False,

        "model_parameters_created":
            False,

        "optimizer_steps_performed":
            0,

        "scientific_metrics_computed":
            False,

        "smoke_batch_status":
            "prepared_and_verified",

        "next_phase":
            (
                "4E-R3C Tier C v4 model-family "
                "forward-backward-rollout smoke execution"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "smoke_batch_provenance.json",
        provenance,
    )

    print(
        "Tier C v4 development-only smoke batch "
        "prepared successfully."
    )

    print(
        json.dumps(
            provenance,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
