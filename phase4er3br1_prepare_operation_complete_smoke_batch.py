from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase4er3b_prepare_tier_c_v4_smoke_batch as base


PROTOCOL_VERSION = "tier_c_v4"

TARGET_TRAJECTORY_COUNT = 12
PHYSICAL_MICROBATCH_SIZE = 4

REQUIRED_SEQUENCE_LENGTHS = {
    1,
    2,
    3,
    4,
}

SOURCE_PROVENANCE_PATH = (
    Path(
        "outputs/phase4er3b_tier_c_v4_smoke_batch"
    )
    / "smoke_batch_provenance.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4er3br1_tier_c_v4_operation_complete_smoke_batch"
)

OUTPUT_BATCH_PATH = (
    OUTPUT_DIR
    / "tier_c_v4_operation_complete_smoke_batch.npz"
)

OUTPUT_PROVENANCE_PATH = (
    OUTPUT_DIR
    / "operation_complete_smoke_batch_provenance.json"
)


def require(
    condition,
    message,
):
    if not condition:
        raise AssertionError(message)


def candidate_from_row(
    row,
    sequence_registry,
):
    cell_sequence_index = int(
        row["cell_sequence_index"]
    )

    sequence = sequence_registry[
        cell_sequence_index
    ]

    return {
        "trajectory_index":
            int(row["trajectory_index"]),

        "cell_sequence_index":
            cell_sequence_index,

        "source_sequence_id":
            int(row["source_sequence_id"]),

        "point_start":
            int(row["point_start"]),

        "point_count":
            int(row["point_count"]),

        "sequence_length":
            int(row["sequence_length"]),

        "operations":
            list(sequence["operations"]),
    }


def build_unique_sequence_candidates(
    trajectory_rows,
    sequence_registry,
):
    """
    Retain only the first trajectory for each sequence.

    This prevents the smoke batch from selecting multiple carrier
    realizations of the same operation sequence when distinct
    sequences are available.
    """

    candidates = []
    seen_sequence_indices = set()

    sorted_rows = sorted(
        trajectory_rows,
        key=lambda row: int(
            row["trajectory_index"]
        ),
    )

    for row in sorted_rows:
        cell_sequence_index = int(
            row["cell_sequence_index"]
        )

        if (
            cell_sequence_index
            in seen_sequence_indices
        ):
            continue

        seen_sequence_indices.add(
            cell_sequence_index
        )

        candidate = candidate_from_row(
            row=row,
            sequence_registry=sequence_registry,
        )

        if candidate["sequence_length"] not in {
            1,
            2,
            3,
            4,
        }:
            continue

        require(
            candidate["point_count"]
            == candidate["sequence_length"] + 1,
            (
                "Candidate point count does not equal "
                "sequence length plus one."
            ),
        )

        candidates.append(candidate)

    return candidates


def select_operation_complete_batch(
    candidates,
    operation_to_index,
):
    selected = []
    selected_trajectory_ids = set()
    selected_sequence_ids = set()

    def add_candidate(candidate):
        trajectory_index = candidate[
            "trajectory_index"
        ]

        sequence_index = candidate[
            "cell_sequence_index"
        ]

        require(
            trajectory_index
            not in selected_trajectory_ids,
            (
                "Duplicate trajectory selected: "
                f"{trajectory_index}"
            ),
        )

        require(
            sequence_index
            not in selected_sequence_ids,
            (
                "Duplicate sequence selected: "
                f"{sequence_index}"
            ),
        )

        selected.append(candidate)

        selected_trajectory_ids.add(
            trajectory_index
        )

        selected_sequence_ids.add(
            sequence_index
        )

    # --------------------------------------------------------
    # 1. Prefer one exact single-step probe for each primitive.
    #
    # Some frozen primitives, especially identity, may not have
    # a standalone one-step sequence in the development manifest.
    # In that case, select the shortest available sequence that
    # contains that primitive.
    # --------------------------------------------------------

    primitive_names = [
        name
        for name, _
        in sorted(
            operation_to_index.items(),
            key=lambda item: item[1],
        )
    ]

    for primitive_name in primitive_names:
        exact_single_step = [
            candidate
            for candidate in candidates
            if candidate["sequence_length"] == 1
            and candidate["operations"]
            == [primitive_name]
            and candidate["trajectory_index"]
            not in selected_trajectory_ids
            and candidate["cell_sequence_index"]
            not in selected_sequence_ids
        ]

        if exact_single_step:
            add_candidate(
                exact_single_step[0]
            )

            continue

        containing_candidates = [
            candidate
            for candidate in candidates
            if primitive_name
            in candidate["operations"]
            and candidate["trajectory_index"]
            not in selected_trajectory_ids
            and candidate["cell_sequence_index"]
            not in selected_sequence_ids
        ]

        require(
            containing_candidates,
            (
                "No development trajectory contains "
                f"primitive {primitive_name}."
            ),
        )

        containing_candidates.sort(
            key=lambda candidate: (
                candidate["sequence_length"],
                candidate["operations"].index(
                    primitive_name
                ),
                candidate["trajectory_index"],
            )
        )

        add_candidate(
            containing_candidates[0]
        )

    # --------------------------------------------------------
    # 2. Explicit variable-length probes for lengths 2, 3, 4.
    # --------------------------------------------------------

    for required_length in (
        2,
        3,
        4,
    ):
        matching = [
            candidate
            for candidate in candidates
            if candidate["sequence_length"]
            == required_length
            and candidate["trajectory_index"]
            not in selected_trajectory_ids
            and candidate["cell_sequence_index"]
            not in selected_sequence_ids
        ]

        require(
            matching,
            (
                "No distinct development trajectory "
                f"was found for length {required_length}."
            ),
        )

        # Prefer sequences with greater primitive diversity.
        matching.sort(
            key=lambda candidate: (
                -len(
                    set(
                        candidate[
                            "operations"
                        ]
                    )
                ),
                candidate[
                    "trajectory_index"
                ],
            )
        )

        add_candidate(
            matching[0]
        )

    # --------------------------------------------------------
    # 3. Fill to exactly 12 trajectories.
    # --------------------------------------------------------

    remaining = [
        candidate
        for candidate in candidates
        if candidate["trajectory_index"]
        not in selected_trajectory_ids
        and candidate["cell_sequence_index"]
        not in selected_sequence_ids
    ]

    remaining.sort(
        key=lambda candidate: (
            candidate[
                "sequence_length"
            ],
            -len(
                set(
                    candidate[
                        "operations"
                    ]
                )
            ),
            candidate[
                "trajectory_index"
            ],
        )
    )

    for candidate in remaining:
        if len(selected) >= (
            TARGET_TRAJECTORY_COUNT
        ):
            break

        add_candidate(candidate)

    require(
        len(selected)
        == TARGET_TRAJECTORY_COUNT,
        (
            "Expected "
            f"{TARGET_TRAJECTORY_COUNT} trajectories, "
            f"selected {len(selected)}."
        ),
    )

    require(
        len(selected)
        % PHYSICAL_MICROBATCH_SIZE
        == 0,
        (
            "Smoke batch is not divisible by the "
            "frozen physical microbatch size."
        ),
    )

    return selected


def validate_operation_coverage(
    selected,
    candidates,
    operation_to_index,
):
    observed_operations = {
        operation_name
        for candidate in selected
        for operation_name
        in candidate["operations"]
    }

    expected_operations = set(
        operation_to_index
    )

    missing_operations = sorted(
        expected_operations
        - observed_operations
    )

    unexpected_operations = sorted(
        observed_operations
        - expected_operations
    )

    primitive_order = sorted(
        expected_operations,
        key=lambda name:
            operation_to_index[name],
    )

    single_step_available = {
        operation_name:
            any(
                candidate[
                    "sequence_length"
                ] == 1
                and candidate[
                    "operations"
                ] == [operation_name]
                for candidate
                in candidates
            )
        for operation_name
        in primitive_order
    }

    single_step_selected = {
        operation_name:
            any(
                candidate[
                    "sequence_length"
                ] == 1
                and candidate[
                    "operations"
                ] == [operation_name]
                for candidate
                in selected
            )
        for operation_name
        in primitive_order
    }

    single_step_unavailable = [
        operation_name
        for operation_name
        in primitive_order
        if not single_step_available[
            operation_name
        ]
    ]

    missing_available_single_step = [
        operation_name
        for operation_name
        in primitive_order
        if single_step_available[
            operation_name
        ]
        and not single_step_selected[
            operation_name
        ]
    ]

    primitive_coverage_modes = {}

    for operation_name in primitive_order:
        if single_step_selected[
            operation_name
        ]:
            primitive_coverage_modes[
                operation_name
            ] = "exact_single_step"

        elif operation_name in (
            observed_operations
        ):
            shortest_selected_length = min(
                candidate[
                    "sequence_length"
                ]
                for candidate
                in selected
                if operation_name
                in candidate[
                    "operations"
                ]
            )

            primitive_coverage_modes[
                operation_name
            ] = (
                "contained_in_length_"
                f"{shortest_selected_length}_sequence"
            )

        else:
            primitive_coverage_modes[
                operation_name
            ] = "missing"

    observed_lengths = {
        candidate[
            "sequence_length"
        ]
        for candidate in selected
    }

    missing_lengths = sorted(
        REQUIRED_SEQUENCE_LENGTHS
        - observed_lengths
    )

    require(
        not missing_operations,
        (
            "Primitive coverage remains incomplete: "
            f"{missing_operations}"
        ),
    )

    require(
        not unexpected_operations,
        (
            "Unexpected primitive labels detected: "
            f"{unexpected_operations}"
        ),
    )

    require(
        not missing_available_single_step,
        (
            "Available single-step primitive probes "
            "were not selected: "
            f"{missing_available_single_step}"
        ),
    )

    require(
        not missing_lengths,
        (
            "Variable-length coverage remains "
            f"incomplete: {missing_lengths}"
        ),
    )

    return {
        "observed_operations":
            primitive_order,

        "missing_operations":
            missing_operations,

        "unexpected_operations":
            unexpected_operations,

        "single_step_available":
            single_step_available,

        "single_step_selected":
            single_step_selected,

        "single_step_unavailable_primitives":
            single_step_unavailable,

        "missing_available_single_step_probes":
            missing_available_single_step,

        "primitive_coverage_modes":
            primitive_coverage_modes,

        "observed_sequence_lengths":
            sorted(
                observed_lengths
            ),

        "missing_sequence_lengths":
            missing_lengths,

        "all_primitives_present":
            True,

        "all_primitives_have_single_step_probe":
            all(
                single_step_selected.values()
            ),

        "all_available_single_step_probes_present":
            True,

        "all_required_sequence_lengths_present":
            True,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = (
        SOURCE_PROVENANCE_PATH,
        base.CONTRACT_SUMMARY_PATH,
        base.INTERFACE_SUMMARY_PATH,
        base.CELL_MANIFEST_PATH,
        base.SEQUENCE_MANIFEST_PATH,
        base.TRAJECTORY_INDEX_PATH,
        base.FIELD_PATH,
        base.NOISE_LEVEL_PATH,
        base.TRANSITION_TABLE_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

        base.assert_development_only(
            path
        )

    print(
        "[1/6] Validating the original smoke-batch record"
    )

    original_provenance = base.load_json(
        SOURCE_PROVENANCE_PATH
    )

    require(
        original_provenance[
            "smoke_batch_status"
        ] == "prepared_and_verified",
        "Original smoke batch was not verified.",
    )

    require(
        original_provenance[
            "optimizer_steps_performed"
        ] == 0,
        (
            "Original smoke-batch preparation "
            "performed optimizer steps."
        ),
    )

    require(
        original_provenance[
            "test_fields_read"
        ] is False,
        "Original preparation read test fields.",
    )

    print(
        "[2/6] Validating frozen execution authorization"
    )

    contract_summary = base.load_json(
        base.CONTRACT_SUMMARY_PATH
    )

    interface_summary = base.load_json(
        base.INTERFACE_SUMMARY_PATH
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
        "Smoke execution is not authorized.",
    )

    require(
        contract_summary[
            "tuning_execution_authorized"
        ] is False,
        "Tuning was already authorized.",
    )

    require(
        contract_summary[
            "test_open_count"
        ] == 0,
        "Test data was opened.",
    )

    require(
        interface_summary[
            "interface_snapshot_status"
        ] == "complete",
        "Interface snapshot is incomplete.",
    )

    print(
        "[3/6] Loading operation and development registries"
    )

    operation_to_index, element_ids = (
        base.load_operation_registry()
    )

    sequence_registry = (
        base.load_sequence_registry()
    )

    trajectory_rows = base.read_csv(
        base.TRAJECTORY_INDEX_PATH
    )

    cell_manifest = base.load_json(
        base.CELL_MANIFEST_PATH
    )

    base.validate_complete_cell(
        sequence_registry=sequence_registry,
        trajectory_rows=trajectory_rows,
        cell_manifest=cell_manifest,
    )

    print(
        "[4/6] Selecting operation-complete trajectories"
    )

    candidates = (
        build_unique_sequence_candidates(
            trajectory_rows=trajectory_rows,
            sequence_registry=(
                sequence_registry
            ),
        )
    )

    selected = (
        select_operation_complete_batch(
            candidates=candidates,
            operation_to_index=(
                operation_to_index
            ),
        )
    )

    coverage = (
        validate_operation_coverage(
            selected=selected,
            candidates=candidates,
            operation_to_index=(
                operation_to_index
            ),
        )
    )

    print(
        "[5/6] Materializing the corrected smoke batch"
    )

    batch = base.materialize_smoke_batch(
        selected=selected,
        operation_to_index=(
            operation_to_index
        ),
    )

    np.savez(
        OUTPUT_BATCH_PATH,

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
            base.SMOKE_NOISE_FRACTION,
            dtype=np.float32,
        ),
    )

    print(
        "[6/6] Freezing corrected smoke-batch provenance"
    )

    physical_microbatch_count = (
        TARGET_TRAJECTORY_COUNT
        // PHYSICAL_MICROBATCH_SIZE
    )

    provenance = {
        "phase":
            (
                "4E-R3B-R1 Tier C v4 "
                "operation-complete smoke-batch preparation"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "correction_reason":
            (
                "The original four-trajectory smoke batch "
                "contained only identity and cycle. It was "
                "data-valid but did not exercise all six "
                "primitive operation paths."
            ),

        "original_smoke_batch_modified":
            False,

        "original_smoke_batch_path":
            original_provenance[
                "smoke_batch_path"
            ],

        "original_smoke_batch_sha256":
            original_provenance[
                "smoke_batch_sha256"
            ],

        "source_cell":
            "train_joint",

        "source_noise_fraction":
            base.SMOKE_NOISE_FRACTION,

        "source_noise_index":
            batch["noise_index"],

        "selected_trajectory_count":
            len(selected),

        "physical_microbatch_size":
            PHYSICAL_MICROBATCH_SIZE,

        "physical_microbatch_count":
            physical_microbatch_count,

        "selected_trajectories":
            selected,

        "operation_to_index":
            operation_to_index,

        "operation_to_element_id":
            element_ids,

        "coverage":
            coverage,

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

            "sequence_lengths":
                list(
                    batch[
                        "sequence_lengths"
                    ].shape
                ),
        },

        "smoke_batch_path":
            str(
                OUTPUT_BATCH_PATH
            ),

        "smoke_batch_sha256":
            base.sha256_file(
                OUTPUT_BATCH_PATH
            ),

        "development_data_only":
            True,

        "zero_length_sequences_excluded":
            True,

        "all_six_primitives_covered":
            coverage[
                "all_primitives_present"
            ],

        "all_six_single_step_probes_present":
            coverage[
                "all_primitives_have_single_step_probe"
            ],

        "all_available_single_step_probes_present":
            coverage[
                "all_available_single_step_probes_present"
            ],

        "single_step_unavailable_primitives":
            coverage[
                "single_step_unavailable_primitives"
            ],

        "variable_length_sequences_covered":
            coverage[
                "all_required_sequence_lengths_present"
            ],

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

        "operation_complete_smoke_batch_status":
            "prepared_and_verified",

        "next_phase":
            (
                "4E-R3C Tier C v4 model-family "
                "forward-backward-rollout smoke execution"
            ),
    }

    base.write_json(
        OUTPUT_PROVENANCE_PATH,
        provenance,
    )

    print(
        "Operation-complete Tier C v4 smoke batch "
        "prepared successfully."
    )

    print(
        json.dumps(
            provenance,
            indent=2,
        )
    )

    print(
        f"Outputs written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
