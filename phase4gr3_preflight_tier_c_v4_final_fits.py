from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "tier_c_v4"

TRAINABLE_MODEL_IDS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

FINAL_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

NOISE_FRACTIONS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

EXPECTED_ORIGINAL_FINAL_ROWS = 135
EXPECTED_TRAINABLE_FINAL_FITS = 130
EXPECTED_PERSISTENCE_ROWS = 5

EXPECTED_TRAINABLE_COUNTS = {
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
    "OCM": 25,
}


PROTOCOL_ROOT = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol"
)

FINAL_RUN_REGISTRY_PATH = (
    PROTOCOL_ROOT
    / "tier_c_v4_final_run_registry.csv"
)

MODEL_REGISTRY_PATH = (
    PROTOCOL_ROOT
    / "tier_c_v4_model_registry.csv"
)

PROTOCOL_SUMMARY_PATH = (
    PROTOCOL_ROOT
    / "phase4dr3_predictive_protocol_summary.json"
)

TEST_OPENING_POLICY_PATH = (
    PROTOCOL_ROOT
    / "tier_c_v4_test_opening_policy.json"
)

EXECUTION_CONTRACT_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "tier_c_v4_execution_contract.json"
)

EXECUTION_CONTRACT_SUMMARY_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "phase4dr32_execution_contract_summary.json"
)

TUNING_SELECTION_SUMMARY_PATH = Path(
    "outputs/"
    "phase4fr3c_tier_c_v4_tuning_selection/"
    "phase4fr3c_tuning_selection_summary.json"
)

SELECTED_TUNING_REGISTRY_PATH = Path(
    "outputs/"
    "phase4fr3c_tier_c_v4_tuning_selection/"
    "selected_tuning_registry.json"
)

REPAIRED_DATA_PREFLIGHT_PATH = Path(
    "outputs/"
    "phase4fr3ar1_tier_c_v4_visible_trajectory_indexes/"
    "phase4fr3ar1_repaired_tuning_preflight_summary.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3_tier_c_v4_final_fit_preflight"
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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    require(
        bool(rows),
        f"No rows supplied for {path}.",
    )

    fieldnames: list[str] = []

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


def normalized_false(value: str) -> bool:
    return (
        value
        .strip()
        .lower()
        in {
            "false",
            "0",
            "no",
        }
    )


def noise_token(
    noise_fraction: float,
) -> str:
    return (
        f"{noise_fraction:g}"
        .replace(".", "p")
    )


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


def validate_protocol_sources() -> dict[str, Any]:
    required_paths = (
        FINAL_RUN_REGISTRY_PATH,
        MODEL_REGISTRY_PATH,
        PROTOCOL_SUMMARY_PATH,
        TEST_OPENING_POLICY_PATH,
        EXECUTION_CONTRACT_PATH,
        EXECUTION_CONTRACT_SUMMARY_PATH,
        TUNING_SELECTION_SUMMARY_PATH,
        SELECTED_TUNING_REGISTRY_PATH,
        REPAIRED_DATA_PREFLIGHT_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    protocol_summary = load_json(
        PROTOCOL_SUMMARY_PATH
    )

    execution_contract = load_json(
        EXECUTION_CONTRACT_PATH
    )

    execution_summary = load_json(
        EXECUTION_CONTRACT_SUMMARY_PATH
    )

    tuning_selection_summary = load_json(
        TUNING_SELECTION_SUMMARY_PATH
    )

    selected_registry = load_json(
        SELECTED_TUNING_REGISTRY_PATH
    )

    test_policy = load_json(
        TEST_OPENING_POLICY_PATH
    )

    data_preflight = load_json(
        REPAIRED_DATA_PREFLIGHT_PATH
    )

    require(
        protocol_summary[
            "planned_final_run_count"
        ] == EXPECTED_ORIGINAL_FINAL_ROWS,
        "Original final-run count is not 135.",
    )

    require(
        protocol_summary[
            "planned_trainable_final_fit_count"
        ] == EXPECTED_TRAINABLE_FINAL_FITS,
        "Trainable final-fit count is not 130.",
    )

    require(
        protocol_summary[
            "planned_persistence_evaluation_count"
        ] == EXPECTED_PERSISTENCE_ROWS,
        "Persistence count is not five.",
    )

    require(
        [
            int(value)
            for value
            in protocol_summary[
                "final_seeds"
            ]
        ] == list(FINAL_SEEDS),
        "Frozen final seeds changed.",
    )

    require(
        all(
            math.isclose(
                float(observed),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for observed, expected
            in zip(
                protocol_summary[
                    "noise_fractions"
                ],
                NOISE_FRACTIONS,
            )
        ),
        "Frozen noise fractions changed.",
    )

    require(
        protocol_summary[
            "field_codec_pretraining"
        ] is False,
        "Field-codec pretraining was enabled.",
    )

    require(
        execution_summary[
            "phase4dr32_status"
        ] == "execution_contract_frozen",
        "Execution contract is not frozen.",
    )

    duration = execution_contract[
        "training_duration"
    ]

    require(
        duration[
            "maximum_epochs"
        ] == 200,
        "Maximum epoch count changed.",
    )

    require(
        duration[
            "early_stopping_patience"
        ] == 25,
        "Early-stopping patience changed.",
    )

    require(
        duration[
            "checkpoint_selection_cell"
        ] == "val_joint",
        "Checkpoint-selection cell changed.",
    )

    require(
        duration[
            "checkpoint_selection_metric"
        ] == "noisy_target_rollout_mse",
        "Checkpoint-selection metric changed.",
    )

    batching = execution_contract[
        "batching"
    ]

    require(
        batching[
            "physical_microbatch_size_trajectories"
        ] == 4,
        "Physical microbatch size changed.",
    )

    require(
        batching[
            "gradient_accumulation_steps"
        ] == 64,
        "Gradient accumulation changed.",
    )

    require(
        batching[
            "effective_batch_size_trajectories"
        ] == 256,
        "Effective batch size changed.",
    )

    require(
        execution_contract[
            "gradient_clipping_policy"
        ][
            "maximum_norm"
        ] == 5.0,
        "Gradient clipping changed.",
    )

    require(
        execution_contract[
            "mixed_precision_policy"
        ][
            "enabled"
        ] is False,
        "Mixed precision was enabled.",
    )

    require(
        execution_contract[
            "codec_initialization_policy"
        ][
            "seed_before_model_construction"
        ] is True,
        (
            "Seed-before-model-construction "
            "policy changed."
        ),
    )

    b1_amendment = execution_contract[
        "b1_contract_amendment"
    ]

    require(
        int(
            b1_amendment[
                "fixed_seed"
            ]
        ) == 11,
        "B1 fixed seed is not 11.",
    )

    require(
        b1_amendment[
            "one_fit_per_noise_condition"
        ] is True,
        (
            "B1 one-fit-per-noise policy "
            "is not frozen."
        ),
    )

    require(
        tuning_selection_summary[
            "phase4fr3c_status"
        ] == "tuning_selection_frozen",
        "Tuning selection is not frozen.",
    )

    require(
        tuning_selection_summary[
            "selected_registry_frozen"
        ] is True,
        "Selected registry is not frozen.",
    )

    require(
        tuning_selection_summary[
            "additional_tuning_authorized"
        ] is False,
        "Additional tuning is still authorized.",
    )

    require(
        tuning_selection_summary[
            "final_fit_execution_authorized"
        ] is True,
        "Final fitting is not authorized.",
    )

    require(
        tuning_selection_summary[
            "test_artifact_generation_authorized"
        ] is False,
        "Test generation is already authorized.",
    )

    require(
        tuning_selection_summary[
            "test_evaluation_authorized"
        ] is False,
        "Test evaluation is already authorized.",
    )

    require(
        tuning_selection_summary[
            "test_open_count"
        ] == 0,
        "Test data was opened.",
    )

    require(
        selected_registry[
            "selected_configuration_count"
        ] == 6,
        "Expected six selected configurations.",
    )

    require(
        selected_registry[
            "test_data_read"
        ] is False,
        "Selected registry records test access.",
    )

    require(
        data_preflight[
            "training_must_use_released_visible_indexes"
        ] is True,
        "Visible structural indexes are not mandatory.",
    )

    require(
        data_preflight[
            "training_may_not_read_privileged_directory"
        ] is True,
        "Privileged-directory prohibition is absent.",
    )

    require(
        test_policy[
            "test_generation_authorized_now"
        ] is False,
        "Test generation is already authorized.",
    )

    require(
        test_policy[
            "test_evaluation_authorized_now"
        ] is False,
        "Test evaluation is already authorized.",
    )

    require(
        test_policy[
            "test_open_count"
        ] == 0,
        "Test opening count is not zero.",
    )

    return {
        "required_paths":
            required_paths,

        "protocol_summary":
            protocol_summary,

        "execution_contract":
            execution_contract,

        "tuning_selection_summary":
            tuning_selection_summary,

        "selected_registry":
            selected_registry,

        "test_policy":
            test_policy,

        "data_preflight":
            data_preflight,
    }


def validate_model_registry() -> list[dict[str, str]]:
    rows = load_csv(
        MODEL_REGISTRY_PATH
    )

    require(
        len(rows) == 7,
        "Expected seven model-registry rows.",
    )

    rows_by_model = {
        row[
            "model_id"
        ]:
            row
        for row in rows
    }

    require(
        set(
            rows_by_model
        )
        == {
            "B0",
            *TRAINABLE_MODEL_IDS,
        },
        "Model registry has unexpected model IDs.",
    )

    require(
        rows_by_model[
            "B1"
        ][
            "final_seed_policy"
        ]
        == "one deterministic fit per noise level",
        "B1 final-seed policy changed.",
    )

    for model_id in (
        "B2",
        "B3",
        "B4",
        "B5",
        "OCM",
    ):
        require(
            rows_by_model[
                model_id
            ][
                "final_seed_policy"
            ]
            == "five frozen seeds per noise level",
            (
                f"{model_id} final-seed policy "
                "changed."
            ),
        )

    require(
        rows_by_model[
            "OCM"
        ][
            "primary_model"
        ].strip().lower()
        == "true",
        "OCM is no longer the primary model.",
    )

    return rows


def load_selected_configurations(
    selected_registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    selected_rows = (
        selected_registry[
            "selected_configurations"
        ]
    )

    selected_by_model = {
        row[
            "model_id"
        ]:
            row
        for row in selected_rows
    }

    require(
        set(
            selected_by_model
        )
        == set(
            TRAINABLE_MODEL_IDS
        ),
        (
            "Selected configurations do not cover "
            "all trainable models."
        ),
    )

    for model_id, row in (
        selected_by_model.items()
    ):
        require(
            row[
                "selected"
            ] is True,
            f"{model_id} was not marked selected.",
        )

        require(
            row[
                "family_rank"
            ] == 1,
            f"{model_id} selected row is not rank one.",
        )

        checkpoint_path = Path(
            row[
                "best_checkpoint_path"
            ]
        )

        require(
            checkpoint_path.exists(),
            (
                f"Selected tuning checkpoint missing "
                f"for {model_id}."
            ),
        )

        require(
            sha256_file(
                checkpoint_path
            )
            == row[
                "best_checkpoint_sha256"
            ],
            (
                f"Selected tuning checkpoint hash "
                f"changed for {model_id}."
            ),
        )

    return selected_by_model


def validate_original_final_registry() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    rows = load_csv(
        FINAL_RUN_REGISTRY_PATH
    )

    require(
        len(rows)
        == EXPECTED_ORIGINAL_FINAL_ROWS,
        (
            "Original final-run registry does not "
            "contain 135 rows."
        ),
    )

    require(
        all(
            normalized_false(
                row[
                    "training_allowed"
                ]
            )
            for row in rows
        ),
        (
            "Original final-run registry contains "
            "a modified training_allowed value."
        ),
    )

    require(
        all(
            row[
                "status"
            ] == "planned"
            for row in rows
        ),
        "Original final-run statuses changed.",
    )

    b0_rows = [
        row
        for row in rows
        if row[
            "model_id"
        ] == "B0"
    ]

    trainable_rows = [
        row
        for row in rows
        if row[
            "model_id"
        ] in TRAINABLE_MODEL_IDS
    ]

    require(
        len(b0_rows)
        == EXPECTED_PERSISTENCE_ROWS,
        "Expected five B0 rows.",
    )

    require(
        len(trainable_rows)
        == EXPECTED_TRAINABLE_FINAL_FITS,
        "Expected 130 trainable final-fit rows.",
    )

    counts = Counter(
        row[
            "model_id"
        ]
        for row in trainable_rows
    )

    require(
        dict(counts)
        == EXPECTED_TRAINABLE_COUNTS,
        (
            "Unexpected trainable final-fit counts: "
            f"{dict(counts)}"
        ),
    )

    noise_by_model: dict[str, set[float]] = (
        defaultdict(set)
    )

    seed_by_model: dict[str, set[str]] = (
        defaultdict(set)
    )

    for row in rows:
        noise_by_model[
            row[
                "model_id"
            ]
        ].add(
            float(
                row[
                    "noise_fraction"
                ]
            )
        )

        seed_by_model[
            row[
                "model_id"
            ]
        ].add(
            row[
                "seed"
            ]
        )

    for model_id in {
        "B0",
        *TRAINABLE_MODEL_IDS,
    }:
        require(
            noise_by_model[
                model_id
            ] == set(
                NOISE_FRACTIONS
            ),
            (
                f"{model_id} does not cover all "
                "five noise levels."
            ),
        )

    require(
        seed_by_model[
            "B1"
        ] == {
            "deterministic"
        },
        "B1 original seed token changed.",
    )

    for model_id in (
        "B2",
        "B3",
        "B4",
        "B5",
        "OCM",
    ):
        require(
            seed_by_model[
                model_id
            ]
            == {
                str(seed)
                for seed
                in FINAL_SEEDS
            },
            (
                f"{model_id} final seed set "
                "changed."
            ),
        )

    return rows, trainable_rows, b0_rows


def build_resolved_registry(
    original_trainable_rows: list[dict[str, str]],
    selected_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved_rows = []

    run_ids = set()

    for original_index, row in enumerate(
        original_trainable_rows
    ):
        model_id = row[
            "model_id"
        ]

        noise_fraction = float(
            row[
                "noise_fraction"
            ]
        )

        selected = selected_by_model[
            model_id
        ]

        if model_id == "B1":
            require(
                row[
                    "seed"
                ] == "deterministic",
                "B1 original seed token changed.",
            )

            effective_seed = 11

            require(
                row[
                    "run_type"
                ]
                == "deterministic_final_fit",
                "B1 run type changed.",
            )

        else:
            effective_seed = int(
                row[
                    "seed"
                ]
            )

            require(
                effective_seed
                in FINAL_SEEDS,
                (
                    f"Unexpected seed "
                    f"{effective_seed} for {model_id}."
                ),
            )

            require(
                row[
                    "run_type"
                ] == "neural_final_fit",
                (
                    f"{model_id} final-fit "
                    "run type changed."
                ),
            )

        run_id = (
            f"{model_id}_"
            f"noise_{noise_token(noise_fraction)}_"
            f"seed_{effective_seed}"
        )

        require(
            run_id not in run_ids,
            f"Duplicate run ID: {run_id}",
        )

        run_ids.add(run_id)

        output_directory = (
            Path(
                "outputs/"
                "phase4gr3b_tier_c_v4_final_fits"
            )
            / model_id
            / run_id
        )

        resolved_rows.append(
            {
                "execution_index":
                    original_index,

                "run_id":
                    run_id,

                "model_id":
                    model_id,

                "run_type":
                    row[
                        "run_type"
                    ],

                "noise_fraction":
                    noise_fraction,

                "noise_index":
                    list(
                        NOISE_FRACTIONS
                    ).index(
                        noise_fraction
                    ),

                "original_seed_token":
                    row[
                        "seed"
                    ],

                "effective_seed":
                    effective_seed,

                "selected_configuration_id":
                    selected[
                        "configuration_id"
                    ],

                "selected_configuration_json":
                    json.dumps(
                        selected[
                            "configuration"
                        ],
                        sort_keys=True,
                    ),

                "selected_tuning_best_epoch":
                    selected[
                        "best_epoch"
                    ],

                "selected_tuning_validation_metric":
                    selected[
                        "best_validation_rollout_mse"
                    ],

                "selected_tuning_checkpoint_path":
                    selected[
                        "best_checkpoint_path"
                    ],

                "selected_tuning_checkpoint_sha256":
                    selected[
                        "best_checkpoint_sha256"
                    ],

                "initialize_from_tuning_checkpoint":
                    False,

                "fresh_seeded_initialization_required":
                    True,

                "training_cell":
                    "train_joint",

                "checkpoint_selection_cell":
                    "val_joint",

                "checkpoint_selection_metric":
                    "noisy_target_rollout_mse",

                "checkpoint_selection_direction":
                    "minimize",

                "maximum_epochs":
                    200,

                "early_stopping_patience":
                    25,

                "selected_tuning_epoch_used_as_fixed_duration":
                    False,

                "independent_early_stopping_required":
                    True,

                "physical_microbatch_size":
                    4,

                "gradient_accumulation_steps":
                    64,

                "effective_batch_size":
                    256,

                "gradient_clip_norm":
                    5.0,

                "mixed_precision":
                    False,

                "runtime_observation_dimension":
                    128,

                "original_training_allowed":
                    False,

                "execution_authorized_by_phase4fr3c":
                    True,

                "final_fit_training_allowed":
                    True,

                "privileged_directory_access_allowed":
                    False,

                "clean_target_access_allowed":
                    False,

                "test_access_allowed":
                    False,

                "output_directory":
                    str(
                        output_directory
                    ),

                "execution_status":
                    "pending",
            }
        )

    require(
        len(resolved_rows)
        == EXPECTED_TRAINABLE_FINAL_FITS,
        "Resolved registry does not contain 130 rows.",
    )

    require(
        len(run_ids)
        == EXPECTED_TRAINABLE_FINAL_FITS,
        "Resolved run IDs are not unique.",
    )

    return resolved_rows


def build_b0_deferred_registry(
    b0_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    deferred_rows = []

    for index, row in enumerate(
        b0_rows
    ):
        noise_fraction = float(
            row[
                "noise_fraction"
            ]
        )

        deferred_rows.append(
            {
                "persistence_index":
                    index,

                "model_id":
                    "B0",

                "noise_fraction":
                    noise_fraction,

                "noise_index":
                    list(
                        NOISE_FRACTIONS
                    ).index(
                        noise_fraction
                    ),

                "seed":
                    "deterministic",

                "run_type":
                    "evaluation_only",

                "training_required":
                    False,

                "included_in_130_trainable_final_fits":
                    False,

                "execution_status":
                    (
                        "deferred_to_dedicated_"
                        "persistence_artifact_phase"
                    ),

                "test_access_allowed":
                    False,
            }
        )

    return deferred_rows


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1/6] Validating frozen protocol sources"
    )

    sources = validate_protocol_sources()

    print(
        "[2/6] Validating model seed policies"
    )

    model_registry_rows = (
        validate_model_registry()
    )

    print(
        "[3/6] Validating original 135-row final registry"
    )

    (
        original_rows,
        original_trainable_rows,
        b0_rows,
    ) = validate_original_final_registry()

    print(
        "[4/6] Resolving selected configurations"
    )

    selected_by_model = (
        load_selected_configurations(
            sources[
                "selected_registry"
            ]
        )
    )

    print(
        "[5/6] Building authorized 130-run execution registry"
    )

    resolved_rows = (
        build_resolved_registry(
            original_trainable_rows,
            selected_by_model,
        )
    )

    b0_deferred_rows = (
        build_b0_deferred_registry(
            b0_rows
        )
    )

    resolved_registry_path = (
        OUTPUT_DIR
        / "resolved_trainable_final_fit_registry.csv"
    )

    write_csv(
        resolved_registry_path,
        resolved_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "deferred_b0_persistence_registry.csv",
        b0_deferred_rows,
    )

    selected_configuration_map = {
        model_id:
            selected_by_model[
                model_id
            ][
                "configuration_id"
            ]
        for model_id in TRAINABLE_MODEL_IDS
    }

    run_counts = Counter(
        row[
            "model_id"
        ]
        for row in resolved_rows
    )

    seed_counts = Counter(
        row[
            "effective_seed"
        ]
        for row in resolved_rows
    )

    noise_counts = Counter(
        row[
            "noise_fraction"
        ]
        for row in resolved_rows
    )

    print(
        "[6/6] Freezing final-fit preflight"
    )

    source_hashes = {
        str(path):
            sha256_file(path)
        for path in sources[
            "required_paths"
        ]
    }

    source_hashes[
        str(
            MODEL_REGISTRY_PATH
        )
    ] = sha256_file(
        MODEL_REGISTRY_PATH
    )

    source_hashes[
        str(
            FINAL_RUN_REGISTRY_PATH
        )
    ] = sha256_file(
        FINAL_RUN_REGISTRY_PATH
    )

    write_json(
        OUTPUT_DIR
        / "final_fit_preflight_source_hashes.json",
        source_hashes,
    )

    summary = {
        "phase":
            (
                "4G-R3 Tier C v4 final-fit "
                "execution preflight"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_original_final_run_count":
            len(
                original_rows
            ),

        "source_trainable_final_fit_count":
            len(
                original_trainable_rows
            ),

        "source_b0_persistence_count":
            len(
                b0_rows
            ),

        "resolved_trainable_final_fit_count":
            len(
                resolved_rows
            ),

        "resolved_run_counts":
            dict(
                sorted(
                    run_counts.items()
                )
            ),

        "resolved_seed_counts":
            {
                str(seed):
                    count
                for seed, count
                in sorted(
                    seed_counts.items()
                )
            },

        "resolved_noise_counts":
            {
                str(noise):
                    count
                for noise, count
                in sorted(
                    noise_counts.items()
                )
            },

        "selected_configuration_ids":
            selected_configuration_map,

        "b1_original_seed_token":
            "deterministic",

        "b1_effective_seed":
            11,

        "b1_fit_count":
            run_counts[
                "B1"
            ],

        "b1_one_fit_per_noise_condition":
            True,

        "neural_final_seeds":
            list(
                FINAL_SEEDS
            ),

        "noise_fractions":
            list(
                NOISE_FRACTIONS
            ),

        "training_cell":
            "train_joint",

        "checkpoint_selection_cell":
            "val_joint",

        "checkpoint_selection_metric":
            "noisy_target_rollout_mse",

        "maximum_epochs":
            200,

        "early_stopping_patience":
            25,

        "selected_tuning_epochs_used_as_fixed_final_epochs":
            False,

        "independent_final_fit_early_stopping_required":
            True,

        "fresh_seeded_initialization_required":
            True,

        "selected_tuning_checkpoint_weights_reused":
            False,

        "selected_tuning_checkpoints_used_for_provenance_only":
            True,

        "physical_microbatch_size":
            4,

        "gradient_accumulation_steps":
            64,

        "effective_batch_size":
            256,

        "gradient_clip_norm":
            5.0,

        "mixed_precision":
            False,

        "runtime_observation_dimension":
            128,

        "original_final_registry_modified":
            False,

        "original_training_allowed_values_preserved":
            True,

        "authorization_source":
            str(
                TUNING_SELECTION_SUMMARY_PATH
            ),

        "final_fit_execution_authorized":
            True,

        "additional_tuning_authorized":
            False,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_open_count":
            0,

        "model_parameters_initialized":
            False,

        "training_batches_read":
            False,

        "validation_batches_read":
            False,

        "optimizer_steps_performed":
            0,

        "final_fit_checkpoints_created":
            False,

        "b0_persistence_execution_completed":
            False,

        "b0_persistence_rows_deferred":
            len(
                b0_deferred_rows
            ),

        "resolved_final_fit_registry_path":
            str(
                resolved_registry_path
            ),

        "resolved_final_fit_registry_sha256":
            sha256_file(
                resolved_registry_path
            ),

        "final_fit_preflight_passed":
            True,

        "phase4gr3_status":
            "ready_for_130_trainable_final_fits",

        "next_phase":
            (
                "4G-R3B Tier C v4 resumable "
                "130-run final-fit execution"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4gr3_final_fit_preflight_summary.json",
        summary,
    )

    print(
        "Phase 4G-R3 final-fit preflight passed."
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
