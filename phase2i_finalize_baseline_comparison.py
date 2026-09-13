from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PHASE2D_DIR = Path(
    "outputs/phase2d_final_ocm"
)

PHASE2I_DIR = Path(
    "outputs/phase2i_final_baselines"
)

RUN_DIR = (
    PHASE2I_DIR / "runs"
)


FROZEN_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

FROZEN_NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

BASELINES = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
)

NEURAL_BASELINES = (
    "B2",
    "B3",
    "B4",
    "B5",
)

EXPECTED_CONFIGURATION_IDS = {
    "B1": "B1_c02",
    "B2": "B2_c01",
    "B3": "B3_c01",
    "B4": "B4_c02",
    "B5": "B5_c02",
}


def noise_slug(
    noise,
):
    return (
        f"{noise:g}"
        .replace(
            ".",
            "p",
        )
    )


def load_json(
    path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def load_csv(
    path,
):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def write_json(
    path,
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
    path,
    rows,
):
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def summarize(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean":
            float(
                values.mean()
            ),

        "std":
            float(
                values.std(
                    ddof=1
                )
            )
            if len(values) > 1
            else 0.0,

        "minimum":
            float(
                values.min()
            ),

        "maximum":
            float(
                values.max()
            ),
    }


def load_baseline_runs():
    rows = []

    for baseline in BASELINES:
        if baseline == "B1":
            for noise in FROZEN_NOISE_LEVELS:
                run_name = (
                    f"B1_noise_"
                    f"{noise_slug(noise)}"
                )

                path = (
                    RUN_DIR
                    / run_name
                    / "result.json"
                )

                row = load_json(
                    path
                )

                rows.append(
                    validate_and_flatten(
                        row=row,
                        baseline=baseline,
                        expected_seed=(
                            "deterministic"
                        ),
                        expected_noise=noise,
                    )
                )

        else:
            for seed in FROZEN_SEEDS:
                for noise in (
                    FROZEN_NOISE_LEVELS
                ):
                    run_name = (
                        f"{baseline}"
                        f"_seed_{seed}"
                        f"_noise_"
                        f"{noise_slug(noise)}"
                    )

                    path = (
                        RUN_DIR
                        / run_name
                        / "result.json"
                    )

                    row = load_json(
                        path
                    )

                    rows.append(
                        validate_and_flatten(
                            row=row,
                            baseline=baseline,
                            expected_seed=seed,
                            expected_noise=noise,
                        )
                    )

    if len(rows) != 105:
        raise AssertionError(
            "Expected exactly 105 baseline runs."
        )

    return rows


def validate_and_flatten(
    row,
    baseline,
    expected_seed,
    expected_noise,
):
    if (
        row[
            "status"
        ]
        != "completed"
    ):
        raise AssertionError(
            "Incomplete baseline run."
        )

    if (
        row[
            "baseline_id"
        ]
        != baseline
    ):
        raise AssertionError(
            "Baseline identity mismatch."
        )

    if (
        row[
            "configuration_id"
        ]
        != EXPECTED_CONFIGURATION_IDS[
            baseline
        ]
    ):
        raise AssertionError(
            "Frozen configuration changed."
        )

    if (
        row[
            "seed"
        ]
        != expected_seed
    ):
        raise AssertionError(
            "Seed mismatch."
        )

    if not np.isclose(
        row[
            "noise_fraction"
        ],
        expected_noise,
    ):
        raise AssertionError(
            "Noise mismatch."
        )

    if (
        row[
            "numerically_valid"
        ]
        is not True
    ):
        raise AssertionError(
            "Numerically invalid run."
        )

    forbidden_true_flags = [
        "test_used_for_checkpoint_selection",
        "clean_targets_used_for_training",
        "structural_metrics_computed",
        "structural_labels_read",
        "blackwell_labels_read",
        "symbolic_states_read",
    ]

    for flag in forbidden_true_flags:
        if row[
            flag
        ] is not False:
            raise AssertionError(
                f"Leakage or forbidden action: {flag}"
            )

    metrics = row[
        "test_metrics"
    ]

    return {
        "model_id":
            baseline,

        "configuration_id":
            row[
                "configuration_id"
            ],

        "seed":
            row[
                "seed"
            ],

        "noise_fraction":
            float(
                row[
                    "noise_fraction"
                ]
            ),

        "parameter_count":
            int(
                row[
                    "parameter_count"
                ]
            ),

        "best_epoch":
            row[
                "best_epoch"
            ],

        "best_validation_rollout_mse":
            float(
                row[
                    "best_validation_rollout_mse"
                ]
            ),

        "model_noisy_target_mse":
            float(
                metrics[
                    "model_noisy_target_mse"
                ]
            ),

        "model_clean_target_mse":
            float(
                metrics[
                    "model_clean_target_mse"
                ]
            ),

        "persistence_noisy_target_mse":
            float(
                metrics[
                    "persistence_noisy_target_mse"
                ]
            ),

        "persistence_clean_target_mse":
            float(
                metrics[
                    "persistence_clean_target_mse"
                ]
            ),
    }


def load_ocm_runs():
    rows = load_csv(
        PHASE2D_DIR
        / "all_run_results.csv"
    )

    output = []

    for row in rows:
        output.append(
            {
                "model_id":
                    "OCM",

                "configuration_id":
                    "c05",

                "seed":
                    int(
                        row[
                            "seed"
                        ]
                    ),

                "noise_fraction":
                    float(
                        row[
                            "noise_fraction"
                        ]
                    ),

                "parameter_count":
                    5992,

                "best_epoch":
                    int(
                        row[
                            "best_epoch"
                        ]
                    ),

                "best_validation_rollout_mse":
                    float(
                        row[
                            "best_validation_rollout_mse"
                        ]
                    ),

                "model_noisy_target_mse":
                    float(
                        row[
                            "model_noisy_target_mse"
                        ]
                    ),

                "model_clean_target_mse":
                    float(
                        row[
                            "model_clean_target_mse"
                        ]
                    ),

                "persistence_noisy_target_mse":
                    float(
                        row[
                            "persistence_noisy_target_mse"
                        ]
                    ),

                "persistence_clean_target_mse":
                    float(
                        row[
                            "persistence_clean_target_mse"
                        ]
                    ),
            }
        )

    if len(output) != 25:
        raise AssertionError(
            "Expected 25 OCM runs."
        )

    return output


def add_persistence_rows(
    ocm_rows,
):
    rows = []

    for noise in FROZEN_NOISE_LEVELS:
        group = [
            row
            for row in ocm_rows
            if np.isclose(
                row[
                    "noise_fraction"
                ],
                noise,
            )
        ]

        noisy_values = {
            row[
                "persistence_noisy_target_mse"
            ]
            for row in group
        }

        clean_values = {
            row[
                "persistence_clean_target_mse"
            ]
            for row in group
        }

        if (
            len(noisy_values) != 1
            or len(clean_values) != 1
        ):
            raise AssertionError(
                "Persistence values differ by seed."
            )

        rows.append(
            {
                "model_id":
                    "B0",

                "configuration_id":
                    "persistence",

                "seed":
                    "deterministic",

                "noise_fraction":
                    noise,

                "parameter_count":
                    0,

                "best_epoch":
                    None,

                "best_validation_rollout_mse":
                    None,

                "model_noisy_target_mse":
                    next(
                        iter(
                            noisy_values
                        )
                    ),

                "model_clean_target_mse":
                    next(
                        iter(
                            clean_values
                        )
                    ),

                "persistence_noisy_target_mse":
                    next(
                        iter(
                            noisy_values
                        )
                    ),

                "persistence_clean_target_mse":
                    next(
                        iter(
                            clean_values
                        )
                    ),
            }
        )

    return rows


def aggregate_models(
    rows,
):
    grouped = defaultdict(
        list
    )

    for row in rows:
        grouped[
            (
                row[
                    "model_id"
                ],
                row[
                    "noise_fraction"
                ],
            )
        ].append(
            row
        )

    output_rows = []
    output_json = {}

    for (
        model_id,
        noise,
    ), group in sorted(
        grouped.items()
    ):
        item = {
            "model_id":
                model_id,

            "noise_fraction":
                noise,

            "seed_count":
                len(
                    group
                ),

            "parameter_count":
                int(
                    group[0][
                        "parameter_count"
                    ]
                ),

            "noisy_target_mse":
                summarize(
                    [
                        row[
                            "model_noisy_target_mse"
                        ]
                        for row in group
                    ]
                ),

            "clean_target_mse":
                summarize(
                    [
                        row[
                            "model_clean_target_mse"
                        ]
                        for row in group
                    ]
                ),
        }

        output_json[
            f"{model_id}|{noise}"
        ] = item

        output_rows.append(
            {
                "model_id":
                    model_id,

                "noise_fraction":
                    noise,

                "seed_count":
                    len(
                        group
                    ),

                "parameter_count":
                    item[
                        "parameter_count"
                    ],

                "noisy_mse_mean":
                    item[
                        "noisy_target_mse"
                    ][
                        "mean"
                    ],

                "noisy_mse_std":
                    item[
                        "noisy_target_mse"
                    ][
                        "std"
                    ],

                "clean_mse_mean":
                    item[
                        "clean_target_mse"
                    ][
                        "mean"
                    ],

                "clean_mse_std":
                    item[
                        "clean_target_mse"
                    ][
                        "std"
                    ],
            }
        )

    return (
        output_rows,
        output_json,
    )


def paired_ocm_comparisons(
    all_rows,
):
    ocm_lookup = {
        (
            int(
                row[
                    "seed"
                ]
            ),
            row[
                "noise_fraction"
            ],
        ):
            row
        for row in all_rows
        if row[
            "model_id"
        ] == "OCM"
    }

    comparison_rows = []

    for baseline in (
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
    ):
        for noise in FROZEN_NOISE_LEVELS:
            baseline_group = [
                row
                for row in all_rows
                if (
                    row[
                        "model_id"
                    ]
                    == baseline
                    and np.isclose(
                        row[
                            "noise_fraction"
                        ],
                        noise,
                    )
                )
            ]

            if baseline in {
                "B0",
                "B1",
            }:
                if len(
                    baseline_group
                ) != 1:
                    raise AssertionError(
                        "Deterministic baseline "
                        "has wrong run count."
                    )

                baseline_by_seed = {
                    seed:
                        baseline_group[
                            0
                        ]
                    for seed in FROZEN_SEEDS
                }

            else:
                baseline_by_seed = {
                    int(
                        row[
                            "seed"
                        ]
                    ):
                        row
                    for row in baseline_group
                }

            noisy_differences = []
            clean_differences = []

            noisy_relative_reductions = []
            clean_relative_reductions = []

            noisy_wins = 0
            clean_wins = 0

            for seed in FROZEN_SEEDS:
                ocm = ocm_lookup[
                    (
                        seed,
                        noise,
                    )
                ]

                baseline_row = (
                    baseline_by_seed[
                        seed
                    ]
                )

                noisy_difference = (
                    baseline_row[
                        "model_noisy_target_mse"
                    ]
                    - ocm[
                        "model_noisy_target_mse"
                    ]
                )

                clean_difference = (
                    baseline_row[
                        "model_clean_target_mse"
                    ]
                    - ocm[
                        "model_clean_target_mse"
                    ]
                )

                noisy_differences.append(
                    noisy_difference
                )

                clean_differences.append(
                    clean_difference
                )

                noisy_relative_reductions.append(
                    noisy_difference
                    / baseline_row[
                        "model_noisy_target_mse"
                    ]
                )

                clean_relative_reductions.append(
                    clean_difference
                    / baseline_row[
                        "model_clean_target_mse"
                    ]
                )

                noisy_wins += int(
                    noisy_difference > 0
                )

                clean_wins += int(
                    clean_difference > 0
                )

            comparison_rows.append(
                {
                    "baseline_id":
                        baseline,

                    "noise_fraction":
                        noise,

                    "paired_seed_count":
                        len(
                            FROZEN_SEEDS
                        ),

                    "ocm_noisy_mse_advantage_mean":
                        float(
                            np.mean(
                                noisy_differences
                            )
                        ),

                    "ocm_clean_mse_advantage_mean":
                        float(
                            np.mean(
                                clean_differences
                            )
                        ),

                    "ocm_relative_noisy_reduction_mean":
                        float(
                            np.mean(
                                noisy_relative_reductions
                            )
                        ),

                    "ocm_relative_clean_reduction_mean":
                        float(
                            np.mean(
                                clean_relative_reductions
                            )
                        ),

                    "ocm_noisy_wins":
                        noisy_wins,

                    "ocm_clean_wins":
                        clean_wins,
                }
            )

    return comparison_rows


def rankings(
    aggregate_rows,
):
    output = []

    for noise in FROZEN_NOISE_LEVELS:
        group = [
            row
            for row in aggregate_rows
            if np.isclose(
                row[
                    "noise_fraction"
                ],
                noise,
            )
        ]

        noisy_sorted = sorted(
            group,
            key=lambda row:
                row[
                    "noisy_mse_mean"
                ],
        )

        clean_sorted = sorted(
            group,
            key=lambda row:
                row[
                    "clean_mse_mean"
                ],
        )

        noisy_rank = {
            row[
                "model_id"
            ]:
                rank
            for rank, row in enumerate(
                noisy_sorted,
                start=1,
            )
        }

        clean_rank = {
            row[
                "model_id"
            ]:
                rank
            for rank, row in enumerate(
                clean_sorted,
                start=1,
            )
        }

        for row in group:
            output.append(
                {
                    "noise_fraction":
                        noise,

                    "model_id":
                        row[
                            "model_id"
                        ],

                    "noisy_target_rank":
                        noisy_rank[
                            row[
                                "model_id"
                            ]
                        ],

                    "clean_target_rank":
                        clean_rank[
                            row[
                                "model_id"
                            ]
                        ],

                    "noisy_mse_mean":
                        row[
                            "noisy_mse_mean"
                        ],

                    "clean_mse_mean":
                        row[
                            "clean_mse_mean"
                        ],
                }
            )

    return output


def main():
    baseline_rows = (
        load_baseline_runs()
    )

    ocm_rows = (
        load_ocm_runs()
    )

    persistence_rows = (
        add_persistence_rows(
            ocm_rows
        )
    )

    all_rows = (
        baseline_rows
        + ocm_rows
        + persistence_rows
    )

    write_csv(
        PHASE2I_DIR
        / "all_predictive_run_results.csv",
        all_rows,
    )

    (
        aggregate_rows,
        aggregate_json,
    ) = aggregate_models(
        all_rows
    )

    write_csv(
        PHASE2I_DIR
        / "model_noise_summary.csv",
        aggregate_rows,
    )

    comparison_rows = (
        paired_ocm_comparisons(
            all_rows
        )
    )

    write_csv(
        PHASE2I_DIR
        / "paired_ocm_comparisons.csv",
        comparison_rows,
    )

    ranking_rows = rankings(
        aggregate_rows
    )

    write_csv(
        PHASE2I_DIR
        / "ranking_by_noise.csv",
        ranking_rows,
    )

    baseline_run_count = len(
        baseline_rows
    )

    deterministic_count = sum(
        row[
            "model_id"
        ]
        == "B1"
        for row in baseline_rows
    )

    neural_count = (
        baseline_run_count
        - deterministic_count
    )

    summary = {
        "phase":
            "2I final predictive baseline comparison",

        "baseline_final_fit_count":
            baseline_run_count,

        "deterministic_baseline_fit_count":
            deterministic_count,

        "neural_baseline_fit_count":
            neural_count,

        "ocm_run_count":
            len(
                ocm_rows
            ),

        "noise_levels":
            list(
                FROZEN_NOISE_LEVELS
            ),

        "neural_seeds":
            list(
                FROZEN_SEEDS
            ),

        "selected_configuration_ids":
            EXPECTED_CONFIGURATION_IDS,

        "aggregated_model_results":
            aggregate_json,

        "test_checkpoint_rule": (
            "Every neural checkpoint was selected "
            "using validation noisy-target rollout "
            "MSE before test evaluation."
        ),

        "test_used_for_checkpoint_selection":
            False,

        "clean_targets_used_for_training":
            False,

        "structural_metrics_computed":
            False,

        "structural_labels_read":
            False,

        "blackwell_labels_read":
            False,

        "completion_status":
            "passed",

        "interpretation_note": (
            "Phase completion requires valid execution "
            "of all frozen fits. It does not require OCM "
            "or any baseline to win."
        ),
    }

    write_json(
        PHASE2I_DIR
        / "phase2i_summary.json",
        summary,
    )

    print(
        "Phase 2I final baseline comparison "
        "aggregated successfully."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
