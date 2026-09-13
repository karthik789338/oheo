from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PHASE2D_DIR = Path(
    "outputs/phase2d_final_ocm"
)

RUN_DIR = (
    PHASE2D_DIR
    / "runs"
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

EXPECTED_CONFIGURATION_ID = "c05"

TUNING_SEED = 11


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
                np.mean(
                    values
                )
            ),

        "std":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            if len(values) > 1
            else 0.0,

        "minimum":
            float(
                np.min(
                    values
                )
            ),

        "maximum":
            float(
                np.max(
                    values
                )
            ),
    }


def collect_runs():
    rows = []

    for seed in FROZEN_SEEDS:

        for noise in FROZEN_NOISE_LEVELS:

            run_name = (
                f"seed_{seed}"
                f"_noise_{noise_slug(noise)}"
            )

            path = (
                RUN_DIR
                / run_name
                / "result.json"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing run: {path}"
                )

            row = load_json(
                path
            )

            if (
                row[
                    "status"
                ]
                != "completed"
            ):
                raise AssertionError(
                    f"{run_name} is incomplete."
                )

            if (
                row[
                    "configuration_id"
                ]
                != EXPECTED_CONFIGURATION_ID
            ):
                raise AssertionError(
                    f"{run_name} used the "
                    "wrong configuration."
                )

            if (
                row[
                    "seed"
                ]
                != seed
            ):
                raise AssertionError(
                    f"{run_name} seed mismatch."
                )

            if not np.isclose(
                row[
                    "noise_fraction"
                ],
                noise,
            ):
                raise AssertionError(
                    f"{run_name} noise mismatch."
                )

            if (
                row[
                    "test_used_for_checkpoint_selection"
                ]
                is not False
            ):
                raise AssertionError(
                    "Test leakage detected."
                )

            forbidden_flags = [
                "structural_labels_read",
                "blackwell_labels_read",
                "information_classes_read",
                "symbolic_states_read",
                "structural_metrics_computed",
            ]

            for flag in forbidden_flags:

                if (
                    row[
                        flag
                    ]
                    is not False
                ):
                    raise AssertionError(
                        f"{flag} was enabled."
                    )

            if (
                row[
                    "numerically_valid"
                ]
                is not True
            ):
                raise AssertionError(
                    f"{run_name} is "
                    "numerically invalid."
                )

            metrics = row[
                "test_metrics"
            ]

            rows.append(
                {
                    "seed":
                        seed,

                    "noise_fraction":
                        noise,

                    "best_epoch":
                        row[
                            "best_epoch"
                        ],

                    "best_validation_rollout_mse":
                        row[
                            "best_validation_rollout_mse"
                        ],

                    "model_noisy_target_mse":
                        metrics[
                            "model_noisy_target_mse"
                        ],

                    "model_clean_target_mse":
                        metrics[
                            "model_clean_target_mse"
                        ],

                    "persistence_noisy_target_mse":
                        metrics[
                            "persistence_noisy_target_mse"
                        ],

                    "persistence_clean_target_mse":
                        metrics[
                            "persistence_clean_target_mse"
                        ],

                    "relative_improvement_vs_persistence_noisy":
                        (
                            metrics[
                                "persistence_noisy_target_mse"
                            ]
                            - metrics[
                                "model_noisy_target_mse"
                            ]
                        )
                        / metrics[
                            "persistence_noisy_target_mse"
                        ],

                    "relative_improvement_vs_persistence_clean":
                        (
                            metrics[
                                "persistence_clean_target_mse"
                            ]
                            - metrics[
                                "model_clean_target_mse"
                            ]
                        )
                        / metrics[
                            "persistence_clean_target_mse"
                        ],
                }
            )

    if len(rows) != 25:
        raise AssertionError(
            "Expected exactly 25 final runs."
        )

    return rows


def write_run_table(
    rows,
):
    path = (
        PHASE2D_DIR
        / "all_run_results.csv"
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


def aggregate_noise(
    rows,
    seeds,
):
    grouped = defaultdict(
        list
    )

    for row in rows:

        if row[
            "seed"
        ] not in seeds:
            continue

        grouped[
            row[
                "noise_fraction"
            ]
        ].append(
            row
        )

    output = {}

    for noise in FROZEN_NOISE_LEVELS:

        group = grouped[
            noise
        ]

        if len(group) != len(
            seeds
        ):
            raise AssertionError(
                f"Noise {noise} has "
                "incorrect seed count."
            )

        output[
            str(
                noise
            )
        ] = {
            "seed_count":
                len(
                    group
                ),

            "model_noisy_target_mse":
                summarize(
                    [
                        row[
                            "model_noisy_target_mse"
                        ]
                        for row in group
                    ]
                ),

            "model_clean_target_mse":
                summarize(
                    [
                        row[
                            "model_clean_target_mse"
                        ]
                        for row in group
                    ]
                ),

            "persistence_noisy_target_mse":
                summarize(
                    [
                        row[
                            "persistence_noisy_target_mse"
                        ]
                        for row in group
                    ]
                ),

            "persistence_clean_target_mse":
                summarize(
                    [
                        row[
                            "persistence_clean_target_mse"
                        ]
                        for row in group
                    ]
                ),

            "relative_improvement_vs_persistence_noisy":
                summarize(
                    [
                        row[
                            "relative_improvement_vs_persistence_noisy"
                        ]
                        for row in group
                    ]
                ),

            "relative_improvement_vs_persistence_clean":
                summarize(
                    [
                        row[
                            "relative_improvement_vs_persistence_clean"
                        ]
                        for row in group
                    ]
                ),

            "best_epoch":
                summarize(
                    [
                        row[
                            "best_epoch"
                        ]
                        for row in group
                    ]
                ),
        }

    return output


def write_noise_summary_csv(
    primary_summary,
):
    rows = []

    for noise in FROZEN_NOISE_LEVELS:

        item = primary_summary[
            str(
                noise
            )
        ]

        rows.append(
            {
                "noise_fraction":
                    noise,

                "model_noisy_mse_mean":
                    item[
                        "model_noisy_target_mse"
                    ][
                        "mean"
                    ],

                "model_noisy_mse_std":
                    item[
                        "model_noisy_target_mse"
                    ][
                        "std"
                    ],

                "model_clean_mse_mean":
                    item[
                        "model_clean_target_mse"
                    ][
                        "mean"
                    ],

                "model_clean_mse_std":
                    item[
                        "model_clean_target_mse"
                    ][
                        "std"
                    ],

                "persistence_noisy_mse_mean":
                    item[
                        "persistence_noisy_target_mse"
                    ][
                        "mean"
                    ],

                "relative_improvement_noisy_mean":
                    item[
                        "relative_improvement_vs_persistence_noisy"
                    ][
                        "mean"
                    ],
            }
        )

    path = (
        PHASE2D_DIR
        / "noise_summary.csv"
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


def main():
    rows = collect_runs()

    write_run_table(
        rows
    )

    primary_summary = (
        aggregate_noise(
            rows=rows,
            seeds=FROZEN_SEEDS,
        )
    )

    sensitivity_seeds = tuple(
        seed
        for seed in FROZEN_SEEDS
        if seed != TUNING_SEED
    )

    sensitivity_summary = (
        aggregate_noise(
            rows=rows,
            seeds=sensitivity_seeds,
        )
    )

    write_noise_summary_csv(
        primary_summary
    )

    summary = {
        "phase":
            "2D final OCM predictive evaluation",

        "configuration_id":
            EXPECTED_CONFIGURATION_ID,

        "total_run_count":
            len(
                rows
            ),

        "noise_levels":
            list(
                FROZEN_NOISE_LEVELS
            ),

        "primary_seeds":
            list(
                FROZEN_SEEDS
            ),

        "secondary_sensitivity_seeds":
            list(
                sensitivity_seeds
            ),

        "primary_five_seed_results":
            primary_summary,

        "four_seed_sensitivity_excluding_tuning_seed":
            sensitivity_summary,

        "test_checkpoint_rule":
            (
                "Every model checkpoint was selected "
                "using validation rollout MSE before "
                "test evaluation."
            ),

        "structural_metrics_computed":
            False,

        "blackwell_labels_read":
            False,

        "symbolic_states_read":
            False,

        "completion_status":
            "passed",

        "interpretation_note":
            (
                "Phase completion means all 25 frozen "
                "runs executed validly. It does not "
                "require OCM to outperform persistence. "
                "Any negative predictive results must "
                "be retained and reported."
            ),
    }

    path = (
        PHASE2D_DIR
        / "phase2d_summary.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "Phase 2D final OCM evaluation "
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
