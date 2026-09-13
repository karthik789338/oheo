from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PHASE1A_DIR = Path(
    "outputs/phase1a_ground_truth"
)

PHASE1C_DIR = Path(
    "outputs/phase1c_continuous_observations"
)

PHASE2D_DIR = Path(
    "outputs/phase2d_final_ocm"
)

PHASE2F_DIR = Path(
    "outputs/phase2f_ood_structural_audit"
)

PHASE2H_DIR = Path(
    "outputs/phase2h_baseline_tuning"
)

PHASE2I_DIR = Path(
    "outputs/phase2i_final_baselines"
)

OUTPUT_DIR = Path(
    "outputs/phase2j_predictive_audit"
)


FROZEN_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

SENSITIVITY_SEEDS = (
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

ALL_MODELS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

LEARNED_MODELS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

PAIRED_BASELINES = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
)

EXPECTED_MODEL_RUN_COUNTS = {
    "B0": 5,
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
    "OCM": 25,
}

EXPECTED_PREDICTIVE_ROW_COUNT = 135
EXPECTED_LENGTH_RUN_COUNT = 130
EXPECTED_TEST_TRAJECTORY_COUNT = 1152


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def load_csv(
    path: Path,
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


def write_csv(
    path: Path,
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

    if values.size == 0:
        raise ValueError(
            "Cannot summarize an empty array."
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


def noise_slug(
    noise: float,
):
    return (
        f"{noise:g}"
        .replace(
            ".",
            "p",
        )
    )


def parse_seed(
    value,
):
    if value == "deterministic":
        return "deterministic"

    return int(
        value
    )


def validate_completed_phases():
    phase2d = load_json(
        PHASE2D_DIR
        / "phase2d_summary.json"
    )

    phase2f = load_json(
        PHASE2F_DIR
        / "phase2f_summary.json"
    )

    phase2h = load_json(
        PHASE2H_DIR
        / "phase2h_summary.json"
    )

    phase2i = load_json(
        PHASE2I_DIR
        / "phase2i_summary.json"
    )

    if (
        phase2d[
            "completion_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2D is not passed."
        )

    if (
        phase2d[
            "total_run_count"
        ]
        != 25
    ):
        raise AssertionError(
            "Phase 2D run count changed."
        )

    if (
        phase2f[
            "phase2f_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2F is not passed."
        )

    if (
        phase2h[
            "phase2h_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2H is not passed."
        )

    if (
        phase2i[
            "completion_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2I is not passed."
        )

    if (
        phase2i[
            "baseline_final_fit_count"
        ]
        != 105
    ):
        raise AssertionError(
            "Phase 2I baseline count changed."
        )

    if (
        phase2i[
            "ocm_run_count"
        ]
        != 25
    ):
        raise AssertionError(
            "Phase 2I OCM count changed."
        )

    return {
        "phase2d":
            phase2d,

        "phase2f":
            phase2f,

        "phase2h":
            phase2h,

        "phase2i":
            phase2i,
    }


def load_predictive_rows():
    rows = load_csv(
        PHASE2I_DIR
        / "all_predictive_run_results.csv"
    )

    parsed = []

    for row in rows:
        best_epoch = row[
            "best_epoch"
        ].strip()

        parsed.append(
            {
                "model_id":
                    row[
                        "model_id"
                    ],

                "configuration_id":
                    row[
                        "configuration_id"
                    ],

                "seed":
                    parse_seed(
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
                    int(
                        row[
                            "parameter_count"
                        ]
                    ),

                "best_epoch":
                    (
                        int(
                            float(
                                best_epoch
                            )
                        )
                        if best_epoch
                        else None
                    ),

                "best_validation_rollout_mse":
                    (
                        float(
                            row[
                                "best_validation_rollout_mse"
                            ]
                        )
                        if row[
                            "best_validation_rollout_mse"
                        ].strip()
                        else None
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

    if (
        len(parsed)
        != EXPECTED_PREDICTIVE_ROW_COUNT
    ):
        raise AssertionError(
            "Unexpected predictive result count: "
            f"{len(parsed)}"
        )

    counts = Counter(
        row[
            "model_id"
        ]
        for row in parsed
    )

    if dict(
        counts
    ) != EXPECTED_MODEL_RUN_COUNTS:
        raise AssertionError(
            "Model run counts changed: "
            f"{dict(counts)}"
        )

    for row in parsed:
        values = (
            row[
                "model_noisy_target_mse"
            ],
            row[
                "model_clean_target_mse"
            ],
            row[
                "persistence_noisy_target_mse"
            ],
            row[
                "persistence_clean_target_mse"
            ],
        )

        if not all(
            np.isfinite(
                value
            )
            for value in values
        ):
            raise FloatingPointError(
                "Non-finite predictive result."
            )

    return parsed


def create_metadata_correction():
    """
    Preserve the original outputs and record a transparent correction.

    The OCM and Phase 2I final-run implementations constructed
    validation loaders containing clean final targets. Their evaluation
    functions calculated both noisy-target and clean-target MSE.

    Only noisy-target MSE affected checkpoint selection.
    """

    correction = {
        "scope": [
            "Phase 2D final OCM runs",
            "Phase 2I B1-B5 final baseline runs",
        ],

        "original_outputs_modified":
            False,

        "clean_targets_used_for_training":
            False,

        "clean_targets_used_in_training_loss":
            False,

        "clean_targets_used_for_checkpoint_selection":
            False,

        "clean_targets_used_for_early_stopping":
            False,

        "clean_validation_targets_accessed":
            True,

        "clean_validation_metrics_computed":
            True,

        "clean_validation_metrics_logged_to_training_history":
            False,

        "test_used_for_checkpoint_selection":
            False,

        "impact_assessment": (
            "Clean validation targets were present in the validation "
            "evaluation path but did not affect optimization, early "
            "stopping, hyperparameter selection, or checkpoint "
            "selection. Predictive test results remain valid, but the "
            "stricter claim that clean validation targets were accessed "
            "only after checkpoint selection is incorrect."
        ),

        "correct_reporting_fields": {
            "clean_targets_used_for_training":
                False,

            "clean_targets_used_for_selection":
                False,

            "clean_targets_accessed_during_validation":
                True,

            "test_used_for_selection":
                False,
        },
    }

    write_json(
        OUTPUT_DIR
        / "metadata_correction.json",
        correction,
    )

    return correction


def read_length_file(
    path: Path,
    model_id,
    seed,
    noise,
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    rows = load_csv(
        path
    )

    output = []
    trajectory_total = 0

    for row in rows:
        trajectory_count = int(
            row[
                "trajectory_count"
            ]
        )

        trajectory_total += (
            trajectory_count
        )

        output.append(
            {
                "model_id":
                    model_id,

                "seed":
                    seed,

                "noise_fraction":
                    float(
                        noise
                    ),

                "sequence_length":
                    int(
                        row[
                            "sequence_length"
                        ]
                    ),

                "trajectory_count":
                    trajectory_count,

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
            }
        )

    if (
        trajectory_total
        != EXPECTED_TEST_TRAJECTORY_COUNT
    ):
        raise AssertionError(
            f"{path} contains {trajectory_total} "
            "test trajectories rather than 1152."
        )

    return output


def collect_length_rows():
    rows = []
    run_count = 0

    for seed in FROZEN_SEEDS:
        for noise in FROZEN_NOISE_LEVELS:
            run_name = (
                f"seed_{seed}"
                f"_noise_{noise_slug(noise)}"
            )

            path = (
                PHASE2D_DIR
                / "runs"
                / run_name
                / "test_mse_by_sequence_length.csv"
            )

            rows.extend(
                read_length_file(
                    path=path,
                    model_id="OCM",
                    seed=seed,
                    noise=noise,
                )
            )

            run_count += 1

    for noise in FROZEN_NOISE_LEVELS:
        run_name = (
            f"B1_noise_{noise_slug(noise)}"
        )

        path = (
            PHASE2I_DIR
            / "runs"
            / run_name
            / "test_mse_by_sequence_length.csv"
        )

        rows.extend(
            read_length_file(
                path=path,
                model_id="B1",
                seed="deterministic",
                noise=noise,
            )
        )

        run_count += 1

    for model_id in (
        "B2",
        "B3",
        "B4",
        "B5",
    ):
        for seed in FROZEN_SEEDS:
            for noise in FROZEN_NOISE_LEVELS:
                run_name = (
                    f"{model_id}"
                    f"_seed_{seed}"
                    f"_noise_{noise_slug(noise)}"
                )

                path = (
                    PHASE2I_DIR
                    / "runs"
                    / run_name
                    / "test_mse_by_sequence_length.csv"
                )

                rows.extend(
                    read_length_file(
                        path=path,
                        model_id=model_id,
                        seed=seed,
                        noise=noise,
                    )
                )

                run_count += 1

    if (
        run_count
        != EXPECTED_LENGTH_RUN_COUNT
    ):
        raise AssertionError(
            "Expected 130 sequence-length files, "
            f"found {run_count}."
        )

    return rows


def aggregate_length_rows(
    rows,
):
    grouped = defaultdict(
        list
    )

    for row in rows:
        key = (
            row[
                "model_id"
            ],
            row[
                "noise_fraction"
            ],
            row[
                "sequence_length"
            ],
        )

        grouped[
            key
        ].append(
            row
        )

    output = []

    for (
        model_id,
        noise,
        sequence_length,
    ), group in sorted(
        grouped.items()
    ):
        noisy_summary = summarize(
            [
                row[
                    "model_noisy_target_mse"
                ]
                for row in group
            ]
        )

        clean_summary = summarize(
            [
                row[
                    "model_clean_target_mse"
                ]
                for row in group
            ]
        )

        trajectory_counts = {
            row[
                "trajectory_count"
            ]
            for row in group
        }

        if len(
            trajectory_counts
        ) != 1:
            raise AssertionError(
                "Trajectory count differs between seeds."
            )

        trajectory_count_per_run = next(
            iter(
                trajectory_counts
            )
        )

        output.append(
            {
                "model_id":
                    model_id,

                "noise_fraction":
                    noise,

                "sequence_length":
                    sequence_length,

                "seed_or_fit_count":
                    len(
                        group
                    ),

                "trajectory_count_per_run":
                    trajectory_count_per_run,

                "total_trajectory_evaluations":
                    (
                        trajectory_count_per_run
                        * len(
                            group
                        )
                    ),

                "noisy_mse_mean":
                    noisy_summary[
                        "mean"
                    ],

                "noisy_mse_std":
                    noisy_summary[
                        "std"
                    ],

                "clean_mse_mean":
                    clean_summary[
                        "mean"
                    ],

                "clean_mse_std":
                    clean_summary[
                        "std"
                    ],
            }
        )

    return output


def create_length_trends(
    aggregate_rows,
):
    grouped = defaultdict(
        list
    )

    for row in aggregate_rows:
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

    output = []

    for (
        model_id,
        noise,
    ), group in sorted(
        grouped.items()
    ):
        group = sorted(
            group,
            key=lambda row:
                row[
                    "sequence_length"
                ],
        )

        lengths = np.asarray(
            [
                row[
                    "sequence_length"
                ]
                for row in group
            ],
            dtype=np.float64,
        )

        noisy_values = np.asarray(
            [
                row[
                    "noisy_mse_mean"
                ]
                for row in group
            ],
            dtype=np.float64,
        )

        clean_values = np.asarray(
            [
                row[
                    "clean_mse_mean"
                ]
                for row in group
            ],
            dtype=np.float64,
        )

        if len(
            lengths
        ) > 1:
            noisy_slope = float(
                np.polyfit(
                    lengths,
                    noisy_values,
                    deg=1,
                )[0]
            )

            clean_slope = float(
                np.polyfit(
                    lengths,
                    clean_values,
                    deg=1,
                )[0]
            )
        else:
            noisy_slope = 0.0
            clean_slope = 0.0

        output.append(
            {
                "model_id":
                    model_id,

                "noise_fraction":
                    noise,

                "minimum_sequence_length":
                    int(
                        lengths.min()
                    ),

                "maximum_sequence_length":
                    int(
                        lengths.max()
                    ),

                "length_count":
                    len(
                        lengths
                    ),

                "noisy_mse_linear_slope":
                    noisy_slope,

                "clean_mse_linear_slope":
                    clean_slope,

                "shortest_length_noisy_mse":
                    float(
                        noisy_values[0]
                    ),

                "longest_length_noisy_mse":
                    float(
                        noisy_values[-1]
                    ),

                "shortest_length_clean_mse":
                    float(
                        clean_values[0]
                    ),

                "longest_length_clean_mse":
                    float(
                        clean_values[-1]
                    ),
            }
        )

    return output


def model_rows_for_noise(
    rows,
    model_id,
    noise,
):
    return [
        row
        for row in rows
        if (
            row[
                "model_id"
            ]
            == model_id
            and np.isclose(
                row[
                    "noise_fraction"
                ],
                noise,
            )
        )
    ]


def build_seed_lookup(
    rows,
    model_id,
    noise,
):
    group = model_rows_for_noise(
        rows,
        model_id,
        noise,
    )

    if model_id in {
        "B0",
        "B1",
    }:
        if len(
            group
        ) != 1:
            raise AssertionError(
                f"{model_id} should be deterministic."
            )

        return {
            seed:
                group[0]
            for seed in FROZEN_SEEDS
        }

    lookup = {
        int(
            row[
                "seed"
            ]
        ):
            row
        for row in group
    }

    if set(
        lookup
    ) != set(
        FROZEN_SEEDS
    ):
        raise AssertionError(
            f"{model_id} seed coverage changed."
        )

    return lookup


def paired_comparisons(
    rows,
):
    output = []

    for cohort_name, cohort_seeds in (
        (
            "all_five_seeds",
            FROZEN_SEEDS,
        ),
        (
            "excluding_tuning_seed",
            SENSITIVITY_SEEDS,
        ),
    ):
        for baseline_id in (
            PAIRED_BASELINES
        ):
            for noise in (
                FROZEN_NOISE_LEVELS
            ):
                ocm_lookup = (
                    build_seed_lookup(
                        rows=rows,
                        model_id="OCM",
                        noise=noise,
                    )
                )

                baseline_lookup = (
                    build_seed_lookup(
                        rows=rows,
                        model_id=baseline_id,
                        noise=noise,
                    )
                )

                noisy_differences = []
                clean_differences = []

                noisy_relative_reductions = []
                clean_relative_reductions = []

                noisy_ocm_wins = 0
                clean_ocm_wins = 0

                noisy_baseline_wins = 0
                clean_baseline_wins = 0

                noisy_ties = 0
                clean_ties = 0

                tolerance = 1e-12

                for seed in cohort_seeds:
                    ocm = ocm_lookup[
                        seed
                    ]

                    baseline = baseline_lookup[
                        seed
                    ]

                    noisy_difference = (
                        baseline[
                            "model_noisy_target_mse"
                        ]
                        - ocm[
                            "model_noisy_target_mse"
                        ]
                    )

                    clean_difference = (
                        baseline[
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
                        / baseline[
                            "model_noisy_target_mse"
                        ]
                    )

                    clean_relative_reductions.append(
                        clean_difference
                        / baseline[
                            "model_clean_target_mse"
                        ]
                    )

                    if noisy_difference > tolerance:
                        noisy_ocm_wins += 1
                    elif noisy_difference < -tolerance:
                        noisy_baseline_wins += 1
                    else:
                        noisy_ties += 1

                    if clean_difference > tolerance:
                        clean_ocm_wins += 1
                    elif clean_difference < -tolerance:
                        clean_baseline_wins += 1
                    else:
                        clean_ties += 1

                noisy_difference_summary = summarize(
                    noisy_differences
                )

                clean_difference_summary = summarize(
                    clean_differences
                )

                noisy_relative_summary = summarize(
                    noisy_relative_reductions
                )

                clean_relative_summary = summarize(
                    clean_relative_reductions
                )

                output.append(
                    {
                        "cohort":
                            cohort_name,

                        "baseline_id":
                            baseline_id,

                        "noise_fraction":
                            noise,

                        "paired_seed_count":
                            len(
                                cohort_seeds
                            ),

                        "ocm_noisy_advantage_mean":
                            noisy_difference_summary[
                                "mean"
                            ],

                        "ocm_noisy_advantage_std":
                            noisy_difference_summary[
                                "std"
                            ],

                        "ocm_clean_advantage_mean":
                            clean_difference_summary[
                                "mean"
                            ],

                        "ocm_clean_advantage_std":
                            clean_difference_summary[
                                "std"
                            ],

                        "ocm_relative_noisy_reduction_mean":
                            noisy_relative_summary[
                                "mean"
                            ],

                        "ocm_relative_clean_reduction_mean":
                            clean_relative_summary[
                                "mean"
                            ],

                        "ocm_noisy_wins":
                            noisy_ocm_wins,

                        "baseline_noisy_wins":
                            noisy_baseline_wins,

                        "noisy_ties":
                            noisy_ties,

                        "ocm_clean_wins":
                            clean_ocm_wins,

                        "baseline_clean_wins":
                            clean_baseline_wins,

                        "clean_ties":
                            clean_ties,
                    }
                )

    expected_count = (
        2
        * len(
            PAIRED_BASELINES
        )
        * len(
            FROZEN_NOISE_LEVELS
        )
    )

    if len(
        output
    ) != expected_count:
        raise AssertionError(
            "Paired comparison count changed."
        )

    return output


def aggregate_model_noise(
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

    output = {}

    for (
        model_id,
        noise,
    ), group in grouped.items():
        output[
            (
                model_id,
                noise,
            )
        ] = {
            "parameter_count":
                int(
                    group[0][
                        "parameter_count"
                    ]
                ),

            "noisy_mse_mean":
                float(
                    np.mean(
                        [
                            row[
                                "model_noisy_target_mse"
                            ]
                            for row in group
                        ]
                    )
                ),

            "clean_mse_mean":
                float(
                    np.mean(
                        [
                            row[
                                "model_clean_target_mse"
                            ]
                            for row in group
                        ]
                    )
                ),
        }

    return output


def parameter_efficiency_table(
    rows,
):
    aggregate = aggregate_model_noise(
        rows
    )

    ranks = {
        model_id: {
            "noisy": [],
            "clean": [],
        }
        for model_id in LEARNED_MODELS
    }

    noisy_win_counts = Counter()
    clean_win_counts = Counter()

    for noise in FROZEN_NOISE_LEVELS:
        group = [
            {
                "model_id":
                    model_id,

                **aggregate[
                    (
                        model_id,
                        noise,
                    )
                ],
            }
            for model_id in LEARNED_MODELS
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

        noisy_win_counts[
            noisy_sorted[0][
                "model_id"
            ]
        ] += 1

        clean_win_counts[
            clean_sorted[0][
                "model_id"
            ]
        ] += 1

        for rank, item in enumerate(
            noisy_sorted,
            start=1,
        ):
            ranks[
                item[
                    "model_id"
                ]
            ][
                "noisy"
            ].append(
                rank
            )

        for rank, item in enumerate(
            clean_sorted,
            start=1,
        ):
            ranks[
                item[
                    "model_id"
                ]
            ][
                "clean"
            ].append(
                rank
            )

    ocm_parameter_count = aggregate[
        (
            "OCM",
            0.0,
        )
    ][
        "parameter_count"
    ]

    output = []

    for model_id in LEARNED_MODELS:
        parameter_counts = {
            aggregate[
                (
                    model_id,
                    noise,
                )
            ][
                "parameter_count"
            ]
            for noise in FROZEN_NOISE_LEVELS
        }

        if len(
            parameter_counts
        ) != 1:
            raise AssertionError(
                f"{model_id} parameter count changed "
                "across noise levels."
            )

        parameter_count = next(
            iter(
                parameter_counts
            )
        )

        noisy_values = [
            aggregate[
                (
                    model_id,
                    noise,
                )
            ][
                "noisy_mse_mean"
            ]
            for noise in FROZEN_NOISE_LEVELS
        ]

        clean_values = [
            aggregate[
                (
                    model_id,
                    noise,
                )
            ][
                "clean_mse_mean"
            ]
            for noise in FROZEN_NOISE_LEVELS
        ]

        output.append(
            {
                "model_id":
                    model_id,

                "parameter_count":
                    parameter_count,

                "parameters_relative_to_ocm":
                    (
                        parameter_count
                        / ocm_parameter_count
                    ),

                "ocm_parameter_reduction_vs_model":
                    (
                        1.0
                        - ocm_parameter_count
                        / parameter_count
                    ),

                "mean_noisy_rank_among_learned_models":
                    float(
                        np.mean(
                            ranks[
                                model_id
                            ][
                                "noisy"
                            ]
                        )
                    ),

                "mean_clean_rank_among_learned_models":
                    float(
                        np.mean(
                            ranks[
                                model_id
                            ][
                                "clean"
                            ]
                        )
                    ),

                "noisy_conditions_won":
                    int(
                        noisy_win_counts[
                            model_id
                        ]
                    ),

                "clean_conditions_won":
                    int(
                        clean_win_counts[
                            model_id
                        ]
                    ),

                "mean_noisy_mse_across_noise_conditions":
                    float(
                        np.mean(
                            noisy_values
                        )
                    ),

                "mean_clean_mse_across_noise_conditions":
                    float(
                        np.mean(
                            clean_values
                        )
                    ),
            }
        )

    return output


def load_clean_prototypes():
    rows = load_csv(
        PHASE1C_DIR
        / "state_prototypes.csv"
    )

    rows.sort(
        key=lambda row:
            int(
                row[
                    "state"
                ]
            )
    )

    dimension_names = [
        name
        for name in rows[0]
        if name.startswith(
            "z"
        )
    ]

    dimension_names.sort(
        key=lambda name:
            int(
                name[1:]
            )
    )

    prototypes = np.asarray(
        [
            [
                float(
                    row[
                        name
                    ]
                )
                for name in dimension_names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    if prototypes.shape != (
        8,
        12,
    ):
        raise AssertionError(
            "Prototype shape changed."
        )

    return prototypes


def load_generators():
    rows = load_json(
        PHASE1A_DIR
        / "generators.json"
    )

    return {
        row[
            "name"
        ]:
            tuple(
                int(
                    value
                )
                for value in row[
                    "mapping"
                ]
            )
        for row in rows
    }


def parse_shortest_word(
    row,
):
    word_length = int(
        row[
            "word_length"
        ]
    )

    if word_length == 0:
        return tuple()

    text = row[
        "shortest_word"
    ].strip()

    if text.startswith(
        "["
    ):
        word = tuple(
            json.loads(
                text
            )
        )
    else:
        word = tuple(
            part.strip()
            for part in text.split(
                "->"
            )
            if part.strip()
        )

    if len(
        word
    ) != word_length:
        raise AssertionError(
            "Shortest-word length mismatch."
        )

    return word


def load_semigroup_elements():
    rows = load_csv(
        PHASE1A_DIR
        / "semigroup_elements.csv"
    )

    output = []

    for row in rows:
        output.append(
            {
                "element_id":
                    row[
                        "element_id"
                    ],

                "mapping":
                    tuple(
                        int(
                            value
                        )
                        for value in json.loads(
                            row[
                                "mapping"
                            ]
                        )
                    ),

                "word":
                    parse_shortest_word(
                        row
                    ),
            }
        )

    if len(
        output
    ) != 104:
        raise AssertionError(
            "Semigroup size changed."
        )

    return output


def fit_affine_map(
    source_points,
    target_points,
):
    augmented = np.concatenate(
        [
            source_points,
            np.ones(
                (
                    len(
                        source_points
                    ),
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    weights, _, _, _ = np.linalg.lstsq(
        augmented,
        target_points,
        rcond=None,
    )

    matrix = weights[
        :-1,
        :
    ]

    bias = weights[
        -1,
        :
    ]

    prediction = (
        source_points
        @ matrix
        + bias
    )

    residual = (
        prediction
        - target_points
    )

    return {
        "matrix":
            matrix,

        "bias":
            bias,

        "mse":
            float(
                np.mean(
                    residual
                    * residual
                )
            ),

        "maximum_absolute_error":
            float(
                np.max(
                    np.abs(
                        residual
                    )
                )
            ),
    }


def run_affine_shortcut_audit():
    prototypes = load_clean_prototypes()
    generators = load_generators()
    semigroup = load_semigroup_elements()

    augmented = np.concatenate(
        [
            prototypes,
            np.ones(
                (
                    len(
                        prototypes
                    ),
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    singular_values = np.linalg.svd(
        augmented,
        compute_uv=False,
    )

    design_rank = int(
        np.linalg.matrix_rank(
            augmented
        )
    )

    smallest_nonzero = float(
        singular_values[
            design_rank - 1
        ]
    )

    largest = float(
        singular_values[0]
    )

    condition_number = (
        largest
        / smallest_nonzero
    )

    fitted_generators = {}
    generator_rows = []

    for operation, mapping in (
        generators.items()
    ):
        targets = prototypes[
            np.asarray(
                mapping,
                dtype=np.int64,
            )
        ]

        fitted = fit_affine_map(
            source_points=prototypes,
            target_points=targets,
        )

        fitted_generators[
            operation
        ] = fitted

        generator_rows.append(
            {
                "operation":
                    operation,

                "interpolation_mse":
                    fitted[
                        "mse"
                    ],

                "maximum_absolute_error":
                    fitted[
                        "maximum_absolute_error"
                    ],
            }
        )

    semigroup_rows = []

    for element in semigroup:
        current = prototypes.copy()

        for operation in element[
            "word"
        ]:
            fitted = fitted_generators[
                operation
            ]

            current = (
                current
                @ fitted[
                    "matrix"
                ]
                + fitted[
                    "bias"
                ]
            )

        target = prototypes[
            np.asarray(
                element[
                    "mapping"
                ],
                dtype=np.int64,
            )
        ]

        residual = (
            current
            - target
        )

        semigroup_rows.append(
            {
                "element_id":
                    element[
                        "element_id"
                    ],

                "word_length":
                    len(
                        element[
                            "word"
                        ]
                    ),

                "composition_mse":
                    float(
                        np.mean(
                            residual
                            * residual
                        )
                    ),

                "maximum_absolute_error":
                    float(
                        np.max(
                            np.abs(
                                residual
                            )
                        )
                    ),
            }
        )

    maximum_generator_mse = max(
        row[
            "interpolation_mse"
        ]
        for row in generator_rows
    )

    maximum_generator_error = max(
        row[
            "maximum_absolute_error"
        ]
        for row in generator_rows
    )

    maximum_semigroup_mse = max(
        row[
            "composition_mse"
        ]
        for row in semigroup_rows
    )

    maximum_semigroup_error = max(
        row[
            "maximum_absolute_error"
        ]
        for row in semigroup_rows
    )

    shortcut_confirmed = bool(
        design_rank == 8
        and maximum_generator_mse < 1e-10
        and maximum_semigroup_mse < 1e-8
    )

    write_csv(
        OUTPUT_DIR
        / "affine_generator_interpolation.csv",
        generator_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "affine_semigroup_composition.csv",
        semigroup_rows,
    )

    summary = {
        "prototype_count":
            int(
                prototypes.shape[0]
            ),

        "observation_dimension":
            int(
                prototypes.shape[1]
            ),

        "augmented_affine_design_shape":
            list(
                augmented.shape
            ),

        "augmented_affine_design_rank":
            design_rank,

        "full_row_rank":
            bool(
                design_rank
                == augmented.shape[0]
            ),

        "affine_unknowns_per_output_dimension":
            int(
                augmented.shape[1]
            ),

        "constraints_per_output_dimension":
            int(
                augmented.shape[0]
            ),

        "underdetermined_degrees_per_output_dimension":
            int(
                augmented.shape[1]
                - augmented.shape[0]
            ),

        "nonzero_singular_value_condition_number":
            float(
                condition_number
            ),

        "maximum_generator_interpolation_mse":
            maximum_generator_mse,

        "maximum_generator_absolute_error":
            maximum_generator_error,

        "maximum_semigroup_composition_mse":
            maximum_semigroup_mse,

        "maximum_semigroup_absolute_error":
            maximum_semigroup_error,

        "affine_interpolation_shortcut_confirmed":
            shortcut_confirmed,

        "interpretation": (
            "The eight augmented prototype vectors have full row "
            "rank. Each output coordinate of an affine operator has "
            "13 adjustable coefficients but only eight prototype "
            "constraints. Consequently, an arbitrary mapping of the "
            "eight prototypes can be interpolated exactly by an "
            "affine observation-space operator, and compositions of "
            "the fitted primitive operators reproduce the finite "
            "semigroup on the prototype set."
        ),
    }

    write_json(
        OUTPUT_DIR
        / "affine_shortcut_audit.json",
        summary,
    )

    return summary


def predictive_rankings(
    rows,
):
    aggregate = aggregate_model_noise(
        rows
    )

    output = []
    best_by_noise = {}

    for noise in FROZEN_NOISE_LEVELS:
        group = [
            {
                "model_id":
                    model_id,

                **aggregate[
                    (
                        model_id,
                        noise,
                    )
                ],
            }
            for model_id in ALL_MODELS
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

        noisy_ranks = {
            row[
                "model_id"
            ]:
                rank
            for rank, row in enumerate(
                noisy_sorted,
                start=1,
            )
        }

        clean_ranks = {
            row[
                "model_id"
            ]:
                rank
            for rank, row in enumerate(
                clean_sorted,
                start=1,
            )
        }

        best_by_noise[
            str(
                noise
            )
        ] = {
            "best_noisy_target_model":
                noisy_sorted[0][
                    "model_id"
                ],

            "best_noisy_target_mse":
                noisy_sorted[0][
                    "noisy_mse_mean"
                ],

            "best_clean_target_model":
                clean_sorted[0][
                    "model_id"
                ],

            "best_clean_target_mse":
                clean_sorted[0][
                    "clean_mse_mean"
                ],

            "ocm_noisy_rank":
                noisy_ranks[
                    "OCM"
                ],

            "ocm_clean_rank":
                clean_ranks[
                    "OCM"
                ],
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
                        noisy_ranks[
                            row[
                                "model_id"
                            ]
                        ],

                    "clean_target_rank":
                        clean_ranks[
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

    return (
        output,
        best_by_noise,
    )


def structural_boundary_summary(
    phase2f_summary,
):
    hard_results = phase2f_summary[
        "primary_five_seed_results"
    ][
        "hard"
    ]

    output = {}

    for noise in FROZEN_NOISE_LEVELS:
        item = hard_results[
            str(
                noise
            )
        ]

        output[
            str(
                noise
            )
        ] = {
            "test_exact_transformation_rate":
                item[
                    "test_transformations"
                ][
                    "exact_transformation_rate"
                ][
                    "mean"
                ],

            "test_partition_accuracy":
                item[
                    "test_transformations"
                ][
                    "partition_accuracy"
                ][
                    "mean"
                ],

            "test_involved_relation_accuracy":
                item[
                    "test_involved_relations"
                ][
                    "accuracy"
                ][
                    "mean"
                ],

            "test_involved_relation_balanced_accuracy":
                item[
                    "test_involved_relations"
                ][
                    "balanced_accuracy"
                ][
                    "mean"
                ],

            "test_involved_relation_majority_baseline":
                item[
                    "test_involved_relations"
                ][
                    "majority_baseline"
                ][
                    "mean"
                ],
        }

    return output


def create_final_conclusions(
    rows,
    best_by_noise,
    affine_audit,
    phase2f_summary,
):
    aggregate = aggregate_model_noise(
        rows
    )

    ocm_parameters = aggregate[
        (
            "OCM",
            0.0,
        )
    ][
        "parameter_count"
    ]

    parameter_comparisons = {}

    for model_id in (
        "B2",
        "B3",
        "B4",
        "B5",
    ):
        model_parameters = aggregate[
            (
                model_id,
                0.0,
            )
        ][
            "parameter_count"
        ]

        parameter_comparisons[
            model_id
        ] = {
            "model_parameter_count":
                model_parameters,

            "ocm_parameter_count":
                ocm_parameters,

            "ocm_parameter_reduction":
                float(
                    1.0
                    - ocm_parameters
                    / model_parameters
                ),
        }

    structural_results = (
        structural_boundary_summary(
            phase2f_summary
        )
    )

    stronger_models = {
        best_by_noise[
            str(
                noise
            )
        ][
            "best_noisy_target_model"
        ]
        for noise in FROZEN_NOISE_LEVELS
    }

    predictive_superiority_supported = bool(
        stronger_models
        == {
            "OCM"
        }
    )

    exact_recovery_through_025 = bool(
        all(
            np.isclose(
                structural_results[
                    str(
                        noise
                    )
                ][
                    "test_exact_transformation_rate"
                ],
                1.0,
            )
            for noise in (
                0.0,
                0.1,
                0.25,
            )
        )
    )

    moderate_noise_partial_order = bool(
        structural_results[
            "0.5"
        ][
            "test_involved_relation_accuracy"
        ]
        >
        structural_results[
            "0.5"
        ][
            "test_involved_relation_majority_baseline"
        ]
    )

    severe_noise_structure_failed = bool(
        structural_results[
            "1.0"
        ][
            "test_involved_relation_accuracy"
        ]
        <=
        structural_results[
            "1.0"
        ][
            "test_involved_relation_majority_baseline"
        ]
    )

    conclusions = {
        "predictive_superiority_claim": {
            "claim": (
                "Explicit finite operation-channel structure provides "
                "the best OOD predictive accuracy."
            ),

            "supported":
                predictive_superiority_supported,

            "finding": (
                "Unsupported on Tier A. B2 and B3 achieve lower OOD "
                "prediction error than OCM across several or all "
                "nonzero-noise conditions."
            ),
        },

        "competitive_structured_prediction_claim": {
            "claim": (
                "OCM provides competitive OOD prediction while exposing "
                "an explicit, auditable operation structure."
            ),

            "supported":
                True,

            "qualification": (
                "Behavioral structural audits of B1-B5 are still "
                "required before claiming that only OCM supports "
                "structural recovery."
            ),
        },

        "prediction_identifiability_separation": {
            "supported":
                bool(
                    exact_recovery_through_025
                    and moderate_noise_partial_order
                    and severe_noise_structure_failed
                ),

            "finding": (
                "Prediction remains accurate at severe noise even when "
                "held-out transformation and relation recovery falls "
                "to baseline. Predictive success and structural "
                "identifiability are therefore empirically distinct."
            ),

            "structural_results":
                structural_results,
        },

        "affine_shortcut": {
            "confirmed":
                affine_audit[
                    "affine_interpolation_shortcut_confirmed"
                ],

            "finding": (
                "Tier A admits exact observation-space affine "
                "interpolation of arbitrary mappings on the eight "
                "prototype points. B1's near-zero clean-noise error is "
                "therefore an expected benchmark shortcut rather than "
                "evidence that generic affine physical dynamics are "
                "sufficient."
            ),
        },

        "parameter_comparisons":
            parameter_comparisons,

        "best_models_by_noise":
            best_by_noise,

        "rq3_status": {
            "strong_version":
                "not supported",

            "revised_version":
                "supported so far",

            "revised_statement": (
                "OCM provides competitive unseen-composition prediction "
                "with substantially fewer parameters than several "
                "generic neural baselines while enabling explicit "
                "operation-structure auditing."
            ),
        },

        "next_required_evaluation": (
            "Evaluate B1-B5 behaviorally on primitive mappings, held-out "
            "composite transformations, partitions, ranks, and "
            "informativeness relations using their already frozen "
            "checkpoints. No additional training or tuning is needed."
        ),
    }

    write_json(
        OUTPUT_DIR
        / "final_predictive_conclusions.json",
        conclusions,
    )

    return conclusions


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = validate_completed_phases()

    predictive_rows = (
        load_predictive_rows()
    )

    metadata_correction = (
        create_metadata_correction()
    )

    length_rows = (
        collect_length_rows()
    )

    length_summary = (
        aggregate_length_rows(
            length_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "sequence_length_summary.csv",
        length_summary,
    )

    length_trends = (
        create_length_trends(
            length_summary
        )
    )

    write_csv(
        OUTPUT_DIR
        / "sequence_length_trends.csv",
        length_trends,
    )

    paired_rows = paired_comparisons(
        predictive_rows
    )

    write_csv(
        OUTPUT_DIR
        / "paired_ocm_comparisons.csv",
        paired_rows,
    )

    parameter_rows = (
        parameter_efficiency_table(
            predictive_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "parameter_efficiency.csv",
        parameter_rows,
    )

    (
        ranking_rows,
        best_by_noise,
    ) = predictive_rankings(
        predictive_rows
    )

    write_csv(
        OUTPUT_DIR
        / "final_predictive_rankings.csv",
        ranking_rows,
    )

    affine_audit = (
        run_affine_shortcut_audit()
    )

    conclusions = (
        create_final_conclusions(
            rows=predictive_rows,
            best_by_noise=(
                best_by_noise
            ),
            affine_audit=(
                affine_audit
            ),
            phase2f_summary=(
                completed[
                    "phase2f"
                ]
            ),
        )
    )

    summary = {
        "phase":
            "2J final predictive audit",

        "source_predictive_row_count":
            len(
                predictive_rows
            ),

        "source_model_run_counts":
            EXPECTED_MODEL_RUN_COUNTS,

        "sequence_length_run_count":
            EXPECTED_LENGTH_RUN_COUNT,

        "sequence_length_summary_row_count":
            len(
                length_summary
            ),

        "paired_comparison_row_count":
            len(
                paired_rows
            ),

        "parameter_efficiency_model_count":
            len(
                parameter_rows
            ),

        "metadata_correction": {
            "clean_targets_used_for_training":
                metadata_correction[
                    "clean_targets_used_for_training"
                ],

            "clean_targets_used_for_selection":
                metadata_correction[
                    "clean_targets_used_for_checkpoint_selection"
                ],

            "clean_targets_accessed_during_validation":
                metadata_correction[
                    "clean_validation_targets_accessed"
                ],

            "test_used_for_selection":
                metadata_correction[
                    "test_used_for_checkpoint_selection"
                ],
        },

        "affine_shortcut_confirmed":
            affine_audit[
                "affine_interpolation_shortcut_confirmed"
            ],

        "strong_predictive_superiority_supported":
            conclusions[
                "predictive_superiority_claim"
            ][
                "supported"
            ],

        "rq3_strong_version":
            conclusions[
                "rq3_status"
            ][
                "strong_version"
            ],

        "rq3_revised_version":
            conclusions[
                "rq3_status"
            ][
                "revised_version"
            ],

        "training_performed":
            False,

        "checkpoint_selection_reopened":
            False,

        "checkpoints_modified":
            False,

        "phase2d_outputs_modified":
            False,

        "phase2f_outputs_modified":
            False,

        "phase2i_outputs_modified":
            False,

        "phase2j_status":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase2j_summary.json",
        summary,
    )

    print(
        "Phase 2J predictive audit completed."
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
