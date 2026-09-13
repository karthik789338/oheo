from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


PROTOCOL_VERSION = "tier_c_v4"
PRIMARY_NOISE = 0.25

BASELINE_IDS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
)

PRIMARY_MODEL = "OCM"

VALIDATION_SUMMARY_PATH = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "final_fit_validation_summary.csv"
)

FINAL_FIT_FREEZE_PATH = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "phase4gr3c_final_fit_freeze_summary.json"
)

TEST_POLICY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_test_opening_policy.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3d1_tier_c_v4_primary_comparator"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        VALIDATION_SUMMARY_PATH,
        FINAL_FIT_FREEZE_PATH,
        TEST_POLICY_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    freeze = load_json(
        FINAL_FIT_FREEZE_PATH
    )

    test_policy = load_json(
        TEST_POLICY_PATH
    )

    require(
        freeze["phase4gr3c_status"]
        == "final_fit_artifacts_frozen",
        "Final-fit artifacts are not frozen.",
    )

    require(
        freeze["audited_final_fit_count"] == 130,
        "Expected 130 audited final fits.",
    )

    require(
        freeze["test_data_read"] is False,
        "Test data was already read.",
    )

    require(
        freeze["test_open_count"] == 0,
        "Test open count is not zero.",
    )

    primary = test_policy[
        "primary_confirmatory_condition"
    ]

    require(
        primary["cell"] == "test_joint",
        "Primary test cell changed.",
    )

    require(
        math.isclose(
            float(primary["noise_fraction"]),
            PRIMARY_NOISE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Primary noise fraction changed.",
    )

    require(
        primary["metric"]
        == "noisy_target_rollout_mse",
        "Primary metric changed.",
    )

    require(
        primary["primary_model"] == PRIMARY_MODEL,
        "Primary model changed.",
    )

    require(
        primary["comparator"]
        == (
            "Best B1-B5 baseline selected using "
            "mean val_joint noisy-target rollout MSE "
            "at noise fraction 0.25 before test generation."
        ),
        "Frozen comparator rule changed.",
    )

    with VALIDATION_SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    candidate_rows = []

    for row in rows:
        model_id = row["model_id"]
        noise = float(row["noise_fraction"])

        if (
            model_id in BASELINE_IDS
            and math.isclose(
                noise,
                PRIMARY_NOISE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            metric = float(
                row[
                    "mean_best_validation_rollout_mse"
                ]
            )

            require(
                math.isfinite(metric),
                f"{model_id} metric is non-finite.",
            )

            candidate_rows.append(
                {
                    "model_id":
                        model_id,

                    "noise_fraction":
                        noise,

                    "final_fit_count":
                        int(
                            row["run_count"]
                        ),

                    "mean_validation_rollout_mse":
                        metric,

                    "minimum_validation_rollout_mse":
                        float(
                            row[
                                "minimum_best_validation_rollout_mse"
                            ]
                        ),

                    "maximum_validation_rollout_mse":
                        float(
                            row[
                                "maximum_best_validation_rollout_mse"
                            ]
                        ),
                }
            )

    require(
        len(candidate_rows) == 5,
        "Expected five B1-B5 candidates.",
    )

    require(
        {
            row["model_id"]
            for row in candidate_rows
        } == set(BASELINE_IDS),
        "Baseline candidate set changed.",
    )

    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            row[
                "mean_validation_rollout_mse"
            ],
            BASELINE_IDS.index(
                row["model_id"]
            ),
        ),
    )

    for rank, row in enumerate(
        ranked,
        start=1,
    ):
        row["rank"] = rank

    selected = ranked[0]

    require(
        selected["model_id"] == "B4",
        (
            "Observed comparator differs from "
            "the current frozen-validation result."
        ),
    )

    comparison_margin = (
        ranked[1][
            "mean_validation_rollout_mse"
        ]
        - ranked[0][
            "mean_validation_rollout_mse"
        ]
    )

    artifact = {
        "phase":
            (
                "4G-R3D1 Tier C v4 primary "
                "validation comparator freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "selection_population":
            list(BASELINE_IDS),

        "selection_cell":
            "val_joint",

        "selection_noise_fraction":
            PRIMARY_NOISE,

        "selection_metric":
            "noisy_target_rollout_mse",

        "aggregation_rule":
            (
                "Mean best-validation rollout MSE "
                "across the frozen final fits for "
                "each baseline at noise fraction 0.25."
            ),

        "selection_direction":
            "minimize",

        "candidate_ranking":
            ranked,

        "selected_comparator_model_id":
            selected["model_id"],

        "selected_comparator_mean_validation_rollout_mse":
            selected[
                "mean_validation_rollout_mse"
            ],

        "runner_up_model_id":
            ranked[1]["model_id"],

        "runner_up_mean_validation_rollout_mse":
            ranked[1][
                "mean_validation_rollout_mse"
            ],

        "selection_margin_mse":
            comparison_margin,

        "primary_model":
            PRIMARY_MODEL,

        "primary_model_used_for_comparator_selection":
            False,

        "test_artifact_used_for_selection":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "selection_frozen_before_test_generation":
            True,

        "primary_comparator_frozen":
            True,

        "source_validation_summary_path":
            str(
                VALIDATION_SUMMARY_PATH
            ),

        "source_validation_summary_sha256":
            sha256_file(
                VALIDATION_SUMMARY_PATH
            ),

        "source_final_fit_freeze_sha256":
            sha256_file(
                FINAL_FIT_FREEZE_PATH
            ),

        "source_test_policy_sha256":
            sha256_file(
                TEST_POLICY_PATH
            ),

        "phase4gr3d1_status":
            "primary_comparator_frozen",
    }

    artifact_path = (
        OUTPUT_DIR
        / "primary_validation_comparator.json"
    )

    write_json(
        artifact_path,
        artifact,
    )

    summary = {
        "phase":
            (
                "4G-R3D1 Tier C v4 primary "
                "comparator freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "selected_comparator":
            selected["model_id"],

        "selected_mean_validation_rollout_mse":
            selected[
                "mean_validation_rollout_mse"
            ],

        "runner_up":
            ranked[1]["model_id"],

        "runner_up_mean_validation_rollout_mse":
            ranked[1][
                "mean_validation_rollout_mse"
            ],

        "selection_margin_mse":
            comparison_margin,

        "primary_model":
            PRIMARY_MODEL,

        "test_data_read":
            False,

        "test_open_count":
            0,

        "comparator_artifact_path":
            str(artifact_path),

        "comparator_artifact_sha256":
            sha256_file(
                artifact_path
            ),

        "phase4gr3d1_status":
            "primary_comparator_frozen",

        "next_requirement":
            (
                "Complete five B0 persistence "
                "evaluations before test authorization."
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4gr3d1_primary_comparator_summary.json",
        summary,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
