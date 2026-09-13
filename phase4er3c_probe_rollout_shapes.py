from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

import phase3d_tune_tier_b_models as tier_b


REGISTRY_PATH = Path(
    "outputs/phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_tuning_registry.csv"
)

OUTPUT_PATH = Path(
    "outputs/phase4er3c_runtime_interface_introspection/"
    "rollout_shape_probe.json"
)


OBSERVATION_DIMENSION = 128
OPERATION_COUNT = 6
MAXIMUM_SEQUENCE_LENGTH = 4
BATCH_SIZE = 4


def load_first_configuration_per_model():
    selected = {}

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            model_id = row["model_id"]

            if model_id not in {
                "B2",
                "B3",
                "B4",
                "B5",
            }:
                continue

            if model_id in selected:
                continue

            selected[model_id] = json.loads(
                row[
                    "source_configuration_json"
                ]
            )

    if set(selected) != {
        "B2",
        "B3",
        "B4",
        "B5",
    }:
        raise AssertionError(
            f"Missing configurations: {selected.keys()}"
        )

    return selected


def main():
    torch.manual_seed(11)

    tier_b.OBSERVATION_DIMENSION = (
        OBSERVATION_DIMENSION
    )

    configurations = (
        load_first_configuration_per_model()
    )

    initial = torch.randn(
        BATCH_SIZE,
        OBSERVATION_DIMENSION,
        dtype=torch.float32,
    )

    operation_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [5, 4, 3, 2],
        ],
        dtype=torch.long,
    )

    step_mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )

    results = {}

    with torch.no_grad():
        for model_id in (
            "B2",
            "B3",
            "B4",
            "B5",
        ):
            model = tier_b.build_model(
                configuration=(
                    configurations[
                        model_id
                    ]
                ),
                operation_count=(
                    OPERATION_COUNT
                ),
                maximum_sequence_length=(
                    MAXIMUM_SEQUENCE_LENGTH
                ),
            )

            model.eval()

            prediction = model.rollout(
                initial_observation=initial,
                operation_ids=operation_ids,
                step_mask=step_mask,
            )

            results[model_id] = {
                "configuration_id":
                    configurations[
                        model_id
                    ][
                        "configuration_id"
                    ],

                "model_class":
                    type(model).__name__,

                "initial_shape":
                    list(initial.shape),

                "operation_ids_shape":
                    list(
                        operation_ids.shape
                    ),

                "step_mask_shape":
                    list(
                        step_mask.shape
                    ),

                "rollout_shape":
                    list(
                        prediction.shape
                    ),

                "rollout_dtype":
                    str(
                        prediction.dtype
                    ),

                "rollout_finite":
                    bool(
                        torch.isfinite(
                            prediction
                        ).all()
                    ),

                "parameter_count":
                    sum(
                        parameter.numel()
                        for parameter
                        in model.parameters()
                    ),
            }

    output = {
        "phase":
            (
                "4E-R3C Tier C v4 baseline "
                "rollout-shape probe"
            ),

        "observation_dimension":
            OBSERVATION_DIMENSION,

        "operation_count":
            OPERATION_COUNT,

        "maximum_sequence_length":
            MAXIMUM_SEQUENCE_LENGTH,

        "results":
            results,

        "backward_passes_performed":
            0,

        "optimizer_steps_performed":
            0,

        "scientific_metrics_computed":
            False,

        "privileged_data_read":
            False,

        "test_data_read":
            False,

        "probe_status":
            (
                "passed"
                if all(
                    result[
                        "rollout_finite"
                    ]
                    for result
                    in results.values()
                )
                else "failed"
            ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            output,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
