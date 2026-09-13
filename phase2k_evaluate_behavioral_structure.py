from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from phase2b_operation_channel_model import (
    OperationChannelModel,
)

from phase2h_baseline_models import (
    AffineOperationModel,
)

from phase2h_tune_predictive_baselines import (
    build_neural_model,
)

from phase2e_evaluate_structural_recovery import (
    load_generators,
    load_semigroup,
    load_clean_prototypes,
    evaluate_mapping_family,
    primitive_metrics,
)

from phase2f_audit_ood_structural_generalization import (
    transformation_split_metrics,
    relation_scope_metrics,
)


PHASE1D_DIR = Path(
    "outputs/phase1d_compositional_splits"
)

PHASE2A_DIR = Path(
    "outputs/phase2a_learning_protocol"
)

PHASE2D_DIR = Path(
    "outputs/phase2d_final_ocm"
)

PHASE2F_DIR = Path(
    "outputs/phase2f_ood_structural_audit"
)

PHASE2I_DIR = Path(
    "outputs/phase2i_final_baselines"
)

OUTPUT_DIR = Path(
    "outputs/phase2k_behavioral_structure"
)

RUN_DETAIL_DIR = (
    OUTPUT_DIR / "run_details"
)


STATE_COUNT = 8
OBSERVATION_DIM = 12
LATENT_STATE_COUNT = 8

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

BASELINE_MODELS = (
    "B0",
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

ALL_MODELS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

EXPECTED_CONFIGURATION_IDS = {
    "B1": "B1_c02",
    "B2": "B2_c01",
    "B3": "B3_c01",
    "B4": "B4_c02",
    "B5": "B5_c02",
}

EXPECTED_TRANSFORMATION_COUNTS = {
    "train": 86,
    "val": 9,
    "test": 9,
}

EXPECTED_BEHAVIORAL_RUN_COUNT = 135
EXPECTED_CHECKPOINT_COUNT = 130
EXPECTED_TRANSFORMATION_COUNT = 104


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
):
    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but unavailable."
            )

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


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


def summarize(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot summarize an empty sequence."
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


def load_operation_vocabulary():
    return load_json(
        PHASE2A_DIR
        / "operation_vocabulary.json"
    )


def load_transformation_splits():
    rows = load_csv(
        PHASE1D_DIR
        / "transformation_split.csv"
    )

    split_lookup = {
        row[
            "element_id"
        ]:
            row[
                "split"
            ]
        for row in rows
    }

    counts = {
        split:
            sum(
                value == split
                for value
                in split_lookup.values()
            )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    if counts != EXPECTED_TRANSFORMATION_COUNTS:
        raise AssertionError(
            "Frozen transformation split changed: "
            f"{counts}"
        )

    return split_lookup


def validate_source_phases():
    phase2d = load_json(
        PHASE2D_DIR
        / "phase2d_summary.json"
    )

    phase2f = load_json(
        PHASE2F_DIR
        / "phase2f_summary.json"
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
        phase2f[
            "phase2f_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2F is not passed."
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
            "Phase 2I baseline-fit count changed."
        )

    return phase2f


def build_run_registry():
    rows = []

    # Persistence has no learned checkpoint.
    for noise in FROZEN_NOISE_LEVELS:
        rows.append(
            {
                "model_id":
                    "B0",

                "seed":
                    "deterministic",

                "noise_fraction":
                    noise,

                "checkpoint_required":
                    False,
            }
        )

    # B1 has one deterministic fit for each noise condition.
    for noise in FROZEN_NOISE_LEVELS:
        rows.append(
            {
                "model_id":
                    "B1",

                "seed":
                    "deterministic",

                "noise_fraction":
                    noise,

                "checkpoint_required":
                    True,
            }
        )

    # B2-B5 and OCM use the frozen five seeds.
    for model_id in (
        "B2",
        "B3",
        "B4",
        "B5",
        "OCM",
    ):
        for seed in FROZEN_SEEDS:
            for noise in FROZEN_NOISE_LEVELS:
                rows.append(
                    {
                        "model_id":
                            model_id,

                        "seed":
                            seed,

                        "noise_fraction":
                            noise,

                        "checkpoint_required":
                            True,
                    }
                )

    if len(
        rows
    ) != EXPECTED_BEHAVIORAL_RUN_COUNT:
        raise AssertionError(
            "Expected 135 behavioral evaluations."
        )

    checkpoint_count = sum(
        row[
            "checkpoint_required"
        ]
        for row in rows
    )

    if (
        checkpoint_count
        != EXPECTED_CHECKPOINT_COUNT
    ):
        raise AssertionError(
            "Expected 130 checkpoint evaluations."
        )

    return rows


def baseline_run_name(
    model_id,
    seed,
    noise,
):
    if model_id == "B1":
        return (
            f"B1_noise_"
            f"{noise_slug(noise)}"
        )

    return (
        f"{model_id}"
        f"_seed_{seed}"
        f"_noise_{noise_slug(noise)}"
    )


def ocm_run_name(
    seed,
    noise,
):
    return (
        f"seed_{seed}"
        f"_noise_{noise_slug(noise)}"
    )


def load_baseline_model(
    model_id,
    seed,
    noise,
    operation_count,
    device,
):
    run_name = baseline_run_name(
        model_id=model_id,
        seed=seed,
        noise=noise,
    )

    run_dir = (
        PHASE2I_DIR
        / "runs"
        / run_name
    )

    result_path = (
        run_dir
        / "result.json"
    )

    checkpoint_path = (
        run_dir
        / "best_model.pt"
    )

    if not result_path.exists():
        raise FileNotFoundError(
            result_path
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    result = load_json(
        result_path
    )

    if (
        result[
            "status"
        ]
        != "completed"
    ):
        raise AssertionError(
            f"{run_name} is incomplete."
        )

    if (
        result[
            "numerically_valid"
        ]
        is not True
    ):
        raise AssertionError(
            f"{run_name} is numerically invalid."
        )

    if (
        result[
            "configuration_id"
        ]
        != EXPECTED_CONFIGURATION_IDS[
            model_id
        ]
    ):
        raise AssertionError(
            f"{run_name} used the wrong configuration."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if model_id == "B1":
        state = checkpoint[
            "model_state_dict"
        ]

        model = AffineOperationModel(
            matrices=state[
                "matrices"
            ],
            biases=state[
                "biases"
            ],
        ).to(
            device
        )

        model.load_state_dict(
            state
        )

    else:
        configuration = result[
            "configuration"
        ]

        model = build_neural_model(
            configuration=configuration,
            operation_count=operation_count,
        ).to(
            device
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    model.eval()

    if not all(
        torch.isfinite(
            value
        ).all().item()
        for value
        in model.state_dict().values()
    ):
        raise FloatingPointError(
            f"{run_name} contains non-finite values."
        )

    return {
        "model":
            model,

        "temperature":
            None,

        "parameter_count":
            int(
                result[
                    "parameter_count"
                ]
            ),

        "checkpoint_path":
            str(
                checkpoint_path
            ),

        "result_path":
            str(
                result_path
            ),
    }


def load_ocm_model(
    seed,
    noise,
    operation_count,
    device,
):
    run_name = ocm_run_name(
        seed,
        noise,
    )

    run_dir = (
        PHASE2D_DIR
        / "runs"
        / run_name
    )

    result_path = (
        run_dir
        / "result.json"
    )

    checkpoint_path = (
        run_dir
        / "best_model.pt"
    )

    if not result_path.exists():
        raise FileNotFoundError(
            result_path
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    result = load_json(
        result_path
    )

    if (
        result[
            "status"
        ]
        != "completed"
    ):
        raise AssertionError(
            f"{run_name} is incomplete."
        )

    if (
        result[
            "numerically_valid"
        ]
        is not True
    ):
        raise AssertionError(
            f"{run_name} is numerically invalid."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = OperationChannelModel(
        observation_dim=(
            OBSERVATION_DIM
        ),
        latent_state_count=(
            LATENT_STATE_COUNT
        ),
        operation_count=(
            operation_count
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    if not all(
        torch.isfinite(
            parameter
        ).all().item()
        for parameter in model.parameters()
    ):
        raise FloatingPointError(
            f"{run_name} contains non-finite values."
        )

    return {
        "model":
            model,

        "temperature":
            float(
                checkpoint[
                    "temperature"
                ]
            ),

        "parameter_count":
            int(
                result[
                    "parameter_count"
                ]
            ),

        "checkpoint_path":
            str(
                checkpoint_path
            ),

        "result_path":
            str(
                result_path
            ),
    }


@torch.no_grad()
def predict_word_mapping(
    model_id,
    model,
    prototype_tensor,
    operation_vocabulary,
    word,
    temperature,
):
    """
    Evaluate only externally visible transformation behavior.

    The model receives a clean initial prototype and an operation word.
    Its final continuous prediction is assigned to the nearest prototype.
    """

    if len(
        word
    ) == 0:
        return tuple(
            range(
                STATE_COUNT
            )
        )

    if model_id == "B0":
        return tuple(
            range(
                STATE_COUNT
            )
        )

    operation_indices = [
        operation_vocabulary[
            operation
        ]
        for operation in word
    ]

    if model_id == "OCM":
        current = model.encode(
            prototype_tensor,
            temperature=temperature,
        )

        for operation_id in (
            operation_indices
        ):
            operation_ids = torch.full(
                (
                    STATE_COUNT,
                ),
                fill_value=operation_id,
                dtype=torch.long,
                device=prototype_tensor.device,
            )

            current = model.apply_operation(
                latent_distribution=current,
                operation_ids=operation_ids,
                temperature=temperature,
            )

        prediction = model.decode(
            current
        )

    else:
        operation_ids = torch.tensor(
            operation_indices,
            dtype=torch.long,
            device=prototype_tensor.device,
        ).unsqueeze(
            0
        ).repeat(
            STATE_COUNT,
            1,
        )

        step_mask = torch.ones(
            (
                STATE_COUNT,
                len(
                    operation_indices
                ),
            ),
            dtype=torch.bool,
            device=prototype_tensor.device,
        )

        prediction = model.rollout(
            initial_observation=(
                prototype_tensor
            ),
            operation_ids=(
                operation_ids
            ),
            step_mask=(
                step_mask
            ),
        )

    squared_distances = (
        (
            prediction[
                :,
                None,
                :
            ]
            - prototype_tensor[
                None,
                :,
                :
            ]
        )
        .pow(2)
        .sum(
            dim=-1
        )
    )

    nearest = torch.argmin(
        squared_distances,
        dim=1,
    )

    return tuple(
        int(
            value
        )
        for value in nearest
        .detach()
        .cpu()
        .tolist()
    )


def evaluate_one_run(
    registry_row,
    true_generators,
    semigroup_rows,
    split_lookup,
    prototype_tensor,
    operation_vocabulary,
    device,
):
    model_id = registry_row[
        "model_id"
    ]

    seed = registry_row[
        "seed"
    ]

    noise = registry_row[
        "noise_fraction"
    ]

    if model_id == "B0":
        model = None
        temperature = None
        parameter_count = 0
        checkpoint_path = None
        result_path = None

    elif model_id == "OCM":
        loaded = load_ocm_model(
            seed=seed,
            noise=noise,
            operation_count=len(
                operation_vocabulary
            ),
            device=device,
        )

        model = loaded[
            "model"
        ]

        temperature = loaded[
            "temperature"
        ]

        parameter_count = loaded[
            "parameter_count"
        ]

        checkpoint_path = loaded[
            "checkpoint_path"
        ]

        result_path = loaded[
            "result_path"
        ]

    else:
        loaded = load_baseline_model(
            model_id=model_id,
            seed=seed,
            noise=noise,
            operation_count=len(
                operation_vocabulary
            ),
            device=device,
        )

        model = loaded[
            "model"
        ]

        temperature = None

        parameter_count = loaded[
            "parameter_count"
        ]

        checkpoint_path = loaded[
            "checkpoint_path"
        ]

        result_path = loaded[
            "result_path"
        ]

    predicted_generators = {}

    for operation in true_generators:
        predicted_generators[
            operation
        ] = predict_word_mapping(
            model_id=model_id,
            model=model,
            prototype_tensor=prototype_tensor,
            operation_vocabulary=(
                operation_vocabulary
            ),
            word=(
                operation,
            ),
            temperature=temperature,
        )

    primitive_result = primitive_metrics(
        true_generators=true_generators,
        predicted_generators=(
            predicted_generators
        ),
    )

    predicted_mappings = []

    for semigroup_row in semigroup_rows:
        predicted_mappings.append(
            predict_word_mapping(
                model_id=model_id,
                model=model,
                prototype_tensor=(
                    prototype_tensor
                ),
                operation_vocabulary=(
                    operation_vocabulary
                ),
                word=semigroup_row[
                    "word"
                ],
                temperature=temperature,
            )
        )

    true_mappings = [
        row[
            "mapping"
        ]
        for row in semigroup_rows
    ]

    from phase2e_evaluate_structural_recovery import (
        relation_vector,
    )

    true_relations = relation_vector(
        true_mappings
    )

    full_result = evaluate_mapping_family(
        true_rows=semigroup_rows,
        predicted_mappings=(
            predicted_mappings
        ),
        true_relations=true_relations,
    )

    transformation_group = (
        full_result[
            "transformation_rows"
        ]
    )

    # evaluate_mapping_family stores mappings as JSON strings so they
    # remain machine-readable when written to CSV. The Phase 2F relation
    # audit expects integer sequences instead. Parse a separate in-memory
    # copy without changing the rows retained for output.
    relation_ready_group = []

    for transformation_row in transformation_group:
        parsed_row = dict(
            transformation_row
        )

        for mapping_field in (
            "true_mapping",
            "predicted_mapping",
        ):
            mapping_value = parsed_row[
                mapping_field
            ]

            if isinstance(
                mapping_value,
                str,
            ):
                mapping_value = json.loads(
                    mapping_value
                )

            parsed_row[
                mapping_field
            ] = tuple(
                int(value)
                for value in mapping_value
            )

        relation_ready_group.append(
            parsed_row
        )

    split_metrics = (
        transformation_split_metrics(
            transformation_group,
            split_lookup,
        )
    )

    scope_metrics = (
        relation_scope_metrics(
            relation_ready_group,
            split_lookup,
        )
    )

    relation_result = (
        full_result[
            "relation_metrics"
        ]
    )

    run_row = {
        "model_id":
            model_id,

        "seed":
            seed,

        "noise_fraction":
            float(
                noise
            ),

        "parameter_count":
            parameter_count,

        "temperature":
            temperature,

        "primitive_mapping_accuracy":
            primitive_result[
                "mapping_accuracy"
            ],

        "primitive_exact_rate":
            primitive_result[
                "exact_operation_rate"
            ],

        "transformation_mapping_accuracy":
            full_result[
                "mapping_accuracy"
            ],

        "exact_transformation_rate":
            full_result[
                "exact_transformation_rate"
            ],

        "partition_accuracy":
            full_result[
                "partition_accuracy"
            ],

        "rank_accuracy":
            full_result[
                "rank_accuracy"
            ],

        "distinct_predicted_transformation_count":
            full_result[
                "distinct_predicted_transformation_count"
            ],

        "distinct_predicted_partition_count":
            full_result[
                "distinct_predicted_partition_count"
            ],

        "relation_accuracy":
            relation_result[
                "accuracy"
            ],

        "relation_balanced_accuracy":
            relation_result[
                "balanced_accuracy"
            ],

        "relation_macro_f1":
            relation_result[
                "macro_f1"
            ],

        "relation_majority_baseline":
            relation_result[
                "majority_baseline_accuracy"
            ],

        "checkpoint_path":
            checkpoint_path,

        "result_path":
            result_path,
    }

    split_rows = [
        {
            "model_id":
                model_id,

            "seed":
                seed,

            "noise_fraction":
                float(
                    noise
                ),

            **row,
        }
        for row in split_metrics
    ]

    scope_rows = [
        {
            "model_id":
                model_id,

            "seed":
                seed,

            "noise_fraction":
                float(
                    noise
                ),

            **row,
        }
        for row in scope_metrics
    ]

    transformation_rows = [
        {
            "model_id":
                model_id,

            "seed":
                seed,

            "noise_fraction":
                float(
                    noise
                ),

            **row,
        }
        for row in transformation_group
    ]

    primitive_rows = [
        {
            "model_id":
                model_id,

            "seed":
                seed,

            "noise_fraction":
                float(
                    noise
                ),

            **row,
        }
        for row in primitive_result[
            "rows"
        ]
    ]

    detail = {
        "model_id":
            model_id,

        "seed":
            seed,

        "noise_fraction":
            float(
                noise
            ),

        "parameter_count":
            parameter_count,

        "checkpoint_path":
            checkpoint_path,

        "behavioral_evaluation": (
            "clean initial prototype plus operation "
            "sequence, followed by nearest-prototype "
            "classification"
        ),

        "training_performed":
            False,

        "checkpoint_modified":
            False,

        "checkpoint_selection_reopened":
            False,

        "structural_ground_truth_use":
            "evaluation only",

        "metrics":
            run_row,
    }

    return {
        "run_row":
            run_row,

        "split_rows":
            split_rows,

        "scope_rows":
            scope_rows,

        "transformation_rows":
            transformation_rows,

        "primitive_rows":
            primitive_rows,

        "detail":
            detail,
    }


def expected_seed_count(
    model_id,
    sensitivity,
):
    if model_id in {
        "B0",
        "B1",
    }:
        return 1

    return (
        4
        if sensitivity
        else 5
    )


def aggregate_rows(
    rows,
    key_fields,
    metric_fields,
    sensitivity=False,
):
    filtered = []

    for row in rows:
        model_id = row[
            "model_id"
        ]

        seed = row[
            "seed"
        ]

        if (
            sensitivity
            and model_id
            not in {
                "B0",
                "B1",
            }
            and seed == 11
        ):
            continue

        filtered.append(
            row
        )

    grouped = defaultdict(
        list
    )

    for row in filtered:
        key = tuple(
            row[
                field
            ]
            for field in key_fields
        )

        grouped[
            key
        ].append(
            row
        )

    output = {}

    for key in sorted(
        grouped
    ):
        group = grouped[
            key
        ]

        model_id = group[
            0
        ][
            "model_id"
        ]

        required_count = expected_seed_count(
            model_id=model_id,
            sensitivity=sensitivity,
        )

        if len(
            group
        ) != required_count:
            raise AssertionError(
                f"Group {key} has {len(group)} "
                f"rows rather than {required_count}."
            )

        label = "|".join(
            str(
                value
            )
            for value in key
        )

        item = {
            field:
                value
            for field, value in zip(
                key_fields,
                key,
            )
        }

        item[
            "seed_or_fit_count"
        ] = len(
            group
        )

        for metric in metric_fields:
            item[
                metric
            ] = summarize(
                [
                    row[
                        metric
                    ]
                    for row in group
                ]
            )

        output[
            label
        ] = item

    return output


def build_model_noise_summary(
    run_rows,
    split_rows,
    scope_rows,
    sensitivity=False,
):
    run_metrics = [
        "primitive_mapping_accuracy",
        "primitive_exact_rate",
        "transformation_mapping_accuracy",
        "exact_transformation_rate",
        "partition_accuracy",
        "rank_accuracy",
        "distinct_predicted_transformation_count",
        "distinct_predicted_partition_count",
        "relation_accuracy",
        "relation_balanced_accuracy",
        "relation_macro_f1",
    ]

    split_metrics = [
        "mapping_accuracy",
        "exact_transformation_rate",
        "partition_accuracy",
        "rank_accuracy",
    ]

    relation_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "majority_baseline",
        "accuracy_above_majority",
    ]

    run_aggregate = aggregate_rows(
        rows=run_rows,
        key_fields=[
            "model_id",
            "noise_fraction",
        ],
        metric_fields=run_metrics,
        sensitivity=sensitivity,
    )

    split_aggregate = aggregate_rows(
        rows=split_rows,
        key_fields=[
            "model_id",
            "noise_fraction",
            "split",
        ],
        metric_fields=split_metrics,
        sensitivity=sensitivity,
    )

    relation_aggregate = aggregate_rows(
        rows=scope_rows,
        key_fields=[
            "model_id",
            "noise_fraction",
            "scope",
        ],
        metric_fields=relation_metrics,
        sensitivity=sensitivity,
    )

    output = {}

    for model_id in ALL_MODELS:
        output[
            model_id
        ] = {}

        for noise in FROZEN_NOISE_LEVELS:
            noise_key = str(
                noise
            )

            base_key = (
                f"{model_id}|{noise_key}"
            )

            output[
                model_id
            ][
                noise_key
            ] = {
                "overall":
                    run_aggregate[
                        base_key
                    ],

                "test_transformations":
                    split_aggregate[
                        base_key
                        + "|test"
                    ],

                "test_involved_relations":
                    relation_aggregate[
                        base_key
                        + "|test_involved"
                    ],

                "test_test_relations":
                    relation_aggregate[
                        base_key
                        + "|test_test"
                    ],
            }

    return output


def build_compact_summary_rows(
    structured_summary,
):
    rows = []

    for model_id in ALL_MODELS:
        for noise in FROZEN_NOISE_LEVELS:
            item = structured_summary[
                model_id
            ][
                str(
                    noise
                )
            ]

            overall = item[
                "overall"
            ]

            test_transformations = item[
                "test_transformations"
            ]

            test_relations = item[
                "test_involved_relations"
            ]

            test_test_relations = item[
                "test_test_relations"
            ]

            rows.append(
                {
                    "model_id":
                        model_id,

                    "noise_fraction":
                        noise,

                    "seed_or_fit_count":
                        overall[
                            "seed_or_fit_count"
                        ],

                    "primitive_mapping_accuracy_mean":
                        overall[
                            "primitive_mapping_accuracy"
                        ][
                            "mean"
                        ],

                    "overall_exact_transformation_rate_mean":
                        overall[
                            "exact_transformation_rate"
                        ][
                            "mean"
                        ],

                    "overall_partition_accuracy_mean":
                        overall[
                            "partition_accuracy"
                        ][
                            "mean"
                        ],

                    "overall_relation_accuracy_mean":
                        overall[
                            "relation_accuracy"
                        ][
                            "mean"
                        ],

                    "overall_relation_balanced_accuracy_mean":
                        overall[
                            "relation_balanced_accuracy"
                        ][
                            "mean"
                        ],

                    "test_mapping_accuracy_mean":
                        test_transformations[
                            "mapping_accuracy"
                        ][
                            "mean"
                        ],

                    "test_exact_transformation_rate_mean":
                        test_transformations[
                            "exact_transformation_rate"
                        ][
                            "mean"
                        ],

                    "test_partition_accuracy_mean":
                        test_transformations[
                            "partition_accuracy"
                        ][
                            "mean"
                        ],

                    "test_rank_accuracy_mean":
                        test_transformations[
                            "rank_accuracy"
                        ][
                            "mean"
                        ],

                    "test_involved_relation_accuracy_mean":
                        test_relations[
                            "accuracy"
                        ][
                            "mean"
                        ],

                    "test_involved_relation_balanced_accuracy_mean":
                        test_relations[
                            "balanced_accuracy"
                        ][
                            "mean"
                        ],

                    "test_involved_relation_macro_f1_mean":
                        test_relations[
                            "macro_f1"
                        ][
                            "mean"
                        ],

                    "test_involved_relation_majority_baseline":
                        test_relations[
                            "majority_baseline"
                        ][
                            "mean"
                        ],

                    "test_test_relation_accuracy_mean":
                        test_test_relations[
                            "accuracy"
                        ][
                            "mean"
                        ],

                    "test_test_relation_balanced_accuracy_mean":
                        test_test_relations[
                            "balanced_accuracy"
                        ][
                            "mean"
                        ],
                }
            )

    return rows


def build_behavioral_rankings(
    compact_rows,
):
    output = []

    for noise in FROZEN_NOISE_LEVELS:
        group = [
            row
            for row in compact_rows
            if np.isclose(
                row[
                    "noise_fraction"
                ],
                noise,
            )
        ]

        by_exact = sorted(
            group,
            key=lambda row: (
                -row[
                    "test_exact_transformation_rate_mean"
                ],
                -row[
                    "test_involved_relation_balanced_accuracy_mean"
                ],
                row[
                    "model_id"
                ],
            ),
        )

        by_relation = sorted(
            group,
            key=lambda row: (
                -row[
                    "test_involved_relation_balanced_accuracy_mean"
                ],
                -row[
                    "test_exact_transformation_rate_mean"
                ],
                row[
                    "model_id"
                ],
            ),
        )

        exact_ranks = {
            row[
                "model_id"
            ]:
                rank
            for rank, row in enumerate(
                by_exact,
                start=1,
            )
        }

        relation_ranks = {
            row[
                "model_id"
            ]:
                rank
            for rank, row in enumerate(
                by_relation,
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

                    "test_exact_transformation_rank":
                        exact_ranks[
                            row[
                                "model_id"
                            ]
                        ],

                    "test_relation_balanced_accuracy_rank":
                        relation_ranks[
                            row[
                                "model_id"
                            ]
                        ],

                    "test_exact_transformation_rate":
                        row[
                            "test_exact_transformation_rate_mean"
                        ],

                    "test_relation_accuracy":
                        row[
                            "test_involved_relation_accuracy_mean"
                        ],

                    "test_relation_balanced_accuracy":
                        row[
                            "test_involved_relation_balanced_accuracy_mean"
                        ],

                    "test_relation_macro_f1":
                        row[
                            "test_involved_relation_macro_f1_mean"
                        ],
                }
            )

    return output


def compare_ocm_hard_and_behavioral(
    compact_rows,
    phase2f_summary,
):
    lookup = {
        (
            row[
                "model_id"
            ],
            row[
                "noise_fraction"
            ],
        ):
            row
        for row in compact_rows
    }

    hard_results = phase2f_summary[
        "primary_five_seed_results"
    ][
        "hard"
    ]

    rows = []

    for noise in FROZEN_NOISE_LEVELS:
        ocm_behavior = lookup[
            (
                "OCM",
                noise,
            )
        ]

        baseline_candidates = [
            lookup[
                (
                    model_id,
                    noise,
                )
            ]
            for model_id in BASELINE_MODELS
        ]

        best_baseline = max(
            baseline_candidates,
            key=lambda row: (
                row[
                    "test_involved_relation_balanced_accuracy_mean"
                ],
                row[
                    "test_exact_transformation_rate_mean"
                ],
            ),
        )

        hard = hard_results[
            str(
                noise
            )
        ]

        rows.append(
            {
                "noise_fraction":
                    noise,

                "ocm_hard_test_exact_transformation_rate":
                    hard[
                        "test_transformations"
                    ][
                        "exact_transformation_rate"
                    ][
                        "mean"
                    ],

                "ocm_behavioral_test_exact_transformation_rate":
                    ocm_behavior[
                        "test_exact_transformation_rate_mean"
                    ],

                "ocm_hard_test_relation_accuracy":
                    hard[
                        "test_involved_relations"
                    ][
                        "accuracy"
                    ][
                        "mean"
                    ],

                "ocm_hard_test_relation_balanced_accuracy":
                    hard[
                        "test_involved_relations"
                    ][
                        "balanced_accuracy"
                    ][
                        "mean"
                    ],

                "ocm_behavioral_test_relation_accuracy":
                    ocm_behavior[
                        "test_involved_relation_accuracy_mean"
                    ],

                "ocm_behavioral_test_relation_balanced_accuracy":
                    ocm_behavior[
                        "test_involved_relation_balanced_accuracy_mean"
                    ],

                "best_baseline_behavioral_model":
                    best_baseline[
                        "model_id"
                    ],

                "best_baseline_test_exact_transformation_rate":
                    best_baseline[
                        "test_exact_transformation_rate_mean"
                    ],

                "best_baseline_test_relation_accuracy":
                    best_baseline[
                        "test_involved_relation_accuracy_mean"
                    ],

                "best_baseline_test_relation_balanced_accuracy":
                    best_baseline[
                        "test_involved_relation_balanced_accuracy_mean"
                    ],
            }
        )

    return rows


def main():
    arguments = parse_arguments()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_DETAIL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        arguments.device
    )

    phase2f_summary = (
        validate_source_phases()
    )

    operation_vocabulary = (
        load_operation_vocabulary()
    )

    true_generators = (
        load_generators()
    )

    semigroup_rows = (
        load_semigroup()
    )

    if (
        len(
            semigroup_rows
        )
        != EXPECTED_TRANSFORMATION_COUNT
    ):
        raise AssertionError(
            "Semigroup size changed."
        )

    split_lookup = (
        load_transformation_splits()
    )

    clean_prototypes = (
        load_clean_prototypes()
    )

    prototype_tensor = torch.tensor(
        clean_prototypes,
        dtype=torch.float32,
        device=device,
    )

    registry = build_run_registry()

    run_rows = []
    split_rows = []
    scope_rows = []
    transformation_rows = []
    primitive_rows = []

    for index, registry_row in enumerate(
        registry,
        start=1,
    ):
        output = evaluate_one_run(
            registry_row=registry_row,
            true_generators=true_generators,
            semigroup_rows=semigroup_rows,
            split_lookup=split_lookup,
            prototype_tensor=prototype_tensor,
            operation_vocabulary=(
                operation_vocabulary
            ),
            device=device,
        )

        run_row = output[
            "run_row"
        ]

        run_rows.append(
            run_row
        )

        split_rows.extend(
            output[
                "split_rows"
            ]
        )

        scope_rows.extend(
            output[
                "scope_rows"
            ]
        )

        transformation_rows.extend(
            output[
                "transformation_rows"
            ]
        )

        primitive_rows.extend(
            output[
                "primitive_rows"
            ]
        )

        detail_name = (
            str(
                run_row[
                    "model_id"
                ]
            )
            + "_seed_"
            + str(
                run_row[
                    "seed"
                ]
            )
            + "_noise_"
            + noise_slug(
                run_row[
                    "noise_fraction"
                ]
            )
            + ".json"
        )

        write_json(
            RUN_DETAIL_DIR
            / detail_name,
            output[
                "detail"
            ],
        )

        test_split = next(
            row
            for row in output[
                "split_rows"
            ]
            if row[
                "split"
            ] == "test"
        )

        test_relations = next(
            row
            for row in output[
                "scope_rows"
            ]
            if row[
                "scope"
            ] == "test_involved"
        )

        print(
            f"[{index:03d}/{len(registry):03d}] "
            f"{run_row['model_id']} | "
            f"seed={run_row['seed']} | "
            f"noise={run_row['noise_fraction']} | "
            f"test exact="
            f"{test_split['exact_transformation_rate']:.4f} | "
            f"test relation balanced="
            f"{test_relations['balanced_accuracy']:.4f}"
        )

    if (
        len(
            run_rows
        )
        != EXPECTED_BEHAVIORAL_RUN_COUNT
    ):
        raise AssertionError(
            "Behavioral evaluation count changed."
        )

    write_csv(
        OUTPUT_DIR
        / "all_run_behavioral_metrics.csv",
        run_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "transformation_split_metrics.csv",
        split_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "relation_scope_metrics.csv",
        scope_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "transformation_predictions.csv",
        transformation_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "primitive_operation_predictions.csv",
        primitive_rows,
    )

    primary_summary = (
        build_model_noise_summary(
            run_rows=run_rows,
            split_rows=split_rows,
            scope_rows=scope_rows,
            sensitivity=False,
        )
    )

    sensitivity_summary = (
        build_model_noise_summary(
            run_rows=run_rows,
            split_rows=split_rows,
            scope_rows=scope_rows,
            sensitivity=True,
        )
    )

    compact_rows = (
        build_compact_summary_rows(
            primary_summary
        )
    )

    write_csv(
        OUTPUT_DIR
        / "model_noise_behavioral_summary.csv",
        compact_rows,
    )

    ranking_rows = (
        build_behavioral_rankings(
            compact_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "behavioral_structure_rankings.csv",
        ranking_rows,
    )

    hard_comparison_rows = (
        compare_ocm_hard_and_behavioral(
            compact_rows=compact_rows,
            phase2f_summary=(
                phase2f_summary
            ),
        )
    )

    write_csv(
        OUTPUT_DIR
        / "ocm_hard_vs_behavioral.csv",
        hard_comparison_rows,
    )

    behavioral_summary = {
        "phase":
            "2K behavioral structural comparison",

        "evaluation_definition": (
            "Each model receives a clean initial prototype and "
            "an operation sequence. Its final continuous prediction "
            "is assigned to the nearest clean prototype."
        ),

        "primary_five_seed_results":
            primary_summary,

        "four_seed_sensitivity_excluding_tuning_seed":
            sensitivity_summary,

        "ocm_comparison_note": (
            "OCM behavioral results use the same externally visible "
            "nearest-prototype evaluation as all baselines. OCM hard "
            "channel extraction from Phase 2F is reported separately."
        ),
    }

    write_json(
        OUTPUT_DIR
        / "behavioral_structure_summary.json",
        behavioral_summary,
    )

    phase_summary = {
        "phase":
            "2K behavioral structural comparison",

        "behavioral_run_group_count":
            len(
                run_rows
            ),

        "checkpoint_evaluation_count":
            sum(
                row[
                    "checkpoint_required"
                ]
                for row in registry
            ),

        "persistence_behavior_group_count":
            sum(
                row[
                    "model_id"
                ]
                == "B0"
                for row in registry
            ),

        "baseline_checkpoint_evaluation_count":
            sum(
                row[
                    "model_id"
                ]
                in {
                    "B1",
                    "B2",
                    "B3",
                    "B4",
                    "B5",
                }
                for row in registry
            ),

        "ocm_checkpoint_evaluation_count":
            sum(
                row[
                    "model_id"
                ]
                == "OCM"
                for row in registry
            ),

        "model_ids":
            list(
                ALL_MODELS
            ),

        "noise_levels":
            list(
                FROZEN_NOISE_LEVELS
            ),

        "neural_seeds":
            list(
                FROZEN_SEEDS
            ),

        "transformation_count":
            EXPECTED_TRANSFORMATION_COUNT,

        "transformation_split_counts":
            EXPECTED_TRANSFORMATION_COUNTS,

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

        "structural_ground_truth_used_for":
            "evaluation only",

        "phase2k_status":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase2k_summary.json",
        phase_summary,
    )

    print()
    print(
        "Phase 2K behavioral structural "
        "comparison completed."
    )

    print(
        json.dumps(
            phase_summary,
            indent=2,
        )
    )

    print(
        f"Outputs written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
