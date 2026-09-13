from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "tier_c_v4"

EXPECTED_AUTH_SHA256 = (
    "374973d53afa81175e3f5fa44946b48f"
    "bcdb4b0d89ed77539310fbee3f93d045"
)

AUTHORIZATION_PATH = Path(
    "outputs/"
    "phase4gr3f_tier_c_v4_test_authorization/"
    "one_time_test_opening_authorization.json"
)

AUTHORIZATION_SUMMARY_PATH = Path(
    "outputs/"
    "phase4gr3f_tier_c_v4_test_authorization/"
    "phase4gr3f_test_authorization_summary.json"
)

PHASE4A_SUMMARY_PATH = Path(
    "outputs/"
    "phase4ar3_tier_c_v4_protocol/"
    "phase4ar3_summary.json"
)

TEST_SEALING_POLICY_PATH = Path(
    "outputs/"
    "phase4ar3_tier_c_v4_protocol/"
    "test_sealing_policy.json"
)

PREDICTIVE_TEST_POLICY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_test_opening_policy.json"
)

FROZEN_FINAL_FIT_REGISTRY_PATH = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "frozen_final_fit_registry.csv"
)

SOURCE_FILES = [
    Path(
        "phase4br3_construct_tier_c_v4_development_fields.py"
    ),
    Path(
        "phase4br3_materialize_tier_c_v4_development_dataset.py"
    ),
    Path(
        "phase4fr3b_run_tier_c_v4_tuning.py"
    ),
    Path(
        "phase4gr3b_run_tier_c_v4_final_fits.py"
    ),
    Path(
        "phase4ar3_freeze_tier_c_v4_protocol.py"
    ),
]

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3g0_tier_c_v4_test_execution_interfaces"
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


def literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def function_signature_from_ast(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, Any]:
    args = node.args

    positional = [
        arg.arg
        for arg in (
            list(args.posonlyargs)
            + list(args.args)
        )
    ]

    keyword_only = [
        arg.arg
        for arg in args.kwonlyargs
    ]

    return {
        "name":
            node.name,

        "lineno":
            node.lineno,

        "end_lineno":
            getattr(
                node,
                "end_lineno",
                None,
            ),

        "positional_args":
            positional,

        "keyword_only_args":
            keyword_only,

        "vararg":
            (
                args.vararg.arg
                if args.vararg
                else None
            ),

        "kwarg":
            (
                args.kwarg.arg
                if args.kwarg
                else None
            ),
    }


def class_record(
    node: ast.ClassDef,
) -> dict[str, Any]:
    methods = []

    for child in node.body:
        if isinstance(
            child,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            methods.append(
                function_signature_from_ast(
                    child
                )
            )

    return {
        "name":
            node.name,

        "lineno":
            node.lineno,

        "end_lineno":
            getattr(
                node,
                "end_lineno",
                None,
            ),

        "methods":
            methods,
    }


def inspect_source_file(
    path: Path,
) -> dict[str, Any]:
    source = path.read_text(
        encoding="utf-8",
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    functions = []
    classes = []
    constants = {}
    test_related_strings = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            functions.append(
                function_signature_from_ast(
                    node
                )
            )

        elif isinstance(
            node,
            ast.ClassDef,
        ):
            classes.append(
                class_record(node)
            )

        elif isinstance(
            node,
            ast.Assign,
        ):
            value = literal_value(
                node.value
            )

            for target in node.targets:
                if (
                    isinstance(
                        target,
                        ast.Name,
                    )
                    and target.id.isupper()
                ):
                    constants[
                        target.id
                    ] = value

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            if (
                isinstance(
                    node.target,
                    ast.Name,
                )
                and node.target.id.isupper()
            ):
                constants[
                    node.target.id
                ] = literal_value(
                    node.value
                )

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Constant,
        ) and isinstance(
            node.value,
            str,
        ):
            lower = node.value.lower()

            if any(
                token in lower
                for token in (
                    "test_",
                    "test ",
                    "sealed",
                    "carrier",
                    "manifest",
                    "normalization",
                    "checkpoint",
                )
            ):
                test_related_strings.append(
                    node.value
                )

    # Deduplicate while preserving order.
    seen = set()
    unique_strings = []

    for value in test_related_strings:
        if value not in seen:
            seen.add(value)
            unique_strings.append(
                value
            )

    interesting_functions = [
        record
        for record in functions
        if any(
            token in record[
                "name"
            ].lower()
            for token in (
                "carrier",
                "field",
                "manifest",
                "normal",
                "dataset",
                "noise",
                "cell",
                "batch",
                "model",
                "codec",
                "loss",
                "evaluate",
                "checkpoint",
                "seed",
                "device",
                "save",
                "load",
            )
        )
    ]

    interesting_classes = [
        record
        for record in classes
        if any(
            token in record[
                "name"
            ].lower()
            for token in (
                "trajectory",
                "cell",
                "model",
                "codec",
                "dataset",
                "field",
            )
        )
    ]

    return {
        "path":
            str(path),

        "sha256":
            sha256_file(path),

        "line_count":
            len(
                source.splitlines()
            ),

        "constants":
            constants,

        "functions":
            functions,

        "interesting_functions":
            interesting_functions,

        "classes":
            classes,

        "interesting_classes":
            interesting_classes,

        "test_related_strings":
            unique_strings,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1/7] Verifying one-time authorization"
    )

    required_paths = [
        AUTHORIZATION_PATH,
        AUTHORIZATION_SUMMARY_PATH,
        PHASE4A_SUMMARY_PATH,
        TEST_SEALING_POLICY_PATH,
        PREDICTIVE_TEST_POLICY_PATH,
        FROZEN_FINAL_FIT_REGISTRY_PATH,
        *SOURCE_FILES,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    observed_auth_sha = sha256_file(
        AUTHORIZATION_PATH
    )

    require(
        observed_auth_sha
        == EXPECTED_AUTH_SHA256,
        (
            "One-time authorization SHA256 "
            "does not match the frozen value."
        ),
    )

    authorization = load_json(
        AUTHORIZATION_PATH
    )

    authorization_summary = load_json(
        AUTHORIZATION_SUMMARY_PATH
    )

    require(
        authorization[
            "phase4gr3f_status"
        ] == "one_time_test_opening_authorized",
        "One-time authorization is not active.",
    )

    require(
        authorization[
            "one_time_test_authorization_frozen"
        ] is True,
        "Authorization is not frozen.",
    )

    require(
        authorization[
            "test_generation_authorized"
        ] is True,
        "Test generation is not authorized.",
    )

    require(
        authorization[
            "test_evaluation_authorized"
        ] is True,
        "Test evaluation is not authorized.",
    )

    require(
        authorization[
            "test_open_count_after_this_phase"
        ] == 0,
        "Test opening has already been consumed.",
    )

    require(
        authorization[
            "maximum_test_open_count"
        ] == 1,
        "Maximum test-open count changed.",
    )

    require(
        authorization_summary[
            "test_open_count"
        ] == 0,
        "Authorization summary records test access.",
    )

    print(
        "[2/7] Verifying sealed test scope"
    )

    sealing = load_json(
        TEST_SEALING_POLICY_PATH
    )

    predictive_policy = load_json(
        PREDICTIVE_TEST_POLICY_PATH
    )

    expected_cells = [
        "test_iid_pairing",
        "test_composition",
        "test_carrier",
        "test_joint",
    ]

    require(
        sealing[
            "sealed_test_cells"
        ] == expected_cells,
        "Phase 4A-R3 sealed cell set changed.",
    )

    require(
        predictive_policy[
            "sealed_test_cells"
        ] == expected_cells,
        "Predictive sealed cell set changed.",
    )

    require(
        predictive_policy[
            "test_open_count"
        ] == 0,
        "Predictive policy records test access.",
    )

    require(
        predictive_policy[
            "maximum_test_open_count"
        ] == 1,
        "Predictive maximum test-open count changed.",
    )

    print(
        "[3/7] Freezing generator source interfaces"
    )

    source_records = {
        str(path):
            inspect_source_file(path)
        for path in SOURCE_FILES
    }

    print(
        "[4/7] Verifying frozen generation seeds"
    )

    phase4a_summary = load_json(
        PHASE4A_SUMMARY_PATH
    )

    seed_binding = {
        "fresh_carrier_parameter_seed":
            phase4a_summary[
                "fresh_carrier_parameter_seed"
            ],

        "fresh_field_initialization_seed":
            phase4a_summary[
                "fresh_field_initialization_seed"
            ],

        "fresh_solver_seed":
            phase4a_summary[
                "fresh_solver_seed"
            ],

        "fresh_noise_seed":
            phase4a_summary[
                "fresh_noise_seed"
            ],

        "sealed_test_manifest_seed":
            phase4a_summary[
                "sealed_test_manifest_seed"
            ],
    }

    require(
        all(
            value is not None
            for value in seed_binding.values()
        ),
        "One or more test-generation seeds are missing.",
    )

    print(
        "[5/7] Verifying generator constants"
    )

    field_record = source_records[
        "phase4br3_construct_tier_c_v4_development_fields.py"
    ]

    materialize_record = source_records[
        "phase4br3_materialize_tier_c_v4_development_dataset.py"
    ]

    field_constants = field_record[
        "constants"
    ]

    materialize_constants = materialize_record[
        "constants"
    ]

    # Development-side constants should correspond
    # to the already-frozen Phase 4A-R3 seeds.
    for constant_name, summary_key in (
        (
            "CARRIER_PARAMETER_SEED",
            "fresh_carrier_parameter_seed",
        ),
        (
            "FIELD_INITIALIZATION_SEED",
            "fresh_field_initialization_seed",
        ),
        (
            "SOLVER_SEED",
            "fresh_solver_seed",
        ),
    ):
        require(
            constant_name in field_constants,
            (
                f"Missing field-generator constant "
                f"{constant_name}."
            ),
        )

        require(
            int(
                field_constants[
                    constant_name
                ]
            )
            == int(
                seed_binding[
                    summary_key
                ]
            ),
            (
                f"{constant_name} differs from "
                "the frozen Phase 4A-R3 seed."
            ),
        )

    require(
        "DEVELOPMENT_MANIFEST_SEED"
        in materialize_constants,
        (
            "Development manifest seed constant "
            "was not found."
        ),
    )

    print(
        "[6/7] Freezing inference interfaces"
    )

    tuning_record = source_records[
        "phase4fr3b_run_tier_c_v4_tuning.py"
    ]

    tuning_function_names = {
        record[
            "name"
        ]
        for record in tuning_record[
            "functions"
        ]
    }

    required_inference_functions = {
        "resolve_device",
        "build_model_bundle",
        "compute_losses",
        "evaluate_validation",
    }

    require(
        required_inference_functions
        <= tuning_function_names,
        (
            "Required frozen inference functions "
            "are missing."
        ),
    )

    trajectory_classes = [
        record
        for record in tuning_record[
            "classes"
        ]
        if record[
            "name"
        ] == "TrajectoryCell"
    ]

    require(
        len(
            trajectory_classes
        ) == 1,
        (
            "Expected exactly one TrajectoryCell "
            "implementation."
        ),
    )

    print(
        "[7/7] Writing G0 interface freeze"
    )

    primary = predictive_policy[
        "primary_confirmatory_condition"
    ]

    secondary = predictive_policy[
        "secondary_conditions"
    ]

    interface_report_path = (
        OUTPUT_DIR
        / "test_execution_interface_report.json"
    )

    write_json(
        interface_report_path,
        {
            "protocol_version":
                PROTOCOL_VERSION,

            "authorization_sha256":
                observed_auth_sha,

            "source_records":
                source_records,

            "frozen_generation_seeds":
                seed_binding,

            "primary_confirmatory_condition":
                primary,

            "secondary_conditions":
                secondary,
        },
    )

    source_hashes = {
        str(path):
            sha256_file(path)
        for path in required_paths
    }

    source_hashes_path = (
        OUTPUT_DIR
        / "phase4gr3g0_source_hashes.json"
    )

    write_json(
        source_hashes_path,
        source_hashes,
    )

    summary = {
        "phase":
            (
                "4G-R3G0 Tier C v4 sealed-test "
                "execution interface freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "authorization_sha256":
            observed_auth_sha,

        "authorization_verified":
            True,

        "sealed_test_cells":
            expected_cells,

        "test_carrier_count":
            authorization[
                "test_carrier_count"
            ],

        "frozen_generation_seeds":
            seed_binding,

        "primary_model":
            "OCM",

        "primary_comparator":
            "B4",

        "primary_cell":
            primary[
                "cell"
            ],

        "primary_noise_fraction":
            primary[
                "noise_fraction"
            ],

        "primary_metric":
            primary[
                "metric"
            ],

        "secondary_cells":
            secondary[
                "cells"
            ],

        "secondary_noise_fractions":
            secondary[
                "noise_fractions"
            ],

        "secondary_metrics":
            secondary[
                "metrics"
            ],

        "generator_source_files":
            [
                str(path)
                for path in SOURCE_FILES[
                    :2
                ]
            ],

        "inference_source_file":
            "phase4fr3b_run_tier_c_v4_tuning.py",

        "required_inference_functions_verified":
            sorted(
                required_inference_functions
            ),

        "trajectory_cell_interface_verified":
            True,

        "interface_report_path":
            str(
                interface_report_path
            ),

        "interface_report_sha256":
            sha256_file(
                interface_report_path
            ),

        "source_hashes_path":
            str(
                source_hashes_path
            ),

        "source_hashes_sha256":
            sha256_file(
                source_hashes_path
            ),

        "test_artifacts_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "maximum_test_open_count":
            1,

        "single_opening_consumed":
            False,

        "phase4gr3g0_status":
            "ready_for_single_sealed_test_opening",

        "next_phase":
            (
                "4G-R3G1 Tier C v4 single sealed-test "
                "generation and immutable artifact freeze"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4gr3g0_interface_freeze_summary.json",
        summary,
    )

    print(
        "Phase 4G-R3G0 interface freeze passed."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
