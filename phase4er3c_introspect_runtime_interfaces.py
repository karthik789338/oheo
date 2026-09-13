from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import phase3c_run_tier_b_smoke_tests as tier_b_smoke
import phase3d_tune_tier_b_models as tier_b_tune


OUTPUT_DIR = Path(
    "outputs/phase4er3c_runtime_interface_introspection"
)

SOURCE_PATHS = (
    Path("phase3d_tune_tier_b_models.py"),
    Path("phase3c_run_tier_b_smoke_tests.py"),
    Path("phase2b_operation_channel_model.py"),
    Path("phase2h_baseline_models.py"),
)

CLASS_NAMES = (
    "OperationConditionedMLP",
    "GRUOperationSequenceModel",
    "TransformerOperationSequenceModel",
    "ContinuousLatentOperatorModel",
    "OperationChannelModel",
)

METHOD_NAMES = (
    "forward",
    "rollout",
    "encode",
    "decode",
    "transition",
    "step",
    "apply_operation",
)

FUNCTION_NAMES = (
    "build_model",
    "try_constructor",
    "train_baseline_configuration",
    "train_ocm_configuration",
)

SMOKE_FUNCTION_NAMES = (
    "ocm_forward_losses",
    "masked_vector_mse",
    "train_baseline",
    "train_ocm",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def safe_signature(value):
    try:
        return str(
            inspect.signature(value)
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def safe_source(value):
    try:
        return inspect.getsource(value)

    except (
        TypeError,
        OSError,
    ):
        return None


def find_object(name):
    for module_name, module in (
        (
            "phase3d_tune_tier_b_models",
            tier_b_tune,
        ),
        (
            "phase3c_run_tier_b_smoke_tests",
            tier_b_smoke,
        ),
    ):
        if hasattr(module, name):
            return (
                module_name,
                getattr(module, name),
            )

    return None, None


def inspect_class(name):
    module_name, class_object = find_object(
        name
    )

    if class_object is None:
        return {
            "class_name": name,
            "found": False,
        }

    result = {
        "class_name":
            name,

        "found":
            True,

        "module":
            module_name,

        "class_signature":
            safe_signature(
                class_object
            ),

        "module_path":
            inspect.getsourcefile(
                class_object
            ),
    }

    methods = {}

    for method_name in METHOD_NAMES:
        if not hasattr(
            class_object,
            method_name,
        ):
            continue

        method = getattr(
            class_object,
            method_name,
        )

        methods[method_name] = {
            "signature":
                safe_signature(method),

            "source":
                safe_source(method),
        }

    result["methods"] = methods

    return result


def inspect_function(
    module_name,
    module,
    function_name,
):
    if not hasattr(
        module,
        function_name,
    ):
        return {
            "function_name":
                function_name,

            "found":
                False,

            "module":
                module_name,
        }

    function = getattr(
        module,
        function_name,
    )

    return {
        "function_name":
            function_name,

        "found":
            True,

        "module":
            module_name,

        "signature":
            safe_signature(function),

        "source":
            safe_source(function),

        "source_path":
            inspect.getsourcefile(
                function
            ),
    }


def write_human_readable(
    path: Path,
    report,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "TIER C V4 RUNTIME INTERFACE INTROSPECTION\n"
        )

        handle.write(
            "=" * 100 + "\n\n"
        )

        handle.write(
            "MODULE CONSTANTS\n"
        )

        handle.write(
            "-" * 100 + "\n"
        )

        for key, value in report[
            "module_constants"
        ].items():
            handle.write(
                f"{key}: {value}\n"
            )

        for class_result in report[
            "classes"
        ]:
            handle.write("\n")
            handle.write(
                "=" * 100 + "\n"
            )

            handle.write(
                f"CLASS: "
                f"{class_result['class_name']}\n"
            )

            handle.write(
                "=" * 100 + "\n"
            )

            handle.write(
                f"Found: "
                f"{class_result['found']}\n"
            )

            if not class_result[
                "found"
            ]:
                continue

            handle.write(
                f"Module: "
                f"{class_result['module']}\n"
            )

            handle.write(
                f"Class signature: "
                f"{class_result['class_signature']}\n"
            )

            handle.write(
                f"Source path: "
                f"{class_result['module_path']}\n"
            )

            for method_name, method in (
                class_result[
                    "methods"
                ].items()
            ):
                handle.write("\n")
                handle.write(
                    f"METHOD: {method_name}\n"
                )

                handle.write(
                    "-" * 100 + "\n"
                )

                handle.write(
                    f"Signature: "
                    f"{method['signature']}\n\n"
                )

                if method["source"]:
                    handle.write(
                        method["source"]
                    )

                    handle.write("\n")

        for function_result in report[
            "functions"
        ]:
            handle.write("\n")
            handle.write(
                "=" * 100 + "\n"
            )

            handle.write(
                f"FUNCTION: "
                f"{function_result['function_name']}\n"
            )

            handle.write(
                "=" * 100 + "\n"
            )

            handle.write(
                f"Found: "
                f"{function_result['found']}\n"
            )

            if not function_result[
                "found"
            ]:
                continue

            handle.write(
                f"Module: "
                f"{function_result['module']}\n"
            )

            handle.write(
                f"Signature: "
                f"{function_result['signature']}\n"
            )

            handle.write(
                f"Source path: "
                f"{function_result['source_path']}\n\n"
            )

            if function_result[
                "source"
            ]:
                handle.write(
                    function_result["source"]
                )

                handle.write("\n")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in SOURCE_PATHS:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    print(
        "[1/4] Inspecting module constants"
    )

    module_constants = {
        "phase3d_tune_OBSERVATION_DIMENSION":
            getattr(
                tier_b_tune,
                "OBSERVATION_DIMENSION",
                None,
            ),

        "phase3d_tune_OPERATION_COUNT":
            getattr(
                tier_b_tune,
                "OPERATION_COUNT",
                None,
            ),

        "phase3c_smoke_OBSERVATION_DIMENSION":
            getattr(
                tier_b_smoke,
                "OBSERVATION_DIMENSION",
                None,
            ),

        "phase3c_smoke_OPERATION_COUNT":
            getattr(
                tier_b_smoke,
                "OPERATION_COUNT",
                None,
            ),
    }

    print(
        "[2/4] Inspecting model classes"
    )

    class_results = [
        inspect_class(name)
        for name in CLASS_NAMES
    ]

    print(
        "[3/4] Inspecting builders and loss functions"
    )

    function_results = []

    for function_name in FUNCTION_NAMES:
        function_results.append(
            inspect_function(
                module_name=(
                    "phase3d_tune_tier_b_models"
                ),
                module=tier_b_tune,
                function_name=function_name,
            )
        )

    for function_name in (
        SMOKE_FUNCTION_NAMES
    ):
        function_results.append(
            inspect_function(
                module_name=(
                    "phase3c_run_tier_b_smoke_tests"
                ),
                module=tier_b_smoke,
                function_name=function_name,
            )
        )

    print(
        "[4/4] Writing introspection outputs"
    )

    report = {
        "phase":
            (
                "4E-R3C Tier C v4 runtime "
                "interface introspection"
            ),

        "protocol_version":
            "tier_c_v4",

        "module_constants":
            module_constants,

        "classes":
            class_results,

        "functions":
            function_results,

        "source_hashes": {
            str(path):
                sha256_file(path)
            for path in SOURCE_PATHS
        },

        "models_instantiated":
            False,

        "model_parameters_created":
            False,

        "forward_passes_performed":
            0,

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

        "test_open_count":
            0,

        "introspection_status":
            "complete",
    }

    json_path = (
        OUTPUT_DIR
        / "runtime_interface_report.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    text_path = (
        OUTPUT_DIR
        / "runtime_interface_report.txt"
    )

    write_human_readable(
        text_path,
        report,
    )

    summary = {
        "phase":
            report["phase"],

        "protocol_version":
            "tier_c_v4",

        "module_constants":
            module_constants,

        "class_count_requested":
            len(CLASS_NAMES),

        "class_count_found":
            sum(
                int(result["found"])
                for result in class_results
            ),

        "function_count_requested":
            len(function_results),

        "function_count_found":
            sum(
                int(result["found"])
                for result in function_results
            ),

        "class_signatures": {
            result["class_name"]:
                result.get(
                    "class_signature"
                )
            for result in class_results
        },

        "method_signatures": {
            result["class_name"]: {
                name:
                    value["signature"]
                for name, value
                in result.get(
                    "methods",
                    {}
                ).items()
            }
            for result in class_results
        },

        "function_signatures": {
            result["function_name"]:
                result.get(
                    "signature"
                )
            for result in function_results
        },

        "models_instantiated":
            False,

        "optimizer_steps_performed":
            0,

        "scientific_metrics_computed":
            False,

        "test_data_read":
            False,

        "introspection_status":
            "complete",
    }

    summary_path = (
        OUTPUT_DIR
        / "runtime_interface_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"Outputs written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
