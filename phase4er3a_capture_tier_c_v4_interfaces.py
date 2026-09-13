from __future__ import annotations

import ast
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


CONTRACT_DIR = Path(
    "outputs/phase4dr32_tier_c_v4_execution_contract"
)

CONTRACT_SUMMARY_PATH = (
    CONTRACT_DIR
    / "phase4dr32_execution_contract_summary.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "tier_c_v4_execution_contract.json"
)

DATA_ROOT = Path(
    "outputs/phase4br3_tier_c_v4_development_data"
)

SOURCE_PATHS = (
    Path("phase3d_tune_tier_b_models.py"),
    Path("phase3c_run_tier_b_smoke_tests.py"),
    Path("phase3e_run_final_fits.py"),
    Path(
        "phase4br3_materialize_tier_c_v4_development_dataset.py"
    ),
)

OUTPUT_DIR = Path(
    "outputs/phase4er3a_tier_c_v4_interface_snapshot"
)

RELEVANT_FUNCTION_TERMS = (
    "build_model",
    "train",
    "load",
    "dataset",
    "batch",
    "collate",
    "rollout",
    "loss",
    "evaluate",
    "forward",
    "transition",
)

FORBIDDEN_PATH_TERMS = (
    "test_iid",
    "test_composition",
    "test_carrier",
    "test_joint",
    "sealed_test",
)


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


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

    fieldnames = []

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


def require(
    condition,
    message,
):
    if not condition:
        raise AssertionError(message)


def annotation_text(node):
    if node is None:
        return ""

    try:
        return ast.unparse(node)

    except Exception:
        return ""


def function_signature(
    node,
):
    positional = list(
        node.args.posonlyargs
    ) + list(node.args.args)

    default_offset = (
        len(positional)
        - len(node.args.defaults)
    )

    parameters = []

    for index, argument in enumerate(
        positional
    ):
        value = argument.arg

        annotation = annotation_text(
            argument.annotation
        )

        if annotation:
            value += f": {annotation}"

        if index >= default_offset:
            default_node = node.args.defaults[
                index - default_offset
            ]

            try:
                default_text = ast.unparse(
                    default_node
                )

            except Exception:
                default_text = "<expression>"

            value += f" = {default_text}"

        parameters.append(value)

    if node.args.vararg is not None:
        value = "*" + node.args.vararg.arg

        annotation = annotation_text(
            node.args.vararg.annotation
        )

        if annotation:
            value += f": {annotation}"

        parameters.append(value)

    elif node.args.kwonlyargs:
        parameters.append("*")

    for argument, default_node in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):
        value = argument.arg

        annotation = annotation_text(
            argument.annotation
        )

        if annotation:
            value += f": {annotation}"

        if default_node is not None:
            try:
                default_text = ast.unparse(
                    default_node
                )

            except Exception:
                default_text = "<expression>"

            value += f" = {default_text}"

        parameters.append(value)

    if node.args.kwarg is not None:
        value = "**" + node.args.kwarg.arg

        annotation = annotation_text(
            node.args.kwarg.annotation
        )

        if annotation:
            value += f": {annotation}"

        parameters.append(value)

    result = (
        f"{node.name}("
        + ", ".join(parameters)
        + ")"
    )

    return_annotation = annotation_text(
        node.returns
    )

    if return_annotation:
        result += f" -> {return_annotation}"

    return result


def source_excerpt(
    lines,
    start_line,
    end_line,
    maximum_lines=120,
):
    start_index = max(
        0,
        start_line - 1,
    )

    end_index = min(
        len(lines),
        end_line,
        start_index + maximum_lines,
    )

    return "\n".join(
        f"{index + 1}: {lines[index]}"
        for index in range(
            start_index,
            end_index,
        )
    )


def inventory_source(
    path: Path,
):
    source = path.read_text(
        encoding="utf-8"
    )

    lines = source.splitlines()

    tree = ast.parse(
        source,
        filename=str(path),
    )

    function_rows = []
    class_rows = []
    excerpt_records = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            lower_name = node.name.lower()

            if not any(
                term in lower_name
                for term in RELEVANT_FUNCTION_TERMS
            ):
                continue

            end_line = int(
                getattr(
                    node,
                    "end_lineno",
                    node.lineno,
                )
            )

            function_rows.append(
                {
                    "source_path":
                        str(path),

                    "function_name":
                        node.name,

                    "signature":
                        function_signature(node),

                    "start_line":
                        int(node.lineno),

                    "end_line":
                        end_line,

                    "is_async":
                        isinstance(
                            node,
                            ast.AsyncFunctionDef,
                        ),
                }
            )

            if (
                node.name == "build_model"
                or "train_" in lower_name
                or "loss" in lower_name
                or "rollout" in lower_name
                or "collate" in lower_name
            ):
                excerpt_records.append(
                    {
                        "source_path":
                            str(path),

                        "symbol":
                            node.name,

                        "start_line":
                            int(node.lineno),

                        "end_line":
                            end_line,

                        "excerpt":
                            source_excerpt(
                                lines,
                                int(node.lineno),
                                end_line,
                            ),
                    }
                )

        elif isinstance(
            node,
            ast.ClassDef,
        ):
            lower_name = node.name.lower()

            if not any(
                term in lower_name
                for term in (
                    "model",
                    "baseline",
                    "transformer",
                    "operator",
                    "ocm",
                    "dataset",
                    "encoder",
                    "decoder",
                )
            ):
                continue

            end_line = int(
                getattr(
                    node,
                    "end_lineno",
                    node.lineno,
                )
            )

            class_rows.append(
                {
                    "source_path":
                        str(path),

                    "class_name":
                        node.name,

                    "base_classes":
                        json.dumps(
                            [
                                annotation_text(base)
                                for base in node.bases
                            ]
                        ),

                    "start_line":
                        int(node.lineno),

                    "end_line":
                        end_line,
                }
            )

            excerpt_records.append(
                {
                    "source_path":
                        str(path),

                    "symbol":
                        node.name,

                    "start_line":
                        int(node.lineno),

                    "end_line":
                        end_line,

                    "excerpt":
                        source_excerpt(
                            lines,
                            int(node.lineno),
                            end_line,
                        ),
                }
            )

    function_rows.sort(
        key=lambda row: (
            row["source_path"],
            row["start_line"],
        )
    )

    class_rows.sort(
        key=lambda row: (
            row["source_path"],
            row["start_line"],
        )
    )

    excerpt_records.sort(
        key=lambda row: (
            row["source_path"],
            row["start_line"],
        )
    )

    return (
        function_rows,
        class_rows,
        excerpt_records,
    )


def forbidden_path(path: Path):
    text = str(path).lower()

    return any(
        term in text
        for term in FORBIDDEN_PATH_TERMS
    )


def inspect_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        first_row = next(
            reader,
            None,
        )

        fieldnames = (
            reader.fieldnames
            if reader.fieldnames is not None
            else []
        )

    return {
        "columns":
            fieldnames,

        "first_row":
            first_row,
    }


def inspect_npy(path: Path):
    array = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    return {
        "shape":
            list(array.shape),

        "dtype":
            str(array.dtype),
    }


def inspect_npz(path: Path):
    arrays = []

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        for key in archive.files:
            try:
                value = archive[key]

            except ValueError:
                arrays.append(
                    {
                        "key":
                            key,

                        "status":
                            "skipped_object_array",
                    }
                )

                continue

            arrays.append(
                {
                    "key":
                        key,

                    "shape":
                        list(value.shape),

                    "dtype":
                        str(value.dtype),

                    "finite":
                        (
                            bool(
                                np.isfinite(value).all()
                            )
                            if np.issubdtype(
                                value.dtype,
                                np.number,
                            )
                            else None
                        ),
                }
            )

    return {
        "arrays":
            arrays,
    }


def inspect_zip_names(path: Path):
    with zipfile.ZipFile(
        path,
        "r",
    ) as archive:
        return archive.namelist()


def inventory_visible_data():
    rows = []
    detailed = []

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            DATA_ROOT
        )

    for path in sorted(
        DATA_ROOT.rglob("*")
    ):
        if not path.is_file():
            continue

        if forbidden_path(path):
            raise AssertionError(
                "Forbidden test-like artifact found "
                f"inside development data root: {path}"
            )

        if "privileged" in {
            part.lower()
            for part in path.parts
        }:
            continue

        suffix = path.suffix.lower()

        row = {
            "path":
                str(path),

            "suffix":
                suffix,

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(path),
        }

        detail = {
            "path":
                str(path),
        }

        try:
            if suffix == ".csv":
                result = inspect_csv(path)

                row["content_type"] = "csv"

                row["column_count"] = len(
                    result["columns"]
                )

                detail.update(result)

            elif suffix == ".npy":
                result = inspect_npy(path)

                row["content_type"] = "npy"

                row["shape"] = json.dumps(
                    result["shape"]
                )

                row["dtype"] = result["dtype"]

                detail.update(result)

            elif suffix == ".npz":
                result = inspect_npz(path)

                row["content_type"] = "npz"

                row["array_count"] = len(
                    result["arrays"]
                )

                detail.update(result)

            elif suffix == ".json":
                value = load_json(path)

                row["content_type"] = "json"

                detail["top_level_type"] = (
                    type(value).__name__
                )

                if isinstance(value, dict):
                    detail[
                        "top_level_keys"
                    ] = sorted(value)

            else:
                row["content_type"] = "other"

        except Exception as error:
            row["inspection_error"] = (
                f"{type(error).__name__}: {error}"
            )

        rows.append(row)
        detailed.append(detail)

    return rows, detailed


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        CONTRACT_SUMMARY_PATH,
        CONTRACT_PATH,
        *SOURCE_PATHS,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    print(
        "[1/5] Validating execution-contract authorization"
    )

    summary = load_json(
        CONTRACT_SUMMARY_PATH
    )

    contract = load_json(
        CONTRACT_PATH
    )

    require(
        summary[
            "phase4dr32_status"
        ] == "execution_contract_frozen",
        "Execution contract is not frozen.",
    )

    require(
        summary[
            "implementation_smoke_test_authorized"
        ] is True,
        "Smoke test is not authorized.",
    )

    require(
        summary[
            "tuning_execution_authorized"
        ] is False,
        "Tuning was already authorized.",
    )

    require(
        summary[
            "test_open_count"
        ] == 0,
        "Test data was already opened.",
    )

    require(
        contract[
            "test_access"
        ][
            "test_generation_authorized"
        ] is False,
        "Test generation was authorized.",
    )

    print(
        "[2/5] Capturing Tier B model and training interfaces"
    )

    function_rows = []
    class_rows = []
    excerpt_records = []

    for source_path in SOURCE_PATHS:
        functions, classes, excerpts = (
            inventory_source(
                source_path
            )
        )

        function_rows.extend(
            functions
        )

        class_rows.extend(
            classes
        )

        excerpt_records.extend(
            excerpts
        )

    write_csv(
        OUTPUT_DIR
        / "source_function_inventory.csv",
        function_rows,
    )

    if class_rows:
        write_csv(
            OUTPUT_DIR
            / "source_class_inventory.csv",
            class_rows,
        )

    write_json(
        OUTPUT_DIR
        / "relevant_source_excerpts.json",
        excerpt_records,
    )

    print(
        "[3/5] Capturing visible Tier C development-data schema"
    )

    data_rows, data_details = (
        inventory_visible_data()
    )

    write_csv(
        OUTPUT_DIR
        / "visible_development_file_inventory.csv",
        data_rows,
    )

    write_json(
        OUTPUT_DIR
        / "visible_development_schema_details.json",
        data_details,
    )

    print(
        "[4/5] Freezing source hashes"
    )

    source_hashes = {
        str(path):
            sha256_file(path)
        for path in (
            CONTRACT_SUMMARY_PATH,
            CONTRACT_PATH,
            *SOURCE_PATHS,
        )
    }

    write_json(
        OUTPUT_DIR
        / "interface_snapshot_source_hashes.json",
        source_hashes,
    )

    print(
        "[5/5] Writing interface snapshot summary"
    )

    build_model_rows = [
        row
        for row in function_rows
        if row["function_name"]
        == "build_model"
    ]

    summary_output = {
        "phase":
            (
                "4E-R3A Tier C v4 predictive "
                "implementation interface snapshot"
            ),

        "protocol_version":
            "tier_c_v4",

        "source_execution_contract_complete":
            True,

        "source_execution_contract_sha256":
            summary[
                "execution_contract_sha256"
            ],

        "source_script_count":
            len(SOURCE_PATHS),

        "function_inventory_count":
            len(function_rows),

        "class_inventory_count":
            len(class_rows),

        "build_model_definition_count":
            len(build_model_rows),

        "visible_development_file_count":
            len(data_rows),

        "visible_development_files_inspected":
            True,

        "privileged_files_inspected":
            False,

        "test_files_inspected":
            False,

        "test_files_generated":
            False,

        "test_open_count":
            0,

        "model_parameters_updated":
            False,

        "optimizer_steps_performed":
            0,

        "scientific_metrics_computed":
            False,

        "implementation_smoke_test_executed":
            False,

        "interface_snapshot_status":
            (
                "complete"
                if build_model_rows
                and data_rows
                else "incomplete"
            ),

        "next_phase":
            (
                "4E-R3B Tier C v4 predictive "
                "forward-backward-rollout smoke execution"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4er3a_interface_snapshot_summary.json",
        summary_output,
    )

    print(
        "Phase 4E-R3A interface snapshot completed."
    )

    print(
        json.dumps(
            summary_output,
            indent=2,
        )
    )

    print(
        f"Outputs written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
