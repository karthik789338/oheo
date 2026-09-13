from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROTOCOL_VERSION = "tier_c_v4"

CODEC_REGISTRY_PATH = Path(
    "outputs/phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_field_codec_registry.json"
)

EXECUTION_CONTRACT_PATH = Path(
    "outputs/phase4dr32_tier_c_v4_execution_contract/"
    "tier_c_v4_execution_contract.json"
)

EXECUTION_SUMMARY_PATH = Path(
    "outputs/phase4dr32_tier_c_v4_execution_contract/"
    "phase4dr32_execution_contract_summary.json"
)

SMOKE_BATCH_PROVENANCE_PATH = Path(
    "outputs/"
    "phase4er3br1_tier_c_v4_operation_complete_smoke_batch/"
    "operation_complete_smoke_batch_provenance.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4dr33_tier_c_v4_codec_microarchitecture"
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


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = (
        CODEC_REGISTRY_PATH,
        EXECUTION_CONTRACT_PATH,
        EXECUTION_SUMMARY_PATH,
        SMOKE_BATCH_PROVENANCE_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    codec = load_json(
        CODEC_REGISTRY_PATH
    )

    contract = load_json(
        EXECUTION_CONTRACT_PATH
    )

    execution_summary = load_json(
        EXECUTION_SUMMARY_PATH
    )

    smoke_batch = load_json(
        SMOKE_BATCH_PROVENANCE_PATH
    )

    require(
        codec["adapter_id"]
        == "tier_c_v4_shared_convolutional_codec",
        "Unexpected field codec.",
    )

    require(
        codec["latent_dimension"] == 128,
        "Codec latent dimension changed.",
    )

    require(
        codec["input_shape"]
        == [4, 32, 32],
        "Codec input shape changed.",
    )

    require(
        codec["pretraining"] is False,
        "Codec pretraining was enabled.",
    )

    require(
        codec["shared_weights_across_models"]
        is False,
        "Codec weights were made shared.",
    )

    require(
        execution_summary[
            "phase4dr32_status"
        ] == "execution_contract_frozen",
        "Execution contract is not frozen.",
    )

    require(
        execution_summary[
            "implementation_smoke_test_authorized"
        ] is True,
        "Implementation smoke test is not authorized.",
    )

    require(
        execution_summary[
            "tuning_execution_authorized"
        ] is False,
        "Tuning has already been authorized.",
    )

    require(
        execution_summary[
            "test_open_count"
        ] == 0,
        "Test data was opened.",
    )

    require(
        smoke_batch[
            "operation_complete_smoke_batch_status"
        ] == "prepared_and_verified",
        "Operation-complete smoke batch is not verified.",
    )

    require(
        smoke_batch[
            "optimizer_steps_performed"
        ] == 0,
        "A predictive optimizer step was already performed.",
    )

    microarchitecture = {
        "phase":
            (
                "4D-R3.3 Tier C v4 field-codec "
                "microarchitecture freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "adapter_id":
            codec["adapter_id"],

        "freeze_timing":
            (
                "Frozen before any Tier C predictive "
                "forward-backward optimizer smoke execution."
            ),

        "input_shape":
            [4, 32, 32],

        "latent_dimension":
            128,

        "activation":
            {
                "type":
                    "torch.nn.GELU",

                "approximate":
                    "none",
            },

        "normalization":
            {
                "type":
                    "torch.nn.GroupNorm",

                "groups":
                    8,

                "epsilon":
                    1e-5,

                "affine":
                    True,

                "placement":
                    (
                        "After each non-final convolution in "
                        "residual main branches and before GELU."
                    ),
            },

        "residual_block": {
            "input_channels_equal_output_channels":
                True,

            "main_branch": [
                {
                    "layer":
                        "Conv2d",

                    "kernel_size":
                        3,

                    "stride":
                        1,

                    "padding":
                        1,

                    "dilation":
                        1,

                    "groups":
                        1,

                    "bias":
                        False,
                },
                {
                    "layer":
                        "GroupNorm",

                    "groups":
                        8,

                    "epsilon":
                        1e-5,

                    "affine":
                        True,
                },
                {
                    "layer":
                        "GELU",

                    "approximate":
                        "none",
                },
                {
                    "layer":
                        "Conv2d",

                    "kernel_size":
                        3,

                    "stride":
                        1,

                    "padding":
                        1,

                    "dilation":
                        1,

                    "groups":
                        1,

                    "bias":
                        False,
                },
                {
                    "layer":
                        "GroupNorm",

                    "groups":
                        8,

                    "epsilon":
                        1e-5,

                    "affine":
                        True,
                },
            ],

            "skip_branch":
                "identity",

            "merge":
                "elementwise_addition",

            "post_merge_activation":
                {
                    "layer":
                        "GELU",

                    "approximate":
                        "none",
                },
        },

        "residual_downsample_block": {
            "main_branch": [
                {
                    "layer":
                        "Conv2d",

                    "kernel_size":
                        3,

                    "stride":
                        2,

                    "padding":
                        1,

                    "dilation":
                        1,

                    "groups":
                        1,

                    "bias":
                        False,
                },
                {
                    "layer":
                        "GroupNorm",

                    "groups":
                        8,

                    "epsilon":
                        1e-5,

                    "affine":
                        True,
                },
                {
                    "layer":
                        "GELU",

                    "approximate":
                        "none",
                },
                {
                    "layer":
                        "Conv2d",

                    "kernel_size":
                        3,

                    "stride":
                        1,

                    "padding":
                        1,

                    "dilation":
                        1,

                    "groups":
                        1,

                    "bias":
                        False,
                },
                {
                    "layer":
                        "GroupNorm",

                    "groups":
                        8,

                    "epsilon":
                        1e-5,

                    "affine":
                        True,
                },
            ],

            "skip_branch": {
                "layer":
                    "Conv2d",

                "kernel_size":
                    1,

                "stride":
                    2,

                "padding":
                    0,

                "dilation":
                    1,

                "groups":
                    1,

                "bias":
                    False,
            },

            "merge":
                "elementwise_addition",

            "post_merge_activation":
                {
                    "layer":
                        "GELU",

                    "approximate":
                        "none",
                },
        },

        "encoder_boundary_layers": {
            "initial_conv": {
                "input_channels":
                    4,

                "output_channels":
                    32,

                "kernel_size":
                    3,

                "stride":
                    1,

                "padding":
                    1,

                "bias":
                    False,
            },

            "initial_group_norm": {
                "groups":
                    8,

                "epsilon":
                    1e-5,

                "affine":
                    True,
            },

            "final_linear": {
                "input_features":
                    2048,

                "output_features":
                    128,

                "bias":
                    True,
            },
        },

        "decoder_boundary_layers": {
            "initial_linear": {
                "input_features":
                    128,

                "output_features":
                    2048,

                "bias":
                    True,
            },

            "conv_transpose_bias":
                False,

            "final_conv": {
                "input_channels":
                    32,

                "output_channels":
                    4,

                "kernel_size":
                    3,

                "stride":
                    1,

                "padding":
                    1,

                "bias":
                    True,

                "activation":
                    "linear",
            },
        },

        "parameter_initialization":
            {
                "policy":
                    "PyTorch module defaults",

                "manual_reinitialization":
                    False,

                "seeded_before_module_construction":
                    True,
            },

        "dropout":
            {
                "enabled":
                    False,

                "probability":
                    0.0,
            },

        "pooling":
            {
                "enabled":
                    False,
            },

        "data_augmentation":
            False,

        "pretraining":
            False,

        "mixed_precision":
            False,

        "architecture_shared_across_models":
            True,

        "weights_shared_across_models":
            False,

        "diagnostic_weights_reused":
            False,

        "test_data_used":
            False,
    }

    microarchitecture_path = (
        OUTPUT_DIR
        / "tier_c_v4_codec_microarchitecture.json"
    )

    write_json(
        microarchitecture_path,
        microarchitecture,
    )

    source_hashes = {
        str(path):
            sha256_file(path)
        for path in required_paths
    }

    write_json(
        OUTPUT_DIR
        / "codec_microarchitecture_source_hashes.json",
        source_hashes,
    )

    summary = {
        "phase":
            microarchitecture["phase"],

        "protocol_version":
            PROTOCOL_VERSION,

        "source_codec_adapter_id":
            codec["adapter_id"],

        "source_codec_latent_dimension":
            codec["latent_dimension"],

        "residual_block_defined":
            True,

        "residual_downsample_block_defined":
            True,

        "convolution_bias_policy_defined":
            True,

        "group_norm_parameters_defined":
            True,

        "activation_parameters_defined":
            True,

        "initialization_policy_defined":
            True,

        "dropout_disabled":
            True,

        "microarchitecture_complete":
            True,

        "microarchitecture_path":
            str(
                microarchitecture_path
            ),

        "microarchitecture_sha256":
            sha256_file(
                microarchitecture_path
            ),

        "predictive_forward_passes_performed":
            0,

        "predictive_backward_passes_performed":
            0,

        "optimizer_steps_performed":
            0,

        "validation_outcomes_used":
            False,

        "test_data_used":
            False,

        "test_open_count":
            0,

        "tuning_execution_authorized":
            False,

        "final_fit_execution_authorized":
            False,

        "implementation_smoke_test_authorized":
            True,

        "phase4dr33_status":
            "codec_microarchitecture_frozen",

        "next_phase":
            (
                "4E-R3C Tier C v4 model-family "
                "forward-backward-rollout smoke execution"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4dr33_codec_microarchitecture_summary.json",
        summary,
    )

    print(
        "Phase 4D-R3.3 codec microarchitecture frozen."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
