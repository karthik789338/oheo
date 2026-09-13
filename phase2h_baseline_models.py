from __future__ import annotations

import torch
from torch import nn


class AffineOperationModel(nn.Module):
    """
    B1: One affine observation-space map per primitive operation.
    """

    def __init__(
        self,
        matrices: torch.Tensor,
        biases: torch.Tensor,
    ) -> None:
        super().__init__()

        if matrices.ndim != 3:
            raise ValueError(
                "Affine matrices must have shape "
                "(operation_count, dimension, dimension)."
            )

        if biases.ndim != 2:
            raise ValueError(
                "Affine biases must have shape "
                "(operation_count, dimension)."
            )

        self.register_buffer(
            "matrices",
            matrices.float(),
        )

        self.register_buffer(
            "biases",
            biases.float(),
        )

    def step(
        self,
        observations: torch.Tensor,
        operation_ids: torch.Tensor,
    ) -> torch.Tensor:
        selected_matrices = self.matrices[
            operation_ids
        ]

        selected_biases = self.biases[
            operation_ids
        ]

        return (
            torch.einsum(
                "...d,...de->...e",
                observations,
                selected_matrices,
            )
            + selected_biases
        )

    def rollout(
        self,
        initial_observation: torch.Tensor,
        operation_ids: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        current = initial_observation

        for step in range(
            operation_ids.shape[1]
        ):
            proposed = self.step(
                current,
                operation_ids[:, step],
            )

            active = step_mask[
                :,
                step,
            ].unsqueeze(-1)

            current = torch.where(
                active,
                proposed,
                current,
            )

        return current


class OperationConditionedMLP(nn.Module):
    """
    B2: Shared observation-space MLP conditioned on the operation.
    """

    def __init__(
        self,
        observation_dim: int,
        operation_count: int,
        hidden_width: int,
        operation_embedding_dim: int,
    ) -> None:
        super().__init__()

        self.operation_embedding = nn.Embedding(
            operation_count,
            operation_embedding_dim,
        )

        self.network = nn.Sequential(
            nn.Linear(
                observation_dim
                + operation_embedding_dim,
                hidden_width,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_width,
                hidden_width,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_width,
                observation_dim,
            ),
        )

    def step(
        self,
        observations: torch.Tensor,
        operation_ids: torch.Tensor,
    ) -> torch.Tensor:
        operation_embedding = (
            self.operation_embedding(
                operation_ids
            )
        )

        inputs = torch.cat(
            [
                observations,
                operation_embedding,
            ],
            dim=-1,
        )

        return self.network(
            inputs
        )

    def rollout(
        self,
        initial_observation: torch.Tensor,
        operation_ids: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        current = initial_observation

        for step in range(
            operation_ids.shape[1]
        ):
            proposed = self.step(
                current,
                operation_ids[:, step],
            )

            active = step_mask[
                :,
                step,
            ].unsqueeze(-1)

            current = torch.where(
                active,
                proposed,
                current,
            )

        return current


class GRUOperationSequenceModel(nn.Module):
    """
    B3: Initial-observation-conditioned GRU.

    The initial observation initializes the recurrent state.
    Primitive operation embeddings are then processed sequentially.
    """

    def __init__(
        self,
        observation_dim: int,
        operation_count: int,
        hidden_size: int,
        operation_embedding_dim: int,
    ) -> None:
        super().__init__()

        self.initial_projection = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_size,
            ),
            nn.Tanh(),
        )

        self.operation_embedding = nn.Embedding(
            operation_count,
            operation_embedding_dim,
        )

        self.recurrent_cell = nn.GRUCell(
            input_size=operation_embedding_dim,
            hidden_size=hidden_size,
        )

        self.output_projection = nn.Linear(
            hidden_size,
            observation_dim,
        )

    def predict_prefix(
        self,
        initial_observation: torch.Tensor,
        operation_ids: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.initial_projection(
            initial_observation
        )

        predictions = []

        for step in range(
            operation_ids.shape[1]
        ):
            embedded_operation = (
                self.operation_embedding(
                    operation_ids[:, step]
                )
            )

            proposed_hidden = (
                self.recurrent_cell(
                    embedded_operation,
                    hidden,
                )
            )

            active = step_mask[
                :,
                step,
            ].unsqueeze(-1)

            hidden = torch.where(
                active,
                proposed_hidden,
                hidden,
            )

            predictions.append(
                self.output_projection(
                    hidden
                )
            )

        return torch.stack(
            predictions,
            dim=1,
        )

    def rollout(
        self,
        initial_observation: torch.Tensor,
        operation_ids: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.initial_projection(
            initial_observation
        )

        for step in range(
            operation_ids.shape[1]
        ):
            embedded_operation = (
                self.operation_embedding(
                    operation_ids[:, step]
                )
            )

            proposed_hidden = (
                self.recurrent_cell(
                    embedded_operation,
                    hidden,
                )
            )

            active = step_mask[
                :,
                step,
            ].unsqueeze(-1)

            hidden = torch.where(
                active,
                proposed_hidden,
                hidden,
            )

        prediction = self.output_projection(
            hidden
        )

        has_steps = (
            step_mask.sum(
                dim=1
            ) > 0
        ).unsqueeze(-1)

        return torch.where(
            has_steps,
            prediction,
            initial_observation,
        )


class TransformerOperationSequenceModel(nn.Module):
    """
    B4: Causal Transformer over an initial-observation token
    followed by primitive-operation tokens.
    """

    def __init__(
        self,
        observation_dim: int,
        operation_count: int,
        model_dimension: int,
        attention_heads: int,
        layer_count: int,
        feedforward_dimension: int,
        maximum_sequence_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.maximum_sequence_length = (
            maximum_sequence_length
        )

        self.initial_projection = nn.Linear(
            observation_dim,
            model_dimension,
        )

        self.operation_embedding = nn.Embedding(
            operation_count,
            model_dimension,
        )

        self.position_embedding = nn.Embedding(
            maximum_sequence_length + 1,
            model_dimension,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=attention_heads,
            dim_feedforward=(
                feedforward_dimension
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=layer_count,
        )

        self.output_projection = nn.Linear(
            model_dimension,
            observation_dim,
        )

    def predict_prefix(
        self,
        initial_observation: torch.Tensor,
        operation_ids: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = (
            initial_observation.shape[0]
        )

        operation_length = (
            operation_ids.shape[1]
        )

        if (
            operation_length
            > self.maximum_sequence_length
        ):
            raise ValueError(
                "Operation sequence exceeds the "
                "frozen maximum sequence length."
            )

        initial_token = (
            self.initial_projection(
                initial_observation
            ).unsqueeze(1)
        )

        operation_tokens = (
            self.operation_embedding(
                operation_ids
            )
        )

        tokens = torch.cat(
            [
                initial_token,
                operation_tokens,
            ],
            dim=1,
        )

        positions = torch.arange(
            operation_length + 1,
            device=tokens.device,
        ).unsqueeze(0)

        tokens = (
            tokens
            + self.position_embedding(
                positions
            )
        )

        initial_padding = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=tokens.device,
        )

        padding_mask = torch.cat(
            [
                initial_padding,
                ~step_mask,
            ],
            dim=1,
        )

        sequence_length = (
            operation_length + 1
        )

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )

        encoded = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=(
                padding_mask
            ),
        )

        return self.output_projection(
            encoded[
                :,
                1:,
                :
            ]
        )

    def rollout(
        self,
        initial_observation: torch.Tensor,
        operation_ids: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        prefix_predictions = (
            self.predict_prefix(
                initial_observation,
                operation_ids,
                step_mask,
            )
        )

        step_counts = step_mask.sum(
            dim=1
        )

        final_indices = (
            step_counts - 1
        ).clamp_min(
            0
        )

        batch_indices = torch.arange(
            initial_observation.shape[0],
            device=initial_observation.device,
        )

        prediction = prefix_predictions[
            batch_indices,
            final_indices,
            :
        ]

        has_steps = (
            step_counts > 0
        ).unsqueeze(-1)

        return torch.where(
            has_steps,
            prediction,
            initial_observation,
        )


class ContinuousLatentOperatorModel(nn.Module):
    """
    B5: Continuous encoder/decoder with one unrestricted nonlinear
    latent transformation per primitive operation.

    Unlike OCM, these operations are not constrained to be
    stochastic finite-state channels.
    """

    def __init__(
        self,
        observation_dim: int,
        operation_count: int,
        latent_dimension: int,
        encoder_width: int,
        operator_hidden_width: int,
    ) -> None:
        super().__init__()

        self.operation_count = operation_count
        self.latent_dimension = latent_dimension

        self.encoder = nn.Sequential(
            nn.Linear(
                observation_dim,
                encoder_width,
            ),
            nn.GELU(),
            nn.Linear(
                encoder_width,
                encoder_width,
            ),
            nn.GELU(),
            nn.Linear(
                encoder_width,
                latent_dimension,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                latent_dimension,
                encoder_width,
            ),
            nn.GELU(),
            nn.Linear(
                encoder_width,
                encoder_width,
            ),
            nn.GELU(),
            nn.Linear(
                encoder_width,
                observation_dim,
            ),
        )

        self.operation_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        latent_dimension,
                        operator_hidden_width,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        operator_hidden_width,
                        operator_hidden_width,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        operator_hidden_width,
                        latent_dimension,
                    ),
                )
                for _ in range(
                    operation_count
                )
            ]
        )

    def encode(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            observations
        )

    def decode(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(
            latent
        )

    def apply_operation(
        self,
        latent: torch.Tensor,
        operation_ids: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = latent.shape

        flat_latent = latent.reshape(
            -1,
            self.latent_dimension,
        )

        flat_operation_ids = (
            operation_ids.reshape(
                -1
            )
        )

        flat_output = torch.empty_like(
            flat_latent
        )

        for operation_id in range(
            self.operation_count
        ):
            selected = (
                flat_operation_ids
                == operation_id
            )

            if selected.any():
                flat_output[
                    selected
                ] = self.operation_networks[
                    operation_id
                ](
                    flat_latent[
                        selected
                    ]
                )

        return flat_output.reshape(
            original_shape
        )

    def rollout(
        self,
        initial_observation: torch.Tensor,
        operation_ids: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        current = self.encode(
            initial_observation
        )

        for step in range(
            operation_ids.shape[1]
        ):
            proposed = self.apply_operation(
                current,
                operation_ids[:, step],
            )

            active = step_mask[
                :,
                step,
            ].unsqueeze(-1)

            current = torch.where(
                active,
                proposed,
                current,
            )

        return self.decode(
            current
        )
