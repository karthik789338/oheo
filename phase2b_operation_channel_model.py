from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class OperationChannelModel(nn.Module):
    """
    Operation Channel Model (OCM).

    The model represents every observation as a categorical
    distribution over latent states. Each primitive operation is a
    learned row-stochastic latent transition channel.
    """

    def __init__(
        self,
        observation_dim: int,
        latent_state_count: int,
        operation_count: int,
        hidden_sizes: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()

        hidden_1, hidden_2 = hidden_sizes

        self.observation_dim = observation_dim
        self.latent_state_count = latent_state_count
        self.operation_count = operation_count

        self.encoder = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_1,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_1,
                hidden_2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_2,
                latent_state_count,
            ),
        )

        # Each latent state has a learnable continuous prototype.
        self.codebook = nn.Parameter(
            torch.empty(
                latent_state_count,
                observation_dim,
            )
        )

        # One K x K transition matrix per primitive operation.
        # Row softmax converts these logits into stochastic channels.
        self.operation_logits = nn.Parameter(
            torch.empty(
                operation_count,
                latent_state_count,
                latent_state_count,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.codebook,
            mean=0.0,
            std=0.1,
        )

        nn.init.normal_(
            self.operation_logits,
            mean=0.0,
            std=0.02,
        )

    def initialize_codebook(
        self,
        centers: torch.Tensor,
    ) -> None:
        """
        Initialize the latent codebook using unsupervised centers
        computed only from training observations.
        """

        expected_shape = (
            self.latent_state_count,
            self.observation_dim,
        )

        if tuple(centers.shape) != expected_shape:
            raise ValueError(
                "Codebook initialization has the wrong shape. "
                f"Expected {expected_shape}, "
                f"received {tuple(centers.shape)}."
            )

        with torch.no_grad():
            self.codebook.copy_(
                centers.to(
                    device=self.codebook.device,
                    dtype=self.codebook.dtype,
                )
            )

    def encode(
        self,
        observations: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        logits = self.encoder(
            observations
        )

        return F.softmax(
            logits / temperature,
            dim=-1,
        )

    def decode(
        self,
        latent_distribution: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(
            latent_distribution,
            self.codebook,
        )

    def operation_channels(
        self,
        temperature: float,
    ) -> torch.Tensor:
        return F.softmax(
            self.operation_logits / temperature,
            dim=-1,
        )

    def apply_operation(
        self,
        latent_distribution: torch.Tensor,
        operation_ids: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Apply one operation to each latent distribution.

        latent_distribution:
            (..., K)

        operation_ids:
            (...)

        output:
            (..., K)
        """

        channels = self.operation_channels(
            temperature
        )

        selected_channels = channels[
            operation_ids
        ]

        return torch.einsum(
            "...i,...ij->...j",
            latent_distribution,
            selected_channels,
        )

    def rollout(
        self,
        initial_distribution: torch.Tensor,
        operation_ids: torch.Tensor,
        step_mask: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Roll forward from the initial latent state without seeing
        intermediate target observations.

        operation_ids:
            B x L

        step_mask:
            B x L
        """

        current = initial_distribution

        maximum_length = (
            operation_ids.shape[1]
        )

        for step in range(
            maximum_length
        ):
            proposed = self.apply_operation(
                latent_distribution=current,
                operation_ids=operation_ids[:, step],
                temperature=temperature,
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

    def operation_row_entropy(
        self,
        temperature: float,
    ) -> torch.Tensor:
        """
        Mean entropy of primitive-operation channel rows.

        Lower entropy means a more deterministic learned channel.
        """

        channels = self.operation_channels(
            temperature
        )

        log_channels = torch.log(
            channels.clamp_min(
                1e-8
            )
        )

        entropy = -(
            channels
            * log_channels
        ).sum(
            dim=-1
        )

        return entropy.mean()


def jensen_shannon_divergence(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """
    Jensen-Shannon divergence for categorical distributions.

    Returns one value for every distribution pair.
    """

    epsilon = 1e-8

    first = first.clamp_min(
        epsilon
    )

    second = second.clamp_min(
        epsilon
    )

    midpoint = 0.5 * (
        first + second
    )

    first_kl = (
        first
        * (
            torch.log(first)
            - torch.log(midpoint)
        )
    ).sum(
        dim=-1
    )

    second_kl = (
        second
        * (
            torch.log(second)
            - torch.log(midpoint)
        )
    ).sum(
        dim=-1
    )

    return 0.5 * (
        first_kl
        + second_kl
    )
