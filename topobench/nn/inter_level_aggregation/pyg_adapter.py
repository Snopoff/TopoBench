"""Adapters for PyG inter-level aggregation in TopoTune."""

import inspect

import torch
from torch_geometric.nn.aggr import Aggregation


class PyGAggregationAdapter(torch.nn.Module):
    """Adapt a PyG aggregation module to TopoTune route sequences.

    Parameters
    ----------
    aggregation : torch_geometric.nn.aggr.Aggregation
        Instantiated PyG aggregation module used to reduce the route dimension.
    """

    def __init__(self, aggregation: Aggregation) -> None:
        super().__init__()

        if not isinstance(aggregation, Aggregation):
            raise TypeError(
                "aggregation must be an instantiated torch_geometric.nn.aggr.Aggregation."
            )

        self.aggregation = aggregation
        self.supports_max_num_elements = (
            "max_num_elements"
            in inspect.signature(self.aggregation.forward).parameters
        )

    def forward(self, route_sequences: torch.Tensor) -> torch.Tensor:
        """Aggregate route-wise features with a PyG aggregation module.

        Parameters
        ----------
        route_sequences : torch.Tensor
            Tensor with shape ``[num_cells, seq_len, hidden_dim]``.

        Returns
        -------
        torch.Tensor
            Aggregated tensor with shape ``[num_cells, hidden_dim]``.
        """
        if route_sequences.ndim != 3:
            raise ValueError(
                "Expected route_sequences with shape [num_cells, seq_len, hidden_dim]."
            )

        num_cells, seq_len, hidden_dim = route_sequences.shape
        if seq_len == 1:
            return route_sequences[:, 0, :]

        x = route_sequences.reshape(num_cells * seq_len, hidden_dim)
        index = torch.arange(num_cells, device=x.device, dtype=torch.long)
        index = index.repeat_interleave(seq_len)

        aggregation_kwargs = {"index": index, "dim": 0, "dim_size": num_cells}
        if self.supports_max_num_elements:
            aggregation_kwargs["max_num_elements"] = seq_len

        aggregated = self.aggregation(x, **aggregation_kwargs)
        expected_shape = (num_cells, hidden_dim)
        if aggregated.shape != expected_shape:
            raise ValueError(
                "PyGAggregationAdapter requires the wrapped aggregation to preserve "
                "hidden_dim. "
                f"Expected {expected_shape}, received {tuple(aggregated.shape)}."
            )

        return aggregated
