"""Define the TopoTune class, which, given a choice of hyperparameters, instantiates a GCCN expecting a collection of strictly augmented Hasse graphs as input."""

import copy
from typing import Literal

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from topobench.data.utils import get_routes_from_neighborhoods


class TopoTune(torch.nn.Module):
    """Tunes a GNN model using higher-order relations.

    This class takes a GNN and its kwargs as inputs, and tunes it with specified additional relations.

    Parameters
    ----------
    GNN : torch.nn.Module, a class not an object
        The GNN class to use. ex: GAT, GCN.
    neighborhoods : list of lists
        The neighborhoods of interest.
    layers : int
        The number of layers to use. Each layer contains one GNN.
    use_edge_attr : bool
        Whether to use edge attributes.
    activation : str
        The activation function to use. ex: 'relu', 'tanh', 'sigmoid'.
    rank_to_propagate : int, optional
        Extra rank to include in the final output dictionary even if no route
        targets it.
    inter_level_aggregation : torch.nn.Module or None, optional
        Inter-level aggregation module applied per destination rank. If
        ``None``, TopoTune uses the legacy rank-wise sum path.
    route_execution_mode : {"parallel", "sequential"}, optional
        Controls whether routes in each TopoTune layer are evaluated all at
        once and aggregated by destination rank, or applied one after another
        with immediate write-back to the batch embeddings.
    ordered_neighborhoods : list[str] or None, optional
        Subset of ``neighborhoods`` that should use an order-sensitive route
        model instead of the default route GNN.
    ordered_neighborhood_model : torch.nn.Module or None, optional
        Instantiated order-sensitive route model to deep-copy for ordered
        neighborhoods. This is expected to follow the PyG aggregation-style
        signature ``forward(x, index=..., dim=..., dim_size=...)``.
    """

    def __init__(
        self,
        GNN,
        neighborhoods,
        layers,
        use_edge_attr,
        activation,
        rank_to_propagate: int | None = None,
        inter_level_aggregation: torch.nn.Module | None = None,
        route_execution_mode: Literal["parallel", "sequential"] = "parallel",
        ordered_neighborhoods: list[str] | None = None,
        ordered_neighborhood_model: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.routes = get_routes_from_neighborhoods(neighborhoods)
        self.neighborhoods = neighborhoods
        self.layers = layers
        self.use_edge_attr = use_edge_attr
        self.inter_level_aggregation = inter_level_aggregation
        self.ordered_neighborhoods = (
            list(ordered_neighborhoods)
            if ordered_neighborhoods is not None
            else []
        )
        self.ordered_neighborhoods_set = set(self.ordered_neighborhoods)
        self.ordered_neighborhood_model = ordered_neighborhood_model

        if route_execution_mode not in {"parallel", "sequential"}:
            raise ValueError(
                "route_execution_mode must be either 'parallel' or 'sequential'."
            )
        if (
            route_execution_mode == "sequential"
            and inter_level_aggregation is not None
        ):
            raise ValueError(
                "inter_level_aggregation is only supported when route_execution_mode='parallel'."
            )
        self.route_execution_mode = route_execution_mode
        self.hidden_channels = GNN.hidden_channels
        self.out_channels = GNN.out_channels
        self._validate_ordered_neighborhoods()

        routes_max_rank = max([max(route) for route in self.routes])
        self.max_rank = (
            routes_max_rank
            if rank_to_propagate is None
            else max(routes_max_rank, rank_to_propagate)
        )
        self.route_modules = self._build_route_modules(
            GNN, ordered_neighborhood_model
        )
        self.activation = activation
        self.route_indices_by_dst_rank = self._get_route_indices_by_dst_rank()

        self.inter_level_aggregation_layers = (
            self._build_inter_level_aggregation_layers()
        )

    def _get_route_indices_by_dst_rank(self) -> dict[int, list[int]]:
        """Group route indices by destination rank while preserving config order.

        Returns
        -------
        dict[int, list[int]]
            Mapping from destination rank to the ordered list of route indices
            that contribute to it.
        """
        route_indices_by_dst_rank = {}
        for route_index, (_, dst_rank) in enumerate(self.routes):
            route_indices_by_dst_rank.setdefault(dst_rank, []).append(
                route_index
            )
        return route_indices_by_dst_rank

    def _is_ordered_route(self, route_index: int) -> bool:
        """Check whether a route should use the ordered route model.

        Parameters
        ----------
        route_index : int
            Route index in ``self.routes`` / ``self.neighborhoods``.

        Returns
        -------
        bool
            Whether the route is configured as ordered.
        """
        return (
            self.neighborhoods[route_index] in self.ordered_neighborhoods_set
        )

    def _build_inter_level_aggregation_layers(
        self,
    ) -> torch.nn.ModuleList | None:
        """Instantiate one inter-level aggregation module per TopoTune layer.

        Returns
        -------
        torch.nn.ModuleList or None
            One deep-copied aggregation module per TopoTune layer, or ``None``
            when the legacy sum path is active.
        """
        if (
            self.inter_level_aggregation is None
        ):  # if None, then we use the legacy sum path and don't need to instantiate any module
            return None
        if not isinstance(self.inter_level_aggregation, torch.nn.Module):
            raise TypeError(
                "inter_level_aggregation must be either None for the legacy sum path or an instantiated torch.nn.Module."
            )

        return torch.nn.ModuleList(
            [
                copy.deepcopy(self.inter_level_aggregation)
                for _ in range(self.layers)
            ]
        )

    def _validate_ordered_neighborhoods(self) -> None:
        """Validate the ordered-neighborhood configuration."""
        ordered_neighborhoods = self.ordered_neighborhoods_set
        if not ordered_neighborhoods:
            return

        missing_neighborhoods = ordered_neighborhoods.difference(
            self.neighborhoods
        )
        if missing_neighborhoods:
            raise ValueError(
                f"ordered_neighborhoods must be a subset of neighborhoods. Missing: {sorted(missing_neighborhoods)}."
            )

        if self.ordered_neighborhood_model is None:
            raise ValueError(
                "ordered_neighborhood_model must be provided when ordered_neighborhoods is not empty."
            )
        if not isinstance(self.ordered_neighborhood_model, torch.nn.Module):
            raise TypeError(
                "ordered_neighborhood_model must be an instantiated torch.nn.Module."
            )
        if not hasattr(self.ordered_neighborhood_model, "out_channels"):
            raise ValueError(
                "ordered_neighborhood_model must expose an out_channels attribute."
            )
        if self.ordered_neighborhood_model.out_channels != self.out_channels:
            raise ValueError(
                "ordered_neighborhood_model.out_channels must match GNN.out_channels."
            )

        for neighborhood, route in zip(
            self.neighborhoods, self.routes, strict=False
        ):
            if neighborhood not in ordered_neighborhoods:
                continue
            src_rank, dst_rank = route
            if src_rank == dst_rank or "incidence" not in neighborhood:
                raise ValueError(
                    f"Only inter-rank incidence neighborhoods can be marked ordered, but received {neighborhood}."
                )

    def _build_route_modules(
        self,
        GNN: torch.nn.Module,
        ordered_neighborhood_model: torch.nn.Module | None,
    ) -> torch.nn.ModuleList:
        """Instantiate one route module per route per TopoTune layer.

        Parameters
        ----------
        GNN : torch.nn.Module
            Default unordered route model.
        ordered_neighborhood_model : torch.nn.Module or None
            Default ordered route model.

        Returns
        -------
        torch.nn.ModuleList
            Nested module list indexed as ``[layer_idx][route_index]``.
        """
        route_modules = torch.nn.ModuleList()
        num_routes = len(self.routes)
        for _ in range(self.layers):
            layer_routes = torch.nn.ModuleList()
            for route_index in range(num_routes):
                if self._is_ordered_route(route_index):
                    layer_routes.append(
                        copy.deepcopy(ordered_neighborhood_model)
                    )
                else:
                    layer_routes.append(copy.deepcopy(GNN))
            route_modules.append(layer_routes)
        return route_modules

    def get_route_cache(
        self, batch: Data
    ) -> dict[int, dict[str, torch.Tensor | str]]:
        """Cache per-route tensors needed during the current forward pass.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            The batch object containing the complexes.

        Returns
        -------
        dict[int, dict[str, torch.Tensor | str]]
            Mapping from route index to the cached tensors required by that
            route. Ordered routes cache grouped source indices, while
            unordered inter-rank routes cache expanded lifted-graph tensors.
        """
        route_cache: dict[int, dict[str, torch.Tensor | str]] = {}
        for route_index, (neighborhood, route) in enumerate(
            zip(self.neighborhoods, self.routes, strict=False)
        ):
            src_rank, dst_rank = route
            if src_rank == dst_rank:
                continue

            if self._is_ordered_route(route_index):
                route_connectivity = getattr(batch, neighborhood).coalesce()
                dst_ids, src_ids = route_connectivity.indices()
                if dst_ids.numel() == 0:
                    empty = torch.empty(
                        0, dtype=torch.long, device=dst_ids.device
                    )
                    route_cache[route_index] = {
                        "kind": "ordered",
                        "src_ids": empty,
                        "active_dst_ids": empty,
                        "index": empty,
                    }
                    continue

                active_dst_ids, counts = torch.unique_consecutive(
                    dst_ids, return_counts=True
                )
                index = torch.arange(
                    active_dst_ids.numel(),
                    dtype=torch.long,
                    device=dst_ids.device,
                ).repeat_interleave(counts)
                route_cache[route_index] = {
                    "kind": "ordered",
                    "src_ids": src_ids,
                    "active_dst_ids": active_dst_ids,
                    "index": index,
                }
                continue

            n_dst_nodes = getattr(batch, f"x_{dst_rank}").shape[0]
            route_connectivity = getattr(batch, neighborhood).coalesce()
            edge_index, edge_attr = interrank_boundary_index(
                getattr(batch, f"x_{src_rank}"),
                route_connectivity.indices(),
                n_dst_nodes,
            )
            route_cache[route_index] = {
                "kind": "unordered_interrank",
                "edge_index": edge_index,
                "edge_attr": edge_attr,
            }

        return route_cache

    def intrarank_expand(self, batch: Data, src_rank: int, nbhd: str) -> Data:
        """Expand the complex into an intrarank Hasse graph.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            The batch object containing the complex.
        src_rank : int
            The source rank.
        nbhd : str
            The neighborhood to use.

        Returns
        -------
        torch_geometric.data.Data
            The expanded batch of intrarank Hasse graphs for this route.
        """
        batch_route = Data(
            x=getattr(batch, f"x_{src_rank}"),
            edge_index=getattr(batch, nbhd).indices(),
            edge_weight=getattr(batch, nbhd).values().squeeze(),
            edge_attr=getattr(batch, nbhd).values().squeeze(),
            requires_grad=True,
        )

        return batch_route

    def intrarank_gnn_forward(self, batch_route, layer_idx, route_index):
        """Forward pass of the GNN (one layer) for an intrarank Hasse graph.

        Parameters
        ----------
        batch_route : torch_geometric.data.Data
            The batch of intrarank Hasse graphs for this route.
        layer_idx : int
            The index of the TopoTune layer.
        route_index : int
            The index of the route.

        Returns
        -------
        torch.tensor
            The output of the GNN (updated features).
        """
        if batch_route.x.shape[0] < 2:
            return batch_route.x
        out = self.route_modules[layer_idx][route_index](
            batch_route.x,
            batch_route.edge_index,
            #    batch_route.edge_weight, # TODO Mathilde : some gnns take edge_weight (1d) and some take edge_attr.
            #    batch_route.edge_attr,
        )
        return out

    def interrank_expand(
        self,
        batch: Data,
        src_rank: int,
        dst_rank: int,
        route_cache_entry: dict[str, torch.Tensor | str],
        membership: dict[int, torch.Tensor],
        use_current_dst: bool = False,
    ):
        """Expand the complex into an interrank Hasse graph.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            The batch object containing the complex.
        src_rank : int
            The source rank.
        dst_rank : int
            The destination rank.
        route_cache_entry : dict[str, torch.Tensor | str]
            Cached tensors for this unordered inter-rank route.
        membership : dict
            The batch membership of the graphs per rank.
        use_current_dst : bool, optional
            Whether to seed the destination slice with the current destination embeddings instead of zeros.
            This is used by sequential route execution for later routes to consume earlier updates.

        Returns
        -------
        torch_geometric.data.Data
            The expanded batch of interrank Hasse graphs for this route.
        """
        src_batch = membership[src_rank]
        dst_batch = membership[dst_rank]
        edge_index = route_cache_entry["edge_index"]
        edge_attr = route_cache_entry["edge_attr"]
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(edge_attr, torch.Tensor)
        device = getattr(batch, f"x_{src_rank}").device
        feat_on_dst = (
            getattr(batch, f"x_{dst_rank}")
            if use_current_dst
            else torch.zeros_like(getattr(batch, f"x_{dst_rank}"))
        )
        x_in = torch.vstack([feat_on_dst, getattr(batch, f"x_{src_rank}")])
        batch_expanded = torch.cat([dst_batch, src_batch], dim=0)

        batch_route = Data(
            x=x_in,
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            edge_weight=edge_attr.to(device),
            batch=batch_expanded.to(device),
        )

        return batch_route

    def interrank_gnn_forward(
        self, batch_route, layer_idx, route_index, n_dst_cells
    ):
        """Forward pass of the GNN (one layer) for an interrank Hasse graph.

        Parameters
        ----------
        batch_route : torch_geometric.data.Data
            The batch of interrank Hasse graphs for this route.
        layer_idx : int
            The index of the layer.
        route_index : int
            The index of the route.
        n_dst_cells : int
            The number of destination cells in the whole batch.

        Returns
        -------
        torch.tensor
            The output of the GNN (updated features).
        """
        expanded_out = self.route_modules[layer_idx][route_index](
            batch_route.x,
            batch_route.edge_index,
            #    batch_route.edge_weight, # TODO : some gnns take edge_weight (1d) and some take edge_attr.
            #    batch_route.edge_attr,
        )
        out = expanded_out[:n_dst_cells]
        return out

    def _build_rank_sequences(
        self, x_out_per_route: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Stack route outputs into ordered per-rank sequences.

        Parameters
        ----------
        x_out_per_route : dict[int, torch.Tensor]
            Route-wise GNN outputs keyed by route index.

        Returns
        -------
        dict[int, torch.Tensor]
            Mapping from destination rank to either a single route tensor with
            shape ``[N, C]`` or a stacked route tensor with shape ``[N, T, C]``.
        """
        rank_sequences = {}
        for dst_rank, route_indices in self.route_indices_by_dst_rank.items():
            if len(route_indices) == 1:
                rank_sequences[dst_rank] = x_out_per_route[route_indices[0]]
            else:
                rank_sequences[dst_rank] = torch.stack(
                    [
                        x_out_per_route[route_index]
                        for route_index in route_indices
                    ],
                    dim=1,
                )
        return rank_sequences

    def _aggregate_inter_nbhd_sum(
        self, x_out_per_route: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Legacy method to aggregate the outputs of the GNN for each rank using a simple sum.

        Parameters
        ----------
        x_out_per_route : dict[int, torch.Tensor]
            The outputs of the GNN for each route.

        Returns
        -------
        dict[int, torch.Tensor]
            The aggregated outputs of the GNN for each rank.
        """
        x_out_per_rank = {}
        for route_index, (_, dst_rank) in enumerate(self.routes):
            if dst_rank not in x_out_per_rank:
                x_out_per_rank[dst_rank] = x_out_per_route[route_index]
            else:
                x_out_per_rank[dst_rank] += x_out_per_route[route_index]
        return x_out_per_rank

    def _aggregate_inter_nbhd_module(
        self, x_out_per_route: dict[int, torch.Tensor], layer_idx: int
    ) -> dict[int, torch.Tensor]:
        """Aggregate route outputs with an inter-level aggregation module.

        Parameters
        ----------
        x_out_per_route : dict[int, torch.Tensor]
            Route-wise GNN outputs keyed by route index.
        layer_idx : int
            Index of the TopoTune layer whose inter-level aggregation module
            should be used.

        Returns
        -------
        dict[int, torch.Tensor]
            Aggregated outputs keyed by destination rank.
        """
        inter_level_aggregation = self.inter_level_aggregation_layers[
            layer_idx
        ]  # type: ignore
        rank_sequences = self._build_rank_sequences(x_out_per_route)
        x_out_per_rank = {}

        for dst_rank, route_indices in self.route_indices_by_dst_rank.items():
            if len(route_indices) == 1:
                x_out_per_rank[dst_rank] = x_out_per_route[route_indices[0]]
            else:
                route_sequence = rank_sequences[dst_rank]
                aggregated = inter_level_aggregation(route_sequence)
                expected_shape = x_out_per_route[route_indices[0]].shape
                if aggregated.shape != expected_shape:
                    raise ValueError(
                        "inter_level_aggregation must return a tensor with the same "
                        "shape as a single route output. "
                        f"Expected {tuple(expected_shape)}, received "
                        f"{tuple(aggregated.shape)}."
                    )
                x_out_per_rank[dst_rank] = aggregated

        return x_out_per_rank

    def aggregate_inter_nbhd(
        self,
        x_out_per_route: dict[int, torch.Tensor],
        layer_idx: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """Aggregate the outputs of the GNN for each rank.

        While the GNN takes care of intra-nbhd aggregation,
        this will take care of inter-nbhd aggregation.

        Parameters
        ----------
        x_out_per_route : dict[int, torch.Tensor]
            The outputs of the GNN for each route.

        layer_idx : int | None
            The index of the layer, used to select the inter-level aggregation module if applicable.

        Returns
        -------
        dict[int, torch.Tensor]
            The aggregated outputs of the GNN for each rank.
        """
        if (
            self.inter_level_aggregation is None
        ):  # legacy sum path that doesn't use any module
            return self._aggregate_inter_nbhd_sum(x_out_per_route)
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required for module-based inter-level aggregation."
            )
        return self._aggregate_inter_nbhd_module(x_out_per_route, layer_idx)

    def _execute_unordered_route(
        self,
        batch: Data,
        membership: dict[int, torch.Tensor],
        route_cache: dict[int, dict[str, torch.Tensor | str]],
        layer_idx: int,
        route_index: int,
        use_current_dst: bool = False,
    ) -> tuple[int, torch.Tensor]:
        """Execute a single unordered route on the current batch state.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the current cell embeddings.
        membership : dict[int, torch.Tensor]
            Batch membership vectors keyed by rank.
        route_cache : dict[int, dict[str, torch.Tensor | str]]
            Cached per-route tensors keyed by route index.
        layer_idx : int
            Index of the TopoTune layer being executed.
        route_index : int
            Route index in ``self.routes`` / ``self.neighborhoods``.
        use_current_dst : bool, optional
            Whether an inter-rank expansion should expose the current
            destination embeddings to the route.

        Returns
        -------
        tuple[int, torch.Tensor]
            The destination rank and the route output tensor.
        """
        src_rank, dst_rank = self.routes[route_index]

        if src_rank == dst_rank:
            nbhd = self.neighborhoods[route_index]
            batch_route = self.intrarank_expand(batch, src_rank, nbhd)
            x_out = self.intrarank_gnn_forward(
                batch_route, layer_idx, route_index
            )
            return dst_rank, x_out

        route_cache_entry = route_cache[route_index]
        batch_route = self.interrank_expand(
            batch,
            src_rank,
            dst_rank,
            route_cache_entry,
            membership,
            use_current_dst=use_current_dst,
        )
        x_out = self.interrank_gnn_forward(
            batch_route,
            layer_idx,
            route_index,
            getattr(batch, f"x_{dst_rank}").shape[0],
        )
        return dst_rank, x_out

    def _execute_ordered_route(
        self,
        batch: Data,
        route_cache: dict[int, dict[str, torch.Tensor | str]],
        layer_idx: int,
        route_index: int,
    ) -> tuple[int, torch.Tensor]:
        """Execute a single ordered route on the current batch state.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the current cell embeddings.
        route_cache : dict[int, dict[str, torch.Tensor | str]]
            Cached per-route tensors keyed by route index.
        layer_idx : int
            Index of the TopoTune layer being executed.
        route_index : int
            Route index in ``self.routes`` / ``self.neighborhoods``.

        Returns
        -------
        tuple[int, torch.Tensor]
            The destination rank and the ordered route output tensor.
        """
        src_rank, dst_rank = self.routes[route_index]
        route_cache_entry = route_cache[route_index]
        src_ids = route_cache_entry["src_ids"]
        active_dst_ids = route_cache_entry["active_dst_ids"]
        index = route_cache_entry["index"]
        assert isinstance(src_ids, torch.Tensor)
        assert isinstance(active_dst_ids, torch.Tensor)
        assert isinstance(index, torch.Tensor)
        x_src = getattr(batch, f"x_{src_rank}")
        n_dst_cells = getattr(batch, f"x_{dst_rank}").shape[0]
        src_ids = src_ids.to(x_src.device)
        active_dst_ids = active_dst_ids.to(x_src.device)
        index = index.to(x_src.device)

        x_out = torch.zeros(
            (n_dst_cells, self.out_channels),
            dtype=x_src.dtype,
            device=x_src.device,
        )
        if src_ids.numel() == 0:
            return dst_rank, x_out

        route_out = self.route_modules[layer_idx][route_index](
            x_src[src_ids],
            index=index,
            dim=0,
            dim_size=active_dst_ids.numel(),
        )
        if route_out.ndim != 2:
            raise ValueError(
                "ordered_neighborhood_model must return a tensor with shape [num_active_destinations, out_channels]."
            )
        if route_out.shape != (active_dst_ids.numel(), self.out_channels):
            raise ValueError(
                "ordered_neighborhood_model must return one output per "
                "active destination and preserve out_channels. "
                f"Expected {(active_dst_ids.numel(), self.out_channels)}, "
                f"received {tuple(route_out.shape)}."
            )
        x_out[active_dst_ids] = route_out
        return dst_rank, x_out

    def _execute_route(
        self,
        batch: Data,
        membership: dict[int, torch.Tensor],
        route_cache: dict[int, dict[str, torch.Tensor | str]],
        layer_idx: int,
        route_index: int,
        use_current_dst: bool = False,
    ) -> tuple[int, torch.Tensor]:
        """Execute a single route on the current batch state.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the current cell embeddings.
        membership : dict[int, torch.Tensor]
            Batch membership vectors keyed by rank.
        route_cache : dict[int, dict[str, torch.Tensor | str]]
            Cached per-route tensors keyed by route index.
        layer_idx : int
            Index of the TopoTune layer being executed.
        route_index : int
            Route index in ``self.routes`` / ``self.neighborhoods``.
        use_current_dst : bool, optional
            Whether an unordered inter-rank expansion should expose the
            current destination embeddings to the route.

        Returns
        -------
        tuple[int, torch.Tensor]
            The destination rank and the route output tensor.
        """
        if self._is_ordered_route(route_index):
            return self._execute_ordered_route(
                batch,
                route_cache,
                layer_idx,
                route_index,
            )
        return self._execute_unordered_route(
            batch,
            membership,
            route_cache,
            layer_idx,
            route_index,
            use_current_dst=use_current_dst,
        )

    def _forward_parallel_layer(
        self,
        batch: Data,
        membership: dict[int, torch.Tensor],
        route_cache: dict[int, dict[str, torch.Tensor | str]],
        layer_idx: int,
        act,
    ) -> None:
        """Execute one TopoTune layer with the legacy parallel route semantics.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the current cell embeddings.
        membership : dict[int, torch.Tensor]
            Batch membership vectors keyed by rank.
        route_cache : dict[int, dict[str, torch.Tensor | str]]
            Cached per-route tensors keyed by route index.
        layer_idx : int
            Index of the TopoTune layer being executed.
        act : function
            The activation function to apply to the output features.
        """
        x_out_per_route = {}
        for route_index, _ in enumerate(self.routes):
            _, x_out = self._execute_route(
                batch,
                membership,
                route_cache,
                layer_idx,
                route_index,
            )
            x_out_per_route[route_index] = x_out

        x_out_per_rank = self.aggregate_inter_nbhd(
            x_out_per_route, layer_idx=layer_idx
        )
        for rank, x_out in x_out_per_rank.items():
            setattr(batch, f"x_{rank}", act(x_out))

    def _forward_sequential_layer(
        self,
        batch: Data,
        membership: dict[int, torch.Tensor],
        route_cache: dict[int, dict[str, torch.Tensor | str]],
        layer_idx: int,
        act,
    ) -> None:
        """Execute one TopoTune layer with ordered in-place route updates.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the current cell embeddings.
        membership : dict[int, torch.Tensor]
            Batch membership vectors keyed by rank.
        route_cache : dict[int, dict[str, torch.Tensor | str]]
            Cached per-route tensors keyed by route index.
        layer_idx : int
            Index of the TopoTune layer being executed.
        act : function
            The activation function to apply to the output features.
        """
        for route_index, _ in enumerate(self.routes):
            dst_rank, x_out = self._execute_route(
                batch,
                membership,
                route_cache,
                layer_idx,
                route_index,
                use_current_dst=True,
            )
            setattr(batch, f"x_{dst_rank}", act(x_out))

    def generate_membership_vectors(self, batch: Data):
        """Generate membership vectors based on batch.cell_statistics.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            The batch membership of the graphs per rank.
        """
        max_dim = batch.cell_statistics.shape[1]
        cell_statistics = batch.cell_statistics
        membership = {
            j: torch.tensor(
                [
                    elem
                    for list in [
                        [i] * x for i, x in enumerate(cell_statistics[:, j])
                    ]
                    for elem in list
                ]
            )
            for j in range(max_dim)
        }
        return membership

    def forward(self, batch: Data) -> dict[int, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        batch : Complex or ComplexBatch(Complex)
            The input data.

        Returns
        -------
        dict
            The output hidden states of the model per rank.
        """
        act = get_activation(self.activation)

        route_cache = self.get_route_cache(batch)
        membership = self.generate_membership_vectors(batch)

        for layer_idx in range(self.layers):
            if self.route_execution_mode == "parallel":
                self._forward_parallel_layer(
                    batch,
                    membership,
                    route_cache,
                    layer_idx,
                    act,
                )
            else:
                self._forward_sequential_layer(
                    batch,
                    membership,
                    route_cache,
                    layer_idx,
                    act,
                )

        return {
            rank: getattr(batch, f"x_{rank}")
            for rank in range(self.max_rank + 1)
        }


def interrank_boundary_index(x_src, boundary_index, n_dst_nodes):
    """
    Recover lifted graph.

    Edge-to-node boundary relationships of a graph with n_nodes and n_edges
    can be represented as up-adjacency node relations. There are n_nodes+n_edges nodes in this lifted graph.
    Desgiend to work for regular (edge-to-node and face-to-edge) boundary relationships.

    Parameters
    ----------
    x_src : torch.tensor
        Source node features. Shape [n_src_nodes, n_features]. Should represent edge or face features.
    boundary_index : list of lists or list of tensors
        List boundary_index[0] stores node ids in the boundary of edge stored in boundary_index[1].
        List boundary_index[1] stores list of edges.
    n_dst_nodes : int
        Number of destination nodes.

    Returns
    -------
    edge_index : list of lists
        The edge_index[0][i] and edge_index[1][i] are the two nodes of edge i.
    edge_attr : tensor
        Edge features are given by feature of bounding node represnting an edge. Shape [n_edges, n_features].
    """
    node_ids = (
        boundary_index[0]
        if torch.is_tensor(boundary_index[0])
        else torch.tensor(boundary_index[0], dtype=torch.int32)
    )
    edge_ids = (
        boundary_index[1]
        if torch.is_tensor(boundary_index[1])
        else torch.tensor(boundary_index[1], dtype=torch.int32)
    )

    max_node_id = n_dst_nodes
    adjusted_edge_ids = edge_ids + max_node_id

    edge_index = torch.zeros((2, node_ids.numel()), dtype=node_ids.dtype)
    edge_index[0, :] = node_ids
    edge_index[1, :] = adjusted_edge_ids

    edge_attr = x_src[edge_ids].squeeze()

    return edge_index, edge_attr


def get_activation(nonlinearity, return_module=False):
    """Activation resolver from CWN.

    Parameters
    ----------
    nonlinearity : str
        The nonlinearity to use.
    return_module : bool
        Whether to return the module or the function.

    Returns
    -------
    module or function
        The module or the function.
    """
    if nonlinearity == "relu":
        module = torch.nn.ReLU
        function = F.relu
    elif nonlinearity == "elu":
        module = torch.nn.ELU
        function = F.elu
    elif nonlinearity == "id":
        module = torch.nn.Identity

        def function(x):
            return x
    elif nonlinearity == "sigmoid":
        module = torch.nn.Sigmoid
        function = F.sigmoid
    elif nonlinearity == "tanh":
        module = torch.nn.Tanh
        function = torch.tanh
    else:
        raise NotImplementedError(
            f"Nonlinearity {nonlinearity} is not currently supported."
        )
    if return_module:
        return module
    return function
