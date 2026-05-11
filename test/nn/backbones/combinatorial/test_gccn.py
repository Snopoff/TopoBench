"""Unit tests for TopoTune."""

import hydra
import pytest
import rootutils
import torch
from omegaconf import OmegaConf
from test._utils.nn_module_auto_test import NNModuleAutoTest
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import GRUAggregation, SumAggregation

from topobench.nn.backbones.combinatorial.gccn import (
    TopoTune,
    get_activation,
    interrank_boundary_index,
)
from topobench.nn.inter_level_aggregation import PyGAggregationAdapter

ROOT = rootutils.find_root(__file__, indicator=".project-root")

class MockGNN(torch.nn.Module):
    """Mock GNN module for testing purposes.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
        """Forward pass of the MockGNN.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_index : torch.Tensor
            Edge indices.

        Returns
        -------
        torch.Tensor
            Output of the GCN layer.
        """
        return self.conv(x, edge_index)


class OrderedRouteMockGNN(torch.nn.Module):
    """Deterministic route-local update used to test ordered execution.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    out_channels : int
        Number of output channels.
    dst_scale : float, optional
        Multiplicative factor applied to the current features before the
        source-driven accumulation is added.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dst_scale: float = 2.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dst_scale = dst_scale

    def forward(self, x, edge_index):
        """Scale current features and add source features onto edge_index[0].

        Parameters
        ----------
        x : torch.Tensor
            Input node features for the temporary route graph.
        edge_index : torch.Tensor
            Route graph connectivity where source features are accumulated into
            the destination rows.

        Returns
        -------
        torch.Tensor
            Updated node features after deterministic accumulation.
        """
        out = self.dst_scale * x.clone()
        aggregated = torch.zeros_like(x)
        if edge_index.numel() > 0:
            aggregated.index_add_(0, edge_index[0], x[edge_index[1]])
        return out + aggregated


class OrderedNeighborhoodMockModel(torch.nn.Module):
    """Deterministic ordered route model used to test ordered neighborhoods.

    Parameters
    ----------
    in_channels : int
        Number of input channels for each ordered source feature.
    out_channels : int
        Number of output channels for each destination sequence.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = out_channels
        self.out_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor | None = None,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        """Compute a position-weighted sum for each destination sequence.

        Parameters
        ----------
        x : torch.Tensor
            Flattened ordered source features.
        index : torch.Tensor, optional
            Destination-group indices for the flattened ordered source
            features.
        dim : int, optional
            Aggregation dimension. Unused in the mock implementation.
        dim_size : int, optional
            Number of destination groups. Unused in the mock implementation.

        Returns
        -------
        torch.Tensor
            One deterministic output per destination sequence.
        """
        del dim, dim_size
        if index is None:
            raise ValueError("index is required for OrderedNeighborhoodMockModel.")
        outputs = []
        for group_id in torch.unique_consecutive(index).tolist():
            sequence = x[index == group_id]
            weights = torch.arange(
                1,
                sequence.shape[0] + 1,
                dtype=x.dtype,
                device=x.device,
            ).unsqueeze(-1)
            outputs.append((sequence * weights).sum(dim=0, keepdim=True))
        return torch.cat(outputs, dim=0)


def create_mock_complex_batch(hidden_dim: int = 16):
    """Create a mock complex batch for testing.

    Parameters
    ----------
    hidden_dim : int, optional
        Feature dimension used for all ranks in the synthetic complex.

    Returns
    -------
    Data
        A PyTorch Geometric Data object representing a mock complex batch.
    """
    # 3 nodes, 3 edges, 1 face
    x_0 = torch.randn(3, hidden_dim)  # 3 nodes
    x_1 = torch.randn(3, hidden_dim)  # 3 edges
    x_2 = torch.randn(1, hidden_dim)  # 1 face
    
    batch = Data(x_0=x_0, x_1=x_1, x_2=x_2)

    # Incidence matrices
    incidence_1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 1, 2, 0, 2],  # node indices
                              [0, 0, 1, 1, 2, 2]]),  # edge indices
        values=torch.ones(6),
        size=(3, 3)  # (num_nodes, num_edges)
    ).coalesce()
    batch["down_incidence-1"] = incidence_1

    incidence_2 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2],  # edge indices
                              [0, 0, 0]]),  # face index
        values=torch.ones(3),
        size=(3, 1)  # (num_edges, num_faces)
    ).coalesce()
    batch["down_incidence-2"] = incidence_2

    # Adjacency matrices (remain unchanged)
    adjacency_0 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1, 1, 2, 2],
                              [1, 2, 0, 2, 0, 1]]),
        values=torch.ones(6),
        size=(3, 3)  # (num_nodes, num_nodes)
    ).coalesce()
    batch["up_adjacency-0"] = adjacency_0

    adjacency_1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1, 1, 2, 2],
                              [1, 2, 0, 2, 0, 1]]),
        values=torch.ones(6),
        size=(3, 3)  # (num_edges, num_edges)
    ).coalesce()
    batch["up_adjacency-1"] = adjacency_1

    adjacency_2 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [0]]),
        values=torch.ones(1),
        size=(1, 1)  # (num_faces, num_faces)
    ).coalesce()
    batch["up_adjacency-2"] = adjacency_2

    cell_statistics = torch.tensor([[3, 3, 1]]) 
    batch["cell_statistics"] = cell_statistics
    return batch

class ModifiedNNModuleAutoTest(NNModuleAutoTest):
    """Modified NNModuleAutoTest class for TopoTune testing."""

    def assert_return_tensor(self, result):
        """Assert that the result contains a dictionary with tensor values.

        Parameters
        ----------
        result : Any
            The result to check.
        """
        assert any(isinstance(r, dict) and any(isinstance(v, torch.Tensor) for v in r.values()) for r in result)

    def assert_equal_output(self, module, result, result_2):
        """Assert that two outputs are equal.

        Parameters
        ----------
        module : torch.nn.Module
            The module being tested.
        result : Any
            The first result to compare.
        result_2 : Any
            The second result to compare.
        """
        assert len(result) == len(result_2)

        for i, r1 in enumerate(result):
            r2 = result_2[i]
            if isinstance(r1, dict) and isinstance(r2, dict):
                assert r1.keys() == r2.keys(), f"Dictionaries have different keys at index {i}"
                for key in r1.keys():
                    assert torch.allclose(r1[key], r2[key], atol=1e-6), f"Tensors not equal for key {key} at index {i}"
            elif isinstance(r1, torch.Tensor):
                assert torch.allclose(r1, r2, atol=1e-6), f"Tensors not equal at index {i}"
            else:
                assert r1 == r2, f"Values not equal at index {i}"

def test_topotune():
    """Test the TopoTune module using ModifiedNNModuleAutoTest."""
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    neighborhoods = OmegaConf.create(["up_adjacency-0", "up_adjacency-1", "down_incidence-1", "down_incidence-2"])#[[[0, 0], "adjacency"], [[1, 1], "adjacency"], [[1, 0], "boundary"], [[2, 1], "boundary"]])
    
    auto_test = ModifiedNNModuleAutoTest([
        {
            "module": TopoTune,
            "init": {
                "GNN": gnn,
                "neighborhoods": neighborhoods,
                "layers": 2,
                "use_edge_attr": False,
                "activation": "relu"
            },
            "forward": (batch,),
        }
    ])
    auto_test.run()

def test_topotune_methods():
    """Test individual methods of the TopoTune module."""
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    neighborhoods = OmegaConf.create(["up_adjacency-0", "down_incidence-1"])#[[[0, 0], "adjacency"], [[1, 0], "boundary"]])
    topotune = TopoTune(
        GNN=gnn, 
        neighborhoods=neighborhoods, 
        layers=2, 
        use_edge_attr=False, 
        activation="relu", 
    )

    # Test generate_membership_vectors
    membership = topotune.generate_membership_vectors(batch)
    assert 0 in membership and 1 in membership and 2 in membership
    assert membership[0].shape == (batch.x_0.shape[0],)
    assert membership[1].shape == (batch.x_1.shape[0],)
    assert membership[2].shape == (batch.x_2.shape[0],)

    # Test get_route_cache
    route_cache = topotune.get_route_cache(batch)
    assert 1 in route_cache
    assert route_cache[1]["kind"] == "unordered_interrank"
    assert isinstance(route_cache[1]["edge_index"], torch.Tensor)
    assert isinstance(route_cache[1]["edge_attr"], torch.Tensor)

    # Test intrarank_expand
    expanded = topotune.intrarank_expand(batch, 0, "up_adjacency-0")
    assert isinstance(expanded, Data)
    assert expanded.x.shape == (3, 16)
    assert expanded.edge_index.shape[0] == 2

    # Test intrarank_gnn_forward
    output = topotune.intrarank_gnn_forward(expanded, 0, 0)
    assert output.shape == (3, 16) 

    # Test interrank_expand
    membership = topotune.generate_membership_vectors(batch)
    expanded = topotune.interrank_expand(batch, 1, 0, route_cache[1], membership)
    assert isinstance(expanded, Data)
    assert expanded.x.shape[1] == 16
    assert expanded.edge_index.shape[0] == 2

    # Test interrank_gnn_forward
    output = topotune.interrank_gnn_forward(expanded, 0, 0, 3)
    assert output.shape == (3, 16)  

    # Test aggregate_inter_nbhd
    x_out_per_route = {0: torch.randn(3, 16), 1: torch.randn(3, 16)}
    aggregated = topotune.aggregate_inter_nbhd(x_out_per_route, layer_idx=0)
    print(aggregated)
    assert 0 in aggregated
    assert aggregated[0].shape == (3, 16)

def test_interrank_boundary_index():
    """Test the interrank_boundary_index function."""
    x_src = torch.randn(15, 16)
    boundary_index = [torch.randint(0, 10, (30,)), torch.randint(0, 15, (30,))]
    n_dst_nodes = 10
    
    edge_index, edge_attr = interrank_boundary_index(x_src, boundary_index, n_dst_nodes)
    
    assert edge_index.shape == (2, 30)
    assert edge_attr.shape == (30, 16)

def test_get_activation():
    """Test the get_activation function."""
    relu_func = get_activation("relu")
    assert callable(relu_func)
    
    relu_module = get_activation("relu", return_module=True)
    assert issubclass(relu_module, torch.nn.Module)
    
    with pytest.raises(NotImplementedError):
        get_activation("invalid_activation")


@pytest.mark.parametrize("activation", ["relu", "elu", "tanh", "id"])
def test_topotune_different_activations(activation):
    """
    Test TopoTune with multiple activations to improve coverage of get_activation.

     Parameters
    ----------
    activation : str
        Activation function.
    """
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    
    neighborhoods = OmegaConf.create(["up_adjacency-0", "down_incidence-1"])
    model = TopoTune(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,          # single layer to keep test simpler
        use_edge_attr=False,
        activation=activation,
    )

    output = model(batch)
    # We expect a dict of updated features for each rank in the batch
    assert isinstance(output, dict)
    for rank, feat in output.items():
        assert isinstance(feat, torch.Tensor)
        # The shape should match the original x_rank shape
        original_feat = getattr(batch, f"x_{rank}")
        assert feat.shape == original_feat.shape


def test_topotune_use_edge_attr_true():
    """
    Test TopoTune with use_edge_attr=True to ensure that edge attributes flow through properly.
    """
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    
    # Add more complex neighborhoods to ensure both interrank and intrarank expansions
    neighborhoods = OmegaConf.create([
        "up_adjacency-0",   # intrarank route rank=0->0
        "up_adjacency-1",   # intrarank route rank=1->1
        "down_incidence-1", # interrank route rank=1->0
        "down_incidence-2", # interrank route rank=2->1
    ])
    model = TopoTune(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=2,
        use_edge_attr=True,
        activation="relu",
    )

    output = model(batch)
    assert isinstance(output, dict)
    # Check that each rank in [0,1,2] got updated
    for rank in range(3):
        assert rank in output
        assert isinstance(output[rank], torch.Tensor)
        # The shape should match the original x_rank shape
        original_feat = getattr(batch, f"x_{rank}")
        assert output[rank].shape == original_feat.shape


def test_topotune_single_node_per_rank():
    """
    Test corner case: each rank has only 1 cell, ensuring the path that returns early in intrarank_gnn_forward (x.shape[0] < 2).
    """
    # Create a batch with just 1 node, 1 edge, 1 face
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    
    neighborhoods = OmegaConf.create(["up_adjacency-0", "down_incidence-1"])
    model = TopoTune(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=False,
        activation="relu",
    )
    output = model(batch)
    # Since we have exactly 1 cell in each rank, intrarank_gnn_forward
    # should skip the GNN pass and return the original features
    assert isinstance(output, dict)
    for rank, feat in output.items():
        # Should remain the same as the input
        assert torch.allclose(feat, getattr(batch, f"x_{rank}"), atol=1e-6)


def test_topotune_multiple_layers():
    """
    Test TopoTune with multiple layers > 2 to ensure repeated forward passes.
    """
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    
    neighborhoods = OmegaConf.create(["up_adjacency-0", "down_incidence-1"])
    model = TopoTune(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=3,  # more than 2
        use_edge_attr=False,
        activation="relu",
    )

    output = model(batch)
    assert isinstance(output, dict)
    # By default, the final shape should still be (N, 16) per rank
    for rank, feat in output.items():
        original_feat = getattr(batch, f"x_{rank}")
        assert feat.shape == original_feat.shape


def test_topotune_src_rank_larger_than_dst_rank():
    """
    Test a scenario where src_rank > dst_rank for an interrank route.
    """
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    # Force a route from rank=2 -> rank=0, for instance
    neighborhoods = OmegaConf.create(["down_incidence-1", "down_incidence-2"])
    # topotune will interpret these strings as routes:
    #   (1->0) from down_incidence-1
    #   (2->1) from down_incidence-2
    # Let's force an additional route from 2->0 by customizing the route logic if you want
    # but as is, 2->0 won't happen automatically unless your `get_routes_from_neighborhoods`
    # is coded that way. We'll just rely on existing logic for (2->1).

    model = TopoTune(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=False,
        activation="relu",
    )

    output = model(batch)
    assert isinstance(output, dict)
    # Ranks 0, 1, 2 should exist in the final output dictionary
    for rank in [0, 1, 2]:
        assert rank in output
        assert output[rank].shape == getattr(batch, f"x_{rank}").shape


def test_topotune_sequential_route_order_propagates_across_ranks():
    """Sequential routing should expose earlier route outputs to later ranks."""
    batch_a = create_mock_complex_batch(hidden_dim=1)
    batch_b = create_mock_complex_batch(hidden_dim=1)
    for batch in (batch_a, batch_b):
        batch.x_0 = torch.zeros_like(batch.x_0)
        batch.x_1 = torch.zeros_like(batch.x_1)
        batch.x_2 = torch.ones_like(batch.x_2)

    model_a = TopoTune(
        GNN=OrderedRouteMockGNN(1, 1, 1),
        neighborhoods=OmegaConf.create(
            ["down_incidence-2", "down_incidence-1"]
        ),
        layers=1,
        use_edge_attr=False,
        activation="id",
        route_execution_mode="sequential",
    )
    model_b = TopoTune(
        GNN=OrderedRouteMockGNN(1, 1, 1),
        neighborhoods=OmegaConf.create(
            ["down_incidence-1", "down_incidence-2"]
        ),
        layers=1,
        use_edge_attr=False,
        activation="id",
        route_execution_mode="sequential",
    )

    out_a = model_a(batch_a)
    out_b = model_b(batch_b)

    assert torch.allclose(out_a[0], torch.full((3, 1), 2.0))
    assert torch.allclose(out_b[0], torch.zeros(3, 1))
    assert not torch.allclose(out_a[0], out_b[0])


def test_topotune_sequential_route_order_refines_same_destination_rank():
    """Sequential routing should let later inter-rank routes read current x_dst."""
    batch_a = create_mock_complex_batch(hidden_dim=1)
    batch_b = create_mock_complex_batch(hidden_dim=1)
    for batch in (batch_a, batch_b):
        batch.x_0 = torch.tensor([[1.0], [2.0], [4.0]])
        batch.x_1 = torch.zeros_like(batch.x_1)
        batch.x_2 = torch.tensor([[10.0]])
        batch["up_incidence-0"] = (
            batch["down_incidence-1"].transpose(0, 1).coalesce()
        )

    model_a = TopoTune(
        GNN=OrderedRouteMockGNN(1, 1, 1),
        neighborhoods=OmegaConf.create(
            ["up_incidence-0", "down_incidence-2"]
        ),
        layers=1,
        use_edge_attr=False,
        activation="id",
        route_execution_mode="sequential",
    )
    model_b = TopoTune(
        GNN=OrderedRouteMockGNN(1, 1, 1),
        neighborhoods=OmegaConf.create(
            ["down_incidence-2", "up_incidence-0"]
        ),
        layers=1,
        use_edge_attr=False,
        activation="id",
        route_execution_mode="sequential",
    )

    out_a = model_a(batch_a)
    out_b = model_b(batch_b)

    assert torch.allclose(out_a[1], torch.tensor([[16.0], [22.0], [20.0]]))
    assert torch.allclose(out_b[1], torch.tensor([[23.0], [26.0], [25.0]]))
    assert not torch.allclose(out_a[1], out_b[1])


def test_topotune_sequential_mode_rejects_inter_level_aggregation():
    """Sequential execution and inter-level aggregation should not be mixed."""
    with pytest.raises(
        ValueError,
        match=(
            "inter_level_aggregation is only supported when "
            "route_execution_mode='parallel'"
        ),
    ):
        TopoTune(
            GNN=MockGNN(16, 32, 16),
            neighborhoods=OmegaConf.create(
                ["up_adjacency-0", "down_incidence-1"]
            ),
            layers=1,
            use_edge_attr=False,
            activation="relu",
            inter_level_aggregation=PyGAggregationAdapter(
                SumAggregation()
            ),
            route_execution_mode="sequential",
        )


def test_topotune_ordered_neighborhoods_must_be_subset():
    """Ordered neighborhoods must be part of the configured neighborhoods."""
    with pytest.raises(
        ValueError,
        match="ordered_neighborhoods must be a subset of neighborhoods",
    ):
        TopoTune(
            GNN=MockGNN(16, 32, 16),
            neighborhoods=OmegaConf.create(["down_incidence-1"]),
            ordered_neighborhoods=["up_incidence-0"],
            ordered_neighborhood_model=OrderedNeighborhoodMockModel(16, 16),
            layers=1,
            use_edge_attr=False,
            activation="relu",
        )


def test_topotune_ordered_neighborhood_model_is_required():
    """Ordered neighborhoods require an ordered route model."""
    with pytest.raises(
        ValueError,
        match="ordered_neighborhood_model must be provided",
    ):
        TopoTune(
            GNN=MockGNN(16, 32, 16),
            neighborhoods=OmegaConf.create(["down_incidence-1"]),
            ordered_neighborhoods=["down_incidence-1"],
            layers=1,
            use_edge_attr=False,
            activation="relu",
        )


def test_topotune_ordered_neighborhoods_require_interrank_incidence():
    """Only inter-rank incidence routes can be marked as ordered."""
    with pytest.raises(
        ValueError,
        match="Only inter-rank incidence neighborhoods can be marked ordered",
    ):
        TopoTune(
            GNN=MockGNN(16, 32, 16),
            neighborhoods=OmegaConf.create(["up_adjacency-0"]),
            ordered_neighborhoods=["up_adjacency-0"],
            ordered_neighborhood_model=OrderedNeighborhoodMockModel(16, 16),
            layers=1,
            use_edge_attr=False,
            activation="relu",
        )


def test_topotune_ordered_neighborhood_model_must_match_out_channels():
    """Ordered route models must preserve the backbone output width."""
    with pytest.raises(
        ValueError,
        match="ordered_neighborhood_model.out_channels must match",
    ):
        TopoTune(
            GNN=MockGNN(16, 32, 16),
            neighborhoods=OmegaConf.create(["down_incidence-1"]),
            ordered_neighborhoods=["down_incidence-1"],
            ordered_neighborhood_model=OrderedNeighborhoodMockModel(16, 8),
            layers=1,
            use_edge_attr=False,
            activation="relu",
        )


def test_topotune_parallel_ordered_route_uses_source_id_order():
    """Parallel ordered routes should follow the source-cell ordering."""
    batch = create_mock_complex_batch(hidden_dim=1)
    batch.x_0 = torch.tensor([[1.0], [2.0], [4.0]])
    batch.x_1 = torch.zeros_like(batch.x_1)
    batch["up_incidence-0"] = (
        batch["down_incidence-1"].transpose(0, 1).coalesce()
    )

    model = TopoTune(
        GNN=OrderedRouteMockGNN(1, 1, 1),
        neighborhoods=OmegaConf.create(["up_incidence-0"]),
        ordered_neighborhoods=["up_incidence-0"],
        ordered_neighborhood_model=OrderedNeighborhoodMockModel(1, 1),
        layers=1,
        use_edge_attr=False,
        activation="id",
        route_execution_mode="parallel",
    )

    out = model(batch)
    assert torch.allclose(out[1], torch.tensor([[5.0], [10.0], [9.0]]))


def test_topotune_parallel_ordered_route_accepts_pyg_gru_aggregation():
    """Parallel ordered routes should work with raw PyG GRUAggregation."""
    torch.manual_seed(0)
    batch = create_mock_complex_batch(hidden_dim=4)
    batch["up_incidence-0"] = (
        batch["down_incidence-1"].transpose(0, 1).coalesce()
    )

    model = TopoTune(
        GNN=OrderedRouteMockGNN(4, 4, 4),
        neighborhoods=OmegaConf.create(["up_incidence-0"]),
        ordered_neighborhoods=["up_incidence-0"],
        ordered_neighborhood_model=GRUAggregation(4, 4),
        layers=1,
        use_edge_attr=False,
        activation="id",
        route_execution_mode="parallel",
    )

    out = model(batch)
    assert out[1].shape == batch.x_1.shape


def test_topotune_parallel_ordered_routes_keep_zero_for_missing_destinations():
    """Keep zero route outputs for destinations without ordered members."""
    batch = create_mock_complex_batch(hidden_dim=1)
    batch.x_0 = torch.tensor([[1.0], [2.0], [4.0]])
    batch.x_1 = torch.zeros(4, 1)
    batch["up_incidence-0"] = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2]]),
        values=torch.ones(6),
        size=(4, 3),
    ).coalesce()
    batch["cell_statistics"] = torch.tensor([[3, 4, 1]])

    model = TopoTune(
        GNN=OrderedRouteMockGNN(1, 1, 1),
        neighborhoods=OmegaConf.create(["up_incidence-0"]),
        ordered_neighborhoods=["up_incidence-0"],
        ordered_neighborhood_model=OrderedNeighborhoodMockModel(1, 1),
        layers=1,
        use_edge_attr=False,
        activation="id",
        route_execution_mode="parallel",
    )

    out = model(batch)
    assert torch.allclose(out[1], torch.tensor([[5.0], [10.0], [9.0], [0.0]]))


def test_topotune_sequential_ordered_route_feeds_later_unordered_route():
    """Sequential mode should let ordered route outputs feed later routes."""
    batch = create_mock_complex_batch(hidden_dim=1)
    batch.x_0 = torch.tensor([[1.0], [2.0], [4.0]])
    batch.x_1 = torch.zeros_like(batch.x_1)
    batch["up_incidence-0"] = (
        batch["down_incidence-1"].transpose(0, 1).coalesce()
    )

    model = TopoTune(
        GNN=OrderedRouteMockGNN(1, 1, 1),
        neighborhoods=OmegaConf.create(
            ["up_incidence-0", "down_incidence-1"]
        ),
        ordered_neighborhoods=["up_incidence-0"],
        ordered_neighborhood_model=OrderedNeighborhoodMockModel(1, 1),
        layers=1,
        use_edge_attr=False,
        activation="id",
        route_execution_mode="sequential",
    )

    out = model(batch)
    assert torch.allclose(out[1], torch.tensor([[5.0], [10.0], [9.0]]))
    assert torch.allclose(out[0], torch.tensor([[16.0], [19.0], [27.0]]))


def test_topotune_parallel_mixes_ordered_and_unordered_routes():
    """Parallel mode should aggregate ordered and unordered routes together."""
    batch = create_mock_complex_batch(hidden_dim=1)
    batch.x_0 = torch.tensor([[1.0], [2.0], [4.0]])
    batch.x_1 = torch.zeros_like(batch.x_1)
    batch.x_2 = torch.tensor([[10.0]])
    batch["up_incidence-0"] = (
        batch["down_incidence-1"].transpose(0, 1).coalesce()
    )

    model = TopoTune(
        GNN=OrderedRouteMockGNN(1, 1, 1),
        neighborhoods=OmegaConf.create(
            ["up_incidence-0", "down_incidence-2"]
        ),
        ordered_neighborhoods=["up_incidence-0"],
        ordered_neighborhood_model=OrderedNeighborhoodMockModel(1, 1),
        layers=1,
        use_edge_attr=False,
        activation="id",
        route_execution_mode="parallel",
    )

    out = model(batch)
    assert torch.allclose(out[1], torch.tensor([[15.0], [20.0], [19.0]]))


def test_topotune_hydra_instantiation_with_sequential_routes():
    """Hydra should expose the sequential route execution mode."""
    overrides = [
        "dataset=graph/MUTAG",
        "model=combinatorial/topotune",
        "transforms=no_transform",
        "++model.backbone.route_execution_mode=sequential",
    ]
    config_dir = str(ROOT / "configs")
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=config_dir,
        job_name="test_topotune_sequential_route_hydra",
    ):
        cfg = hydra.compose(config_name="run.yaml", overrides=overrides)
        backbone = hydra.utils.instantiate(cfg.model.backbone)

    assert backbone.route_execution_mode == "sequential"


def test_topotune_hydra_instantiation_with_ordered_neighborhoods():
    """Hydra should expose the ordered neighborhood config keys."""
    overrides = [
        "dataset=graph/MUTAG",
        "model=combinatorial/topotune",
        "transforms=no_transform",
        "++model.backbone.ordered_neighborhoods=[up_incidence-0]",
        (
            "++model.backbone.ordered_neighborhood_model._target_="
            "torch_geometric.nn.aggr.GRUAggregation"
        ),
        "++model.backbone.ordered_neighborhood_model.in_channels=32",
        "++model.backbone.ordered_neighborhood_model.out_channels=32",
    ]
    config_dir = str(ROOT / "configs")
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=config_dir,
        job_name="test_topotune_ordered_neighborhood_hydra",
    ):
        cfg = hydra.compose(config_name="run.yaml", overrides=overrides)
        backbone = hydra.utils.instantiate(cfg.model.backbone)

    assert backbone.ordered_neighborhoods == ["up_incidence-0"]
