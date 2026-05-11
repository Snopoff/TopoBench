"""Integration tests for TopoTune with PyG inter-level aggregation."""

import copy

import hydra
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.nn.aggr import GRUAggregation, MaxAggregation, SumAggregation

from test.nn.backbones.combinatorial.test_gccn import (
    MockGNN,
    create_mock_complex_batch,
)
from topobench.nn.backbones.combinatorial.gccn import TopoTune
from topobench.nn.inter_level_aggregation import PyGAggregationAdapter

import rootutils

ROOT = rootutils.find_root(__file__, indicator=".project-root")


def make_inline_inter_level_aggregation_overrides(
    aggregation_target: str,
) -> list[str]:
    """Build Hydra overrides for inline inter-level aggregation config.

    Parameters
    ----------
    aggregation_target : str
        Fully qualified PyG aggregation target to inject into the inline
        config.

    Returns
    -------
    list[str]
        Hydra override strings that create an inline
        ``model.inter_level_aggregation`` adapter config.
    """
    overrides = [
        "++model.inter_level_aggregation._target_=topobench.nn.inter_level_aggregation.PyGAggregationAdapter",
        f"++model.inter_level_aggregation.aggregation._target_={aggregation_target}",
    ]
    if aggregation_target.endswith("GRUAggregation"):
        overrides.extend(
            [
                "++model.inter_level_aggregation.aggregation.in_channels=32",
                "++model.inter_level_aggregation.aggregation.out_channels=32",
            ]
        )
    return overrides


def test_topotune_pyg_sum_aggregation_matches_manual_sum():
    """PyG sum aggregation should match the manual rank-wise sum."""
    gnn = MockGNN(16, 32, 16)
    neighborhoods = OmegaConf.create(
        ["up_adjacency-0", "down_incidence-1", "down_incidence-2"]
    )
    topotune = TopoTune(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=False,
        activation="relu",
        inter_level_aggregation=PyGAggregationAdapter(SumAggregation()),
    )
    x_out_per_route = {
        0: torch.ones(3, 16),
        1: 2 * torch.ones(3, 16),
        2: 3 * torch.ones(3, 16),
    }

    aggregated = topotune.aggregate_inter_nbhd(x_out_per_route, layer_idx=0)

    assert torch.allclose(aggregated[0], x_out_per_route[0] + x_out_per_route[1])
    assert torch.allclose(aggregated[1], x_out_per_route[2])


def test_topotune_pyg_max_aggregation_matches_manual_max():
    """PyG max aggregation should match the manual rank-wise max."""
    gnn = MockGNN(16, 32, 16)
    neighborhoods = OmegaConf.create(
        ["up_adjacency-0", "down_incidence-1", "down_incidence-2"]
    )
    topotune = TopoTune(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=False,
        activation="relu",
        inter_level_aggregation=PyGAggregationAdapter(MaxAggregation()),
    )
    x_out_per_route = {
        0: torch.tensor([[1.0, 4.0], [3.0, 2.0], [0.0, 5.0]]).repeat(1, 8),
        1: torch.tensor([[2.0, 3.0], [1.0, 6.0], [7.0, 1.0]]).repeat(1, 8),
        2: torch.randn(3, 16),
    }

    aggregated = topotune.aggregate_inter_nbhd(x_out_per_route, layer_idx=0)

    assert torch.allclose(
        aggregated[0],
        torch.max(
            torch.stack([x_out_per_route[0], x_out_per_route[1]], dim=1),
            dim=1,
        ).values,
    )
    assert torch.allclose(aggregated[1], x_out_per_route[2])


def test_topotune_pyg_gru_aggregation_preserves_shapes_and_bypasses_single_route():
    """PyG GRU aggregation should preserve shapes and bypass single-route ranks."""
    gnn = MockGNN(16, 32, 16)
    neighborhoods = OmegaConf.create(
        ["up_adjacency-0", "down_incidence-1", "down_incidence-2"]
    )
    topotune = TopoTune(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=False,
        activation="relu",
        inter_level_aggregation=PyGAggregationAdapter(
            GRUAggregation(in_channels=16, out_channels=16)
        ),
    )
    x_out_per_route = {
        0: torch.randn(3, 16),
        1: torch.randn(3, 16),
        2: torch.randn(3, 16),
    }

    aggregated = topotune.aggregate_inter_nbhd(x_out_per_route, layer_idx=0)

    assert aggregated[0].shape == (3, 16)
    assert aggregated[1].shape == (3, 16)
    assert torch.allclose(aggregated[1], x_out_per_route[2])


def test_topotune_pyg_gru_aggregation_is_order_sensitive():
    """PyG GRU aggregation should react to route order changes."""
    base_batch = create_mock_complex_batch()
    neighborhoods_a = OmegaConf.create(
        ["up_adjacency-0", "down_incidence-1", "up_adjacency-1"]
    )
    neighborhoods_b = OmegaConf.create(
        ["down_incidence-1", "up_adjacency-0", "up_adjacency-1"]
    )

    torch.manual_seed(0)
    model_a = TopoTune(
        GNN=MockGNN(16, 32, 16),
        neighborhoods=neighborhoods_a,
        layers=1,
        use_edge_attr=False,
        activation="relu",
        inter_level_aggregation=PyGAggregationAdapter(
            GRUAggregation(in_channels=16, out_channels=16)
        ),
    )
    torch.manual_seed(0)
    model_b = TopoTune(
        GNN=MockGNN(16, 32, 16),
        neighborhoods=neighborhoods_b,
        layers=1,
        use_edge_attr=False,
        activation="relu",
        inter_level_aggregation=PyGAggregationAdapter(
            GRUAggregation(in_channels=16, out_channels=16)
        ),
    )

    out_a = model_a(copy.deepcopy(base_batch))
    out_b = model_b(copy.deepcopy(base_batch))

    assert not torch.allclose(out_a[0], out_b[0], atol=1e-6)


@pytest.mark.parametrize(
    ("model_name", "aggregation_target"),
    [
        ("combinatorial/topotune", "torch_geometric.nn.aggr.SumAggregation"),
        ("cell/topotune", "torch_geometric.nn.aggr.MaxAggregation"),
        ("simplicial/topotune", "torch_geometric.nn.aggr.GRUAggregation"),
    ],
)
def test_topotune_hydra_instantiation_with_pyg_aggregation(
    model_name: str, aggregation_target: str
):
    """Hydra should instantiate TopoTune with adapter-backed inter-level aggregation.

    Parameters
    ----------
    model_name : str
        TopoTune model preset to compose.
    aggregation_target : str
        Fully qualified PyG aggregation target to inject into the inline
        model config.
    """
    overrides = [
        "dataset=graph/MUTAG",
        f"model={model_name}",
        "transforms=no_transform",
    ]
    overrides.extend(make_inline_inter_level_aggregation_overrides(aggregation_target))
    config_dir = str(ROOT / "configs")
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=config_dir,
        job_name=(
            f"test_topotune_{model_name.replace('/', '_')}_"
            f"{aggregation_target.split('.')[-1]}"
        ),
    ):
        cfg = hydra.compose(config_name="run.yaml", overrides=overrides)
        backbone = hydra.utils.instantiate(cfg.model.backbone)

    assert isinstance(backbone.inter_level_aggregation_layers[0], PyGAggregationAdapter)

    hidden_dim = cfg.model.feature_encoder.out_channels
    rows_by_rank = {0: 3, 1: 4, 2: 2, 3: 1}
    x_out_per_route = {
        route_index: torch.randn(rows_by_rank[dst_rank], hidden_dim)
        for route_index, (_, dst_rank) in enumerate(backbone.routes)
    }

    aggregated = backbone.aggregate_inter_nbhd(x_out_per_route, layer_idx=0)

    for dst_rank in backbone.route_indices_by_dst_rank:
        assert aggregated[dst_rank].shape == (rows_by_rank[dst_rank], hidden_dim)
