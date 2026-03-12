"""Tests for the PyG inter-level aggregation adapter."""

import hydra
import pytest
import rootutils
import torch
from torch_geometric.nn.aggr import GRUAggregation, MaxAggregation, SumAggregation

from topobench.nn.inter_level_aggregation import PyGAggregationAdapter

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


def test_pyg_aggregation_adapter_sum_matches_manual_sum():
    """Sum aggregation should match a manual reduction along the route axis."""
    route_sequences = torch.randn(5, 3, 16)
    model = PyGAggregationAdapter(SumAggregation())

    aggregated = model(route_sequences)

    assert torch.allclose(aggregated, route_sequences.sum(dim=1))


def test_pyg_aggregation_adapter_max_matches_manual_max():
    """Max aggregation should match a manual reduction along the route axis."""
    route_sequences = torch.randn(5, 3, 16)
    model = PyGAggregationAdapter(MaxAggregation())

    aggregated = model(route_sequences)

    assert torch.allclose(aggregated, route_sequences.max(dim=1).values)


def test_pyg_aggregation_adapter_single_route_bypasses_aggregation():
    """A single contributing route should be returned unchanged."""
    route_sequences = torch.randn(5, 1, 16)
    model = PyGAggregationAdapter(SumAggregation())

    aggregated = model(route_sequences)

    assert torch.equal(aggregated, route_sequences[:, 0, :])


def test_pyg_aggregation_adapter_raises_on_output_width_mismatch():
    """Width-changing aggregations should fail fast."""
    route_sequences = torch.randn(5, 3, 16)
    model = PyGAggregationAdapter(GRUAggregation(in_channels=16, out_channels=8))

    with pytest.raises(ValueError, match="Expected \\(5, 16\\), received \\(5, 8\\)"):
        model(route_sequences)


@pytest.mark.parametrize(
    "aggregation_target",
    [
        "torch_geometric.nn.aggr.SumAggregation",
        "torch_geometric.nn.aggr.MaxAggregation",
        "torch_geometric.nn.aggr.GRUAggregation",
    ],
)
def test_pyg_aggregation_adapter_hydra_configs_run_forward(aggregation_target: str):
    """Hydra configs should instantiate adapter-backed inter-level aggregation.

    Parameters
    ----------
    aggregation_target : str
        Fully qualified PyG aggregation target to inject into the inline
        model config.
    """
    overrides = [
        "dataset=graph/MUTAG",
        "model=combinatorial/topotune",
        "transforms=no_transform",
    ]
    overrides.extend(make_inline_inter_level_aggregation_overrides(aggregation_target))

    config_dir = str(ROOT / "configs")
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=config_dir,
        job_name=f"test_{aggregation_target}_adapter",
    ):
        cfg = hydra.compose(config_name="run.yaml", overrides=overrides)
        backbone = hydra.utils.instantiate(cfg.model.backbone)

    assert isinstance(backbone.inter_level_aggregation_layers[0], PyGAggregationAdapter)

    hidden_dim = cfg.model.feature_encoder.out_channels
    route_sequences = torch.randn(4, 3, hidden_dim)
    aggregated = backbone.inter_level_aggregation_layers[0](route_sequences)

    assert aggregated.shape == (4, hidden_dim)
