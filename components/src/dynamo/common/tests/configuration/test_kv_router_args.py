# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared KV router CLI configuration."""

import argparse

import pytest

from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class ParsedKvRouterConfig(KvRouterConfigBase):
    pass


def parse_kv_router_config(argv: list[str]) -> ParsedKvRouterConfig:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    return ParsedKvRouterConfig.from_cli_args(parser.parse_args(argv))


def test_remote_g2_router_args_are_cli_only(monkeypatch):
    monkeypatch.setenv("DYN_REMOTE_G2_REUSE_ENABLED", "0")
    monkeypatch.setenv("DYN_REMOTE_G2_SCORE_TAX_BLOCKS", "999")

    config = parse_kv_router_config([])

    assert config.remote_g2_reuse_enabled is False
    assert config.remote_g2_min_planned_blocks == 0
    assert config.remote_g2_score_tax_blocks == 64
    assert config.remote_g2_score_cost_per_block == 0.0
    assert config.remote_g2_score_cap_blocks is None
    assert config.remote_g2_score_max_local_gap_blocks is None


def test_remote_g2_router_args_parse_into_kv_router_kwargs():
    config = parse_kv_router_config(
        [
            "--remote-g2-reuse",
            "--remote-g2-min-planned-blocks",
            "4",
            "--remote-g2-score-tax-blocks",
            "512",
            "--remote-g2-score-cost-per-block",
            "0.25",
            "--remote-g2-score-cap-blocks",
            "32",
            "--remote-g2-score-max-local-gap-blocks",
            "16.5",
        ]
    )

    kwargs = config.kv_router_kwargs()

    assert kwargs["remote_g2_reuse_enabled"] is True
    assert kwargs["remote_g2_min_planned_blocks"] == 4
    assert kwargs["remote_g2_score_tax_blocks"] == 512
    assert kwargs["remote_g2_score_cost_per_block"] == 0.25
    assert kwargs["remote_g2_score_cap_blocks"] == 32
    assert kwargs["remote_g2_score_max_local_gap_blocks"] == 16.5


def test_remote_g2_router_args_help_has_no_product_env_knobs():
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    help_text = parser.format_help()

    assert "DYN_REMOTE_G2" not in help_text
    assert "--remote-g2-reuse" in help_text
    assert "--remote-g2-score-cost-per-block" in help_text
