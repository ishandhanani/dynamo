# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def clear_direct_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DYN_DIRECT",
        "DYN_DIRECT_BACKEND",
        "DYN_ENGINE_URLS",
        "DYN_PREFILL_ENGINE_URLS",
        "DYN_ENGINE_SELECTOR",
        "DYN_PREFILL_ENGINE_SELECTOR",
        "DYN_ENGINE_NAMESPACE",
        "DYN_ENGINE_GRPC_PORT",
        "DYN_DISCOVERY_BACKEND",
        "DYN_TOOL_CALL_PARSER",
        "DYN_REASONING_PARSER",
    ):
        monkeypatch.delenv(name, raising=False)


def parse_frontend_config(args: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(args))
    config.validate()
    return config


def test_direct_with_prefill_and_dynamo_parsers() -> None:
    config = parse_frontend_config(
        [
            "--direct",
            "--direct-backend",
            "sglang",
            "--prefill-engine-url",
            "http://127.0.0.1:18701",
            "--engine-url",
            "http://127.0.0.1:18711",
            "--engine-url",
            "http://127.0.0.1:18721",
            "--dyn-tool-call-parser",
            "hermes",
            "--dyn-reasoning-parser",
            "qwen3",
        ]
    )

    assert config.direct is True
    assert config.direct_backend == "sglang"
    assert config.prefill_engine_urls == ["http://127.0.0.1:18701"]
    assert config.engine_urls == ["http://127.0.0.1:18711", "http://127.0.0.1:18721"]
    assert config.dyn_tool_call_parser == "hermes"
    assert config.dyn_reasoning_parser == "qwen3"


def test_direct_defaults_to_sglang_without_parsers() -> None:
    config = parse_frontend_config(
        ["--direct", "--engine-url", "http://127.0.0.1:30001"]
    )

    assert config.direct_backend == "sglang"
    assert config.dyn_tool_call_parser is None
    assert config.dyn_reasoning_parser is None


def test_direct_requires_an_engine_url() -> None:
    with pytest.raises(ValueError, match="--engine-url"):
        parse_frontend_config(["--direct"])


def test_engine_urls_require_direct() -> None:
    with pytest.raises(ValueError, match="require --direct"):
        parse_frontend_config(["--engine-url", "http://127.0.0.1:30001"])


def test_dynamo_parsers_require_direct() -> None:
    with pytest.raises(ValueError, match="--dyn-tool-call-parser"):
        parse_frontend_config(["--dyn-reasoning-parser", "qwen3"])


def test_direct_rejects_the_interactive_frontend() -> None:
    with pytest.raises(ValueError, match="--interactive"):
        parse_frontend_config(
            ["--direct", "--engine-url", "http://127.0.0.1:30001", "--interactive"]
        )


def test_direct_engines_from_kubernetes_pods() -> None:
    config = parse_frontend_config(
        [
            "--direct",
            "--discovery-backend",
            "kubernetes-pods",
            "--engine-selector",
            "app=sglang,role=decode",
            "--prefill-engine-selector",
            "app=sglang,role=prefill",
            "--engine-namespace",
            "llm",
            "--engine-grpc-port",
            "30001",
        ]
    )

    assert config.discovery_backend == "kubernetes-pods"
    assert config.engine_selector == "app=sglang,role=decode"
    assert config.prefill_engine_selector == "app=sglang,role=prefill"
    assert config.engine_namespace == "llm"
    assert config.engine_grpc_port == 30001
    assert config.engine_urls == []


def test_kubernetes_pods_requires_an_engine_selector() -> None:
    with pytest.raises(ValueError, match="--engine-selector"):
        parse_frontend_config(["--direct", "--discovery-backend", "kubernetes-pods"])


def test_kubernetes_pods_requires_direct() -> None:
    with pytest.raises(ValueError, match="require --direct"):
        parse_frontend_config(
            [
                "--discovery-backend",
                "kubernetes-pods",
                "--engine-selector",
                "app=sglang",
            ]
        )


def test_engine_selector_requires_kubernetes_pods() -> None:
    with pytest.raises(ValueError, match="kubernetes-pods"):
        parse_frontend_config(["--direct", "--engine-selector", "app=sglang"])


def test_kubernetes_pods_rejects_engine_urls() -> None:
    with pytest.raises(ValueError, match="--engine-url"):
        parse_frontend_config(
            [
                "--direct",
                "--discovery-backend",
                "kubernetes-pods",
                "--engine-selector",
                "app=sglang",
                "--engine-url",
                "http://127.0.0.1:30001",
            ]
        )
