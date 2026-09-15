# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-affinity settings shared by KV and builtin routing modes."""

from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument

_MAX_SESSION_AFFINITY_TTL_SECS = 31_536_000


class SessionAffinityConfigBase(ConfigBase):
    session_affinity_ttl_secs: Optional[int] = None
    session_affinity_mode: str = "hard"

    def session_affinity_kwargs(self) -> dict:
        return {
            "session_affinity_ttl_secs": self.session_affinity_ttl_secs,
            "session_affinity_mode": self.session_affinity_mode,
        }

    def validate_session_affinity(self) -> None:
        if self.session_affinity_ttl_secs is not None and not (
            1 <= self.session_affinity_ttl_secs <= _MAX_SESSION_AFFINITY_TTL_SECS
        ):
            raise ValueError(
                "--router-session-affinity-ttl-secs must be between 1 and "
                f"{_MAX_SESSION_AFFINITY_TTL_SECS}"
            )
        if self.session_affinity_mode not in ("hard", "soft"):
            raise ValueError("--router-session-affinity-mode must be hard or soft")


class SessionAffinityArgGroup(ArgGroup):
    def add_arguments(self, parser) -> None:
        g = parser.add_argument_group("Session Affinity Options")

        add_argument(
            g,
            flag_name="--router-session-affinity-ttl-secs",
            env_var="DYN_ROUTER_SESSION_AFFINITY_TTL_SECS",
            default=None,
            help=(
                "Enable session affinity with this router-local idle TTL in seconds. "
                "Bindings synchronize across router replicas on a best-effort basis. "
                "Affinity is disabled when this option is omitted. "
                "This is independent of KV prediction TTL settings."
            ),
            arg_type=int,
            dest="session_affinity_ttl_secs",
        )
        add_argument(
            g,
            flag_name="--router-session-affinity-mode",
            env_var="DYN_ROUTER_SESSION_AFFINITY_MODE",
            default="hard",
            help=(
                "How an existing session binding participates in worker selection. "
                "hard makes the binding an exact constraint; soft exposes it as a "
                "policy-visible preference."
            ),
            choices=("hard", "soft"),
            dest="session_affinity_mode",
        )
