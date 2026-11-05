# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static prompts and config writers for frontend coding-agent smoke tests."""

import json
from pathlib import Path

LIST_DIRECTORY_PROMPT = (
    "What files exist in the current working directory? Use your shell tool to run "
    "ls and report each filename verbatim from the output."
)

CODEX_LIST_DIRECTORY_PROMPT = (
    'Call exec_command with the complete argument object {"cmd":"ls"}. '
    "Use the default sandbox permissions; do not request escalated permissions. "
    "Do not add justification, sandbox_permissions, or any other field. If the "
    "tool refuses a call because you requested escalated permissions, call it "
    'again with only {"cmd":"ls"}, without escalated permissions. Then report '
    "each filename verbatim from the output."
)


def write_codex_config(
    codex_home: Path, frontend_port: int, *, enable_multi_agent: bool = False
) -> None:
    """Emit a minimal ~/.codex/config.toml pointing Codex at Dynamo."""
    codex_home.mkdir(parents=True, exist_ok=True)
    multi_agent_config = (
        """[features.multi_agent_v2]
enabled = true
max_concurrent_threads_per_session = 2
non_code_mode_only = false
"""
        if enable_multi_agent
        else ""
    )
    (codex_home / "config.toml").write_text(
        f"""
model_max_output_tokens = 4096
# Dynamo's Responses adapter supports function tools, not hosted web search.
# Disable Codex's default web-search tool for both parent and child smokes.
web_search = "disabled"

{multi_agent_config}

[model_providers.local]
name = "local-dynamo"
base_url = "http://localhost:{frontend_port}/v1"
wire_api = "responses"
env_key = "LOCAL_API_KEY"
        """.lstrip()
    )


def codex_subagent_prompt() -> str:
    """Prompt Codex to invoke one full-history child agent."""
    return (
        "Make these two tool calls in order, with no text response between them: "
        'first spawn_agent with exactly task_name="dynamo_subagent_smoke" and '
        'message="Return exactly OK."; then wait_agent with timeout_ms=60000. '
        "Do not set agent_type or fork_turns. Do not perform the task yourself. "
        "Do not return a text response until wait_agent returns."
    )


def claude_subagent_definition(subagent_name: str) -> str:
    """Return a session-local Claude Code subagent definition."""
    return json.dumps(
        {
            subagent_name: {
                "description": "Returns a fixed smoke response.",
                "prompt": "Return exactly OK immediately.",
                "tools": [],
            }
        }
    )


def claude_subagent_prompt(subagent_name: str) -> str:
    """Prompt Claude Code to invoke the named smoke subagent."""
    return (
        f"Use the Agent tool exactly once with subagent_type={subagent_name}, "
        'description="run smoke test", and prompt="Return exactly OK." '
        "Do not treat the subagent as a skill. Use no other tool."
    )
