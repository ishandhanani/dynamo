// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coding-agent request metadata recognized at Dynamo's HTTP boundary.

use axum::http::HeaderMap;
use serde::Deserialize;

use crate::protocols::common::extensions::{
    AgentActor, AgentCompaction, AgentHarness, AgentInvocation, AgentOperation, AgentRequestClass,
    AgentTurn,
};

pub(crate) const HEADER_CLAUDE_CODE_SESSION_ID: &str = "x-claude-code-session-id";
pub(crate) const HEADER_CLAUDE_CODE_AGENT_ID: &str = "x-claude-code-agent-id";
pub(crate) const HEADER_CLAUDE_CODE_PARENT_AGENT_ID: &str = "x-claude-code-parent-agent-id";
pub(crate) const HEADER_CLAUDE_CODE_COMPACTION: &str = "x-claude-code-compaction";
pub(crate) const HEADER_CLAUDE_CODE_CONTEXT_COMPACTED: &str = "x-claude-code-context-compacted";
pub(crate) const HEADER_CLAUDE_CODE_REQUEST_CLASS: &str = "x-claude-code-request-class";
pub(crate) const HEADER_CLAUDE_CODE_AGENT_TYPE: &str = "x-claude-code-agent-type";
pub(crate) const HEADER_CLAUDE_CODE_PREV_TOOL_DURATIONS: &str = "x-claude-code-prev-tool-durations";
pub(crate) const HEADER_CODEX_SESSION_ID: &str = "session-id";
pub(crate) const HEADER_CODEX_THREAD_ID: &str = "thread-id";
pub(crate) const HEADER_CODEX_INSTALLATION_ID: &str = "x-codex-installation-id";
pub(crate) const HEADER_CODEX_PARENT_THREAD_ID: &str = "x-codex-parent-thread-id";
pub(crate) const HEADER_CODEX_TURN_METADATA: &str = "x-codex-turn-metadata";
pub(crate) const HEADER_CODEX_WINDOW_ID: &str = "x-codex-window-id";
pub(crate) const HEADER_CODEX_SUBAGENT: &str = "x-openai-subagent";
pub(crate) const HEADER_OPENCODE_SESSION_ID: &str = "x-session-id";
pub(crate) const HEADER_OPENCODE_PARENT_SESSION_ID: &str = "x-parent-session-id";
pub const HEADER_DYNAMO_SESSION_ID: &str = "x-dynamo-session-id";
pub(crate) const HEADER_DYNAMO_PARENT_SESSION_ID: &str = "x-dynamo-parent-session-id";
pub(crate) const HEADER_DYNAMO_SESSION_FINAL: &str = "x-dynamo-session-final";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AgentHeaderMapping {
    root_session_header: &'static str,
    child_session_header: Option<&'static str>,
    parent_session_header: Option<&'static str>,
    infer_parent_from_session_for_child: bool,
}

const AGENT_HEADER_MAPPINGS: &[AgentHeaderMapping] = &[
    AgentHeaderMapping {
        root_session_header: HEADER_CLAUDE_CODE_SESSION_ID,
        child_session_header: Some(HEADER_CLAUDE_CODE_AGENT_ID),
        parent_session_header: Some(HEADER_CLAUDE_CODE_PARENT_AGENT_ID),
        infer_parent_from_session_for_child: true,
    },
    AgentHeaderMapping {
        root_session_header: HEADER_CODEX_THREAD_ID,
        child_session_header: None,
        parent_session_header: Some(HEADER_CODEX_PARENT_THREAD_ID),
        infer_parent_from_session_for_child: false,
    },
    AgentHeaderMapping {
        root_session_header: HEADER_OPENCODE_SESSION_ID,
        child_session_header: None,
        parent_session_header: Some(HEADER_OPENCODE_PARENT_SESSION_ID),
        infer_parent_from_session_for_child: false,
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AgentContextHeaderValues {
    pub(crate) session_id: String,
    pub(crate) parent_session_id: Option<String>,
    pub(crate) session_final: Option<bool>,
    pub(crate) compaction: Option<AgentCompaction>,
    pub(crate) invocation: Option<AgentInvocation>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct CodexTurnMetadata {
    installation_id: Option<String>,
    session_id: Option<String>,
    agent_name: Option<String>,
    turn_id: Option<String>,
    parent_turn_id: Option<String>,
    root_turn_id: Option<String>,
    context_window_id: Option<String>,
    window_id: Option<String>,
    window_number: Option<u64>,
    request_kind: Option<String>,
    forked_from_thread_id: Option<String>,
    forked_from_ordinal_exclusive: Option<u64>,
    subagent_kind: Option<String>,
    turn_trigger: Option<String>,
    turn_started_at_unix_ms: Option<i64>,
    compaction: Option<AgentCompaction>,
}

fn borrowed_header_value<'a>(headers: &'a HeaderMap, header_name: &str) -> Option<&'a str> {
    let value = headers.get(header_name)?.to_str().ok()?.trim();
    (!value.is_empty()).then_some(value)
}

pub(crate) fn agent_context_header_values(headers: &HeaderMap) -> Option<AgentContextHeaderValues> {
    let session_final = header_bool(headers, HEADER_DYNAMO_SESSION_FINAL);
    let codex_metadata = codex_turn_metadata_header_value(headers);
    let compaction = claude_compaction_header_value(headers).or_else(|| {
        borrowed_header_value(headers, HEADER_CODEX_THREAD_ID)
            .and(codex_metadata.as_ref())
            .filter(|metadata| metadata.request_kind.as_deref() == Some("compaction"))
            .map(|metadata| metadata.compaction.clone().unwrap_or_default())
    });
    let invocation =
        agent_invocation_header_value(headers, codex_metadata.as_ref(), compaction.is_some());

    if let Some(session_id) = borrowed_header_value(headers, HEADER_DYNAMO_SESSION_ID) {
        return Some(AgentContextHeaderValues {
            parent_session_id: borrowed_header_value(headers, HEADER_DYNAMO_PARENT_SESSION_ID)
                .filter(|parent_session_id| *parent_session_id != session_id)
                .map(str::to_owned),
            session_id: session_id.to_owned(),
            session_final,
            compaction,
            invocation,
        });
    }

    for mapping in AGENT_HEADER_MAPPINGS {
        let Some(root_session_id) = borrowed_header_value(headers, mapping.root_session_header)
        else {
            continue;
        };
        let session_id = mapping
            .child_session_header
            .and_then(|child_session_header| borrowed_header_value(headers, child_session_header))
            .unwrap_or(root_session_id);
        let parent_session_id = mapping
            .parent_session_header
            .and_then(|parent_header| borrowed_header_value(headers, parent_header))
            .filter(|parent_session_id| *parent_session_id != session_id)
            .filter(|_| {
                !mapping.infer_parent_from_session_for_child || session_id != root_session_id
            })
            .or_else(|| {
                (mapping.infer_parent_from_session_for_child && session_id != root_session_id)
                    .then_some(root_session_id)
            })
            .map(str::to_owned);
        return Some(AgentContextHeaderValues {
            session_id: session_id.to_owned(),
            parent_session_id,
            session_final,
            compaction,
            invocation,
        });
    }
    None
}

fn claude_compaction_header_value(headers: &HeaderMap) -> Option<AgentCompaction> {
    borrowed_header_value(headers, HEADER_CLAUDE_CODE_COMPACTION).map(|trigger| AgentCompaction {
        trigger: Some(trigger.to_owned()),
        ..Default::default()
    })
}

fn codex_turn_metadata_header_value(headers: &HeaderMap) -> Option<CodexTurnMetadata> {
    let raw = borrowed_header_value(headers, HEADER_CODEX_TURN_METADATA)?;
    serde_json::from_str(raw).ok()
}

fn agent_invocation_header_value(
    headers: &HeaderMap,
    codex_metadata: Option<&CodexTurnMetadata>,
    has_compaction: bool,
) -> Option<AgentInvocation> {
    if borrowed_header_value(headers, HEADER_CLAUDE_CODE_SESSION_ID).is_some()
        || borrowed_header_value(headers, HEADER_CLAUDE_CODE_AGENT_ID).is_some()
    {
        return Some(claude_invocation_header_value(headers, has_compaction));
    }

    if borrowed_header_value(headers, HEADER_CODEX_THREAD_ID).is_some() || codex_metadata.is_some()
    {
        return Some(codex_invocation_header_value(headers, codex_metadata));
    }

    borrowed_header_value(headers, HEADER_OPENCODE_SESSION_ID).map(|_| AgentInvocation {
        harness: AgentHarness::OpenCode,
        request_class: None,
        operation: None,
        actor: None,
        context_replaced: None,
        previous_tool_durations_ms: Default::default(),
        turn: None,
    })
}

fn claude_invocation_header_value(headers: &HeaderMap, has_compaction: bool) -> AgentInvocation {
    let request_class_header = borrowed_header_value(headers, HEADER_CLAUDE_CODE_REQUEST_CLASS);
    let request_class = request_class_header.and_then(claude_request_class);
    let agent_type =
        borrowed_header_value(headers, HEADER_CLAUDE_CODE_AGENT_TYPE).map(str::to_owned);
    AgentInvocation {
        harness: AgentHarness::ClaudeCode,
        request_class,
        operation: Some(
            if has_compaction || request_class_header == Some("compaction") {
                AgentOperation::Compaction
            } else {
                AgentOperation::Inference
            },
        ),
        actor: agent_type.map(|agent_type| AgentActor {
            name: None,
            agent_type: Some(agent_type),
        }),
        context_replaced: headers
            .contains_key(HEADER_CLAUDE_CODE_CONTEXT_COMPACTED)
            .then_some(true),
        previous_tool_durations_ms: borrowed_header_value(
            headers,
            HEADER_CLAUDE_CODE_PREV_TOOL_DURATIONS,
        )
        .map(parse_tool_durations_ms)
        .unwrap_or_default(),
        turn: None,
    }
}

fn claude_request_class(value: &str) -> Option<AgentRequestClass> {
    match value {
        "main" => Some(AgentRequestClass::Primary),
        "subagent" => Some(AgentRequestClass::Subagent),
        "workflow" => Some(AgentRequestClass::Workflow),
        "auxiliary" | "compaction" => Some(AgentRequestClass::Auxiliary),
        _ => None,
    }
}

fn parse_tool_durations_ms(value: &str) -> std::collections::BTreeMap<String, u64> {
    value
        .split(';')
        .filter_map(|entry| {
            let (tool, duration) = entry.split_once('=')?;
            let tool = tool.trim();
            let duration = duration.trim().parse().ok()?;
            (!tool.is_empty()).then_some((tool.to_owned(), duration))
        })
        .collect()
}

fn codex_invocation_header_value(
    headers: &HeaderMap,
    metadata: Option<&CodexTurnMetadata>,
) -> AgentInvocation {
    let request_kind = metadata.and_then(|metadata| metadata.request_kind.as_deref());
    let subagent_kind = metadata
        .and_then(|metadata| metadata.subagent_kind.as_deref())
        .or_else(|| borrowed_header_value(headers, HEADER_CODEX_SUBAGENT));
    let operation = codex_operation(request_kind);
    let request_class = match operation {
        Some(AgentOperation::Compaction | AgentOperation::Prewarm | AgentOperation::Memory) => {
            Some(AgentRequestClass::Auxiliary)
        }
        Some(AgentOperation::Inference) | None if subagent_kind.is_some() => {
            Some(AgentRequestClass::Subagent)
        }
        Some(AgentOperation::Inference) | None => Some(AgentRequestClass::Primary),
    };
    let agent_name = metadata.and_then(|metadata| metadata.agent_name.clone());
    let agent_type = subagent_kind.map(str::to_owned);
    let actor = (agent_name.is_some() || agent_type.is_some()).then_some(AgentActor {
        name: agent_name,
        agent_type,
    });
    let turn =
        metadata.map(|metadata| AgentTurn {
            client_instance_id: metadata.installation_id.clone().or_else(|| {
                borrowed_header_value(headers, HEADER_CODEX_INSTALLATION_ID).map(str::to_owned)
            }),
            client_session_id: metadata.session_id.clone().or_else(|| {
                borrowed_header_value(headers, HEADER_CODEX_SESSION_ID).map(str::to_owned)
            }),
            id: metadata.turn_id.clone(),
            parent_id: metadata.parent_turn_id.clone(),
            root_id: metadata.root_turn_id.clone(),
            context_id: metadata.context_window_id.clone(),
            window_id: metadata.window_id.clone().or_else(|| {
                borrowed_header_value(headers, HEADER_CODEX_WINDOW_ID).map(str::to_owned)
            }),
            context_sequence: metadata.window_number,
            trigger: metadata.turn_trigger.clone(),
            started_at_unix_ms: metadata.turn_started_at_unix_ms,
            forked_from_session_id: metadata.forked_from_thread_id.clone(),
            forked_after_history_ordinal: metadata.forked_from_ordinal_exclusive,
        });
    AgentInvocation {
        harness: AgentHarness::Codex,
        request_class,
        operation,
        actor,
        context_replaced: None,
        previous_tool_durations_ms: Default::default(),
        turn,
    }
}

fn codex_operation(request_kind: Option<&str>) -> Option<AgentOperation> {
    match request_kind {
        Some("turn") => Some(AgentOperation::Inference),
        Some("compaction") => Some(AgentOperation::Compaction),
        Some("prewarm") => Some(AgentOperation::Prewarm),
        Some("memory") => Some(AgentOperation::Memory),
        _ => None,
    }
}

pub(crate) fn session_affinity_header_value(headers: &HeaderMap) -> Option<String> {
    if let Some(session_id) = borrowed_header_value(headers, HEADER_DYNAMO_SESSION_ID) {
        return Some(session_id.to_owned());
    }
    for mapping in AGENT_HEADER_MAPPINGS {
        let Some(root_session_id) = borrowed_header_value(headers, mapping.root_session_header)
        else {
            continue;
        };
        let session_id = mapping
            .child_session_header
            .and_then(|child_session_header| borrowed_header_value(headers, child_session_header))
            .unwrap_or(root_session_id);
        return Some(session_id.to_owned());
    }
    None
}

fn header_bool(headers: &HeaderMap, header_name: &str) -> Option<bool> {
    let value = borrowed_header_value(headers, header_name)?;
    dynamo_runtime::config::parse_bool_opt(value)
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderMap;

    use crate::protocols::common::extensions::{
        AgentContext, AgentHarness, AgentOperation, AgentRequestClass,
    };

    use super::{
        HEADER_CLAUDE_CODE_AGENT_TYPE, HEADER_CLAUDE_CODE_COMPACTION,
        HEADER_CLAUDE_CODE_CONTEXT_COMPACTED, HEADER_CLAUDE_CODE_PREV_TOOL_DURATIONS,
        HEADER_CLAUDE_CODE_REQUEST_CLASS, HEADER_CLAUDE_CODE_SESSION_ID,
        HEADER_CODEX_INSTALLATION_ID, HEADER_CODEX_PARENT_THREAD_ID, HEADER_CODEX_SESSION_ID,
        HEADER_CODEX_SUBAGENT, HEADER_CODEX_THREAD_ID, HEADER_CODEX_TURN_METADATA,
        HEADER_CODEX_WINDOW_ID, agent_context_header_values,
    };

    #[test]
    fn claude_gateway_hints_normalize_agent_invocation() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_CLAUDE_CODE_SESSION_ID,
            "claude-session".parse().unwrap(),
        );
        headers.insert(HEADER_CLAUDE_CODE_COMPACTION, "reactive".parse().unwrap());
        headers.insert(HEADER_CLAUDE_CODE_CONTEXT_COMPACTED, "1".parse().unwrap());
        headers.insert(
            HEADER_CLAUDE_CODE_REQUEST_CLASS,
            "workflow".parse().unwrap(),
        );
        headers.insert(HEADER_CLAUDE_CODE_AGENT_TYPE, "Plan".parse().unwrap());
        headers.insert(
            HEADER_CLAUDE_CODE_PREV_TOOL_DURATIONS,
            "Bash=742;Read=9;invalid;Other=not-a-duration"
                .parse()
                .unwrap(),
        );

        let context = agent_context_header_values(&headers).expect("agent context");
        assert_eq!(context.session_id, "claude-session");
        assert_eq!(
            context
                .compaction
                .as_ref()
                .and_then(|compaction| compaction.trigger.as_deref()),
            Some("reactive")
        );

        let invocation = context.invocation.expect("agent invocation");
        assert_eq!(invocation.harness, AgentHarness::ClaudeCode);
        assert_eq!(invocation.request_class, Some(AgentRequestClass::Workflow));
        assert_eq!(invocation.operation, Some(AgentOperation::Compaction));
        assert_eq!(
            invocation
                .actor
                .as_ref()
                .and_then(|actor| actor.agent_type.as_deref()),
            Some("Plan")
        );
        assert_eq!(invocation.context_replaced, Some(true));
        assert_eq!(
            invocation.previous_tool_durations_ms,
            [("Bash".to_owned(), 742), ("Read".to_owned(), 9)].into()
        );
    }

    #[test]
    fn codex_metadata_normalizes_agent_invocation() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_CODEX_SESSION_ID, "codex-session".parse().unwrap());
        headers.insert(HEADER_CODEX_THREAD_ID, "codex-thread".parse().unwrap());
        headers.insert(HEADER_CODEX_INSTALLATION_ID, "install-1".parse().unwrap());
        headers.insert(HEADER_CODEX_WINDOW_ID, "window-1".parse().unwrap());
        headers.insert(
            HEADER_CODEX_PARENT_THREAD_ID,
            "parent-thread".parse().unwrap(),
        );
        headers.insert(HEADER_CODEX_SUBAGENT, "collab_spawn".parse().unwrap());
        headers.insert(
            HEADER_CODEX_TURN_METADATA,
            r#"{"installation_id":"install-1","session_id":"codex-session","thread_id":"codex-thread","agent_name":"/root","turn_id":"turn-1","window_id":"window-1","window_number":4,"context_window_id":"context-window-1","request_kind":"compaction","forked_from_thread_id":"fork-thread","forked_from_ordinal_exclusive":3,"parent_thread_id":"parent-thread","parent_turn_id":"parent-turn","root_turn_id":"root-turn","subagent_kind":"thread_spawn","thread_source":{"kind":"collaboration"},"turn_trigger":"user_message","sandbox":"workspace-write","sandbox_mode":"workspace-write","auto_review_enabled":true,"node_repl_auto_review_required":false,"node_repl_disabled":false,"workspaces":{"/repo":{"has_changes":true}},"tool_namespaces_info":{"functions":{}},"turn_started_at_unix_ms":1700000000123,"history_ingest_requested":true,"analytics_enabled":false,"compaction":{"trigger":"manual","reason":"user_requested","implementation":"responses","phase":"standalone_turn","strategy":"memento"},"future_field":"retained"}"#
                .parse()
                .unwrap(),
        );

        let context = agent_context_header_values(&headers).expect("agent context");
        assert_eq!(context.session_id, "codex-thread");
        assert_eq!(context.parent_session_id.as_deref(), Some("parent-thread"));
        assert_eq!(
            context
                .compaction
                .as_ref()
                .and_then(|compaction| compaction.implementation.as_deref()),
            Some("responses")
        );

        let invocation = context.invocation.as_ref().expect("agent invocation");
        assert_eq!(invocation.harness, AgentHarness::Codex);
        assert_eq!(invocation.request_class, Some(AgentRequestClass::Auxiliary));
        assert_eq!(invocation.operation, Some(AgentOperation::Compaction));
        assert_eq!(
            invocation
                .actor
                .as_ref()
                .and_then(|actor| actor.name.as_deref()),
            Some("/root")
        );
        assert_eq!(
            invocation
                .actor
                .as_ref()
                .and_then(|actor| actor.agent_type.as_deref()),
            Some("thread_spawn")
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.client_instance_id.as_deref()),
            Some("install-1")
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.client_session_id.as_deref()),
            Some("codex-session")
        );
        assert_eq!(
            invocation.turn.as_ref().and_then(|turn| turn.id.as_deref()),
            Some("turn-1")
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.parent_id.as_deref()),
            Some("parent-turn")
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.root_id.as_deref()),
            Some("root-turn")
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.context_id.as_deref()),
            Some("context-window-1")
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.window_id.as_deref()),
            Some("window-1")
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.context_sequence),
            Some(4)
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.forked_from_session_id.as_deref()),
            Some("fork-thread")
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.forked_after_history_ordinal),
            Some(3)
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.trigger.as_deref()),
            Some("user_message")
        );
        assert_eq!(
            invocation
                .turn
                .as_ref()
                .and_then(|turn| turn.started_at_unix_ms),
            Some(1_700_000_000_123)
        );

        let wire = serde_json::to_value(AgentContext::from(context)).expect("serialize wire");
        assert_eq!(wire["invocation"]["turn"]["id"], "turn-1");
        assert!(wire.get("agent_headers").is_none());
    }

    #[test]
    fn codex_subagent_turn_normalizes_to_inference() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_CODEX_THREAD_ID, "child-thread".parse().unwrap());
        headers.insert(
            HEADER_CODEX_TURN_METADATA,
            r#"{"request_kind":"turn","subagent_kind":"review"}"#
                .parse()
                .unwrap(),
        );

        let invocation = agent_context_header_values(&headers)
            .expect("agent context")
            .invocation
            .expect("agent invocation");
        assert_eq!(invocation.harness, AgentHarness::Codex);
        assert_eq!(invocation.request_class, Some(AgentRequestClass::Subagent));
        assert_eq!(invocation.operation, Some(AgentOperation::Inference));
        assert_eq!(
            invocation
                .actor
                .as_ref()
                .and_then(|actor| actor.agent_type.as_deref()),
            Some("review")
        );
    }
}
