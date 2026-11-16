// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool calls of the eight unified-parser families replayed through the HTTP service: the
//! preprocessor's replay engine behind a worker set, read back over SSE and unary responses.

use std::sync::Arc;

use dynamo_model_card::model_card::ModelDeploymentCard;
use futures::StreamExt;

use dynamo_preprocessor::OpenAIPreprocessor;
use dynamo_preprocessor::test_fixtures::UnifiedReplayEngine;

async fn start_unified_replay(
    family: &str,
    native: String,
    arguments: String,
    release: Option<Arc<tokio::sync::Semaphore>>,
) -> (
    u16,
    dynamo_runtime::CancellationToken,
    tokio::task::JoinHandle<anyhow::Result<()>>,
) {
    use crate::http::service::service_v2::HttpService;
    let (tool, reason) = match family {
        "qwen3" => ("qwen3_coder", "qwen3"),
        "kimi_k2" => ("kimi_k2", "kimi_k25"),
        "glm47" => ("glm47", "glm45"),
        other => (other, other),
    };
    let mut card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    card.runtime_config.tool_call_parser = Some(tool.into());
    card.runtime_config.reasoning_parser = Some(reason.into());
    card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
    card.needs.clear();
    let preprocessor = OpenAIPreprocessor::new(card.clone()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let service = HttpService::builder()
        .host("127.0.0.1")
        .port(port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(false)
        .build()
        .unwrap();
    // HTTP aggregation reads parser configuration from the selected worker
    // card, just as discovery does for a deployed model.
    let mut workers =
        dynamo_discovery::WorkerSet::new("replay-workers".into(), "replay-checksum".into(), card);
    workers.chat_engine = Some(Arc::new(UnifiedReplayEngine {
        preprocessor,
        native,
        arguments,
        release,
    }));
    assert!(
        service
            .model_manager()
            .add_worker_set("replay", "replay-workers", workers)
    );
    let cancel = dynamo_runtime::CancellationToken::new();
    let join = service.spawn_with_listener(cancel.clone(), listener).await;
    (port, cancel, join)
}

// The backend cannot emit either continuation until the client releases it.
// This catches transport buffering even when the final aggregate is correct.
async fn paused_http_tool_progress(guided: bool) {
    use dynamo_frontend_protocols::openai::chat_completions::unified_parser;
    assert!(
        unified_parser::parsers_v2_selected(),
        "run with DYN_PARSER_VERSION=2"
    );
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut failures = Vec::new();
    for (shape, fields) in unified_parser::test_fixtures::write_file_progress_cases() {
        let expected_arguments: serde_json::Value =
            serde_json::from_str(&unified_parser::test_fixtures::ordered_arguments(&fields))
                .unwrap();
        for (family, native) in unified_parser::test_fixtures::native_write_file_calls(&fields) {
            for named in if guided {
                vec![false, true]
            } else {
                vec![false]
            } {
                let release = Arc::new(tokio::sync::Semaphore::new(0));
                let (port, cancel, join) = start_unified_replay(
                    family,
                    native.clone(),
                    unified_parser::test_fixtures::ordered_arguments(&fields),
                    Some(release.clone()),
                )
                .await;
                let choice = if !guided {
                    serde_json::json!("auto")
                } else if named {
                    serde_json::json!({"type":"function","function":{"name":"write_file"}})
                } else {
                    serde_json::json!("required")
                };
                let request = serde_json::json!({"model":"replay","messages":[{"role":"user","content":"write it"}],"stream":true,"tool_choice":choice,"tools":[{"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"count":{"type":"integer"},"path":{"type":"string"},"content":{"type":"string"}}}}}]});
                let response = client
                    .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
                    .json(&request)
                    .send()
                    .await
                    .unwrap();
                assert!(response.status().is_success());
                let mut source = response.bytes_stream();
                let mut body = String::new();
                let mut pending = Vec::new();
                let mut arguments = String::new();
                let mut names = String::new();
                let mut ids = Vec::new();
                let mut finishes = Vec::new();
                let mut progress = Vec::new();
                for expected in ["first content", "second content"] {
                    let reached = tokio::time::timeout(std::time::Duration::from_secs(2), async {
                        while let Some(bytes) = source.next().await {
                            pending.extend_from_slice(&bytes.unwrap());
                            while let Some(end) = pending.iter().position(|byte| *byte == b'\n') {
                                let line =
                                    String::from_utf8(pending.drain(..=end).collect()).unwrap();
                                body.push_str(&line);
                                read_replay_delta(
                                    &line,
                                    &mut names,
                                    &mut arguments,
                                    &mut ids,
                                    &mut finishes,
                                );
                            }
                            if names == "write_file" && arguments.contains(expected) {
                                return true;
                            }
                        }
                        false
                    })
                    .await
                    .unwrap_or(false);
                    progress.push(reached);
                    release.add_permits(1);
                }
                tokio::time::timeout(std::time::Duration::from_secs(10), async {
                    while let Some(bytes) = source.next().await {
                        pending.extend_from_slice(&bytes.unwrap());
                        while let Some(end) = pending.iter().position(|byte| *byte == b'\n') {
                            let line = String::from_utf8(pending.drain(..=end).collect()).unwrap();
                            body.push_str(&line);
                            read_replay_delta(
                                &line,
                                &mut names,
                                &mut arguments,
                                &mut ids,
                                &mut finishes,
                            );
                        }
                    }
                })
                .await
                .unwrap();
                cancel.cancel();
                tokio::time::timeout(std::time::Duration::from_secs(5), join)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                assert!(body.contains("[DONE]"));
                assert_eq!(names, "write_file", "{family}/{choice}");
                assert_eq!(ids.len(), 1, "{family}/{choice}");
                assert_eq!(finishes, ["tool_calls"], "{family}/{choice}");
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
                    expected_arguments,
                    "{family}/{choice}"
                );
                eprintln!(
                    "paused HTTP {family}/{shape}/{choice}: first={} continued={} final=true",
                    progress[0], progress[1]
                );
                if progress != [true, true] {
                    failures.push(format!("{family}/{shape}/{choice}: {progress:?}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "paused HTTP progress failed: {failures:?}"
    );
}

fn read_replay_delta(
    line: &str,
    names: &mut String,
    arguments: &mut String,
    ids: &mut Vec<String>,
    finishes: &mut Vec<String>,
) {
    let Some(data) = line.trim_end().strip_prefix("data: ") else {
        return;
    };
    if data == "[DONE]" {
        return;
    }
    let event: serde_json::Value = serde_json::from_str(data).unwrap();
    for choice in event["choices"].as_array().unwrap() {
        if let Some(finish) = choice["finish_reason"].as_str() {
            finishes.push(finish.into());
        }
        if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
            for call in calls {
                assert_eq!(call["index"], 0);
                if let Some(id) = call["id"].as_str() {
                    ids.push(id.into());
                }
                if let Some(name) = call["function"]["name"].as_str() {
                    names.push_str(name);
                }
                if let Some(value) = call["function"]["arguments"].as_str() {
                    arguments.push_str(value);
                }
            }
        }
    }
}

#[tokio::test]
#[ignore = "Run explicitly with DYN_PARSER_VERSION=2; measures upstream native buffering"]
async fn eight_family_paused_native_http_progress() {
    paused_http_tool_progress(false).await;
}

#[tokio::test]
#[ignore = "Run explicitly with DYN_PARSER_VERSION=2; measures installed forced-tool modes"]
async fn eight_family_paused_forced_http_progress() {
    paused_http_tool_progress(true).await;
}

#[tokio::test]
async fn eight_family_http_sse_replay() {
    use dynamo_frontend_protocols::openai::chat_completions::unified_parser;
    // Run with the experiment enabled in a separate process: its flag is cached.
    if unified_parser::selected_family(Some("qwen3_coder"), Some("qwen3")).is_none() {
        return;
    }
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    for (family, native) in unified_parser::test_fixtures::eight_family_calls("é replay") {
        let (port, cancel, join) = start_unified_replay(
            family,
            native,
            serde_json::json!({"count":2,"content":"é replay"}).to_string(),
            None,
        )
        .await;
        for streaming in [false, true] {
            for choice in [
                serde_json::Value::Null,
                serde_json::json!("auto"),
                serde_json::json!("required"),
                serde_json::json!({"type":"function","function":{"name":"write_file"}}),
                serde_json::json!("none"),
            ] {
                let response = client.post(format!("http://127.0.0.1:{port}/v1/chat/completions")).json(&serde_json::json!({"model":"replay","messages":[{"role":"user","content":"write it"}],"stream":streaming,"tool_choice":choice,"tools":[{"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"count":{"type":"integer"},"content":{"type":"string"}}}}}]})).send().await.unwrap();
                let status = response.status();
                let body = response.text().await.unwrap();
                assert!(status.is_success(), "{family} {choice} {status}: {body}");
                let mut arguments = String::new();
                if streaming {
                    assert!(body.contains("[DONE]"), "{family}");
                    for line in body
                        .lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .filter(|line| *line != "[DONE]")
                    {
                        let event: serde_json::Value = serde_json::from_str(line).unwrap();
                        if let Some(calls) = event["choices"][0]["delta"]["tool_calls"].as_array() {
                            for call in calls {
                                arguments
                                    .push_str(call["function"]["arguments"].as_str().unwrap_or(""));
                            }
                        }
                    }
                } else {
                    let result: serde_json::Value = serde_json::from_str(&body).unwrap();
                    if let Some(calls) = result["choices"][0]["message"]["tool_calls"].as_array() {
                        assert_eq!(calls.len(), 1, "{family}");
                        arguments.push_str(calls[0]["function"]["arguments"].as_str().unwrap());
                    }
                }
                if choice == "none" {
                    assert!(arguments.is_empty(), "{family}");
                } else {
                    assert_eq!(
                        serde_json::from_str::<serde_json::Value>(&arguments).unwrap_or_else(
                            |error| panic!(
                                "{family} {choice} stream={streaming}: {error}; body={body}"
                            )
                        ),
                        serde_json::json!({"count":2,"content":"é replay"}),
                        "{family} {choice}"
                    );
                }
            }
        }
        cancel.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(5), join)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}
