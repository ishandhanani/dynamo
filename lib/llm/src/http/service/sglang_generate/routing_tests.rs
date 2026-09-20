// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

const BODY: &[u8] = br#"{"input_ids": [[1,2], [3]], "stream":false, "routed_dp_rank":1, "sampling_params":{"n":2,"future_option":1e400}, "future_field":{"number":1e400,"float":1.234567890123456789} }"#;

#[test]
fn native_routing_uses_first_prompt_without_expanding_samples() {
    let input = Projection::read(BODY, &HeaderMap::new())
        .unwrap()
        .routing_input(None, true)
        .unwrap();
    assert_eq!(&*input.0, &[1, 2]);
    let input = Projection::read(
        br#"{"image_data":"opaque","sampling_params":{"n":128}}"#,
        &HeaderMap::new(),
    )
    .unwrap();
    assert!(input.clone().routing_input(None, true).is_err());
    assert!(input.routing_input(None, false).unwrap().0.is_empty());
}

#[test]
fn native_routing_controls_preserve_engine_fields() {
    let projection = Projection::read(BODY, &HeaderMap::new()).unwrap();
    let body = projection
        .with_controls([("routed_dp_rank", Value::from(2))])
        .unwrap();
    let result = Projection::read(&body, &HeaderMap::new()).unwrap();
    assert_eq!(result.dp_rank, Some(2));
    for key in ["input_ids", "sampling_params", "future_field"] {
        assert_eq!(
            projection.raw(key).unwrap().get(),
            result.raw(key).unwrap().get()
        );
    }
}
