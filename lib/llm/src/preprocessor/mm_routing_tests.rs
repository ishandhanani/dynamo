// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preprocessor-level checks of the multimodal routing specs in `dynamo-local-model`.

use std::sync::Arc;

use super::mm_routing::image::testing::failing_counter as test_counter;
use super::{
    ModelDeploymentCard, NvCreateChatCompletionRequest, OpenAIPreprocessor, PreprocessedRequest,
    RoutingImagePromptLayout,
};
use crate::model_card::ModelDeploymentCardLoader;

#[tokio::test]
async fn image_count_failure_discards_partial_routing_and_metrics() {
    use base64::Engine;

    let image_urls: Vec<_> = [1, 2]
        .into_iter()
        .map(|width| {
            let mut png = std::io::Cursor::new(Vec::new());
            ::image::DynamicImage::new_rgb8(width, 1)
                .write_to(&mut png, ::image::ImageFormat::Png)
                .unwrap();
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(png.into_inner())
            )
        })
        .collect();
    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test-image-counter",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_urls[0]}},
            {"type": "image_url", "image_url": {"url": image_urls[1]}}
        ]}]
    }))
    .unwrap();
    let ids = vec![7, 10, 8, 10, 9];
    // Include a successful control and failures before and after a valid
    // count, so neither partial totals nor cache keys can survive an error.
    for failing_width in [None, Some(1), Some(2)] {
        let mut mdc = ModelDeploymentCard::load_from_disk(
            "tests/data/sample-models/mock-llama-3.1-8b-instruct",
            None,
        )
        .unwrap();
        mdc.kv_cache_block_size = 16;
        let mut preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let preprocessor = Arc::get_mut(&mut preprocessor).unwrap();
        preprocessor.image_token_counter = Some(test_counter(failing_width));
        preprocessor.routing_image_token_id = Some(10);
        preprocessor.routing_image_prompt_layout = Some(RoutingImagePromptLayout::RepeatedPad);
        let mut builder = PreprocessedRequest::builder();
        builder
            .model("test-image-counter".to_string())
            .token_ids(ids.clone())
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default());
        let (_, image_tokens) = preprocessor
            .gather_multi_modal_data_with_image_tokens(&request, &mut builder, None, &ids)
            .await
            .unwrap();
        let result = builder.build().unwrap();
        let success = failing_width.is_none();
        assert_eq!(image_tokens, success.then_some(16));
        assert_eq!(result.mm_routing_info.is_some(), success);
        let extra = result.extra_args.as_ref().unwrap();
        assert_eq!(extra.get("mm_hashes").is_some(), success);
        assert!(extra.get("mm_hashes_by_modality").is_none());
        assert_eq!(result.token_ids.as_ref(), &ids);
        assert_eq!(
            result.multi_modal_data.as_ref().unwrap()["image_url"].len(),
            2
        );
    }
}
