// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::StreamExt;

use super::*;

#[path = "logprobs.rs"]
mod logprobs;

impl NativeGenerateBinding {
    pub(super) async fn forward_disaggregated(
        &self,
        router: &PrefillRouter,
        request: &mut NativeRequest,
    ) -> anyhow::Result<Response> {
        let prefill = router.native_binding()?;
        let prefill_admission = prefill
            .reserve(request, RequestPhase::Prefill, Default::default())
            .await?;
        let target = prefill_admission.target;
        let rank = target
            .dp_rank
            .ok_or_else(|| anyhow::anyhow!("native prefill selection did not choose a DP rank"))?;
        let (bootstrap, constraints) =
            router.native_bootstrap(&prefill.target.endpoint.id(), target.worker_id)?;
        let (host, port) = bootstrap
            .bootstrap_host
            .zip(bootstrap.bootstrap_port)
            .ok_or_else(|| anyhow::anyhow!("native prefill bootstrap address is missing"))?;
        if constraints.is_some() && self.host.kv_router_if_enabled().is_none() {
            anyhow::bail!("native topology-constrained decode requires KV routing");
        }
        let decode_admission = self
            .reserve(
                request,
                RequestPhase::Decode,
                constraints.unwrap_or_default(),
            )
            .await?;
        let projection = &request.input;
        let room = projection
            .field::<Value>("bootstrap_room")?
            .unwrap_or_else(|| Value::from(bootstrap_room()));
        let bootstrap = [
            ("bootstrap_host", Value::from(host)),
            ("bootstrap_port", Value::from(port)),
            ("bootstrap_room", room),
            ("disagg_prefill_dp_rank", Value::from(rank)),
        ];
        let collect = projection
            .raw("return_logprob")
            .is_some_and(|v| v.get() == "true")
            && !projection.field::<bool>("stream")?.unwrap_or(false);
        if collect {
            request
                .wire
                .headers
                .retain(|(name, _)| name != "accept-encoding");
            request
                .wire
                .headers
                .push(("accept-encoding".into(), Bytes::from_static(b"identity")));
        }
        let prefill = prefill.send(request, prefill_admission, bootstrap.to_vec());
        let decode = self.send(request, decode_admission, bootstrap.to_vec());
        let (decode, prompt) = dispatch(prefill, decode, collect).await?;
        match prompt {
            Some(prompt) => logprobs::merge_response(decode, prompt).await,
            None => Ok(decode),
        }
    }
}

fn bootstrap_room() -> u64 {
    // Stock SGLang's torch scalar setter requires an int64-compatible value.
    (uuid::Uuid::new_v4().as_u128() as u64) & (i64::MAX as u64) & !4095
}

// Poll both stages concurrently: prefill may need the decode receiver before
// completing. Each future/response owns its HTTP connection and routing guard;
// an error drops its peer without cancelling the returned rejection body.
async fn dispatch(
    prefill: impl std::future::Future<Output = anyhow::Result<Response>>,
    decode: impl std::future::Future<Output = anyhow::Result<Response>>,
    collect: bool,
) -> anyhow::Result<(Response, Option<Bytes>)> {
    let prefill = async {
        let response = prefill.await?;
        if !response.status().is_success() {
            return Ok::<_, anyhow::Error>(Err(response));
        }
        let mut body = response.into_body().into_data_stream();
        let mut prompt = bytes::BytesMut::new();
        while let Some(chunk) = body.next().await {
            let chunk = chunk?;
            if collect {
                anyhow::ensure!(
                    prompt.len().saturating_add(chunk.len()) <= logprobs::MAX_BODY,
                    "prefill logprobs exceed body limit"
                );
                prompt.extend_from_slice(&chunk);
            }
        }
        Ok(Ok(collect.then(|| prompt.freeze())))
    };
    tokio::pin!(prefill, decode);
    let (response, prompt) = tokio::select! {
        result = &mut prefill => match result? {
            Err(response) => return Ok((response, None)),
            Ok(prompt) => (decode.await?, prompt),
        },
        response = &mut decode => {
            let response = response?;
            if !response.status().is_success() { return Ok((response, None)); }
            match prefill.await? {
                Err(rejection) => return Ok((rejection, None)),
                Ok(prompt) => (response, prompt),
            }
        }
    };
    let prompt = response.status().is_success().then_some(prompt).flatten();
    Ok((response, prompt))
}
