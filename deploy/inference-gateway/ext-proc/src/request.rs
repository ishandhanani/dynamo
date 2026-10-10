// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The remote renderer owns prompt preparation; inspect only the fields the EPP
//! needs before forwarding the original body. Message text, tools and media are
//! borrowed or skipped, never copied into a second request tree.

use std::borrow::Cow;
use std::fmt;

use dynamo_llm::protocols::common::extensions::{InputTrigger, NvExt};
use dynamo_llm::protocols::common::input_trigger::classify_chat_role;
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::value::RawValue;

use crate::picker::PickError;

#[derive(Deserialize)]
pub(crate) struct RemoteRequest<'a> {
    #[serde(borrow)]
    pub model: Cow<'a, str>,
    pub nvext: Option<NvExt>,
    #[serde(borrow, rename = "cache_salt")]
    pub cache_namespace: Option<Cow<'a, str>>,
    pub messages: Option<ChatMetadata>,
    #[serde(borrow)]
    pub prompt: Option<&'a RawValue>,
    pub n: Option<u32>,
    pub prompt_embeds: Option<serde::de::IgnoredAny>,
    // SGLang's explicit adapter identity is separate from the served model.
    pub lora_path: Option<serde::de::IgnoredAny>,
}

impl RemoteRequest<'_> {
    pub fn is_completion(&self, headers: &[(String, String)]) -> Result<bool, PickError> {
        let completion = match (self.messages.as_ref(), self.prompt) {
            (Some(messages), None) if messages.count > 0 => false,
            (None, Some(_)) => true,
            _ => {
                return Err(PickError::InvalidRequest(
                    "expected either non-empty messages or a completion prompt".into(),
                ));
            }
        };
        if let Some((_, path)) = headers.iter().find(|(name, _)| name == ":path") {
            let path = path.split('?').next().unwrap_or(path);
            let expected = if completion {
                "/v1/completions"
            } else {
                "/v1/chat/completions"
            };
            if path != expected {
                return Err(PickError::InvalidRequest(
                    "request body does not match a supported completion endpoint".into(),
                ));
            }
        }
        if self.n.is_some_and(|n| n != 1) || self.lora_path.is_some() {
            return Err(PickError::InvalidRequest(
                "standalone EPP supports one output and no per-request LoRA adapter".into(),
            ));
        }
        if self.prompt_embeds.is_some() {
            return Err(PickError::InvalidRequest(
                "EPP cannot route prompt embeddings without token metadata".into(),
            ));
        }
        if self
            .messages
            .as_ref()
            .is_some_and(|messages| messages.has_media)
        {
            return Err(PickError::InvalidRequest(
                "renderer protocol does not provide multimodal routing hashes".into(),
            ));
        }
        if let Some(prompt) = self.prompt {
            let shape: SinglePrompt = serde_json::from_str(prompt.get())
                .map_err(|_| PickError::InvalidRequest("invalid completion prompt".into()))?;
            if !shape.0 {
                return Err(PickError::InvalidRequest(
                    "standalone EPP does not support batched completion prompts".into(),
                ));
            }
        }
        Ok(completion)
    }

    pub fn input_trigger(&self) -> InputTrigger {
        self.messages
            .as_ref()
            .map_or(InputTrigger::Other, |messages| messages.input_trigger)
    }
}

pub(crate) struct ChatMetadata {
    count: usize,
    has_media: bool,
    input_trigger: InputTrigger,
}

impl<'de> Deserialize<'de> for ChatMetadata {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Messages;
        impl<'de> Visitor<'de> for Messages {
            type Value = ChatMetadata;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("chat messages")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                #[derive(Deserialize)]
                struct Message<'a> {
                    #[serde(borrow)]
                    role: Cow<'a, str>,
                    #[serde(borrow)]
                    content: Option<&'a RawValue>,
                    audio: Option<serde::de::IgnoredAny>,
                }
                let mut metadata = ChatMetadata {
                    count: 0,
                    has_media: false,
                    input_trigger: InputTrigger::Other,
                };
                while let Some(message) = seq.next_element::<Message<'de>>()? {
                    metadata.count += 1;
                    metadata.input_trigger = classify_chat_role(&message.role);
                    metadata.has_media |= message.audio.is_some();
                    if let Some(content) = message
                        .content
                        .filter(|content| content.get().starts_with('['))
                    {
                        metadata.has_media |=
                            serde_json::from_str::<ContentMetadata>(content.get())
                                .map_err(serde::de::Error::custom)?
                                .0;
                    }
                }
                Ok(metadata)
            }
        }
        deserializer.deserialize_seq(Messages)
    }
}

struct ContentMetadata(bool);
impl<'de> Deserialize<'de> for ContentMetadata {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Parts;
        impl<'de> Visitor<'de> for Parts {
            type Value = ContentMetadata;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("message content parts")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                #[derive(Deserialize)]
                struct Part<'a> {
                    #[serde(borrow, rename = "type")]
                    kind: Cow<'a, str>,
                }
                let mut media = false;
                while let Some(part) = seq.next_element::<Part<'de>>()? {
                    media |= !matches!(part.kind.as_ref(), "text" | "refusal");
                }
                Ok(ContentMetadata(media))
            }
        }
        deserializer.deserialize_seq(Parts)
    }
}

struct SinglePrompt(bool);
impl<'de> Deserialize<'de> for SinglePrompt {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Prompt;
        impl<'de> Visitor<'de> for Prompt {
            type Value = SinglePrompt;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("text or token prompt")
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Self::Value, E> {
                Ok(SinglePrompt(true))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut count = 0;
                let mut batch = false;
                while let Some(value) = seq.next_element::<&'de RawValue>()? {
                    count += 1;
                    batch |= value.get().starts_with(['"', '[']);
                }
                Ok(SinglePrompt(count > 0 && (!batch || count == 1)))
            }
        }
        deserializer.deserialize_any(Prompt)
    }
}
