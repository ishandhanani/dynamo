// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Model Deployment Card
//!
//! The ModelDeploymentCard (MDC) is the primary model configuration structure that will be available to any
//! component that needs to interact with the model or its dependent artifacts.
//!
//! The ModelDeploymentCard contains LLM model deployment configuration information:
//! - Display name and service name for the model
//! - Model information (ModelInfoType)
//! - Tokenizer configuration (TokenizerKind)
//! - Prompt formatter settings (PromptFormatterArtifact)

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::common::checked_file::CheckedFile;
use crate::entrypoint::RouterConfig;
use crate::local_model::runtime_config::{
    ModelRuntimeConfig, TokenizerBackend, VLLM_NEMOTRON_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
};
use crate::model_type::{ModelInput, ModelType};
use anyhow::{Context, Result};
use derive_builder::Builder;
use dynamo_kv_router::identity::{ExplicitIdentityMap, IndexerIdentitySpec};
use dynamo_runtime::{slug::Slug, storage::kv};
use serde::{Deserialize, Serialize};
use tensor::TensorModelConfig;
use tokenizers::Tokenizer as HfTokenizer;

use crate::protocols::TokenIdType;
use media::{MediaDecoderConfig, MediaFetcherConfig};

pub mod loader;
pub mod media;
pub mod tensor;

pub use loader::ModelInfo;
pub(crate) use loader::is_weight_file;

fn append_runtime_contract_checksum(
    bytes: &mut Vec<u8>,
    runtime_config: &ModelRuntimeConfig,
    key: &str,
) {
    let Some(value) = runtime_config.runtime_data.get(key) else {
        return;
    };
    let mut value = value.clone();
    canonicalize_json_object_keys(&mut value);
    let value = serde_json::to_vec(&value).expect("serializing serde_json::Value cannot fail");

    // This contract controls model-visible media prompt expansion. Workers
    // with different contracts must not share a cohort whose preprocessor is
    // built from one representative card.
    bytes.extend_from_slice(b"\0dynamo/model-card/runtime-contract/v1\0");
    bytes.extend_from_slice(&(key.len() as u64).to_be_bytes());
    bytes.extend_from_slice(key.as_bytes());
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(&value);
}

fn canonicalize_json_object_keys(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            values.iter_mut().for_each(canonicalize_json_object_keys);
        }
        serde_json::Value::Object(map) => {
            map.values_mut().for_each(canonicalize_json_object_keys);
            map.sort_keys();
        }
        _ => {}
    }
}

fn append_indexer_identity_checksum(bytes: &mut Vec<u8>, spec: &IndexerIdentitySpec) {
    bytes.extend_from_slice(b"dynamo/model-card/indexer-identity/v1");
    append_identity_dimension(bytes, spec.semantics());
    append_identity_dimension(bytes, spec.routing_scope());
}

fn append_identity_dimension(bytes: &mut Vec<u8>, dimension: Option<&ExplicitIdentityMap>) {
    let Some(dimension) = dimension else {
        bytes.push(0);
        return;
    };
    bytes.push(1);
    bytes.extend_from_slice(&(dimension.entries().len() as u32).to_le_bytes());
    for (key, value) in dimension.entries() {
        append_framed_identity_value(bytes, key.as_bytes());
        append_framed_identity_value(bytes, value.as_bytes());
    }
}

fn append_framed_identity_value(bytes: &mut Vec<u8>, value: &[u8]) {
    let len = u32::try_from(value.len()).expect("validated identity values fit u32");
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(value);
}

/// Identify model deployment cards in the key-value store
pub const ROOT_PATH: &str = "v1/mdc";

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ModelInfoType {
    HfConfigJson(CheckedFile),
}

impl ModelInfoType {
    pub fn checksum(&self) -> String {
        match self {
            ModelInfoType::HfConfigJson(c) => c.checksum().to_string(),
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            ModelInfoType::HfConfigJson(c) => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            ModelInfoType::HfConfigJson(c) => c.update_dir(dir),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    HfTokenizerJson(CheckedFile),
    TikTokenModel(CheckedFile),
}

impl TokenizerKind {
    pub fn checksum(&self) -> String {
        match self {
            TokenizerKind::HfTokenizerJson(c) | TokenizerKind::TikTokenModel(c) => {
                c.checksum().to_string()
            }
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            TokenizerKind::HfTokenizerJson(c) | TokenizerKind::TikTokenModel(c) => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            TokenizerKind::HfTokenizerJson(c) | TokenizerKind::TikTokenModel(c) => {
                c.update_dir(dir)
            }
        }
    }
}

/// Supported types of prompt formatters.
///
/// We need a way to associate the prompt formatter template definition with an associated
/// data model which is expected for rendering.
///
/// All current prompt formatters are Jinja2 templates which use the OpenAI ChatCompletionRequest
/// format. However, we currently do not have a discovery path to know if the model supports tool use
/// unless we inspect the template.
///
/// TODO(): Add an enum for the PromptFormatDataModel with at minimum arms for:
/// - OaiChat
/// - OaiChatToolUse
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PromptFormatterArtifact {
    HfTokenizerConfigJson(CheckedFile),
    #[serde(rename = "hf_chat_template", alias = "hf_chat_template_jinja")]
    HfChatTemplateJinja {
        is_custom: bool,
        file: CheckedFile,
    },
    HfChatTemplateJson {
        is_custom: bool,
        file: CheckedFile,
    },
}

impl PromptFormatterArtifact {
    pub fn checksum(&self) -> String {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(c) => c.checksum().to_string(),
            PromptFormatterArtifact::HfChatTemplateJinja { file: c, .. }
            | PromptFormatterArtifact::HfChatTemplateJson { file: c, .. } => {
                c.checksum().to_string()
            }
        }
    }

    /// Is this file available locally
    pub fn is_local(&self) -> bool {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(c) => c.is_local(),
            PromptFormatterArtifact::HfChatTemplateJinja { file: c, .. }
            | PromptFormatterArtifact::HfChatTemplateJson { file: c, .. } => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(c) => c.update_dir(dir),
            PromptFormatterArtifact::HfChatTemplateJinja { file: c, .. }
            | PromptFormatterArtifact::HfChatTemplateJson { file: c, .. } => c.update_dir(dir),
        }
    }

    pub fn is_custom(&self) -> bool {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(_) => false,
            PromptFormatterArtifact::HfChatTemplateJinja { is_custom, .. }
            | PromptFormatterArtifact::HfChatTemplateJson { is_custom, .. } => *is_custom,
        }
    }
}

// `PromptContextMixin` is owned by the `dynamo-renderer` crate (it drives
// chat-template rendering); the MDC's `prompt_context` field is typed with it.
use dynamo_renderer::PromptContextMixin;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum GenerationConfig {
    HfGenerationConfigJson(CheckedFile),
}

impl GenerationConfig {
    pub fn checksum(&self) -> String {
        match self {
            GenerationConfig::HfGenerationConfigJson(c) => c.checksum().to_string(),
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            GenerationConfig::HfGenerationConfigJson(c) => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            GenerationConfig::HfGenerationConfigJson(c) => c.update_dir(dir),
        }
    }
}

fn pf_checked_file(p: &PromptFormatterArtifact) -> &CheckedFile {
    match p {
        PromptFormatterArtifact::HfTokenizerConfigJson(cf)
        | PromptFormatterArtifact::HfChatTemplateJinja { file: cf, .. }
        | PromptFormatterArtifact::HfChatTemplateJson { file: cf, .. } => cf,
    }
}

fn pf_checked_file_mut(p: &mut PromptFormatterArtifact) -> &mut CheckedFile {
    match p {
        PromptFormatterArtifact::HfTokenizerConfigJson(cf)
        | PromptFormatterArtifact::HfChatTemplateJinja { file: cf, .. }
        | PromptFormatterArtifact::HfChatTemplateJson { file: cf, .. } => cf,
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Builder, Default)]
pub struct ModelDeploymentCard {
    /// Human readable model name, e.g. "Meta Llama 3.1 8B Instruct"
    pub display_name: String,

    // Cache the Slugified display_name so we can share references to it
    slug: Slug,

    /// Original Hugging Face repository, NGC URI, or local model path.
    /// When `display_name` is customized (e.g., via `--served-model-name`),
    /// this field preserves the original repository path needed for downloads.
    /// Falls back to `display_name` if not set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,

    /// Model information
    pub model_info: Option<ModelInfoType>,

    /// Tokenizer configuration
    pub tokenizer: Option<TokenizerKind>,

    /// Prompt Formatter configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_formatter: Option<PromptFormatterArtifact>,

    /// chat template may be stored as a separate file instead of in `prompt_formatter`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_file: Option<PromptFormatterArtifact>,

    /// Generation config - default sampling params
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gen_config: Option<GenerationConfig>,

    /// Prompt Formatter Config
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_context: Option<Vec<PromptContextMixin>>,

    /// Architectural context maximum derived from model or tokenizer metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub architectural_max_context_length: Option<u32>,

    /// Deprecated v1.2 MDC wire field.
    ///
    /// This is only a deserialization fallback and serialization projection. Canonical state
    /// remains in `runtime_config.context_length` and `architectural_max_context_length`.
    ///
    /// TODO(v1.5): Remove after the temporary v1.2-to-v1.4 upgrade exception expires.
    #[serde(
        default,
        rename = "context_length",
        skip_serializing_if = "Option::is_none"
    )]
    #[builder(default, setter(skip))]
    legacy_context_length: Option<u32>,

    /// Size of a KV cache block.
    /// Passed to the engine, KV router, and trace replay hash path.
    pub kv_cache_block_size: u32,

    /// How many times a request can be migrated to another worker if the HTTP server lost
    /// connection to the current worker.
    pub migration_limit: u32,

    /// Specifies whether the model is a chat, completions, etc model.
    pub model_type: ModelType,

    /// Specifies the model input type.
    /// `Tokens` for engines that expect pre-processed input.
    /// `Text` for engines that take care of pre-processing themselves.
    pub model_input: ModelInput,

    /// Processing stage this worker handles (Prefill, Decode, Encode, Aggregated).
    /// Orthogonal to `model_type` (which describes endpoints exposed).
    ///
    /// Every worker must set this explicitly. `None` means the worker has
    /// not declared a worker type and is treated as misconfiguration:
    /// `Model::ws_type_and_needs` returns `None`, the serving-readiness
    /// gate refuses to vouch for the namespace, and `register_model`
    /// rejects such cards outright. The `Option<>` type and
    /// `#[serde(default)]` are kept so older cards still deserialize, but
    /// downstream readers treat them as not-ready.
    #[serde(default)]
    pub worker_type: Option<crate::worker_type::WorkerType>,

    /// Peer worker types this worker requires to serve traffic, in DNF form.
    /// The outer `Vec` is OR; each inner `Vec` is an AND-set of required
    /// worker types. Empty outer `Vec` means "no peers required."
    ///
    /// Examples:
    /// - Prefill worker: `[[Decode]]` — needs a Decode peer.
    /// - Encode worker: `[[Prefill, Decode], [Aggregated]]` — needs either a
    ///   P+D pair or a single Aggregated peer.
    #[serde(default)]
    pub needs: Vec<Vec<crate::worker_type::WorkerType>>,

    /// LoRA metadata for routing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraInfo>,

    /// Additional names this model responds to (aliases).
    /// Requests using any of these names will be routed to this worker.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,

    /// User-defined metadata for custom worker behavior
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_data: Option<serde_json::Value>,

    #[serde(default)]
    pub runtime_config: ModelRuntimeConfig,

    /// Tensor model configuration for tensor-serving protocols.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub tensor_model_config: Option<TensorModelConfig>,

    /// Media decoding configuration
    #[serde(default)]
    pub media_decoder: Option<MediaDecoderConfig>,

    /// Media fetching configuration
    #[serde(default)]
    pub media_fetcher: Option<MediaFetcherConfig>,

    /// Per-worker-set router configuration override.
    /// When set, the frontend watcher uses this instead of the global frontend router config.
    /// Falls back to the frontend-level config when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router_config: Option<RouterConfig>,

    /// Optional authoritative KV-indexer compatibility and isolation material.
    ///
    /// A present dimension replaces its component-derived default. Entry labels are deliberately
    /// uninterpreted so engine-specific compatibility facts do not expand this schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub indexer_identity: Option<IndexerIdentitySpec>,

    /// Sibling files (e.g. `preprocessor_config.json`) the worker
    /// advertises alongside the typed slots.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_files: Vec<CheckedFile>,

    #[serde(skip, default)]
    checksum: OnceLock<String>,

    /// Set when this card loads its tokenizer. See `hf_token_id_bound`.
    #[serde(skip, default)]
    tokenizer_id_bound: OnceLock<Option<usize>>,
}

/// LoRA adapter information for routing decisions
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoraInfo {
    /// LoRA adapter name (e.g., "customer-123-v2")
    pub name: String,

    /// Maximum number of LoRA adapters that can be loaded at once on a single GPU
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_gpu_lora_count: Option<u32>,
}

impl ModelDeploymentCard {
    /// Number of typed metadata slots (`model_info`, `tokenizer`,
    /// `prompt_formatter`, `chat_template_file`, `gen_config`). Used as
    /// a capacity hint for [`Self::iter_metadata_files`].
    const TYPED_SLOT_COUNT: usize = 5;

    pub fn builder() -> ModelDeploymentCardBuilder {
        ModelDeploymentCardBuilder::default()
    }

    /// Return this card's explicit worker role, with compatibility for legacy cards.
    ///
    /// Before `worker_type` was added, prefill cards used `ModelType::Prefill`; every other
    /// worker was a full-request worker.
    pub fn effective_worker_type(&self) -> crate::worker_type::WorkerType {
        Self::resolve_worker_type(self.worker_type, self.model_type)
    }

    /// Resolve an explicit or legacy worker role without constructing a model card.
    pub fn resolve_worker_type(
        worker_type: Option<crate::worker_type::WorkerType>,
        model_type: ModelType,
    ) -> crate::worker_type::WorkerType {
        worker_type.unwrap_or_else(|| {
            if model_type.supports_prefill() {
                crate::worker_type::WorkerType::Prefill
            } else {
                crate::worker_type::WorkerType::Aggregated
            }
        })
    }

    /// Create a ModelDeploymentCard where only the name is filled in.
    ///
    /// Single-process setups don't need an MDC to communicate model details, but it
    /// simplifies the code to assume we always have one. This is how you get one in those
    /// cases. A quasi-null object: <https://en.wikipedia.org/wiki/Null_object_pattern>
    pub fn with_name_only(name: &str) -> ModelDeploymentCard {
        ModelDeploymentCard {
            display_name: name.to_string(),
            slug: Slug::from_string(name),
            ..Default::default()
        }
    }

    /// Load a model deployment card from a JSON file
    pub fn load_from_json_file<P: AsRef<Path>>(file: P) -> std::io::Result<Self> {
        let contents = std::fs::read_to_string(&file)?;
        Ok(serde_json::from_str(&contents).inspect_err(|err| {
            crate::log_json_err(&file.as_ref().display().to_string(), &contents, err)
        })?)
    }

    /// Load a model deployment card from a JSON string
    pub fn load_from_json_str(contents: &str) -> Result<Self, anyhow::Error> {
        Ok(serde_json::from_str(contents)
            .inspect_err(|err| crate::log_json_err("unknown", contents, err))?)
    }

    //
    // Methods
    //

    /// Save the model deployment card to a JSON file
    pub fn save_to_json_file(&self, file: &str) -> Result<(), anyhow::Error> {
        std::fs::write(file, self.to_json()?)?;
        Ok(())
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.display_name
    }

    #[inline]
    pub fn slug(&self) -> &Slug {
        &self.slug
    }

    /// Effective serving context: runtime engine limit, then architectural maximum.
    pub fn effective_context_length(&self) -> u32 {
        self.runtime_config
            .context_length
            .or(self.architectural_max_context_length)
            .or(self.legacy_context_length)
            .unwrap_or(0)
    }

    pub(crate) fn for_mdc_wire(&self) -> Self {
        let mut card = self.clone();
        card.legacy_context_length = Some(self.effective_context_length());
        card.runtime_config.normalize_structural_tag_compatibility();
        card
    }

    /// Serialize the model deployment card to a JSON string
    pub fn to_json(&self) -> Result<String, anyhow::Error> {
        Ok(serde_json::to_string(&self.for_mdc_wire())?)
    }

    pub fn mdcsum(&self) -> &str {
        self.checksum
            .get_or_init(|| {
                // Only include the important fields
                let mut bytes_to_hash: Vec<u8> = Vec::with_capacity(512);
                bytes_to_hash.extend(self.display_name.as_bytes());
                if let Some(source_path) = self.source_path.as_ref() {
                    bytes_to_hash.extend(source_path.as_bytes());
                }

                // The files can be either a URL or a local path, so we ignore that and hash their
                // checksum instead, which won't change wherever they are.

                if let Some(model_info) = self.model_info.as_ref() {
                    bytes_to_hash.extend(model_info.checksum().as_bytes());
                }
                if let Some(tokenizer) = self.tokenizer.as_ref() {
                    bytes_to_hash.extend(tokenizer.checksum().as_bytes());
                }
                if let Some(prompt_formatter) = self.prompt_formatter.as_ref() {
                    bytes_to_hash.extend(prompt_formatter.checksum().as_bytes());
                }
                if let Some(chat_template) = self.chat_template_file.as_ref() {
                    bytes_to_hash.extend(chat_template.checksum().as_bytes());
                }
                if let Some(gen_config) = self.gen_config.as_ref() {
                    bytes_to_hash.extend(gen_config.checksum().as_bytes());
                }

                // extra_files: hash sorted (basename, checksum) pairs so
                // (a) workers with identical siblings produce the same
                // mdcsum regardless of `read_dir` order, and (b) the same
                // bytes under different filenames don't collide — otherwise
                // the frontend cache could serve a local_dir missing siblings.
                let mut extras: Vec<(&str, &str)> = self
                    .extra_files
                    .iter()
                    .map(|cf| (cf.basename().unwrap_or(""), cf.checksum().hash()))
                    .collect();
                extras.sort_unstable();
                for (name, h) in &extras {
                    bytes_to_hash.extend(name.as_bytes());
                    bytes_to_hash.push(0);
                    bytes_to_hash.extend(h.as_bytes());
                }

                if let Some(prompt_context_vec) = self.prompt_context.as_ref() {
                    // Paste it as the bytes of the debug format. It's a Vec of enum, so this should be
                    // fine. If the debug representation changes that only happens in a new release.
                    bytes_to_hash.extend(format!("{prompt_context_vec:?}").as_bytes());
                }
                bytes_to_hash.extend(self.effective_context_length().to_be_bytes());
                bytes_to_hash.extend(self.kv_cache_block_size.to_be_bytes());

                // worker_type/needs participate in the checksum so that a rolling
                // update that changes only those is correctly
                // rejected as incompatible with the existing WorkerSet (forcing
                // drain-and-redeploy) instead of silently joining and serving
                // stale readiness data.
                //
                // worker_type discriminator: 0 = None, then the variant ordinal.
                match self.worker_type {
                    None => bytes_to_hash.push(0),
                    Some(crate::worker_type::WorkerType::Prefill) => bytes_to_hash.push(1),
                    Some(crate::worker_type::WorkerType::Decode) => bytes_to_hash.push(2),
                    Some(crate::worker_type::WorkerType::Encode) => bytes_to_hash.push(3),
                    Some(crate::worker_type::WorkerType::Aggregated) => bytes_to_hash.push(4),
                }
                // needs is DNF: hash length(outer) || for each alt { length(inner) || each variant }
                bytes_to_hash.extend((self.needs.len() as u32).to_be_bytes());
                for alt in &self.needs {
                    bytes_to_hash.extend((alt.len() as u32).to_be_bytes());
                    for w in alt {
                        let v: u8 = match w {
                            crate::worker_type::WorkerType::Prefill => 1,
                            crate::worker_type::WorkerType::Decode => 2,
                            crate::worker_type::WorkerType::Encode => 3,
                            crate::worker_type::WorkerType::Aggregated => 4,
                        };
                        bytes_to_hash.push(v);
                    }
                }

                if let Some(router_config) = self.router_config.as_ref()
                    && let Ok(bytes) = serde_json::to_vec(router_config)
                {
                    // Hash router_config separately so we extend bytes_to_hash with a
                    // fixed-size digest (32 bytes) rather than the full JSON payload.
                    // [gluo TODO] take checksum() approach that is the same as above,
                    // along with this effort, we should reorganize where RouterConfig
                    // should be defined.
                    bytes_to_hash.extend(blake3::hash(&bytes).as_bytes());
                }

                if let Some(identity) = self.indexer_identity.as_ref() {
                    append_indexer_identity_checksum(&mut bytes_to_hash, identity);
                }

                // Aliases participate in the checksum. Every worker in a
                // deployment carries the same static --served-model-name list,
                // so their checksums still match and they share one WorkerSet;
                // changing the alias list rolls a new WorkerSet (consistent per
                // set) rather than mutating a live one in place. `aliases` holds
                // only the alternate names (the primary is `display_name`, hashed
                // above), so order within the list still matters — hash in order.
                // Skipped entirely when empty so a card without aliases keeps the
                // same checksum as before this field existed (no spurious
                // WorkerSet split on upgrade); this is the last hashed field, so
                // omission is unambiguous.
                if !self.aliases.is_empty() {
                    bytes_to_hash.extend((self.aliases.len() as u32).to_be_bytes());
                    for alias in &self.aliases {
                        bytes_to_hash.extend((alias.len() as u32).to_be_bytes());
                        bytes_to_hash.extend(alias.as_bytes());
                    }
                }

                // Tower/connector LoRA changes vLLM's multimodal cache identity.
                // Isolate such workers from the default/missing=false WorkerSet
                // without changing checksums for existing deployments.
                if self.runtime_config.runtime_flag_enabled(
                    crate::local_model::runtime_config::VLLM_ENABLE_TOWER_CONNECTOR_LORA_RUNTIME_KEY,
                ) {
                    bytes_to_hash.extend_from_slice(b"\0vllm_enable_tower_connector_lora\0true");
                }

                if self.runtime_config.runtime_flag_enabled(
                    crate::local_model::runtime_config::VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
                ) {
                    bytes_to_hash
                        .extend_from_slice(b"\0vllm_inference_v1_generate\0true");
                }

                // The Qwen video contract is resolved per cohort, not per card.
                // Nemotron contracts still partition WorkerSets by checksum.
                append_runtime_contract_checksum(
                    &mut bytes_to_hash,
                    &self.runtime_config,
                    VLLM_NEMOTRON_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
                );

                // TODO: Do we want any other user_data or runtime_config?

                blake3::hash(&bytes_to_hash).to_string()
            })
            .as_ref()
    }

    /// Is this a full model card with tokenizer?
    /// There are cases where we have a placeholder card (see `with_name_only`).
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    pub(crate) fn set_source_path(&mut self, source_path: PathBuf) {
        self.source_path = Some(source_path.display().to_string());
    }

    /// Allow user to override the name we register this model under.
    /// Corresponds to vllm's `--served-model-name`.
    pub fn set_name(&mut self, name: &str) {
        self.display_name = name.to_string();
        self.slug = Slug::from_string(name);
    }

    pub fn source_path(&self) -> &str {
        self.source_path.as_ref().unwrap_or(&self.display_name)
    }

    /// Set additional names (aliases) this model responds to.
    pub fn set_aliases(&mut self, aliases: Vec<String>) {
        self.aliases = aliases;
    }

    pub fn requires_preprocessing(&self) -> bool {
        matches!(self.model_input, ModelInput::Tokens)
    }

    /// Iterate populated metadata slots in deterministic order:
    /// model_info, tokenizer, prompt_formatter, chat_template_file,
    /// gen_config, then any `extra_files` siblings the worker harvested
    /// (preprocessor_config.json, special_tokens_map.json, etc.). Each entry
    /// is `(file, is_custom)` — `is_custom` is only ever true for
    /// operator-supplied chat templates, which can't fall back to HF.
    pub fn iter_metadata_files(&self) -> Vec<(&CheckedFile, bool)> {
        let mut out: Vec<(&CheckedFile, bool)> =
            Vec::with_capacity(Self::TYPED_SLOT_COUNT + self.extra_files.len());
        if let Some(ModelInfoType::HfConfigJson(cf)) = self.model_info.as_ref() {
            out.push((cf, false));
        }
        if let Some(TokenizerKind::HfTokenizerJson(cf) | TokenizerKind::TikTokenModel(cf)) =
            self.tokenizer.as_ref()
        {
            out.push((cf, false));
        }
        if let Some(p) = self.prompt_formatter.as_ref() {
            out.push((pf_checked_file(p), p.is_custom()));
        }
        if let Some(c) = self.chat_template_file.as_ref() {
            out.push((pf_checked_file(c), c.is_custom()));
        }
        if let Some(GenerationConfig::HfGenerationConfigJson(cf)) = self.gen_config.as_ref() {
            out.push((cf, false));
        }
        for cf in &self.extra_files {
            out.push((cf, false));
        }
        out
    }

    /// Mutable mirror of [`Self::iter_metadata_files`].
    pub fn iter_metadata_files_mut(&mut self) -> Vec<(&mut CheckedFile, bool)> {
        let mut out: Vec<(&mut CheckedFile, bool)> =
            Vec::with_capacity(Self::TYPED_SLOT_COUNT + self.extra_files.len());
        if let Some(ModelInfoType::HfConfigJson(cf)) = self.model_info.as_mut() {
            out.push((cf, false));
        }
        if let Some(TokenizerKind::HfTokenizerJson(cf) | TokenizerKind::TikTokenModel(cf)) =
            self.tokenizer.as_mut()
        {
            out.push((cf, false));
        }
        if let Some(p) = self.prompt_formatter.as_mut() {
            let is_custom = p.is_custom();
            out.push((pf_checked_file_mut(p), is_custom));
        }
        if let Some(c) = self.chat_template_file.as_mut() {
            let is_custom = c.is_custom();
            out.push((pf_checked_file_mut(c), is_custom));
        }
        if let Some(GenerationConfig::HfGenerationConfigJson(cf)) = self.gen_config.as_mut() {
            out.push((cf, false));
        }
        for cf in &mut self.extra_files {
            out.push((cf, false));
        }
        out
    }
}

impl PartialEq for ModelDeploymentCard {
    fn eq(&self, other: &ModelDeploymentCard) -> bool {
        self.mdcsum() == other.mdcsum()
    }
}

/// Model-card registration is create-only. The discovery taint API owns the
/// narrow exception for updating runtime_config.taints on an existing record.
impl kv::Versioned for ModelDeploymentCard {
    fn revision(&self) -> u64 {
        0
    }

    fn set_revision(&mut self, _revision: u64) {}
}

impl fmt::Display for ModelDeploymentCard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.slug())
    }
}
