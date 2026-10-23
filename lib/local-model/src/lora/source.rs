// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::cache::LoRACache;
use crate::hub::{self, HfRepoSpec};
use anyhow::Result;
use async_trait::async_trait;
use hf_hub::Cache;
use std::path::{Path, PathBuf};

/// Minimal trait for LoRA sources
/// Users can implement this in Rust for custom sources
#[async_trait]
pub trait LoRASource: Send + Sync {
    /// Returns whether this source handles the URI.
    fn supports(&self, _lora_uri: &str) -> bool {
        true
    }

    /// Download LoRA from source to destination path
    /// Returns the actual path where files were written
    async fn download(&self, lora_uri: &str, dest_path: &Path) -> Result<PathBuf>;

    /// Check if LoRA exists in this source
    async fn exists(&self, lora_uri: &str) -> Result<bool>;

    /// Return a complete source-owned cache path without network access.
    fn cached_path(&self, _lora_uri: &str) -> Result<Option<PathBuf>> {
        Ok(None)
    }
}

/// Hugging Face Hub LoRA source.
///
/// Downloads repositories into the standard Hugging Face cache and returns the
/// immutable snapshot directory, avoiding a second copy under `DYN_LORA_PATH`.
pub struct HuggingFaceLoRASource {
    cache: Cache,
}

impl Default for HuggingFaceLoRASource {
    fn default() -> Self {
        Self::from_env()
    }
}

impl HuggingFaceLoRASource {
    pub fn from_env() -> Self {
        Self {
            cache: hub::huggingface_cache(),
        }
    }

    #[cfg(test)]
    fn with_cache(cache: Cache) -> Self {
        Self { cache }
    }

    fn cached_snapshot(&self, spec: &HfRepoSpec) -> Result<Option<PathBuf>> {
        let Some(snapshot) = hub::cached_hf_snapshot(&self.cache, spec, "adapter_config.json")
        else {
            return Ok(None);
        };
        Ok(LoRACache::validate_path(&snapshot)?.then_some(snapshot))
    }
}

#[async_trait]
impl LoRASource for HuggingFaceLoRASource {
    fn supports(&self, lora_uri: &str) -> bool {
        lora_uri.starts_with("hf://")
    }

    async fn download(&self, hf_uri: &str, _dest_path: &Path) -> Result<PathBuf> {
        let spec = HfRepoSpec::from_uri(hf_uri)?;
        if let Some(snapshot) = self.cached_snapshot(&spec)? {
            tracing::debug!(uri = hf_uri, path = %snapshot.display(), "using cached Hugging Face LoRA");
            return Ok(snapshot);
        }

        tracing::info!(uri = hf_uri, "downloading LoRA from Hugging Face Hub");
        let snapshot = hub::download_hf_snapshot(&self.cache, &spec).await?;
        if !LoRACache::validate_path(&snapshot)? {
            anyhow::bail!(
                "Hugging Face repository {hf_uri} is not a valid LoRA: expected adapter_config.json and adapter weights"
            );
        }
        hub::finalize_hf_snapshot(&self.cache, &spec, &snapshot)?;
        Ok(snapshot)
    }

    async fn exists(&self, hf_uri: &str) -> Result<bool> {
        HfRepoSpec::from_uri(hf_uri)?;
        Ok(true)
    }

    fn cached_path(&self, hf_uri: &str) -> Result<Option<PathBuf>> {
        if !hf_uri.starts_with("hf://") {
            return Ok(None);
        }
        self.cached_snapshot(&HfRepoSpec::from_uri(hf_uri)?)
    }
}

/// Local filesystem LoRA source
/// For file:// URIs, just validates the path exists
pub struct LocalLoRASource;

impl Default for LocalLoRASource {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalLoRASource {
    pub fn new() -> Self {
        Self
    }

    /// Parse file:// URI to extract local path
    /// Format: file:///absolute/path/to/lora
    fn parse_file_uri(uri: &str) -> Result<PathBuf> {
        if !uri.starts_with("file://") {
            anyhow::bail!("Invalid file URI scheme: expected file://");
        }

        let path_str = uri.strip_prefix("file://").unwrap();
        Ok(PathBuf::from(path_str))
    }
}

#[async_trait]
impl LoRASource for LocalLoRASource {
    fn supports(&self, lora_uri: &str) -> bool {
        lora_uri.starts_with("file://")
    }

    async fn download(&self, file_uri: &str, _dest_path: &Path) -> Result<PathBuf> {
        let source_path = Self::parse_file_uri(file_uri)?;

        if !source_path.exists() {
            anyhow::bail!("LoRA path does not exist: {}", source_path.display());
        }

        if !source_path.is_dir() {
            anyhow::bail!("LoRA path is not a directory: {}", source_path.display());
        }

        tracing::info!("Using local LoRA at: {:?}", source_path);

        Ok(source_path)
    }

    async fn exists(&self, file_uri: &str) -> Result<bool> {
        let source_path = Self::parse_file_uri(file_uri)?;
        Ok(source_path.exists() && source_path.is_dir())
    }
}

#[cfg(feature = "lora-s3")]
mod s3;
#[cfg(feature = "lora-s3")]
pub use s3::S3LoRASource;

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::Cache;
    use mockito::Matcher;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_parse_file_uri() {
        let uri = "file:///path/to/lora";
        let path = LocalLoRASource::parse_file_uri(uri).unwrap();
        assert_eq!(path, PathBuf::from("/path/to/lora"));
    }

    #[test]
    fn test_parse_file_uri_invalid() {
        let uri = "http://example.com/lora";
        assert!(LocalLoRASource::parse_file_uri(uri).is_err());
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn hf_source_reuses_valid_native_snapshot_in_offline_mode() {
        let temp = TempDir::new().unwrap();
        let repo_dir = temp.path().join("models--org--adapter");
        let snapshot = repo_dir.join("snapshots/abc123");
        fs::create_dir_all(repo_dir.join("refs")).unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(repo_dir.join("refs/main"), "abc123").unwrap();
        fs::write(snapshot.join("adapter_config.json"), "{}").unwrap();
        fs::write(snapshot.join("adapter_model.safetensors"), "weights").unwrap();
        fs::write(snapshot.join(".dynamo_lora_complete"), "1\n").unwrap();

        let source = HuggingFaceLoRASource::with_cache(Cache::new(temp.path().to_path_buf()));
        assert_eq!(
            source.cached_path("hf://org/adapter").unwrap(),
            Some(snapshot.clone())
        );
        let result = temp_env::async_with_vars(
            [("HF_HUB_OFFLINE", Some("1"))],
            source.download("hf://org/adapter", Path::new("unused")),
        )
        .await
        .unwrap();

        assert_eq!(result, snapshot);
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn hf_source_reuses_commit_snapshot_without_ref_file() {
        let temp = TempDir::new().unwrap();
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let snapshot = temp
            .path()
            .join("models--org--adapter/snapshots")
            .join(revision);
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("adapter_config.json"), "{}").unwrap();
        fs::write(snapshot.join("adapter_model.safetensors"), "weights").unwrap();
        fs::write(snapshot.join(".dynamo_lora_complete"), "1\n").unwrap();

        let source = HuggingFaceLoRASource::with_cache(Cache::new(temp.path().to_path_buf()));
        let result = temp_env::async_with_vars(
            [("HF_HUB_OFFLINE", Some("1"))],
            source.download(
                format!("hf://org/adapter@{revision}").as_str(),
                Path::new("unused"),
            ),
        )
        .await
        .unwrap();

        assert_eq!(result, snapshot);
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn hf_source_downloads_complete_revision_pinned_snapshot() {
        let mut server = mockito::Server::new_async().await;
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let token = "test-token";
        let info = server
            .mock("GET", "/api/models/org/adapter/revision/main")
            .match_header("authorization", format!("Bearer {token}").as_str())
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(
                r#"{{"siblings":[{{"rfilename":"adapter_config.json"}},{{"rfilename":"adapter_model.safetensors"}}],"sha":"{commit}"}}"#
            ))
            .create_async()
            .await;
        let config = server
            .mock(
                "GET",
                format!("/org/adapter/resolve/{commit}/adapter_config.json").as_str(),
            )
            .match_header("range", Matcher::Regex("bytes=0-.*".to_string()))
            .with_status(200)
            .with_header("x-repo-commit", commit)
            .with_header("etag", "config-etag")
            .with_header("content-range", "bytes 0-1/2")
            .with_body("{}")
            .expect(2)
            .create_async()
            .await;
        let weights = server
            .mock(
                "GET",
                format!("/org/adapter/resolve/{commit}/adapter_model.safetensors").as_str(),
            )
            .match_header("range", Matcher::Regex("bytes=0-.*".to_string()))
            .with_status(200)
            .with_header("x-repo-commit", commit)
            .with_header("etag", "weights-etag")
            .with_header("content-range", "bytes 0-6/7")
            .with_body("weights")
            .expect(2)
            .create_async()
            .await;

        let temp = TempDir::new().unwrap();
        let token_path = temp.path().join("token");
        fs::write(&token_path, token).unwrap();
        let source = HuggingFaceLoRASource::with_cache(Cache::new(temp.path().to_path_buf()));
        let snapshot = temp_env::async_with_vars(
            [
                ("HF_ENDPOINT", Some(server.url().as_str())),
                ("HF_TOKEN", None),
                ("HUGGING_FACE_HUB_TOKEN", None),
                ("HF_TOKEN_PATH", token_path.to_str()),
                ("HF_HUB_OFFLINE", None),
            ],
            source.download("hf://org/adapter", Path::new("unused")),
        )
        .await
        .unwrap();

        assert_eq!(
            snapshot,
            temp.path()
                .join("models--org--adapter/snapshots")
                .join(commit)
        );
        assert_eq!(
            fs::read(snapshot.join("adapter_config.json")).unwrap(),
            b"{}"
        );
        assert_eq!(
            fs::read(snapshot.join("adapter_model.safetensors")).unwrap(),
            b"weights"
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("models--org--adapter/refs/main")).unwrap(),
            commit
        );
        assert_eq!(
            fs::read_to_string(snapshot.join(".dynamo_lora_complete")).unwrap(),
            "1\n"
        );

        info.assert_async().await;
        config.assert_async().await;
        weights.assert_async().await;
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn hf_source_does_not_mark_partial_commit_snapshot_complete() {
        let mut server = mockito::Server::new_async().await;
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let info = server
            .mock(
                "GET",
                format!("/api/models/org/adapter/revision/{commit}").as_str(),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(
                r#"{{"siblings":[{{"rfilename":"adapter_config.json"}},{{"rfilename":"adapter_model.safetensors"}},{{"rfilename":"README.md"}}],"sha":"{commit}"}}"#
            ))
            .create_async()
            .await;
        let config = server
            .mock(
                "GET",
                format!("/org/adapter/resolve/{commit}/adapter_config.json").as_str(),
            )
            .match_header("range", Matcher::Regex("bytes=0-.*".to_string()))
            .with_status(200)
            .with_header("x-repo-commit", commit)
            .with_header("etag", "config-etag")
            .with_header("content-range", "bytes 0-1/2")
            .with_body("{}")
            .expect(2)
            .create_async()
            .await;
        let weights = server
            .mock(
                "GET",
                format!("/org/adapter/resolve/{commit}/adapter_model.safetensors").as_str(),
            )
            .match_header("range", Matcher::Regex("bytes=0-.*".to_string()))
            .with_status(200)
            .with_header("x-repo-commit", commit)
            .with_header("etag", "weights-etag")
            .with_header("content-range", "bytes 0-6/7")
            .with_body("weights")
            .expect(2)
            .create_async()
            .await;
        let readme = server
            .mock(
                "GET",
                format!("/org/adapter/resolve/{commit}/README.md").as_str(),
            )
            .match_header("range", Matcher::Regex("bytes=0-.*".to_string()))
            .with_status(500)
            .expect(1)
            .create_async()
            .await;

        let temp = TempDir::new().unwrap();
        let source = HuggingFaceLoRASource::with_cache(Cache::new(temp.path().to_path_buf()));
        let result = temp_env::async_with_vars(
            [
                ("HF_ENDPOINT", Some(server.url().as_str())),
                ("HF_TOKEN", None),
                ("HUGGING_FACE_HUB_TOKEN", None),
                ("HF_TOKEN_PATH", None),
                ("HF_HUB_OFFLINE", None),
            ],
            source.download(
                format!("hf://org/adapter@{commit}").as_str(),
                Path::new("unused"),
            ),
        )
        .await;

        assert!(result.is_err());
        let snapshot = temp
            .path()
            .join("models--org--adapter/snapshots")
            .join(commit);
        assert!(snapshot.join("adapter_config.json").is_file());
        assert!(snapshot.join("adapter_model.safetensors").is_file());
        assert!(!snapshot.join(".dynamo_lora_complete").exists());

        let offline_result = temp_env::async_with_vars(
            [("HF_HUB_OFFLINE", Some("1"))],
            source.download(
                format!("hf://org/adapter@{commit}").as_str(),
                Path::new("unused"),
            ),
        )
        .await;
        assert!(offline_result.is_err());

        info.assert_async().await;
        config.assert_async().await;
        weights.assert_async().await;
        readme.assert_async().await;
    }
}
