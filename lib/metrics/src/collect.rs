// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scrape-time collection from live sources.
//!
//! A stage keeps its counters in atomics and knows nothing about Prometheus. A host-side
//! collector holds a [`SourceSet`] of the handles it exports and, on each scrape, reads every
//! live handle into fresh metric families. A handle leaves the set when its last strong
//! reference drops, or when the collector's `keep` predicate rejects it (cancelled, closed).

use std::sync::{Arc, Weak};

use parking_lot::Mutex;
use prometheus::core::{Collector, Desc};

/// The live sources a scrape-time collector reads. Cheap to clone; clones share the set.
pub struct SourceSet<S> {
    sources: Arc<Mutex<Vec<Weak<S>>>>,
}

impl<S> Clone for SourceSet<S> {
    fn clone(&self) -> Self {
        Self {
            sources: Arc::clone(&self.sources),
        }
    }
}

impl<S> Default for SourceSet<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> SourceSet<S> {
    pub fn new() -> Self {
        Self {
            sources: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Keep `source` in the scrape until its last strong reference drops.
    pub fn attach(&self, source: &Arc<S>) {
        let mut sources = self.sources.lock();
        sources.retain(|source| source.strong_count() > 0);
        sources.push(Arc::downgrade(source));
    }

    /// The sources a scrape reads: every live one that `keep` accepts. Dead and rejected
    /// sources leave the set.
    pub fn live(&self, mut keep: impl FnMut(&S) -> bool) -> Vec<Arc<S>> {
        let mut sources = self.sources.lock();
        let mut live = Vec::with_capacity(sources.len());
        sources.retain(|weak| match weak.upgrade() {
            Some(source) if keep(&source) => {
                live.push(source);
                true
            }
            _ => false,
        });
        live
    }

    /// Sources still referenced, including ones `live` would reject.
    pub fn len(&self) -> usize {
        let mut sources = self.sources.lock();
        sources.retain(|source| source.strong_count() > 0);
        sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The descriptors a collector reports: those of a fresh instance of the vector it builds
/// per scrape.
pub fn descs_of<C: Collector>(template: &C) -> Vec<Desc> {
    template.desc().into_iter().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    struct Source {
        value: AtomicU64,
        cancelled: AtomicBool,
    }

    fn source(value: u64) -> Arc<Source> {
        Arc::new(Source {
            value: AtomicU64::new(value),
            cancelled: AtomicBool::new(false),
        })
    }

    #[test]
    fn live_sums_attached_and_forgets_dropped_or_rejected() {
        let set = SourceSet::new();
        let a = source(1);
        let b = source(2);
        let c = source(4);
        set.attach(&a);
        set.attach(&b);
        set.attach(&c);
        let total = |set: &SourceSet<Source>| {
            set.live(|s| !s.cancelled.load(Ordering::Relaxed))
                .iter()
                .map(|s| s.value.load(Ordering::Relaxed))
                .sum::<u64>()
        };
        assert_eq!(total(&set), 7);

        drop(b);
        assert_eq!(total(&set), 5, "a dropped source leaves the scrape");
        assert_eq!(set.len(), 2);

        c.cancelled.store(true, Ordering::Relaxed);
        assert_eq!(total(&set), 1, "a rejected source leaves the scrape");
        assert_eq!(set.len(), 1, "and the set");

        let a2 = Arc::clone(&a);
        drop(a);
        assert_eq!(total(&set), 1, "a live clone keeps its source");
        drop(a2);
        assert_eq!(total(&set), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn clones_share_the_set() {
        let set = SourceSet::new();
        let twin = set.clone();
        let a = source(3);
        twin.attach(&a);
        assert_eq!(set.live(|_| true).len(), 1);
    }
}
