// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online model of how long a worker takes to load KV pages from the shared store.
//!
//! Every prefetch reply reports the tokens the scheduler loaded and the wall-clock the
//! load took. Per worker, this fits `L(n) = fixed_ms + per_page_ms * n` for `n` pages,
//! in pages: `loaded_tokens` is converted with the page size the controller was
//! built with (`HintConfig::page_tokens`), so a page is the unit the scheduler loads.
//! Replies that loaded nothing carry no load time and are ignored.
//!
//! The fit is two EWMAs, one per coefficient, so it tolerates the noise of a
//! handful of samples. A reply for a small load refines the fixed cost; a reply for
//! a larger load refines the per-page cost, each after subtracting the other
//! coefficient's current contribution. Until a worker has [`MIN_SAMPLES`] replies its
//! estimate is the measured prior: about 40 ms fixed and 0.3 ms per page.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::WorkerTarget;

/// Fixed load cost assumed before a worker has enough replies, in milliseconds.
pub const PRIOR_FIXED_MS: f64 = 40.0;
/// Per-page load cost assumed before a worker has enough replies, in milliseconds.
pub const PRIOR_PER_PAGE_MS: f64 = 0.3;
/// Replies a worker needs before its own fit replaces the prior.
pub const MIN_SAMPLES: u32 = 3;
/// EWMA weight of a new sample.
const ALPHA: f64 = 0.2;
/// Loads of at most this many pages refine the fixed cost; larger ones the per-page cost.
const SMALL_LOAD_PAGES: u64 = 8;

/// Coefficients of `L(n) = fixed_ms + per_page_ms * n`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadFit {
    pub fixed_ms: f64,
    pub per_page_ms: f64,
}

impl LoadFit {
    pub const PRIOR: Self = Self {
        fixed_ms: PRIOR_FIXED_MS,
        per_page_ms: PRIOR_PER_PAGE_MS,
    };

    /// Expected milliseconds to load `pages`.
    pub fn estimate_ms(&self, pages: u64) -> f64 {
        self.fixed_ms + self.per_page_ms * pages as f64
    }

    fn observe(&mut self, pages: u64, elapsed_ms: f64) {
        let pages = pages.max(1);
        if pages <= SMALL_LOAD_PAGES {
            let fixed = (elapsed_ms - self.per_page_ms * pages as f64).max(0.0);
            self.fixed_ms += ALPHA * (fixed - self.fixed_ms);
        } else {
            let per_page = (elapsed_ms - self.fixed_ms).max(0.0) / pages as f64;
            self.per_page_ms += ALPHA * (per_page - self.per_page_ms);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WorkerLoadFit {
    fit: LoadFit,
    samples: u32,
}

impl WorkerLoadFit {
    fn effective(&self) -> LoadFit {
        if self.samples >= MIN_SAMPLES {
            self.fit
        } else {
            LoadFit::PRIOR
        }
    }
}

/// Per-worker load fits; see the module docs.
#[derive(Debug)]
pub struct LoadModel {
    page_tokens: usize,
    workers: Mutex<HashMap<WorkerTarget, WorkerLoadFit>>,
}

impl LoadModel {
    pub fn new(page_tokens: usize) -> Self {
        Self {
            page_tokens: page_tokens.max(1),
            workers: Mutex::new(HashMap::new()),
        }
    }

    pub fn pages(&self, tokens: usize) -> u64 {
        tokens.div_ceil(self.page_tokens) as u64
    }

    /// Record a reply that loaded `loaded_tokens` in `elapsed_ms`; nothing loaded, nothing learned.
    pub fn observe(&self, worker: WorkerTarget, loaded_tokens: u64, elapsed_ms: f64) {
        if loaded_tokens == 0 || !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
            return;
        }
        let pages = (loaded_tokens as usize).div_ceil(self.page_tokens) as u64;
        let mut workers = self.workers.lock().unwrap_or_else(|e| e.into_inner());
        let entry = workers.entry(worker).or_insert(WorkerLoadFit {
            fit: LoadFit::PRIOR,
            samples: 0,
        });
        entry.fit.observe(pages, elapsed_ms);
        entry.samples = entry.samples.saturating_add(1);
    }

    /// The fit used for `worker`: its own once it has [`MIN_SAMPLES`] replies, else the prior.
    pub fn fit(&self, worker: WorkerTarget) -> LoadFit {
        self.workers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&worker)
            .map_or(LoadFit::PRIOR, WorkerLoadFit::effective)
    }

    /// Expected milliseconds for `worker` to load `tokens` from the shared store.
    pub fn estimate_ms(&self, worker: WorkerTarget, tokens: usize) -> f64 {
        self.fit(worker).estimate_ms(self.pages(tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) -> bool {
        (actual - expected).abs() <= tolerance * expected.abs()
    }

    #[test]
    fn prior_until_three_samples() {
        let model = LoadModel::new(16);
        let worker = WorkerTarget::new(1, Some(0));
        // 124 pages at the prior: 40 + 0.3 * 124.
        assert_eq!(model.estimate_ms(worker, 124 * 16), 40.0 + 0.3 * 124.0);
        // Tokens round up to whole pages.
        assert_eq!(model.pages(17), 2);

        model.observe(worker, 128 * 16, 500.0);
        model.observe(worker, 128 * 16, 500.0);
        assert_eq!(
            model.fit(worker),
            LoadFit::PRIOR,
            "two samples keep the prior"
        );
        model.observe(worker, 128 * 16, 500.0);
        assert!(
            model.estimate_ms(worker, 128 * 16) > 200.0,
            "third sample unlocks the fit"
        );

        // Other workers and empty loads leave the model alone.
        let other = WorkerTarget::new(2, Some(0));
        model.observe(other, 0, 999.0);
        assert_eq!(model.fit(other), LoadFit::PRIOR);
    }

    #[test]
    fn fit_converges_on_a_synthetic_line() {
        let page_tokens = 64;
        let model = LoadModel::new(page_tokens);
        let worker = WorkerTarget::new(7, Some(1));
        let (fixed, per_page) = (55.0, 0.4);
        let load = |pages: u64| fixed + per_page * pages as f64;
        // Alternate small and large loads, as a mix of short and long prompts would.
        for round in 0..40u64 {
            for &pages in &[2 + round % 3, 64 + 32 * (round % 5)] {
                model.observe(worker, pages * page_tokens as u64, load(pages));
            }
        }
        let fit = model.fit(worker);
        assert!(close(fit.fixed_ms, fixed, 0.05), "fixed {fit:?}");
        assert!(close(fit.per_page_ms, per_page, 0.05), "per page {fit:?}");
        assert!(close(
            model.estimate_ms(worker, 200 * page_tokens),
            load(200),
            0.05
        ));
    }

    #[test]
    fn large_loads_alone_still_predict_large_loads() {
        let model = LoadModel::new(64);
        let worker = WorkerTarget::new(3, None);
        // Slower store than the prior: 190 ms for 124 pages, 300 ms for 378.
        for _ in 0..20 {
            model.observe(worker, 124 * 64, 190.0);
            model.observe(worker, 378 * 64, 300.0);
        }
        let estimate = model.estimate_ms(worker, 250 * 64);
        assert!((200.0..=300.0).contains(&estimate), "{estimate}");
    }
}
