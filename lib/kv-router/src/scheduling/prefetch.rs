// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deadline-led prefetch hints from the admission queue.
//!
//! A queued request can have its KV pages loaded onto a worker before it is
//! admitted: `lead` before its `due_at` (or at enqueue, for a request flagged
//! `prefetch_on_arrival` or when the trigger fires `on_queue`), while the entry is still queued, the actor selects a
//! worker without booking and hands the request's shared-store-resident prefix
//! to the sink. When a hint was sent, that worker becomes the entry's
//! `affinity_target`, with that field's usual semantics: the default selector
//! treats it as exclusive while the worker stays eligible, like session
//! affinity, and falls back to a free selection otherwise; custom policies see
//! it as advisory. No hint, no affinity.
//!
//! A hint only helps when it lands before the request reaches the scheduler;
//! otherwise the scheduler loads the same pages itself and the hint's load is
//! pure extra work. With a [`PrefetchGate`], entries the `on_queue` trigger parked
//! compare their predicted wait (a per-class [`WaitEstimator`]) with the sink's
//! predicted load time and send the full prefix, a prefix sized to land in time,
//! or nothing. Wake-up and deadline entries are not gated.

use std::sync::Arc;
use std::time::Duration;

use crate::protocols::WorkerWithDpRank;

/// Receives prefetch hints chosen by the admission queue. Implementations must
/// return immediately: the call runs on the scheduler actor.
pub trait PrefetchHintSink: Send + Sync {
    fn prefetch(&self, worker: WorkerWithDpRank, tokens: &[u32]);

    /// Expected milliseconds for `worker` to load a `tokens`-long hint from the
    /// shared store, when the sink can tell. `None` means unknown.
    fn load_estimate_ms(&self, _worker: WorkerWithDpRank, _tokens: usize) -> Option<f64> {
        None
    }

    /// Gate decisions and the waits of hinted entries, for metrics. Dropped by default.
    fn observe(&self, _observation: PrefetchObservation) {}
}

/// What the gate did with one entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchDecision {
    /// The whole shared-resident prefix was sent.
    Full,
    /// A shorter prefix, sized to land before the predicted admission, was sent.
    Partial,
    /// Nothing was sent: not even the minimum page count would land in time.
    Skip,
}

impl PrefetchDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Partial => "partial",
            Self::Skip => "skip",
        }
    }
}

/// One observation from the prefetch path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefetchObservation {
    /// A gated entry was decided. `load_predicted_ms` is for the pages sent (the
    /// full prefix when nothing was sent).
    Decision {
        decision: PrefetchDecision,
        wait_predicted_ms: f64,
        load_predicted_ms: f64,
    },
    /// An entry that received a hint was admitted after `wait_observed_ms`.
    Admitted {
        wait_predicted_ms: f64,
        wait_observed_ms: f64,
    },
}

/// Wait-versus-load gate for entries the `on_queue` trigger parked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefetchGate {
    /// The predicted wait must cover `margin` times the predicted load.
    pub margin: f64,
    /// A partial hint shorter than this many pages is not worth sending.
    pub min_pages: u32,
}

impl Default for PrefetchGate {
    fn default() -> Self {
        Self {
            margin: 1.2,
            min_pages: 16,
        }
    }
}

/// Installed once per scheduler; see the module docs.
#[derive(Clone)]
pub struct PrefetchTrigger {
    /// How long before an entry's `due_at` the hint fires.
    pub lead: Duration,
    /// Fire at enqueue for every parked entry, deadline or not: a request that
    /// is waiting anyway can have its prefix loaded while it waits.
    pub on_queue: bool,
    /// Size `on_queue` hints to the predicted wait; `None` always sends the full prefix.
    pub gate: Option<PrefetchGate>,
    pub sink: Arc<dyn PrefetchHintSink>,
}

/// Decide how many of `missing_pages` to send when the entry is predicted to wait
/// `wait_ms` and `load_ms(n)` is the predicted time to load `n` pages (non-decreasing
/// in `n`). Returns the decision and the pages to send.
pub(crate) fn gate_decision(
    gate: PrefetchGate,
    wait_ms: f64,
    missing_pages: u32,
    load_ms: impl Fn(u32) -> f64,
) -> (PrefetchDecision, u32) {
    let budget_ms = wait_ms / gate.margin.max(f64::EPSILON);
    if load_ms(missing_pages) <= budget_ms {
        return (PrefetchDecision::Full, missing_pages);
    }
    // First page count whose load misses the budget; every smaller count fits.
    let (mut lo, mut hi) = (0u32, missing_pages);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if load_ms(mid) <= budget_ms {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let pages = lo.saturating_sub(1);
    if pages == 0 || pages < gate.min_pages {
        (PrefetchDecision::Skip, 0)
    } else {
        (PrefetchDecision::Partial, pages)
    }
}

/// Prior wait rate until a class has [`WAIT_MIN_SAMPLES`] admissions: 1 ms per 40
/// tokens of work ahead, from the ~40k input tokens/s the reference deployment
/// prefilled while its queue was non-empty.
pub(crate) const PRIOR_WAIT_MS_PER_TOKEN: f64 = 1.0 / 40.0;
/// Admissions a class needs before its learned rate replaces the prior.
pub(crate) const WAIT_MIN_SAMPLES: u32 = 3;
/// EWMA weight of a new wait sample.
const WAIT_ALPHA: f64 = 0.2;

/// Per-class predictor of how long a parked entry waits per token of work ahead
/// of it: an EWMA of `wait_ms / tokens_ahead` over admitted entries.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WaitEstimator {
    rate_ms_per_token: f64,
    samples: u32,
}

impl Default for WaitEstimator {
    fn default() -> Self {
        Self {
            rate_ms_per_token: PRIOR_WAIT_MS_PER_TOKEN,
            samples: 0,
        }
    }
}

impl WaitEstimator {
    /// Learn from an admitted entry; an entry with nothing ahead of it says
    /// nothing about the rate.
    pub(crate) fn observe(&mut self, wait_ms: f64, tokens_ahead: usize) {
        if tokens_ahead == 0 || !wait_ms.is_finite() || wait_ms < 0.0 {
            return;
        }
        let rate = wait_ms / tokens_ahead as f64;
        if self.samples == 0 {
            self.rate_ms_per_token = rate;
        } else {
            self.rate_ms_per_token += WAIT_ALPHA * (rate - self.rate_ms_per_token);
        }
        self.samples = self.samples.saturating_add(1);
    }

    pub(crate) fn rate_ms_per_token(&self) -> f64 {
        if self.samples >= WAIT_MIN_SAMPLES {
            self.rate_ms_per_token
        } else {
            PRIOR_WAIT_MS_PER_TOKEN
        }
    }

    /// Predicted wait for an entry with `tokens_ahead` tokens of work ahead of it.
    pub(crate) fn estimate_ms(&self, tokens_ahead: usize) -> f64 {
        self.rate_ms_per_token() * tokens_ahead as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(fixed_ms: f64, per_page_ms: f64) -> impl Fn(u32) -> f64 {
        move |pages| fixed_ms + per_page_ms * pages as f64
    }

    #[test]
    fn gate_sends_everything_when_the_wait_covers_the_load() {
        let gate = PrefetchGate::default();
        // 40 + 0.3 * 124 = 77.2 ms; margin 1.2 needs a 92.64 ms wait.
        assert_eq!(
            gate_decision(gate, 93.0, 124, linear(40.0, 0.3)),
            (PrefetchDecision::Full, 124)
        );
        assert_ne!(
            gate_decision(gate, 92.0, 124, linear(40.0, 0.3)).0,
            PrefetchDecision::Full
        );
    }

    #[test]
    fn gate_trims_to_the_largest_prefix_that_lands_in_time() {
        let gate = PrefetchGate {
            margin: 1.0,
            min_pages: 16,
        };
        // Budget 70 ms: 40 + 0.3n <= 70 holds up to n = 100.
        assert_eq!(
            gate_decision(gate, 70.0, 124, linear(40.0, 0.3)),
            (PrefetchDecision::Partial, 100)
        );
        // Margin scales the budget: 84 ms / 1.2 = 70 ms again.
        let gate = PrefetchGate {
            margin: 1.2,
            min_pages: 16,
        };
        assert_eq!(
            gate_decision(gate, 84.0, 124, linear(40.0, 0.3)),
            (PrefetchDecision::Partial, 100)
        );
        // Exactly at the boundary the page still fits.
        assert_eq!(
            gate_decision(gate, 84.0, 124, linear(40.0, 0.25)),
            (PrefetchDecision::Partial, 120)
        );
    }

    #[test]
    fn gate_skips_below_the_minimum_pages_or_the_fixed_cost() {
        let gate = PrefetchGate {
            margin: 1.0,
            min_pages: 16,
        };
        // 40 + 0.3n <= 44 gives n = 13 < 16.
        assert_eq!(
            gate_decision(gate, 44.0, 124, linear(40.0, 0.3)),
            (PrefetchDecision::Skip, 0)
        );
        // 15 pages fit exactly; still under the minimum.
        assert_eq!(
            gate_decision(gate, 44.5, 124, linear(40.0, 0.3)),
            (PrefetchDecision::Skip, 0)
        );
        // 16 pages fit: the smallest partial.
        assert_eq!(
            gate_decision(gate, 44.8, 124, linear(40.0, 0.3)),
            (PrefetchDecision::Partial, 16)
        );
        // Not even the fixed cost fits, and a zero wait never sends.
        assert_eq!(
            gate_decision(gate, 30.0, 124, linear(40.0, 0.3)),
            (PrefetchDecision::Skip, 0)
        );
        assert_eq!(
            gate_decision(gate, 0.0, 124, linear(40.0, 0.3)),
            (PrefetchDecision::Skip, 0)
        );
        // A minimum of zero never sends an empty hint either.
        let loose = PrefetchGate {
            margin: 1.0,
            min_pages: 0,
        };
        assert_eq!(
            gate_decision(loose, 40.1, 124, linear(40.0, 0.3)),
            (PrefetchDecision::Skip, 0)
        );
        assert_eq!(
            gate_decision(loose, 40.3, 124, linear(40.0, 0.3)),
            (PrefetchDecision::Partial, 1)
        );
    }

    #[test]
    fn wait_estimator_uses_the_prior_until_three_samples_then_learns() {
        let mut waits = WaitEstimator::default();
        assert_eq!(waits.estimate_ms(4_000), 100.0, "1 ms per 40 tokens");

        // Nothing ahead teaches nothing.
        waits.observe(500.0, 0);
        assert_eq!(waits.estimate_ms(4_000), 100.0);

        // A slower class: 1 ms per 10 tokens.
        waits.observe(400.0, 4_000);
        waits.observe(200.0, 2_000);
        assert_eq!(
            waits.estimate_ms(4_000),
            100.0,
            "two samples keep the prior"
        );
        waits.observe(800.0, 8_000);
        assert!((waits.estimate_ms(4_000) - 400.0).abs() < 1e-9);

        // The EWMA tracks a change in rate.
        for _ in 0..40 {
            waits.observe(100.0, 4_000);
        }
        assert!((waits.rate_ms_per_token() - 0.025).abs() < 1e-3);
    }
}
