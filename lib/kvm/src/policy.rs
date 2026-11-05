// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How long to pin a session's pages, and how much they are worth.
//!
//! A [`PinPolicy`] turns what the host knows when a response calls a tool into a
//! [`PinDecision`]. It is three replaceable parts:
//!
//! - [`ToolKeyer`]: which bucket a tool call's duration belongs to. The default,
//!   [`ToolNameKeyer`], is the tool's name; a classifier of shell commands can
//!   replace it.
//! - [`DurationPredictor`]: learns tool durations per key and predicts the next.
//!   The default, [`QuantilePredictor`], keeps a bounded log-scale histogram per
//!   key and answers a quantile of it.
//! - [`PriorityScorer`]: what keeping the session's pages is worth. The default,
//!   [`RecomputeTokensScorer`], scores the tokens the session would recompute
//!   (ThunderAgent-style in-flight tokens), on a log scale.
//!
//! The host reports a tool call's start and end ([`DefaultPinPolicy::tool_started`],
//! [`DefaultPinPolicy::tool_ended`]); nothing here reads a clock or configuration.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;

/// What the host knows about a tool call when the response makes it.
#[derive(Debug, Clone, Default)]
pub struct ToolCallInfo<'a> {
    /// The tool's name, as the model called it.
    pub name: Option<&'a str>,
    /// The call's arguments as the model wrote them, when the keyer wants them.
    pub arguments: Option<&'a str>,
    /// A harness's own estimate (for example Shed's `expected_duration_ms`); it
    /// wins over the prediction.
    pub expected_duration: Option<Duration>,
}

/// What the host knows about the session. Every field is optional so new
/// signals can be added without changing scorers that ignore them.
#[derive(Debug, Clone, Default)]
pub struct SessionSignals {
    /// Tokens the session's next request would recompute if its pages were lost.
    pub recompute_tokens: Option<u64>,
    /// Requests the session has made so far.
    pub turns: Option<u32>,
    /// An externally assigned weight (an RL rollout's value, a tenant class).
    pub weight: Option<f64>,
}

/// How to pin: for how long, and how much the pages are worth. Higher priority is
/// kept longer when pinned pages must be evicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinDecision {
    pub ttl: Duration,
    pub priority: u32,
}

pub trait ToolKeyer: Send + Sync {
    /// The bucket for this call's duration; `None` falls back to the global one.
    fn key(&self, call: &ToolCallInfo<'_>) -> Option<String>;
}

pub trait DurationPredictor: Send + Sync {
    fn observe(&self, key: Option<&str>, duration: Duration);
    /// The predicted duration for a call keyed `key`, if there is evidence.
    fn predict(&self, key: Option<&str>) -> Option<Duration>;
}

pub trait PriorityScorer: Send + Sync {
    fn score(&self, session: &SessionSignals) -> u32;
}

pub trait PinPolicy: Send + Sync {
    fn decide(&self, call: &ToolCallInfo<'_>, session: &SessionSignals) -> PinDecision;
}

/// Keys a call by its tool name.
#[derive(Debug, Default)]
pub struct ToolNameKeyer;

impl ToolKeyer for ToolNameKeyer {
    fn key(&self, call: &ToolCallInfo<'_>) -> Option<String> {
        call.name.map(str::to_string)
    }
}

/// Bounds and shape of [`QuantilePredictor`].
#[derive(Debug, Clone)]
pub struct QuantileConfig {
    /// The quantile answered, for example `0.9`.
    pub quantile: f64,
    /// Observations a key needs before its own histogram is used.
    pub min_samples: u64,
    /// Keys tracked; past this, the least recently observed key is dropped.
    pub max_keys: usize,
}

impl Default for QuantileConfig {
    fn default() -> Self {
        Self {
            quantile: 0.9,
            min_samples: 5,
            max_keys: 1024,
        }
    }
}

/// Log-scale buckets from 1 ms: bucket `i` holds durations in `[2^(i/4), 2^((i+1)/4))`
/// ms, about 19% wide, up to about 2^28 ms (3 days).
const BUCKETS: usize = 112;

#[derive(Debug, Clone)]
struct Histogram {
    counts: [u64; BUCKETS],
    total: u64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            counts: [0; BUCKETS],
            total: 0,
        }
    }

    fn bucket(duration: Duration) -> usize {
        let ms = duration.as_secs_f64() * 1000.0;
        if ms <= 1.0 {
            return 0;
        }
        ((ms.log2() * 4.0) as usize).min(BUCKETS - 1)
    }

    fn observe(&mut self, duration: Duration) {
        self.counts[Self::bucket(duration)] += 1;
        self.total += 1;
    }

    /// The upper edge of the bucket holding quantile `q`, so the answer errs long.
    fn quantile(&self, q: f64) -> Option<Duration> {
        if self.total == 0 {
            return None;
        }
        let rank = ((self.total as f64) * q.clamp(0.0, 1.0)).ceil().max(1.0) as u64;
        let mut seen = 0;
        for (index, count) in self.counts.iter().enumerate() {
            seen += count;
            if seen >= rank {
                let upper_ms = 2f64.powf((index + 1) as f64 / 4.0);
                return Some(Duration::from_secs_f64(upper_ms / 1000.0));
            }
        }
        None
    }
}

struct QuantileState {
    global: Histogram,
    keys: HashMap<String, Histogram>,
    /// Least recently observed first, for eviction past `max_keys`.
    order: VecDeque<String>,
}

/// Per-key duration histograms answering a quantile; a key with too few samples
/// falls back to all keys together.
pub struct QuantilePredictor {
    config: QuantileConfig,
    state: Mutex<QuantileState>,
}

impl QuantilePredictor {
    pub fn new(config: QuantileConfig) -> Self {
        Self {
            config,
            state: Mutex::new(QuantileState {
                global: Histogram::new(),
                keys: HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }
}

impl DurationPredictor for QuantilePredictor {
    fn observe(&self, key: Option<&str>, duration: Duration) {
        let mut state = self.state.lock().unwrap();
        state.global.observe(duration);
        let Some(key) = key else {
            return;
        };
        if let Some(position) = state.order.iter().position(|k| k == key) {
            let key = state.order.remove(position).unwrap();
            state.order.push_back(key);
        } else {
            while state.order.len() >= self.config.max_keys.max(1) {
                if let Some(old) = state.order.pop_front() {
                    state.keys.remove(&old);
                }
            }
            state.order.push_back(key.to_string());
        }
        state
            .keys
            .entry(key.to_string())
            .or_insert_with(Histogram::new)
            .observe(duration);
    }

    fn predict(&self, key: Option<&str>) -> Option<Duration> {
        let state = self.state.lock().unwrap();
        let own = key
            .and_then(|key| state.keys.get(key))
            .filter(|histogram| histogram.total >= self.config.min_samples);
        let histogram = own.unwrap_or(&state.global);
        if histogram.total < self.config.min_samples {
            return None;
        }
        histogram.quantile(self.config.quantile)
    }
}

/// Scores the tokens a session would recompute, `log2(tokens / unit)`, capped.
#[derive(Debug, Clone)]
pub struct RecomputeTokensScorer {
    /// Tokens worth one priority step at the bottom of the scale.
    pub unit_tokens: u64,
    pub max_priority: u32,
}

impl Default for RecomputeTokensScorer {
    fn default() -> Self {
        Self {
            unit_tokens: 1024,
            max_priority: 16,
        }
    }
}

impl PriorityScorer for RecomputeTokensScorer {
    fn score(&self, session: &SessionSignals) -> u32 {
        let tokens = session.recompute_tokens.unwrap_or(0);
        let units = tokens / self.unit_tokens.max(1);
        let mut priority = if units == 0 {
            0
        } else {
            64 - units.leading_zeros()
        };
        if let Some(weight) = session.weight {
            priority = (priority as f64 * weight.max(0.0)).round() as u32;
        }
        priority.clamp(1, self.max_priority.max(1))
    }
}

/// Bounds on the TTL [`DefaultPinPolicy`] turns a predicted duration into.
#[derive(Debug, Clone)]
pub struct TtlBounds {
    /// Used until there is evidence for a call.
    pub default: Duration,
    /// Multiplies the prediction, which is already a high quantile.
    pub safety_factor: f64,
    /// Added after the factor, for the client's turnaround and the next prefill.
    pub margin: Duration,
    pub min: Duration,
    pub max: Duration,
}

impl Default for TtlBounds {
    fn default() -> Self {
        Self {
            default: Duration::from_secs(120),
            safety_factor: 1.5,
            margin: Duration::from_secs(5),
            min: Duration::from_secs(5),
            max: Duration::from_secs(30 * 60),
        }
    }
}

/// Keyer, predictor and scorer combined; each part is replaceable.
pub struct DefaultPinPolicy {
    keyer: Box<dyn ToolKeyer>,
    predictor: Box<dyn DurationPredictor>,
    scorer: Box<dyn PriorityScorer>,
    bounds: TtlBounds,
}

impl Default for DefaultPinPolicy {
    fn default() -> Self {
        Self::new(
            Box::new(ToolNameKeyer),
            Box::new(QuantilePredictor::new(QuantileConfig::default())),
            Box::new(RecomputeTokensScorer::default()),
            TtlBounds::default(),
        )
    }
}

impl DefaultPinPolicy {
    pub fn new(
        keyer: Box<dyn ToolKeyer>,
        predictor: Box<dyn DurationPredictor>,
        scorer: Box<dyn PriorityScorer>,
        bounds: TtlBounds,
    ) -> Self {
        Self {
            keyer,
            predictor,
            scorer,
            bounds,
        }
    }

    /// The key a call's duration is recorded under; the host keeps it with the
    /// session until the call ends.
    pub fn tool_started(&self, call: &ToolCallInfo<'_>) -> Option<String> {
        self.keyer.key(call)
    }

    /// Record how long a call keyed `key` took, as the client saw it.
    pub fn tool_ended(&self, key: Option<&str>, duration: Duration) {
        self.predictor.observe(key, duration);
    }

    fn ttl(&self, call: &ToolCallInfo<'_>) -> Duration {
        let expected = call.expected_duration.or_else(|| {
            let key = self.keyer.key(call);
            self.predictor.predict(key.as_deref())
        });
        let Some(expected) = expected else {
            return self.bounds.default;
        };
        let scaled = expected.mul_f64(self.bounds.safety_factor.max(1.0)) + self.bounds.margin;
        scaled.clamp(self.bounds.min, self.bounds.max)
    }
}

impl PinPolicy for DefaultPinPolicy {
    fn decide(&self, call: &ToolCallInfo<'_>, session: &SessionSignals) -> PinDecision {
        PinDecision {
            ttl: self.ttl(call),
            priority: self.scorer.score(session),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str) -> ToolCallInfo<'_> {
        ToolCallInfo {
            name: Some(name),
            ..ToolCallInfo::default()
        }
    }

    #[test]
    fn cold_start_uses_the_default_ttl() {
        let policy = DefaultPinPolicy::default();
        let decision = policy.decide(&call("Bash"), &SessionSignals::default());
        assert_eq!(decision.ttl, Duration::from_secs(120));
        assert_eq!(decision.priority, 1);
    }

    #[test]
    fn ttl_follows_the_tool_and_falls_back_to_all_tools() {
        let policy = DefaultPinPolicy::default();
        for _ in 0..20 {
            policy.tool_ended(Some("Read"), Duration::from_millis(50));
            policy.tool_ended(Some("Bash"), Duration::from_secs(20));
        }
        let read = policy.decide(&call("Read"), &SessionSignals::default()).ttl;
        let bash = policy.decide(&call("Bash"), &SessionSignals::default()).ttl;
        // Read: about 50 ms x 1.5 + the 5 s margin. Bash: about 20 s x 1.5 + 5 s.
        assert!(
            read >= Duration::from_secs(5) && read < Duration::from_millis(5_200),
            "{read:?}"
        );
        assert!(
            bash >= Duration::from_secs(35) && bash <= Duration::from_secs(45),
            "{bash:?}"
        );
        // An unseen tool uses every tool's p90, which is Bash-like here.
        let unseen = policy.decide(&call("Grep"), &SessionSignals::default()).ttl;
        assert!(unseen >= Duration::from_secs(35), "{unseen:?}");
    }

    #[test]
    fn a_harness_estimate_wins_and_is_bounded() {
        let policy = DefaultPinPolicy::default();
        let decision = policy.decide(
            &ToolCallInfo {
                name: Some("Bash"),
                expected_duration: Some(Duration::from_secs(4 * 3600)),
                ..ToolCallInfo::default()
            },
            &SessionSignals::default(),
        );
        assert_eq!(decision.ttl, Duration::from_secs(30 * 60));
    }

    #[test]
    fn priority_grows_with_recompute_tokens_and_weight() {
        let scorer = RecomputeTokensScorer::default();
        let score = |tokens, weight| {
            scorer.score(&SessionSignals {
                recompute_tokens: Some(tokens),
                weight,
                ..SessionSignals::default()
            })
        };
        assert_eq!(score(500, None), 1);
        assert!(score(16_000, None) < score(128_000, None));
        // 16,000 tokens is 15 units of 1,024: priority floor(log2(15)) + 1 = 4.
        assert_eq!(score(16_000, None), 4);
        assert_eq!(score(16_000, Some(2.0)), 8);
        assert_eq!(score(u64::MAX, None), 16);
    }

    #[test]
    fn predictor_keeps_a_bounded_number_of_keys() {
        let predictor = QuantilePredictor::new(QuantileConfig {
            max_keys: 2,
            min_samples: 1,
            ..QuantileConfig::default()
        });
        predictor.observe(Some("a"), Duration::from_secs(1));
        predictor.observe(Some("b"), Duration::from_secs(1));
        predictor.observe(Some("c"), Duration::from_secs(60));
        // "a" was dropped; it falls back to the global histogram, which has all three.
        let state = predictor.state.lock().unwrap();
        assert!(!state.keys.contains_key("a"));
        assert_eq!(state.global.total, 3);
    }

    #[test]
    fn histogram_quantile_errs_long_within_a_bucket() {
        let mut histogram = Histogram::new();
        for _ in 0..10 {
            histogram.observe(Duration::from_secs(10));
        }
        let p90 = histogram.quantile(0.9).unwrap();
        assert!(
            p90 >= Duration::from_secs(10) && p90 < Duration::from_secs(12),
            "{p90:?}"
        );
    }
}
