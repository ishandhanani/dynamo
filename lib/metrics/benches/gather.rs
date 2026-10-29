// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a scrape costs when stage counters are read at gather time.
//!
//! `collector` builds a 32-series counter family from N attached sources per scrape;
//! `registered` is the same family as one directly registered `IntCounterVec`, the shape
//! the counters had when stages owned Prometheus objects.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dynamo_metrics::collect::{SourceSet, descs_of};
use prometheus::core::{Collector, Desc};
use prometheus::proto::MetricFamily;
use prometheus::{IntCounterVec, Opts, Registry, TextEncoder};

const SERIES: usize = 32;
const LABELS: &[&str] = &["kind"];

struct Source([AtomicU64; SERIES]);

#[derive(Clone)]
struct StageCounters {
    opts: Opts,
    descs: Vec<Desc>,
    sources: SourceSet<Source>,
}

impl StageCounters {
    fn new() -> Self {
        let opts = Opts::new("bench_stage_events_total", "events");
        let descs = descs_of(&IntCounterVec::new(opts.clone(), LABELS).unwrap());
        Self {
            opts,
            descs,
            sources: SourceSet::new(),
        }
    }
}

impl Collector for StageCounters {
    fn desc(&self) -> Vec<&Desc> {
        self.descs.iter().collect()
    }

    fn collect(&self) -> Vec<MetricFamily> {
        let vec = IntCounterVec::new(self.opts.clone(), LABELS).unwrap();
        let sources = self.sources.live(|_| true);
        let labels: Vec<String> = (0..SERIES).map(|i| format!("k{i}")).collect();
        for (i, label) in labels.iter().enumerate() {
            let counter = vec.with_label_values(&[label.as_str()]);
            for source in &sources {
                counter.inc_by(source.0[i].load(Ordering::Relaxed));
            }
        }
        vec.collect()
    }
}

fn gather(c: &mut Criterion) {
    let mut group = c.benchmark_group("gather");
    for sources in [1usize, 16, 256] {
        let registry = Registry::new();
        let collector = StageCounters::new();
        registry.register(Box::new(collector.clone())).unwrap();
        let handles: Vec<Arc<Source>> = (0..sources)
            .map(|n| {
                let source = Arc::new(Source(std::array::from_fn(|i| {
                    AtomicU64::new((n * i) as u64)
                })));
                collector.sources.attach(&source);
                source
            })
            .collect();
        group.bench_with_input(BenchmarkId::new("collector", sources), &sources, |b, _| {
            b.iter(|| registry.gather())
        });
        group.bench_with_input(
            BenchmarkId::new("collector_encoded", sources),
            &sources,
            |b, _| {
                b.iter(|| {
                    TextEncoder::new()
                        .encode_to_string(&registry.gather())
                        .unwrap()
                })
            },
        );
        drop(handles);
    }
    let registry = Registry::new();
    let vec = IntCounterVec::new(Opts::new("bench_stage_events_total", "events"), LABELS).unwrap();
    for i in 0..SERIES {
        vec.with_label_values(&[&format!("k{i}")]).inc_by(i as u64);
    }
    registry.register(Box::new(vec)).unwrap();
    group.bench_function("registered", |b| b.iter(|| registry.gather()));
    group.finish();
}

criterion_group!(benches, gather);
criterion_main!(benches);
