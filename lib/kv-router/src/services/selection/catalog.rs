// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use crate::identity::RoutingPartitionId;
use crate::protocols::{WorkerId, WorkerWithDpRank};

use super::error::SelectionError;
use super::types::{
    SelectionWorkerConfig, WorkerCatalogRecord, WorkerLifecycle, WorkerPatchRequest, WorkerRequest,
};

/// A worker's registration lease: it must heartbeat before `deadline`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Lease {
    pub ttl: Duration,
    pub deadline: Instant,
}

#[derive(Debug, Default)]
pub(super) struct WorkerCatalog {
    workers: RwLock<HashMap<WorkerId, WorkerCatalogRecord>>,
    leases: RwLock<HashMap<WorkerId, Lease>>,
}

pub(super) fn lease_ttl(ttl_secs: f64) -> Result<Duration, SelectionError> {
    if !ttl_secs.is_finite() || ttl_secs <= 0.0 {
        return Err(SelectionError::BadRequest(
            "ttl_secs must be a positive number of seconds".to_string(),
        ));
    }
    Ok(Duration::from_secs_f64(ttl_secs))
}

impl WorkerCatalog {
    pub(super) fn upsert(
        &self,
        req: WorkerRequest,
    ) -> (Option<WorkerCatalogRecord>, WorkerCatalogRecord) {
        let lease = req.ttl_secs.and_then(|ttl_secs| lease_ttl(ttl_secs).ok());
        let mut workers = self.workers.write();
        let previous = workers.get(&req.worker_id).cloned();
        let record = WorkerCatalogRecord::new(req);
        workers.insert(record.worker_id, record.clone());
        let mut leases = self.leases.write();
        match lease {
            Some(ttl) => {
                leases.insert(
                    record.worker_id,
                    Lease {
                        ttl,
                        deadline: Instant::now() + ttl,
                    },
                );
            }
            None => {
                leases.remove(&record.worker_id);
            }
        }
        (previous, record)
    }

    /// Renew `worker_id`'s lease, optionally changing its TTL. A worker that
    /// registered without a TTL needs one here to start a lease.
    pub(super) fn heartbeat(
        &self,
        worker_id: WorkerId,
        ttl_secs: Option<f64>,
        now: Instant,
    ) -> Result<(WorkerCatalogRecord, Lease), SelectionError> {
        let mut workers = self.workers.write();
        let Some(record) = workers.get_mut(&worker_id) else {
            return Err(SelectionError::NotFound(format!(
                "worker {worker_id} not found"
            )));
        };
        let mut leases = self.leases.write();
        let ttl = match (ttl_secs, leases.get(&worker_id)) {
            (Some(ttl_secs), _) => lease_ttl(ttl_secs)?,
            (None, Some(lease)) => lease.ttl,
            (None, None) => {
                return Err(SelectionError::BadRequest(format!(
                    "worker {worker_id} has no lease; supply ttl_secs to start one"
                )));
            }
        };
        let lease = Lease {
            ttl,
            deadline: now + ttl,
        };
        leases.insert(worker_id, lease);
        record.ttl_secs = Some(ttl.as_secs_f64());
        Ok((record.clone(), lease))
    }

    #[cfg(test)]
    pub(super) fn lease(&self, worker_id: WorkerId) -> Option<Lease> {
        self.leases.read().get(&worker_id).copied()
    }

    /// Workers whose lease deadline has passed, oldest first. Their leases are
    /// removed so an expiry is reported once; the record stays until deleted.
    pub(super) fn take_expired_leases(&self, now: Instant) -> Vec<WorkerId> {
        let mut leases = self.leases.write();
        let mut expired: Vec<(Instant, WorkerId)> = leases
            .iter()
            .filter(|(_, lease)| lease.deadline <= now)
            .map(|(worker_id, lease)| (lease.deadline, *worker_id))
            .collect();
        expired.sort();
        for (_, worker_id) in &expired {
            leases.remove(worker_id);
        }
        expired
            .into_iter()
            .map(|(_, worker_id)| worker_id)
            .collect()
    }

    /// Earliest lease deadline, for sizing the sweep interval.
    pub(super) fn next_lease_deadline(&self) -> Option<Instant> {
        self.leases
            .read()
            .values()
            .map(|lease| lease.deadline)
            .min()
    }

    pub(super) fn patch(
        &self,
        worker_id: WorkerId,
        patch: WorkerPatchRequest,
    ) -> Result<(WorkerCatalogRecord, WorkerCatalogRecord), SelectionError> {
        let mut workers = self.workers.write();
        let Some(record) = workers.get_mut(&worker_id) else {
            return Err(SelectionError::NotFound(format!(
                "worker {worker_id} not found"
            )));
        };
        let previous = record.clone();
        record.apply_patch(patch);
        record.lifecycle = WorkerLifecycle::Incomplete;
        record.not_schedulable_reasons.clear();
        Ok((previous, record.clone()))
    }

    pub(super) fn get(&self, worker_id: WorkerId) -> Option<WorkerCatalogRecord> {
        self.workers.read().get(&worker_id).cloned()
    }

    pub(super) fn set_lifecycle(
        &self,
        worker_id: WorkerId,
        lifecycle: WorkerLifecycle,
        reasons: Vec<String>,
    ) -> Option<WorkerCatalogRecord> {
        let mut workers = self.workers.write();
        let record = workers.get_mut(&worker_id)?;
        record.lifecycle = lifecycle;
        record.not_schedulable_reasons = reasons;
        Some(record.clone())
    }

    pub(super) fn list(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<WorkerCatalogRecord> {
        let mut records: Vec<_> = self
            .workers
            .read()
            .values()
            .filter(|record| {
                model_name.is_none_or(|model_name| record.model_name == model_name)
                    && routing_group
                        .is_none_or(|routing_group| record.routing_group == routing_group)
            })
            .cloned()
            .collect();
        records.sort_by_key(|record| {
            (
                record.model_name.clone(),
                record.routing_group.clone(),
                record.worker_id,
            )
        });
        records
    }

    pub(super) fn has_schedulable_for_key(&self, key: &RoutingPartitionId) -> bool {
        self.workers.read().values().any(|record| {
            record.lifecycle == WorkerLifecycle::Schedulable
                && record.model_name == key.model_name
                && record.routing_group == key.routing_group
        })
    }

    pub(super) fn schedulable_worker_ids_for_key(&self, key: &RoutingPartitionId) -> Vec<WorkerId> {
        self.workers
            .read()
            .values()
            .filter(|record| {
                record.lifecycle == WorkerLifecycle::Schedulable
                    && record.model_name == key.model_name
                    && record.routing_group == key.routing_group
            })
            .map(|record| record.worker_id)
            .collect()
    }

    /// `total_kv_blocks` published by a schedulable worker in `key`'s partition.
    pub(super) fn total_kv_blocks(
        &self,
        worker_id: WorkerId,
        key: &RoutingPartitionId,
    ) -> Option<u64> {
        let workers = self.workers.read();
        let record = workers.get(&worker_id)?;
        (record.lifecycle == WorkerLifecycle::Schedulable
            && record.model_name == key.model_name
            && record.routing_group == key.routing_group)
            .then_some(record.total_kv_blocks)
            .flatten()
    }

    /// Whether any schedulable worker in `key`'s partition can consume router
    /// hints. Worker-level metadata, so one representative rank suffices.
    pub(super) fn has_router_hint_capable_workers(&self, key: &RoutingPartitionId) -> bool {
        self.workers.read().values().any(|record| {
            record.lifecycle == WorkerLifecycle::Schedulable
                && record.model_name == key.model_name
                && record.routing_group == key.routing_group
                && record
                    .router_hint_worker_type
                    .as_deref()
                    .is_some_and(|worker_type| !worker_type.is_empty())
        })
    }

    pub(super) fn scheduler_configs_for_key(
        &self,
        key: &RoutingPartitionId,
    ) -> HashMap<WorkerId, SelectionWorkerConfig> {
        self.workers
            .read()
            .values()
            .filter(|record| {
                record.lifecycle == WorkerLifecycle::Schedulable
                    && record.model_name == key.model_name
                    && record.routing_group == key.routing_group
            })
            .filter_map(|record| {
                record
                    .scheduler_config()
                    .map(|config| (record.worker_id, config))
            })
            .collect()
    }

    pub(super) fn schedulable_count(&self) -> usize {
        self.workers
            .read()
            .values()
            .filter(|record| record.lifecycle == WorkerLifecycle::Schedulable)
            .count()
    }

    pub(super) fn schedulable_endpoint(
        &self,
        worker_id: WorkerId,
        key: &RoutingPartitionId,
    ) -> Option<String> {
        let workers = self.workers.read();
        let record = workers.get(&worker_id)?;
        if record.lifecycle != WorkerLifecycle::Schedulable
            || record.model_name != key.model_name
            || record.routing_group != key.routing_group
        {
            return None;
        }
        record.endpoint.clone()
    }

    pub(super) fn schedulable_worker_endpoint(
        &self,
        worker: WorkerWithDpRank,
        key: &RoutingPartitionId,
    ) -> Option<String> {
        let workers = self.workers.read();
        let record = workers.get(&worker.worker_id)?;
        if record.lifecycle != WorkerLifecycle::Schedulable
            || record.model_name != key.model_name
            || record.routing_group != key.routing_group
            || !record.dp_ranks().any(|rank| rank == worker.dp_rank)
        {
            return None;
        }
        record.endpoint.clone()
    }
}
