// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

const TTL: Duration = Duration::from_secs(10);

fn table() -> SessionAffinity {
    SessionAffinity::with_config(SessionAffinityConfig::new(TTL)).expect("affinity table")
}

fn initialize(table: &SessionAffinity) -> AffinityInitialization {
    match table.try_acquire("s", None).expect("acquire") {
        AcquireStep::Held(Hold::Initialize(init)) => init,
        AcquireStep::Held(Hold::Bound { .. }) => panic!("session is already bound"),
        AcquireStep::Wait(_) => panic!("session is being initialized"),
    }
}

#[tokio::test(start_paused = true)]
async fn joined_lease_released_before_the_commit_is_not_counted() {
    let table = table();
    let target = AffinityTarget::new(2, Some(0));
    let init = initialize(&table);
    let joined = table
        .join_initializing("s", target)
        .expect("join an initializing session");
    assert_eq!(table.lease_count("s"), Some(2));

    // The joiner finishes while the initializer is still queued.
    drop(joined);
    assert_eq!(
        table.lease_count("s"),
        Some(1),
        "early release leaves the initializer's use"
    );

    tokio::time::advance(TTL + Duration::from_secs(1)).await;
    assert_eq!(table.query_target("s", None).expect("query"), Some(target));

    let lease = init.commit(target).expect("commit");
    assert_eq!(
        table.lease_count("s"),
        Some(1),
        "commit counts only the initializer"
    );
    drop(lease);
    assert_eq!(table.lease_count("s"), Some(0));
    tokio::time::advance(TTL + Duration::from_secs(1)).await;
    assert_eq!(
        table.query_target("s", None).expect("query"),
        None,
        "a binding with no live leases idles out"
    );
}

#[tokio::test(start_paused = true)]
async fn joined_lease_counts_once_the_competing_initialization_commits() {
    let table = table();
    let target = AffinityTarget::new(2, Some(0));
    let init = initialize(&table);
    let AcquireStep::Wait(waiter) = table.try_acquire("s", None).expect("waiter") else {
        panic!("initialization must have a waiter");
    };
    let joined = table
        .join_initializing("s", target)
        .expect("join an initializing session");
    tokio::time::timeout(Duration::from_millis(1), waiter)
        .await
        .expect("joining must wake existing waiters");
    assert_eq!(table.lease_count("s"), Some(2));
    assert!(
        table.join_initializing("x", target).is_none(),
        "no join without an initializer"
    );

    let lease = init.commit(target).expect("commit");
    assert_eq!(table.lease_count("s"), Some(2));
    assert_eq!(table.query_target("s", None).expect("query"), Some(target));

    drop(lease);
    assert_eq!(table.lease_count("s"), Some(1));
    // The joiner's release is a use of the binding: it refreshes the idle
    // deadline, so the binding outlives the initializer's own TTL.
    tokio::time::advance(TTL - Duration::from_secs(1)).await;
    drop(joined);
    assert_eq!(table.lease_count("s"), Some(0));
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(
        table.query_target("s", None).expect("query"),
        Some(target),
        "joined lease release must refresh the idle deadline"
    );
    tokio::time::advance(TTL).await;
    assert_eq!(table.query_target("s", None).expect("query"), None);
    assert!(
        matches!(
            table.try_acquire("s", None).expect("acquire"),
            AcquireStep::Held(Hold::Initialize(_))
        ),
        "an idled binding re-initializes"
    );
}

#[tokio::test(start_paused = true)]
async fn joined_binding_survives_the_initializer_dropping() {
    let table = table();
    let init = initialize(&table);
    let target = AffinityTarget::new(2, Some(0));
    let joined = table
        .join_initializing("s", target)
        .expect("join an initializing session");
    assert_eq!(table.entry_count(), 1);

    drop(init);
    assert_eq!(table.entry_count(), 1);
    assert_eq!(table.query_target("s", None).expect("query"), Some(target));
    assert_eq!(table.lease_count("s"), Some(1));
    drop(joined);
    assert_eq!(table.lease_count("s"), Some(0));
    tokio::time::advance(TTL + Duration::from_secs(1)).await;
    assert_eq!(table.query_target("s", None).expect("query"), None);

    // Once the completed booking's TTL expires, another target can bind.
    let init = initialize(&table);
    let lease = init
        .commit(AffinityTarget::new(3, Some(0)))
        .expect("commit");
    assert_eq!(table.lease_count("s"), Some(1));
    drop(lease);
    assert_eq!(table.lease_count("s"), Some(0));
}

#[test]
fn config_validates_the_ttl_range_once_for_every_host() {
    for secs in [-1.0, 0.0, 0.5, f64::NAN, f64::INFINITY, 31_536_001.0, 1e300] {
        let error = SessionAffinityConfig::ttl_from_secs_f64(secs)
            .expect_err("out-of-range or non-finite TTLs are rejected");
        assert!(matches!(error, AffinityError::InvalidArgument(_)));
        assert!(
            error.to_string().contains("session affinity TTL"),
            "{error}"
        );
    }
    for secs in [1.0, 1.5, 31_536_000.0] {
        let ttl = SessionAffinityConfig::ttl_from_secs_f64(secs).expect("in range");
        SessionAffinityConfig::new(ttl).validate().expect("valid");
    }
    assert!(
        SessionAffinityConfig::new(Duration::ZERO)
            .validate()
            .is_err()
    );
    assert!(
        SessionAffinityConfig {
            max_entries: 0,
            ..SessionAffinityConfig::new(TTL)
        }
        .validate()
        .is_err()
    );
    assert!(
        SessionAffinityConfig {
            max_session_id_bytes: 0,
            ..SessionAffinityConfig::new(TTL)
        }
        .validate()
        .is_err()
    );
}

#[tokio::test]
async fn initializer_cannot_change_the_joined_rank_in_hard_mode() {
    let table = table();
    let init = initialize(&table);
    let target = AffinityTarget::new(2, Some(0));
    let joined = table.join_initializing("s", target).expect("join");
    assert!(init.commit(AffinityTarget::new(2, Some(1))).is_err());
    assert_eq!(table.query_target("s", None).expect("query"), Some(target));
    assert_eq!(table.lease_count("s"), Some(1));
    drop(joined);
    assert_eq!(table.lease_count("s"), Some(0));
}

#[tokio::test]
async fn soft_initializer_follows_dispatch_and_keeps_worker_only_affinity() {
    let table = SessionAffinity::with_config(
        SessionAffinityConfig::new(TTL).with_mode(SessionAffinityMode::Soft),
    )
    .expect("table");
    let init = initialize(&table);
    let joined = table
        .join_initializing("s", AffinityTarget::new(2, None))
        .expect("join");
    let lease = init
        .commit(AffinityTarget::new(3, Some(0)))
        .expect("commit");
    assert_eq!(
        table.query_target("s", None).expect("query"),
        Some(AffinityTarget::new(3, None))
    );
    assert_eq!(table.lease_count("s"), Some(2));
    drop(lease);
    drop(joined);
    assert_eq!(table.lease_count("s"), Some(0));
}

fn target(worker_id: u64, dp_rank: Option<u32>) -> AffinityTarget {
    AffinityTarget::new(worker_id, dp_rank)
}

async fn bind(table: &SessionAffinity, target: AffinityTarget) {
    let hold = table.acquire("s", None).await.unwrap();
    drop(table.commit(hold, target).unwrap());
}

async fn assert_binding_expires_after_refreshed_ttl(table: &SessionAffinity) {
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        table.query_target("s", None).unwrap(),
        Some(target(7, Some(0)))
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(table.query_target("s", None).unwrap(), None);
}

fn enable_replica(
    table: &SessionAffinity,
    writer_id: u64,
    capacity: usize,
) -> tokio::sync::mpsc::Receiver<AffinityBindingEvent> {
    let (tx, rx) = tokio::sync::mpsc::channel(capacity);
    assert!(table.enable_replication(
        writer_id,
        replica_sink(crate::RoutingPartitionId::new("model", "group"), tx)
    ));
    rx
}

#[tokio::test(start_paused = true)]
async fn session_affinity_initialization_is_atomic() {
    let affinity = table();
    let first = affinity.acquire("s", None).await.unwrap();
    let Hold::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_affinity = affinity.clone();
    let waiter = tokio::spawn(async move { waiter_affinity.acquire("s", None).await });
    affinity.wait_for_initializing_waiter().await;
    assert!(!waiter.is_finished());

    let first_lease = first.commit(target(7, Some(0))).unwrap();
    let second = waiter.await.unwrap().unwrap();
    let Hold::Bound {
        target: second_target,
        lease: second_lease,
    } = second
    else {
        panic!("waiter must acquire the committed binding");
    };
    assert_eq!(second_target, target(7, Some(0)));
    drop(first_lease);
    drop(second_lease);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_initializer_cancellation_wakes_waiter() {
    let affinity = table();
    let first = affinity.acquire("s", None).await.unwrap();
    let Hold::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_affinity = affinity.clone();
    let waiter = tokio::spawn(async move { waiter_affinity.acquire("s", None).await });
    affinity.wait_for_initializing_waiter().await;
    drop(first);

    let next = waiter.await.unwrap().unwrap();
    assert!(matches!(&next, Hold::Initialize(_)));
    drop(next);
    assert_eq!(affinity.entry_count(), 0);
    assert!(matches!(
        affinity.acquire("s", None).await.unwrap(),
        Hold::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_validates_worker_and_rank_contract() {
    let affinity = table();
    let Hold::Initialize(initializer) = affinity.acquire("s", Some(target(7, None))).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, None)).unwrap());

    assert!(affinity.acquire("s", Some(target(8, None))).await.is_err());
    assert!(
        affinity
            .acquire("s", Some(target(7, Some(0))))
            .await
            .is_err()
    );
    assert!(affinity.acquire("s", Some(target(7, None))).await.is_ok());
}

#[tokio::test(start_paused = true)]
async fn session_affinity_failed_bound_operation_invalidates_binding() {
    let affinity = table();
    let Hold::Initialize(initializer) = affinity.acquire("s", None).await.unwrap() else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());

    let operation = affinity.acquire("s", None).await.unwrap();
    assert_eq!(operation.target(), Some(target(7, Some(0))));
    operation.invalidate();

    assert_eq!(affinity.query_target("s", None).unwrap(), None);
    assert_eq!(affinity.entry_count(), 0);
    assert!(matches!(
        affinity.acquire("s", None).await.unwrap(),
        Hold::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_bound_lease_drop_refreshes_idle_ttl() {
    let affinity = table();
    let Hold::Initialize(initializer) = affinity.acquire("s", None).await.unwrap() else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());

    tokio::time::advance(Duration::from_secs(9)).await;
    let Hold::Bound { lease, .. } = affinity.acquire("s", None).await.unwrap() else {
        panic!("continuation must acquire the binding");
    };
    tokio::time::advance(Duration::from_secs(2)).await;
    tokio::task::yield_now().await;
    assert_eq!(
        affinity.query_target("s", None).unwrap(),
        Some(target(7, Some(0)))
    );
    drop(lease);

    assert_binding_expires_after_refreshed_ttl(&affinity).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_query_is_read_only() {
    let affinity = table();
    assert_eq!(affinity.query_target("s", None).unwrap(), None);
    assert_eq!(affinity.entry_count(), 0);

    let initializing = affinity.acquire("s", None).await.unwrap();
    assert_eq!(affinity.query_target("s", None).unwrap(), None);
    assert_eq!(affinity.entry_count(), 1);
    drop(initializing);
    assert_eq!(affinity.entry_count(), 0);

    let Hold::Initialize(initializer) = affinity.acquire("s", None).await.unwrap() else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());
    assert_eq!(
        affinity.query_target("s", None).unwrap(),
        Some(target(7, Some(0)))
    );
    affinity.expire_for_test("s");
    assert_eq!(affinity.query_target("s", None).unwrap(), None);
    assert_eq!(affinity.entry_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_reaper_removes_idle_entries_and_stops_on_drop() {
    let affinity = table();
    let cancellation = affinity.cancellation_token();
    let Hold::Initialize(initializer) = affinity.acquire("s", None).await.unwrap() else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());

    affinity.wait_for_reaper().await;
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::task::yield_now().await;
    assert_eq!(affinity.entry_count(), 0);

    drop(affinity);
    cancellation.cancelled().await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_enforces_id_and_entry_limits() {
    let affinity = SessionAffinity::with_config(SessionAffinityConfig {
        max_entries: 1,
        max_session_id_bytes: 8,
        ..SessionAffinityConfig::new(TTL)
    })
    .unwrap();
    let oversized = "123456789";
    let Err(error) = affinity.acquire(oversized, None).await else {
        panic!("oversized session ID must fail");
    };
    assert!(matches!(error, AffinityError::InvalidArgument(_)));
    assert_eq!(affinity.entry_count(), 0);

    let first_id = "first";
    let first = affinity.acquire(first_id, None).await.unwrap();
    let second_id = "second";
    let second = affinity
        .resolve(second_id, None, |_| true, std::future::pending())
        .await
        .expect("a full table is not a request failure");
    assert!(
        second.is_none(),
        "the second session routes without affinity"
    );
    drop(first);
    assert_eq!(affinity.entry_count(), 0);
    assert!(matches!(
        affinity.acquire(second_id, None).await.unwrap(),
        Hold::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_applies_first_live_binding_wins() {
    let affinity = table();
    let local_target = target(7, Some(0));
    let conflicting_target = target(8, Some(1));

    assert_eq!(
        affinity.apply_replica_update(
            "replicated".into(),
            local_target,
            AffinityVersion {
                sequence: 0,
                writer_id: 0
            }
        ),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        affinity.query_target("replicated", None).unwrap(),
        Some(local_target)
    );
    assert_eq!(
        affinity.apply_replica_update(
            "replicated".into(),
            conflicting_target,
            AffinityVersion {
                sequence: 0,
                writer_id: 0
            }
        ),
        ReplicaApplyOutcome::IgnoredConflict
    );

    affinity.expire_for_test("replicated");
    assert_eq!(
        affinity.apply_replica_update(
            "replicated".into(),
            conflicting_target,
            AffinityVersion {
                sequence: 0,
                writer_id: 0
            }
        ),
        ReplicaApplyOutcome::ReplacedExpired
    );
    assert_eq!(
        affinity.query_target("replicated", None).unwrap(),
        Some(conflicting_target)
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_duplicate_refreshes_local_ttl() {
    let affinity = table();
    let replicated_id = "replicated";
    let replicated_target = target(7, Some(0));

    assert_eq!(
        affinity.apply_replica_update(
            "replicated".into(),
            replicated_target,
            AffinityVersion {
                sequence: 0,
                writer_id: 0
            }
        ),
        ReplicaApplyOutcome::Inserted
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        affinity.apply_replica_update(
            "replicated".into(),
            replicated_target,
            AffinityVersion {
                sequence: 0,
                writer_id: 0
            }
        ),
        ReplicaApplyOutcome::Refreshed
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(
        affinity.query_target(replicated_id, None).unwrap(),
        Some(replicated_target)
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(affinity.query_target(replicated_id, None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_ignores_initializing_sessions() {
    let affinity = table();
    let initializing = affinity.acquire("s", None).await.unwrap();

    assert_eq!(
        affinity.apply_replica_update(
            "s".into(),
            target(7, Some(0)),
            AffinityVersion {
                sequence: 0,
                writer_id: 0
            }
        ),
        ReplicaApplyOutcome::IgnoredInitializing
    );
    assert_eq!(affinity.query_target("s", None).unwrap(), None);
    drop(initializing);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_enforces_id_and_entry_limits() {
    let affinity = SessionAffinity::with_config(SessionAffinityConfig {
        max_entries: 1,
        max_session_id_bytes: 8,
        ..SessionAffinityConfig::new(TTL)
    })
    .unwrap();

    assert_eq!(
        affinity.apply_replica_update(
            "123456789".into(),
            target(7, Some(0)),
            AffinityVersion {
                sequence: 0,
                writer_id: 0
            }
        ),
        ReplicaApplyOutcome::RejectedSessionId
    );
    assert_eq!(
        affinity.apply_replica_update(
            "first".into(),
            target(7, Some(0)),
            AffinityVersion {
                sequence: 0,
                writer_id: 0
            }
        ),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        affinity.apply_replica_update(
            "second".into(),
            target(8, Some(0)),
            AffinityVersion {
                sequence: 0,
                writer_id: 0
            }
        ),
        ReplicaApplyOutcome::RejectedCapacity
    );
    assert_eq!(affinity.entry_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_republishes_the_stored_replica_version() {
    let affinity = table();
    let mut updates = enable_replica(&affinity, 99, 1);
    let replicated_target = target(7, Some(0));
    assert_eq!(
        affinity.apply_replica_update(
            "s".into(),
            replicated_target,
            AffinityVersion {
                sequence: 123,
                writer_id: 7
            }
        ),
        ReplicaApplyOutcome::Inserted
    );

    let Hold::Bound { lease, .. } = affinity.acquire("s", None).await.unwrap() else {
        panic!("replicated binding must be acquired");
    };
    drop(lease);

    let update = updates.recv().await.unwrap();
    assert_eq!(update.sequence, 123);
    assert_eq!(update.writer_id, 7);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_worker_only_binding_allows_ranked_dispatch_without_narrowing() {
    let affinity = table();
    let worker_binding = target(7, None);

    let initialization = affinity.acquire("s", None).await.unwrap();
    drop(affinity.commit(initialization, worker_binding).unwrap());

    let continuation = affinity.acquire("s", None).await.unwrap();
    let lease = affinity
        .commit(continuation, target(7, Some(3)))
        .expect("worker-only affinity must allow the scheduler to select a DP rank");

    assert_eq!(
        affinity.query_target("s", None).unwrap(),
        Some(worker_binding)
    );
    drop(lease);
}

#[tokio::test(start_paused = true)]
async fn soft_worker_only_binding_stays_worker_scoped_after_ranked_dispatch() {
    let affinity = SessionAffinity::with_config(
        SessionAffinityConfig::new(TTL).with_mode(SessionAffinityMode::Soft),
    )
    .unwrap();
    let worker_binding = target(7, None);
    bind(&affinity, worker_binding).await;

    let continuation = affinity.acquire("s", None).await.unwrap();
    let lease = affinity.commit(continuation, target(7, Some(3))).unwrap();
    drop(lease);

    assert_eq!(
        affinity.query_target("s", None).unwrap(),
        Some(worker_binding)
    );
}

#[tokio::test(start_paused = true)]
async fn soft_ranked_binding_is_not_widened_by_worker_only_dispatch() {
    let affinity = SessionAffinity::with_config(
        SessionAffinityConfig::new(TTL).with_mode(SessionAffinityMode::Soft),
    )
    .unwrap();
    let ranked_binding = target(7, Some(2));
    bind(&affinity, ranked_binding).await;

    let continuation = affinity.acquire("s", None).await.unwrap();
    let lease = affinity.commit(continuation, target(8, None)).unwrap();
    drop(lease);

    assert_eq!(
        affinity.query_target("s", None).unwrap(),
        Some(ranked_binding)
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_ranked_binding_rejects_mismatched_rank_dispatch() {
    let affinity = table();
    let binding = target(7, Some(2));

    let initialization = affinity.acquire("s", None).await.unwrap();
    drop(affinity.commit(initialization, binding).unwrap());

    let continuation = affinity.acquire("s", None).await.unwrap();
    assert!(affinity.commit(continuation, target(7, Some(3))).is_err());
    assert_eq!(affinity.query_target("s", None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn soft_affinity_rebinds_after_dispatch() {
    let affinity = SessionAffinity::with_config(
        SessionAffinityConfig::new(TTL).with_mode(SessionAffinityMode::Soft),
    )
    .unwrap();
    let original = target(7, Some(0));
    let replacement = target(8, Some(1));
    bind(&affinity, original).await;

    let failed_attempt = affinity.acquire("s", None).await.unwrap();
    drop(failed_attempt);
    assert_eq!(affinity.query_target("s", None).unwrap(), Some(original));

    let successful_attempt = affinity.acquire("s", None).await.unwrap();
    let lease = affinity.commit(successful_attempt, replacement).unwrap();
    assert_eq!(affinity.query_target("s", None).unwrap(), Some(replacement));
    drop(lease);
}

#[tokio::test(start_paused = true)]
async fn concurrent_soft_rebind_uses_observed_version_cas() {
    let affinity = SessionAffinity::with_config(
        SessionAffinityConfig::new(TTL).with_mode(SessionAffinityMode::Soft),
    )
    .unwrap();
    let original = target(7, Some(0));
    let winner = target(8, Some(0));
    let stale = target(9, Some(0));
    bind(&affinity, original).await;
    let first = affinity.acquire("s", None).await.unwrap();
    let second = affinity.acquire("s", None).await.unwrap();

    let first_lease = affinity.commit(first, winner).unwrap();
    let second_lease = affinity.commit(second, stale).unwrap();
    drop(first_lease);
    drop(second_lease);
    assert_eq!(affinity.query_target("s", None).unwrap(), Some(winner));
}

#[tokio::test(start_paused = true)]
async fn replica_applies_only_newer_affinity_versions() {
    let affinity = table();
    let first = target(7, Some(0));
    let newer = target(8, Some(0));
    let stale = target(9, Some(0));

    assert_eq!(
        affinity.apply_replica_update(
            "ordered".into(),
            first,
            AffinityVersion {
                sequence: 3,
                writer_id: 10
            }
        ),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        affinity.apply_replica_update(
            "ordered".into(),
            newer,
            AffinityVersion {
                sequence: 4,
                writer_id: 10
            }
        ),
        ReplicaApplyOutcome::ReplacedNewer
    );
    assert_eq!(
        affinity.apply_replica_update(
            "ordered".into(),
            stale,
            AffinityVersion {
                sequence: 3,
                writer_id: 11
            }
        ),
        ReplicaApplyOutcome::IgnoredConflict
    );
    assert_eq!(affinity.query_target("ordered", None).unwrap(), Some(newer));
}

#[tokio::test(start_paused = true)]
async fn stale_lease_cannot_invalidate_or_refresh_newer_replica_binding() {
    let affinity = table();
    let original = target(7, Some(0));
    let replacement = target(8, Some(0));
    bind(&affinity, original).await;
    let stale = affinity.acquire("s", None).await.unwrap();
    let replacement_sequence = u64::MAX / 2;
    affinity.apply_replica_update(
        "s".into(),
        replacement,
        AffinityVersion {
            sequence: replacement_sequence,
            writer_id: 1,
        },
    );
    stale.invalidate();
    assert_eq!(affinity.query_target("s", None).unwrap(), Some(replacement));

    let stale = affinity.acquire("s", None).await.unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    affinity.apply_replica_update(
        "s".into(),
        replacement,
        AffinityVersion {
            sequence: replacement_sequence.saturating_add(1),
            writer_id: 1,
        },
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    drop(stale);
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(affinity.query_target("s", None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_completion_restores_expired_remote_binding() {
    let origin = table();
    let mut updates = enable_replica(&origin, 99, 4);
    let replica = table();
    let replicated_target = target(7, Some(0));
    let operation = origin.acquire("s", None).await.unwrap();
    let lease = origin.commit(operation, replicated_target).unwrap();

    let after_dispatch = updates.recv().await.unwrap();
    let version = after_dispatch.version();
    assert_eq!(
        replica.apply_replica_update(
            after_dispatch.session_id,
            target(after_dispatch.worker_id, after_dispatch.dp_rank),
            version
        ),
        ReplicaApplyOutcome::Inserted
    );
    tokio::time::advance(Duration::from_secs(11)).await;
    assert_eq!(replica.query_target("s", None).unwrap(), None);

    drop(lease);
    let after_completion = updates.recv().await.unwrap();
    let version = after_completion.version();
    assert_eq!(
        replica.apply_replica_update(
            after_completion.session_id,
            target(after_completion.worker_id, after_completion.dp_rank),
            version
        ),
        ReplicaApplyOutcome::ReplacedExpired
    );
    assert_eq!(
        replica.query_target("s", None).unwrap(),
        Some(replicated_target)
    );
}

fn noop_replica_sink() -> Arc<dyn AffinityReplicaSink> {
    Arc::new(|_: &str, _: AffinityTarget, _: AffinityVersion| {})
}

#[tokio::test]
async fn replica_startup_runs_once_and_installs_writer_with_sink() {
    let table = table();
    let starts = AtomicUsize::new(0);
    let start = || async {
        starts.fetch_add(1, Ordering::SeqCst);
        tokio::task::yield_now().await;
        Ok::<_, &'static str>((11, noop_replica_sink()))
    };
    let first = table.enable_replication_with(start());
    let second = table.enable_replication_with(start());
    let (first, second) = tokio::join!(first, second);
    first.unwrap();
    second.unwrap();
    assert_eq!(starts.load(Ordering::SeqCst), 1);
    assert_eq!(table.writer_id(), 11);
    assert_eq!(table.next_version().writer_id, 11);
    assert!(!table.enable_replication(12, noop_replica_sink()));
    assert_eq!(table.writer_id(), 11);
}

#[tokio::test]
async fn failed_replica_startup_allows_retry_and_skips_unused_futures() {
    let table = table();
    let starts = AtomicUsize::new(0);
    let failed = table.enable_replication_with(async {
        starts.fetch_add(1, Ordering::SeqCst);
        Err::<(u64, Arc<dyn AffinityReplicaSink>), _>("transport startup failed")
    });
    assert_eq!(
        starts.load(Ordering::SeqCst),
        0,
        "creating the future starts nothing"
    );
    assert_eq!(failed.await, Err("transport startup failed"));
    assert_eq!(table.writer_id(), 0);
    assert!(table.enable_replication(12, noop_replica_sink()));
    table
        .enable_replication_with(async {
            starts.fetch_add(1, Ordering::SeqCst);
            Err::<(u64, Arc<dyn AffinityReplicaSink>), _>("already installed")
        })
        .await
        .unwrap();
    assert_eq!(
        starts.load(Ordering::SeqCst),
        1,
        "an installed sink never polls a replacement"
    );
    assert_eq!(table.writer_id(), 12);
}

#[tokio::test]
async fn cancelled_replica_startup_drops_its_future_and_allows_retry() {
    let table = table();
    let started = std::sync::atomic::AtomicBool::new(false);
    let (dropped_tx, mut dropped_rx) = tokio::sync::oneshot::channel::<()>();
    let mut startup = Box::pin(table.enable_replication_with(async {
        let _drop_on_cancel = dropped_tx;
        started.store(true, Ordering::SeqCst);
        std::future::pending::<Result<(u64, Arc<dyn AffinityReplicaSink>), &'static str>>().await
    }));
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(startup.as_mut().poll(&mut cx).is_pending());
    assert!(started.load(Ordering::SeqCst));
    assert_eq!(table.writer_id(), 0);
    assert!(
        !table.enable_replication(12, noop_replica_sink()),
        "sync startup cannot overwrite an in-flight initializer"
    );
    drop(startup);
    assert!(matches!(
        dropped_rx.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Closed)
    ));
    table
        .enable_replication_with(async { Ok::<_, &'static str>((13, noop_replica_sink())) })
        .await
        .unwrap();
    assert_eq!(table.writer_id(), 13);
}
