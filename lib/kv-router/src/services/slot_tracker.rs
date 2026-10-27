// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone slot tracker, owned by `dynamo-slot-tracker`; this module keeps the old paths.

pub use dynamo_slot_tracker::http::{SlotTrackerConfig, run_server, server};
pub use dynamo_slot_tracker::registry;
