// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::{KvRouter, RoutingLoadContext};

/// Standalone KV selection surface plus the context that owns its load tasks.
pub struct ManagedKvRouter {
    load_context: Arc<RoutingLoadContext>,
    router: Arc<KvRouter>,
}

impl Clone for ManagedKvRouter {
    fn clone(&self) -> Self {
        Self {
            load_context: self.load_context.clone(),
            router: self.router.clone(),
        }
    }
}

impl std::ops::Deref for ManagedKvRouter {
    type Target = KvRouter;

    fn deref(&self) -> &Self::Target {
        &self.router
    }
}

impl ManagedKvRouter {
    pub fn new(load_context: Arc<RoutingLoadContext>, router: Arc<KvRouter>) -> Self {
        Self {
            load_context,
            router,
        }
    }

    pub fn load_context(&self) -> &Arc<RoutingLoadContext> {
        &self.load_context
    }

    pub fn router(&self) -> &Arc<KvRouter> {
        &self.router
    }
}
