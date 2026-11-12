<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-error

The semantic error contract Dynamo components classify failures with (DEP #14354): `DynamoError`, `ErrorClass`, `ErrorReason`, `Diagnostic`, `PublicDetails`, and the builder. The component closest to a failure classifies it once; protocol boundaries render the class.

A leaf crate: it depends on serde and serde_json and nothing else, so crates below the runtime can classify with the same type. `dynamo-runtime` re-exports it as `dynamo_runtime::error`.

The optional `http` feature adds the `http` module, the HTTP rendering of the contract that every HTTP host shares: the total class-to-status policy (`http_action_for_class`, `http_action_for_error`, `ClientErrorAction`), the lookup that picks the error a response is rendered from (`find_canonical_error_in_chain`), the configured overload status (`overload_status_code`, read from `DYN_HTTP_OVERLOAD_STATUS_CODE`), and the sanitized response vocabulary (`SanitizedError`, `BackendStatusAction`). Its only added dependency is the `http` crate, for `StatusCode`.
