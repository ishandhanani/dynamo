// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

const OUT_DIR: &str = "OUT_DIR";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    build_protos()?;

    Ok(())
}

fn build_protos() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=src/wan/grpc/protocol/relay.proto");
    let descriptor_path = PathBuf::from(env::var(OUT_DIR)?).join("relay_descriptor.bin");
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .bytes(["."])
        .file_descriptor_set_path(descriptor_path)
        .compile_protos(
            &["src/wan/grpc/protocol/relay.proto"],
            &["src/wan/grpc/protocol"],
        )?;
    Ok(())
}
