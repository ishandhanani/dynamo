// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

const DYN_FATBIN_PATH: &str = "DYN_FATBIN_PATH";
const OUT_DIR: &str = "OUT_DIR";

fn main() {
    // Declare our custom cfg flag to avoid unexpected_cfgs warnings
    println!("cargo:rustc-check-cfg=cfg(have_vec_copy_fatbin)");
    // Get FATBIN path and copy it to OUT_DIR for embedding
    if let Some(fatbin_path) = find_fatbin_file() {
        // Copy FATBIN to OUT_DIR so we can include it with a predictable path
        let out_dir = env::var(OUT_DIR).unwrap();
        let dest_path = PathBuf::from(out_dir).join("vectorized_copy.fatbin");

        if let Err(e) = std::fs::copy(&fatbin_path, &dest_path) {
            println!("cargo:warning=Failed to copy FATBIN to OUT_DIR: {}", e);
        } else {
            // Emit cfg flag for conditional compilation
            println!("cargo:rustc-cfg=have_vec_copy_fatbin");
            println!(
                "cargo:warning=CUDA FATBIN found at: {} - copied to OUT_DIR",
                fatbin_path.display()
            );
        }

        // Tell cargo to rerun if FATBIN file changes
        println!("cargo:rerun-if-changed={}", fatbin_path.display());
    } else {
        println!(
            "cargo:warning=CUDA FATBIN not found - run 'make fatbin' in cuda_kernels directory"
        );
        println!("cargo:warning=Set DYN_FATBIN_PATH env var to specify custom location");
    }

    // Rerun build if environment variable changes
    println!("cargo:rerun-if-env-changed=DYN_FATBIN_PATH");
}

fn find_fatbin_file() -> Option<PathBuf> {
    // 1. Check if user specified custom path via environment variable
    if let Ok(custom_path) = env::var(DYN_FATBIN_PATH) {
        let fatbin_file = PathBuf::from(custom_path);
        if fatbin_file.exists() {
            println!(
                "cargo:warning=Using custom FATBIN path: {}",
                fatbin_file.display()
            );
            return Some(fatbin_file);
        } else {
            println!(
                "cargo:warning=Custom FATBIN path does not exist: {}",
                fatbin_file.display()
            );
        }
    }

    // 2. Check standard locations (priority order)
    let default_paths = [
        "./src/block/transfer/kernels/vectorized_copy.fatbin", // Primary: Next to transfer module
    ];

    for path in &default_paths {
        let fatbin_file = PathBuf::from(path);
        if fatbin_file.exists() {
            println!(
                "cargo:warning=Found FATBIN at default location: {}",
                fatbin_file.display()
            );
            return Some(fatbin_file);
        }
    }

    None
}
