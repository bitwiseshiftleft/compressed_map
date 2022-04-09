// from cbindgen docs and
// https://michael-f-bryan.github.io/rust-ffi-guide/cbindgen.html

extern crate cbindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let headers_enabled = env::var_os("CARGO_FEATURE_CFFI").is_some();
    if !headers_enabled { return (); }
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let package_name = env::var("CARGO_PKG_NAME").unwrap();


    let output_file = target_dir()
        .join(format!("{}.h", package_name))
        .display()
        .to_string();

    cbindgen::Builder::new()
      .with_language(cbindgen::Language::C)
      .with_include_version(true)
      .with_include_guard("COMPRESSED_MAP_H")
      .with_crate(crate_dir)
    //   .with_parse_expand(&["compressed_map"])
      .exclude_item("PERMUTE_ZERO")
      .generate()
      .expect("Unable to generate bindings")
      .write_to_file(&output_file);
}

fn target_dir() -> PathBuf {
    if let Ok(target) = env::var("CARGO_TARGET_DIR") {
        PathBuf::from(target)
    } else {
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("target")
    }
}