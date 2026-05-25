fn main() {
    // macOS requires these linker flags to build Python extension modules without
    // unresolved symbol errors, since Python symbols are provided by the interpreter
    // at runtime. Using `cargo:rustc-cdylib-link-arg` ensures we only apply these
    // to the cdylib target, rather than globally masking undefined symbols in tests
    // or binaries.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
