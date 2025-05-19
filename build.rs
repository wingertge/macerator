use cfg_aliases::cfg_aliases;

fn main() {
    cfg_aliases! {
        x86: { any(target_arch = "x86", target_arch = "x86_64") },
        avx512: { all(target_arch = "x86_64", feature = "nightly") },
        fp16: { all(target_arch = "x86_64", feature = "fp16", feature = "nightly") },
        aarch64: { target_arch = "aarch64" },
        wasm32: { target_arch = "wasm32" },
        loong64: { all(target_arch = "loongarch64", feature = "nightly") }
    }
}
