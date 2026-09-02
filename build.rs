use cfg_aliases::cfg_aliases;

fn main() {
    cfg_aliases! {
        x86: { any(target_arch = "x86", target_arch = "x86_64") },
        aarch64: { target_arch = "aarch64" },
        wasm32: { all(target_arch = "wasm32", target_feature = "simd128") },
        loong64: { all(target_arch = "loongarch64", feature = "nightly") },
        avx512: { all(target_arch = "x86_64", feature = "avx512") },
        avx512_fp16: { all(avx512, feature = "fp16", feature = "nightly") },

        // Workaround for Safari
        relaxed_simd: { all(target_arch = "wasm32", target_feature = "simd128", target_feature = "relaxed-simd") },

        x86_v3: { feature = "miri-x86_v3" },
        x86_v4: { feature = "miri-x86_v4" },
    }
}
