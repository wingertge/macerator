use cfg_aliases::cfg_aliases;

fn main() {
    cfg_aliases! {
        x86: { any(target_arch = "x86", target_arch = "x86_64") },
        aarch64: { target_arch = "aarch64" },
        wasm32: { target_arch = "wasm32" }
    }
}
