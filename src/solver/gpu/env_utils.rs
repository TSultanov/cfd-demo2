use std::env;

pub(crate) fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|val| {
            let val = val.to_ascii_lowercase();
            matches!(val.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(default)
}

pub(crate) fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}
