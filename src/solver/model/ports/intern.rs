// String interning for port names.
//
// This module provides a thread-safe string interner that converts `&str` into
// `&'static str` in a controlled, deduplicated way. This allows derived/dynamic
// field names (e.g., `format!("grad_{}", pressure.name())`) to be used with the
// port system without ad-hoc `Box::leak` calls in modules.
//
// The interner is lazy-initialized on first use and uses a DashMap for
// thread-safe concurrent access.

use dashmap::DashMap;
use std::sync::OnceLock;

/// Global string interner for port names.
static INTERNER: OnceLock<DashMap<String, &'static str>> = OnceLock::new();

/// Get or initialize the global string interner.
fn get_interner() -> &'static DashMap<String, &'static str> {
    INTERNER.get_or_init(DashMap::new)
}

/// Intern a string slice, returning a `&'static str`.
///
/// If the string has already been interned, returns the existing static reference.
/// Otherwise, allocates a new `String`, leaks it to get a `&'static str`,
/// stores it in the interner, and returns it.
///
/// This is thread-safe and deduplicates identical strings.
pub fn intern(s: &str) -> &'static str {
    let interner = get_interner();

    // Fast path: check if already interned
    if let Some(entry) = interner.get(s) {
        return entry.value();
    }

    // Slow path: intern the string
    // We need to handle the race condition where another thread interns
    // the same string between our check and insert
    let static_str: &'static str = Box::leak(s.to_string().into_boxed_str());

    // insert returns the old value if the key already existed
    match interner.insert(s.to_string(), static_str) {
        Some(existing) => existing,
        None => static_str,
    }
}

/// Intern a `String`, returning a `&'static str`.
///
/// This is a convenience wrapper around `intern` that takes ownership of the
/// String to avoid an extra allocation when you already have a `String`.
pub fn intern_string(s: String) -> &'static str {
    let interner = get_interner();

    // Fast path: check if already interned
    if let Some(entry) = interner.get(&s) {
        return entry.value();
    }

    // Slow path: intern the string
    let static_str: &'static str = Box::leak(s.clone().into_boxed_str());

    match interner.insert(s, static_str) {
        Some(existing) => existing,
        None => static_str,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_returns_static_str() {
        let s = intern("test_field");
        assert_eq!(s, "test_field");
        // Verify it's actually 'static by using it in a static context
        let _: &'static str = s;
    }

    #[test]
    fn intern_deduplicates_identical_strings() {
        let s1 = intern("dedup_test");
        let s2 = intern("dedup_test");
        // Both should point to the same memory location
        assert_eq!(s1.as_ptr(), s2.as_ptr());
    }

    #[test]
    fn intern_deduplicates_different_allocations() {
        let owned1 = String::from("alloc_test");
        let owned2 = String::from("alloc_test");
        let s1 = intern(&owned1);
        let s2 = intern(&owned2);
        assert_eq!(s1.as_ptr(), s2.as_ptr());
    }

    #[test]
    fn intern_string_deduplicates() {
        let s1 = intern_string(String::from("string_test"));
        let s2 = intern_string(String::from("string_test"));
        assert_eq!(s1.as_ptr(), s2.as_ptr());
    }

    #[test]
    fn intern_different_strings_different_pointers() {
        let s1 = intern("field_a");
        let s2 = intern("field_b");
        assert_ne!(s1.as_ptr(), s2.as_ptr());
    }
}
