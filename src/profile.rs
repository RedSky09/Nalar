//! Lightweight per-layer profiler: wall-clock time and estimated memory.
//!
//! No dependencies: timing via `std::time::Instant`, accumulation via
//! `std::collections::HashMap`. "Memory" here means the size of the tensors
//! involved (parameters and cached forward activations), not measured process
//! memory: `std` alone cannot read RSS without a platform-specific call. Treat
//! it as what the engine's math allocates, not a full memory profile.

use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Profiler {
    time: HashMap<String, Duration>,
    calls: HashMap<String, u64>,
    /// Insertion order, so the report reads in the order labels were first seen
    /// rather than HashMap's unspecified order.
    order: Vec<String>,
}

impl Profiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Times `f` and adds the elapsed duration under `label`. Labels repeat
    /// across calls (e.g. once per batch) and accumulate.
    pub fn time<T>(&mut self, label: impl Into<String>, f: impl FnOnce() -> T) -> T {
        let t0 = Instant::now();
        let out = f();
        self.record(label, t0.elapsed());
        out
    }

    /// Adds an already-measured duration under `label`, for timings that can't
    /// be expressed as a single closure (see `examples/profile.rs`).
    pub fn record(&mut self, label: impl Into<String>, d: Duration) {
        let label = label.into();
        if !self.time.contains_key(&label) {
            self.order.push(label.clone());
        }
        *self.time.entry(label.clone()).or_insert(Duration::ZERO) += d;
        *self.calls.entry(label).or_insert(0) += 1;
    }

    pub fn total_ms(&self, label: &str) -> Option<f64> {
        self.time.get(label).map(|d| d.as_secs_f64() * 1e3)
    }

    /// A formatted table, in first-seen order, plus a total row.
    pub fn report(&self) -> String {
        let mut out = String::new();
        let total: Duration = self.time.values().sum();
        out.push_str(&format!("{:<14} {:>8} {:>12} {:>14}\n", "label", "calls", "total_ms", "per_call_us"));
        for label in &self.order {
            let d = self.time[label];
            let c = self.calls[label];
            out.push_str(&format!(
                "{:<14} {:>8} {:>12.3} {:>14.2}\n",
                label,
                c,
                d.as_secs_f64() * 1e3,
                d.as_secs_f64() * 1e6 / c as f64
            ));
        }
        out.push_str(&format!("{:<14} {:>8} {:>12.3} {:>14}\n", "TOTAL", "-", total.as_secs_f64() * 1e3, "-"));
        out
    }
}

/// Bytes used by `n` `f64` elements (8 bytes each, no padding for a flat array).
pub fn bytes_f64(n: usize) -> usize {
    n * std::mem::size_of::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn accumulates_across_repeated_labels() {
        let mut p = Profiler::new();
        p.time("a", || sleep(Duration::from_millis(2)));
        p.time("a", || sleep(Duration::from_millis(2)));
        p.time("b", || {});
        assert!(p.total_ms("a").unwrap() >= 3.0, "expected >= 3ms, got {:?}", p.total_ms("a"));
        assert_eq!(p.calls["a"], 2);
        assert_eq!(p.order, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn report_has_no_entries_for_unused_profiler() {
        let p = Profiler::new();
        assert!(p.report().contains("TOTAL"));
    }

    #[test]
    fn bytes_f64_is_eight_per_element() {
        assert_eq!(bytes_f64(0), 0);
        assert_eq!(bytes_f64(1000), 8000);
    }
}
