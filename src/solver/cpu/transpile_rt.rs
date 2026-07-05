//! Runtime support (prelude) for the transpiled (compiled-Rust) CPU kernels.
//!
//! The transpiler (`transpile.rs`) emits kernels that target these types and
//! helpers: small vector structs mirroring WGSL `vecN<f32>`, and typed
//! load/store helpers over the relaxed-atomic buffer storage (so compiled
//! kernels share the exact same `Buffers` as the interpreter and are safe to run
//! in parallel over disjoint cells/faces).

use std::sync::atomic::{AtomicU32, Ordering};

const ORD: Ordering = Ordering::Relaxed;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec2 {
    #[inline(always)]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self { x: v, y: v }
    }
}
impl Vec3 {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }
}
impl Vec4 {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v, w: v }
    }
}

macro_rules! vec_ops {
    ($T:ident { $($f:ident),+ }) => {
        impl std::ops::Add for $T {
            type Output = $T;
            #[inline(always)]
            fn add(self, o: $T) -> $T { $T { $($f: self.$f + o.$f),+ } }
        }
        impl std::ops::Sub for $T {
            type Output = $T;
            #[inline(always)]
            fn sub(self, o: $T) -> $T { $T { $($f: self.$f - o.$f),+ } }
        }
        impl std::ops::Neg for $T {
            type Output = $T;
            #[inline(always)]
            fn neg(self) -> $T { $T { $($f: -self.$f),+ } }
        }
        impl std::ops::Mul<f32> for $T {
            type Output = $T;
            #[inline(always)]
            fn mul(self, s: f32) -> $T { $T { $($f: self.$f * s),+ } }
        }
        impl std::ops::Mul<$T> for f32 {
            type Output = $T;
            #[inline(always)]
            fn mul(self, v: $T) -> $T { $T { $($f: self * v.$f),+ } }
        }
        impl std::ops::Mul for $T {
            type Output = $T;
            #[inline(always)]
            fn mul(self, o: $T) -> $T { $T { $($f: self.$f * o.$f),+ } }
        }
        impl std::ops::Div<f32> for $T {
            type Output = $T;
            #[inline(always)]
            fn div(self, s: f32) -> $T { $T { $($f: self.$f / s),+ } }
        }
        impl std::ops::Div for $T {
            type Output = $T;
            #[inline(always)]
            fn div(self, o: $T) -> $T { $T { $($f: self.$f / o.$f),+ } }
        }
    };
}
vec_ops!(Vec2 { x, y });
vec_ops!(Vec3 { x, y, z });
vec_ops!(Vec4 { x, y, z, w });

pub trait Dot {
    fn dot(self, o: Self) -> f32;
}
impl Dot for Vec2 {
    #[inline(always)]
    fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }
}
impl Dot for Vec3 {
    #[inline(always)]
    fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
}
impl Dot for Vec4 {
    #[inline(always)]
    fn dot(self, o: Vec4) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }
}

#[inline(always)]
pub fn dot<T: Dot>(a: T, b: T) -> f32 {
    a.dot(b)
}
#[inline(always)]
pub fn length<T: Dot + Copy>(a: T) -> f32 {
    a.dot(a).sqrt()
}
#[inline(always)]
pub fn distance(a: Vec2, b: Vec2) -> f32 {
    length(a - b)
}
#[inline(always)]
pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}
#[inline(always)]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = (((x - e0) / (e1 - e0)).max(0.0)).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline(always)]
pub fn ldf(b: &[AtomicU32], i: usize) -> f32 {
    f32::from_bits(b[i].load(ORD))
}
#[inline(always)]
pub fn stf(b: &[AtomicU32], i: usize, v: f32) {
    b[i].store(v.to_bits(), ORD);
}
#[inline(always)]
pub fn ldu(b: &[AtomicU32], i: usize) -> u32 {
    b[i].load(ORD)
}
#[inline(always)]
pub fn stu(b: &[AtomicU32], i: usize, v: u32) {
    b[i].store(v, ORD);
}
#[inline(always)]
pub fn ldi(b: &[AtomicU32], i: usize) -> i32 {
    b[i].load(ORD) as i32
}
#[inline(always)]
pub fn sti(b: &[AtomicU32], i: usize, v: i32) {
    b[i].store(v as u32, ORD);
}
#[inline(always)]
pub fn ld2(b: &[AtomicU32], i: usize) -> Vec2 {
    Vec2 { x: ldf(b, 2 * i), y: ldf(b, 2 * i + 1) }
}
#[inline(always)]
pub fn st2(b: &[AtomicU32], i: usize, v: Vec2) {
    stf(b, 2 * i, v.x);
    stf(b, 2 * i + 1, v.y);
}
/// Store one component of a `Vector2` buffer element (`grad_state[i].x = v`).
#[inline(always)]
pub fn st2c(b: &[AtomicU32], i: usize, comp: usize, v: f32) {
    stf(b, 2 * i + comp, v);
}

/// Boundary ghost value for face reconstruction. Dirichlet (kind 1) returns the
/// prescribed value, Neumann (kind 2) extrapolates by the outward gradient,
/// otherwise zero-gradient (owner); interior faces pass `interior` through.
#[inline(always)]
pub fn bc_neighbor_scalar(
    interior: f32,
    owner: f32,
    kind: u32,
    value: f32,
    d_own: f32,
    is_boundary: bool,
) -> f32 {
    let boundary = match kind {
        1 => value,
        2 => owner + value * d_own,
        _ => owner,
    };
    if is_boundary {
        boundary
    } else {
        interior
    }
}

/// Atomic read-modify-write add on an f32 buffer element (CAS loop).
#[inline(always)]
pub fn atomic_add_f32(b: &[AtomicU32], i: usize, v: f32) -> f32 {
    let mut cur = b[i].load(ORD);
    loop {
        let old = f32::from_bits(cur);
        let new = (old + v).to_bits();
        match b[i].compare_exchange_weak(cur, new, ORD, ORD) {
            Ok(_) => return old,
            Err(actual) => cur = actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_arithmetic() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a + b, Vec2::new(4.0, 6.0));
        assert_eq!(b - a, Vec2::new(2.0, 2.0));
        assert_eq!(a * 2.0, Vec2::new(2.0, 4.0));
        assert_eq!(2.0 * a, Vec2::new(2.0, 4.0));
        assert_eq!(-a, Vec2::new(-1.0, -2.0));
        assert_eq!(dot(a, b), 11.0);
        assert_eq!(distance(Vec2::new(0.0, 0.0), Vec2::new(3.0, 4.0)), 5.0);
    }

    #[test]
    fn atomic_load_store() {
        let buf: Vec<AtomicU32> = (0..4).map(|_| AtomicU32::new(0)).collect();
        stf(&buf, 0, 1.5);
        st2(&buf, 1, Vec2::new(2.0, 3.0)); // writes indices 2,3
        assert_eq!(ldf(&buf, 0), 1.5);
        assert_eq!(ld2(&buf, 1), Vec2::new(2.0, 3.0));
        assert_eq!(atomic_add_f32(&buf, 0, 0.5), 1.5);
        assert_eq!(ldf(&buf, 0), 2.0);
    }
}
