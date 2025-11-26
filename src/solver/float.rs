use num_traits::{FromPrimitive, ToPrimitive};
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use wide::{f32x8, f64x4};

pub trait Float:
    num_traits::Float
    + FromPrimitive
    + ToPrimitive
    + Debug
    + Display
    + Sum
    + Copy
    + Send
    + Sync
    + 'static
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    type Simd: Simd<Element = Self>;
    fn val_from_f64(v: f64) -> Self;
    fn val_to_f64(self) -> f64;
}

pub trait Simd:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    type Element;
    const LANES: usize;
    fn splat(val: Self::Element) -> Self;
    fn from_slice(slice: &[Self::Element]) -> Self;
    fn write_to_slice(&self, slice: &mut [Self::Element]);
}

impl Float for f64 {
    type Simd = f64x4;
    fn val_from_f64(v: f64) -> Self {
        v
    }
    fn val_to_f64(self) -> f64 {
        self
    }
}

impl Simd for f64x4 {
    type Element = f64;
    const LANES: usize = 4;
    fn splat(val: Self::Element) -> Self {
        f64x4::splat(val)
    }
    fn from_slice(slice: &[Self::Element]) -> Self {
        f64x4::from(slice)
    }
    fn write_to_slice(&self, slice: &mut [Self::Element]) {
        let arr: [f64; 4] = (*self).into();
        slice.copy_from_slice(&arr);
    }
}

impl Float for f32 {
    type Simd = f32x8;
    fn val_from_f64(v: f64) -> Self {
        v as f32
    }
    fn val_to_f64(self) -> f64 {
        self as f64
    }
}

impl Simd for f32x8 {
    type Element = f32;
    const LANES: usize = 8;
    fn splat(val: Self::Element) -> Self {
        f32x8::splat(val)
    }
    fn from_slice(slice: &[Self::Element]) -> Self {
        f32x8::from(slice)
    }
    fn write_to_slice(&self, slice: &mut [Self::Element]) {
        let arr: [f32; 8] = (*self).into();
        slice.copy_from_slice(&arr);
    }
}
