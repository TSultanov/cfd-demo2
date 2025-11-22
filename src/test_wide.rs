use wide::{f64x4, CmpGe, CmpLt};

fn main() {
    let a = f64x4::splat(1.0);
    let b = f64x4::splat(2.0);
    let c = a.simd_ge(b);
    let any = c.any();
    let mask = c.to_bitmask();
}
