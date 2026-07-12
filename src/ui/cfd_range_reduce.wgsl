// One-workgroup field-range reduction for frame-cadence CFD visualization.
//
// Each lane walks a deterministic strided subset of the packed cell state,
// then the workgroup performs a fixed binary-tree min/max reduction.  One
// dispatch produces pressure, Ux, Uy and |U| ranges together, so changing the
// displayed field never triggers another pass or another state snapshot.

const WORKGROUP_SIZE: u32 = 256u;
const F32_MAX: f32 = 3.402823e+38;

struct Config {
    cell_count: u32,
    stride: u32,
    u_offset: u32,
    p_offset: u32,
    has_u: u32,
    has_p: u32,
    sequence_lo: u32,
    sequence_hi: u32,
};

struct FieldRanges {
    minimum: vec4<f32>,
    maximum: vec4<f32>,
    sequence_lo: u32,
    sequence_hi: u32,
    _padding: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> field_data: array<f32>;
@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(2) var<storage, read_write> output: FieldRanges;

var<workgroup> lane_minimum: array<vec4<f32>, 256>;
var<workgroup> lane_maximum: array<vec4<f32>, 256>;

fn finite(value: f32) -> bool {
    // WGSL has no portable isFinite builtin across all Naga versions.
    return value == value && abs(value) <= F32_MAX;
}

fn finite_magnitude(x: f32, y: f32) -> f32 {
    // Scale first so otherwise-finite large components do not overflow in
    // x*x+y*y. The explicit zero branch also avoids a 0/0 intermediate.
    let scale = max(abs(x), abs(y));
    if (scale == 0.0) {
        return 0.0;
    }
    return scale * sqrt((x / scale) * (x / scale) + (y / scale) * (y / scale));
}

@compute @workgroup_size(256)
fn reduce_ranges(@builtin(local_invocation_index) lane: u32) {
    var minimum = vec4<f32>(F32_MAX);
    var maximum = vec4<f32>(-F32_MAX);

    var cell = lane;
    loop {
        if (cell >= config.cell_count) {
            break;
        }
        let base = cell * config.stride;

        if (config.has_p != 0u) {
            let pressure = field_data[base + config.p_offset];
            if (finite(pressure)) {
                minimum.x = min(minimum.x, pressure);
                maximum.x = max(maximum.x, pressure);
            }
        }

        if (config.has_u != 0u) {
            let ux = field_data[base + config.u_offset];
            let uy = field_data[base + config.u_offset + 1u];
            if (finite(ux)) {
                minimum.y = min(minimum.y, ux);
                maximum.y = max(maximum.y, ux);
            }
            if (finite(uy)) {
                minimum.z = min(minimum.z, uy);
                maximum.z = max(maximum.z, uy);
            }
            if (finite(ux) && finite(uy)) {
                let magnitude = finite_magnitude(ux, uy);
                if (finite(magnitude)) {
                    minimum.w = min(minimum.w, magnitude);
                    maximum.w = max(maximum.w, magnitude);
                }
            }
        }

        cell += WORKGROUP_SIZE;
    }

    lane_minimum[lane] = minimum;
    lane_maximum[lane] = maximum;
    workgroupBarrier();

    // Fixed reduction tree: the same lane combines the same subsets on every
    // run. (Min/max have no floating-point accumulation-order drift anyway.)
    var width = WORKGROUP_SIZE / 2u;
    loop {
        if (width == 0u) {
            break;
        }
        if (lane < width) {
            lane_minimum[lane] = min(lane_minimum[lane], lane_minimum[lane + width]);
            lane_maximum[lane] = max(lane_maximum[lane], lane_maximum[lane + width]);
        }
        workgroupBarrier();
        width /= 2u;
    }

    if (lane == 0u) {
        var reduced_minimum = lane_minimum[0];
        var reduced_maximum = lane_maximum[0];
        for (var field = 0u; field < 4u; field += 1u) {
            let lo = reduced_minimum[field];
            let hi = reduced_maximum[field];
            if (!finite(lo) || !finite(hi) || lo > hi) {
                reduced_minimum[field] = 0.0;
                reduced_maximum[field] = 1.0;
            } else if (abs(hi - lo) < 1.0e-12) {
                let expanded = lo + 1.0;
                if (finite(expanded) && expanded > lo) {
                    reduced_maximum[field] = expanded;
                } else {
                    reduced_minimum[field] = 0.0;
                    reduced_maximum[field] = 1.0;
                }
            } else {
                // Representable-precision floor (keep in sync with
                // `sanitize_range` in cfd_renderer.rs): the state is f32, so a
                // field spanning fewer than ~PRECISION_FLOOR_ULPS quanta of its
                // own magnitude (e.g. sub-Pa acoustics on a 105 kPa absolute
                // pressure) is dominated by representation noise. Never stretch
                // such a span across the full colormap — expand the displayed
                // range so one quantum stays a small fraction of it.
                let magnitude = max(abs(lo), abs(hi));
                let floor_span = 64.0 * 1.1920929e-7 * magnitude;
                let span = hi - lo;
                if (span < floor_span) {
                    let mid = 0.5 * (lo + hi);
                    reduced_minimum[field] = mid - 0.5 * floor_span;
                    reduced_maximum[field] = mid + 0.5 * floor_span;
                }
            }
        }
        output.minimum = reduced_minimum;
        output.maximum = reduced_maximum;
        output.sequence_lo = config.sequence_lo;
        output.sequence_hi = config.sequence_hi;
        output._padding = vec2<u32>(0u);
    }
}
