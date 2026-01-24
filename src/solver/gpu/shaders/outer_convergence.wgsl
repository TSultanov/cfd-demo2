struct Params {
    num_cells: u32,
    stride: u32,
    num_targets: u32,
    _pad0: u32,
}

struct TargetDesc {
    offsets: array<u32, 4>,
    num_comps: u32,
    _pad0: array<u32, 3>,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<TargetDesc>;
@group(0) @binding(2) var<storage, read_write> out_bits: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) {
        return;
    }

    let base = cell * params.stride;
    for (var t: u32 = 0u; t < params.num_targets; t = t + 1u) {
        let desc = targets[t];
        var mag2: f32 = 0.0;
        for (var c: u32 = 0u; c < desc.num_comps; c = c + 1u) {
            let off = desc.offsets[c];
            let v = input[base + off];
            mag2 = mag2 + v * v;
        }
        let mag = sqrt(mag2);
        let bits = bitcast<u32>(mag);
        atomicMax(&out_bits[t], bits);
    }
}
