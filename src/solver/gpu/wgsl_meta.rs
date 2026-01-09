pub(crate) mod generated {
    include!(concat!(env!("OUT_DIR"), "/wgsl_binding_meta.rs"));
}

pub(crate) use generated::*;
