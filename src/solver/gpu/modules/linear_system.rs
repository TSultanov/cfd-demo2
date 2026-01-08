use super::ports::{BufF32, BufU32, Port};

#[derive(Clone, Copy, Debug)]
pub struct LinearSystemPorts {
    pub row_offsets: Port<BufU32>,
    pub col_indices: Port<BufU32>,
    pub values: Port<BufF32>,
    pub rhs: Port<BufF32>,
    pub x: Port<BufF32>,
}

