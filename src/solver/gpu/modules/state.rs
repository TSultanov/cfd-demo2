use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone)]
pub struct PingPongState {
    step_index: Arc<AtomicUsize>,
    buffers: [wgpu::Buffer; 3],
}

impl PingPongState {
    pub fn new(buffers: [wgpu::Buffer; 3]) -> Self {
        Self {
            step_index: Arc::new(AtomicUsize::new(0)),
            buffers,
        }
    }

    pub fn step_handle(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.step_index)
    }

    pub fn step_index(&self) -> usize {
        self.step_index.load(Ordering::Relaxed) % 3
    }

    pub fn set_step_index(&self, idx: usize) {
        self.step_index.store(idx % 3, Ordering::Relaxed);
    }

    pub fn advance(&self) -> usize {
        let next = (self.step_index() + 1) % 3;
        self.step_index.store(next, Ordering::Relaxed);
        next
    }

    pub fn buffers(&self) -> &[wgpu::Buffer; 3] {
        &self.buffers
    }

    pub fn indices(&self) -> (usize, usize, usize) {
        ping_pong_indices(self.step_index())
    }

    pub fn state(&self) -> &wgpu::Buffer {
        let (idx_state, _, _) = self.indices();
        &self.buffers[idx_state]
    }

    pub fn state_old(&self) -> &wgpu::Buffer {
        let (_, idx_old, _) = self.indices();
        &self.buffers[idx_old]
    }

    pub fn state_old_old(&self) -> &wgpu::Buffer {
        let (_, _, idx_old_old) = self.indices();
        &self.buffers[idx_old_old]
    }

    pub fn write_all(&self, queue: &wgpu::Queue, bytes: &[u8]) {
        for buf in &self.buffers {
            queue.write_buffer(buf, 0, bytes);
        }
    }
}

pub fn ping_pong_indices(i: usize) -> (usize, usize, usize) {
    match i % 3 {
        0 => (0, 1, 2),
        1 => (2, 0, 1),
        2 => (1, 2, 0),
        _ => (0, 1, 2),
    }
}
