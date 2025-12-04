// Async Buffer Reading Infrastructure
//
// This module provides non-blocking GPU buffer reading using double-buffered staging.
// Instead of blocking on device.poll(Wait), we use async mapping that allows the
// GPU to continue working while we wait for previous results.

use wgpu;

/// Double-buffered async scalar reader for convergence checks.
/// This allows reading scalar values (like residuals) without blocking the GPU pipeline.
pub struct AsyncScalarReader {
    /// Two staging buffers for double-buffering
    staging_buffers: [wgpu::Buffer; 2],
    /// Which buffer is currently being used for reading (0 or 1)
    current_buffer: usize,
    /// Whether each buffer has a pending read
    pending: [bool; 2],
    /// Last read value (from the completed read)
    last_value: Option<Vec<u8>>,
    /// Receiver for async completion
    receivers: [Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>; 2],
    /// Size of the read
    size: u64,
}

impl AsyncScalarReader {
    /// Create a new async scalar reader
    pub fn new(device: &wgpu::Device, size: u64) -> Self {
        let create_staging = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Async Scalar Staging"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        Self {
            staging_buffers: [create_staging(), create_staging()],
            current_buffer: 0,
            pending: [false, false],
            last_value: None,
            receivers: [None, None],
            size,
        }
    }

    /// Start an async read of a scalar from a GPU buffer.
    /// This submits the copy command and starts the async map, but does NOT block.
    pub fn start_read(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source_buffer: &wgpu::Buffer,
        source_offset: u64,
    ) {
        // Poll device first to give pending operations a chance to complete
        let _ = device.poll(wgpu::PollType::Poll);

        let buffer_idx = self.current_buffer;

        // If the current buffer has a pending read, try to complete it first
        if self.pending[buffer_idx] {
            self.try_complete_read(buffer_idx);
        }

        // If still pending, switch to other buffer
        if self.pending[buffer_idx] {
            self.current_buffer = 1 - self.current_buffer;
            let buffer_idx = self.current_buffer;
            if self.pending[buffer_idx] {
                self.try_complete_read(buffer_idx);
            }

            // If BOTH buffers are still pending, we must wait for one to complete
            // to avoid the "buffer still mapped" error
            if self.pending[buffer_idx] {
                self.wait_for_buffer(device, buffer_idx);
            }
        }

        let buffer_idx = self.current_buffer;
        let staging = &self.staging_buffers[buffer_idx];

        // Submit copy command
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Async Scalar Copy"),
        });
        encoder.copy_buffer_to_buffer(source_buffer, source_offset, staging, 0, self.size);
        queue.submit(Some(encoder.finish()));

        // Start async map
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.receivers[buffer_idx] = Some(rx);
        self.pending[buffer_idx] = true;

        // Switch to the other buffer for next read
        self.current_buffer = 1 - self.current_buffer;
    }

    /// Wait for a specific buffer to complete (blocking)
    fn wait_for_buffer(&mut self, device: &wgpu::Device, buffer_idx: usize) {
        while self.pending[buffer_idx] {
            let _ = device.poll(wgpu::PollType::Poll);
            self.try_complete_read(buffer_idx);
            if self.pending[buffer_idx] {
                std::thread::yield_now();
            }
        }
    }

    /// Try to complete a pending read without blocking
    fn try_complete_read(&mut self, buffer_idx: usize) {
        if !self.pending[buffer_idx] {
            return;
        }

        if let Some(rx) = self.receivers[buffer_idx].take() {
            // Try to receive without blocking
            match rx.try_recv() {
                Ok(Ok(())) => {
                    // Read completed successfully
                    let staging = &self.staging_buffers[buffer_idx];
                    let slice = staging.slice(..);
                    let data = slice.get_mapped_range();
                    let value = data.to_vec();
                    drop(data);
                    staging.unmap();

                    self.last_value = Some(value);
                    self.pending[buffer_idx] = false;
                }
                Ok(Err(_)) => {
                    // Map failed
                    self.pending[buffer_idx] = false;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Still pending, put receiver back
                    self.receivers[buffer_idx] = Some(rx);
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Channel disconnected, shouldn't happen
                    self.pending[buffer_idx] = false;
                }
            }
        }
    }

    /// Poll for completion without blocking. Call device.poll(Poll) before this.
    pub fn poll(&mut self) {
        self.try_complete_read(0);
        self.try_complete_read(1);
    }

    /// Get the last successfully read value as f32, if any
    pub fn get_last_value(&self) -> Option<f32> {
        self.last_value
            .as_ref()
            .map(|v| *bytemuck::from_bytes(&v[0..4]))
    }

    /// Get the last successfully read value as Vec<f32>, if any
    pub fn get_last_value_vec(&self, count: usize) -> Option<Vec<f32>> {
        self.last_value.as_ref().map(|v| {
            let slice: &[f32] = bytemuck::cast_slice(v);
            slice[0..count].to_vec()
        })
    }

    /// Check if a read is currently pending
    pub fn is_pending(&self) -> bool {
        self.pending[0] || self.pending[1]
    }

    /// Block until all pending reads complete (for cleanup or forced sync)
    pub fn flush(&mut self, device: &wgpu::Device) {
        while self.is_pending() {
            let _ = device.poll(wgpu::PollType::Poll);
            self.poll();
        }
    }

    /// Reset the reader state (clear last value)
    pub fn reset(&mut self) {
        self.last_value = None;
    }
}

/// A reusable staging buffer that can be mapped asynchronously
pub struct AsyncStagingBuffer {
    buffer: wgpu::Buffer,
    size: u64,
    receiver: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    is_pending: bool,
}

impl AsyncStagingBuffer {
    pub fn new(device: &wgpu::Device, size: u64, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size,
            receiver: None,
            is_pending: false,
        }
    }

    /// Start copying from source buffer and initiate async map
    pub fn start_read(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) {
        assert!(size <= self.size);

        // Submit copy
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Async Staging Copy"),
        });
        encoder.copy_buffer_to_buffer(source, offset, &self.buffer, 0, size);
        queue.submit(Some(encoder.finish()));

        // Start async map
        let slice = self.buffer.slice(0..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.receiver = Some(rx);
        self.is_pending = true;
    }

    /// Try to get the result without blocking. Returns Some if ready, None if still pending.
    pub fn try_get_result(&mut self) -> Option<Vec<u8>> {
        if !self.is_pending {
            return None;
        }

        if let Some(rx) = self.receiver.take() {
            match rx.try_recv() {
                Ok(Ok(())) => {
                    let slice = self.buffer.slice(..);
                    let data = slice.get_mapped_range();
                    let result = data.to_vec();
                    drop(data);
                    self.buffer.unmap();
                    self.is_pending = false;
                    Some(result)
                }
                Ok(Err(_)) => {
                    self.is_pending = false;
                    None
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    self.receiver = Some(rx);
                    None
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.is_pending = false;
                    None
                }
            }
        } else {
            None
        }
    }

    /// Block until the result is ready
    pub fn wait_for_result(&mut self, device: &wgpu::Device) -> Option<Vec<u8>> {
        while self.is_pending {
            let _ = device.poll(wgpu::PollType::Poll);
            if let Some(result) = self.try_get_result() {
                return Some(result);
            }
            std::thread::yield_now();
        }
        None
    }

    pub fn is_pending(&self) -> bool {
        self.is_pending
    }
}

/// Non-blocking buffer read using async mapping
/// Returns immediately, caller should poll device and check for completion
pub fn start_async_buffer_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
) -> (
    wgpu::Buffer,
    std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
) {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Async Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Async Buffer Copy"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    (staging, rx)
}
