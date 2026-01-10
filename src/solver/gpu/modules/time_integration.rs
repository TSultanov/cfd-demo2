use super::constants::ConstantsModule;

#[derive(Debug, Clone)]
pub struct TimeIntegrationModule {
    pub time: f64,
    pub dt: f32,
    pub dt_old: f32,
    pub step_count: u64,
}

impl TimeIntegrationModule {
    pub fn new() -> Self {
        Self {
            time: 0.0,
            dt: 1e-4,
            dt_old: 1e-4,
            step_count: 0,
        }
    }

    pub fn set_dt(&mut self, dt: f32, constants: &mut ConstantsModule, queue: &wgpu::Queue) {
        self.dt = dt;

        // If this is the very first configuration (time=0, step=0), seed dt_old
        // so that schemes like BDF2 start with a valid ratio (dt/dt_old = 1).
        if self.step_count == 0 && self.time == 0.0 {
            self.dt_old = dt;
        }

        let values = constants.values_mut();
        values.dt = self.dt;
        if self.step_count == 0 && self.time == 0.0 {
            values.dt_old = self.dt_old;
        }
        constants.write(queue);
    }

    /// Prepare for a new time step.
    /// Advances `time` by `dt` (t^{n+1} = t^n + dt).
    pub fn prepare_step(&mut self, constants: &mut ConstantsModule, queue: &wgpu::Queue) {
        self.time += self.dt as f64;

        let values = constants.values_mut();
        values.time = self.time as f32;
        constants.write(queue);
    }

    /// Finalize the current time step.
    /// Rotates `dt` into `dt_old` and increments step count.
    pub fn finalize_step(&mut self, constants: &mut ConstantsModule, queue: &wgpu::Queue) {
        self.dt_old = self.dt;
        self.step_count += 1;

        let values = constants.values_mut();
        values.dt_old = self.dt_old;
        constants.write(queue);
    }

    /// For restart or initialization scenarios
    pub fn initialize(
        &mut self,
        time: f64,
        dt: f32,
        constants: &mut ConstantsModule,
        queue: &wgpu::Queue,
    ) {
        self.time = time;
        self.dt = dt;
        self.dt_old = dt; // Best guess on restart if unknown
                          // step_count? Keep as is or reset? Probably keep as 0 if unknown.

        let values = constants.values_mut();
        values.time = self.time as f32;
        values.dt = self.dt;
        values.dt_old = self.dt_old;
        constants.write(queue);
    }
}

impl Default for TimeIntegrationModule {
    fn default() -> Self {
        Self::new()
    }
}
