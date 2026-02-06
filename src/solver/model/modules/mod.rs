pub mod eos {
    include!("eos.rs");
}

// eos_ports is defined separately to avoid build script issues with proc macros
pub mod eos_ports {
    include!("eos_ports.rs");
}

pub mod rhie_chow {
    include!("rhie_chow.rs");
}

pub mod flux_module {
    include!("flux_module.rs");
}

pub mod generic_coupled {
    include!("generic_coupled.rs");
}

// generic_coupled_ports is defined separately to avoid build script issues with proc macros
pub mod generic_coupled_ports {
    include!("generic_coupled_ports.rs");
}
