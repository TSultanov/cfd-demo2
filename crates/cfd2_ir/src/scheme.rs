#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scheme {
    #[default]
    Upwind,
    SecondOrderUpwind,
    QUICK,

    /// Second-order upwind with MinMod limiting.
    SecondOrderUpwindMinMod,
    /// Second-order upwind with VanLeer-style limiting.
    SecondOrderUpwindVanLeer,

    /// QUICK with MinMod limiting.
    QUICKMinMod,
    /// QUICK with VanLeer-style limiting.
    QUICKVanLeer,

    /// Kinetic-energy-preserving central two-point flux (Kennedy–Gruber
    /// mass/momentum pair with a conserved-average energy flux). No
    /// reconstruction, no limiter, ZERO artificial dissipation — pair with
    /// the structured selective filter for grid-mode stability. Only the
    /// density-based compressible flux modules implement it; other models
    /// fall back to `SecondOrderUpwindVanLeer` behaviour.
    Kep,
    /// SLAU2 all-speed AUSM-family flux (Shima & Kitamura, 2013) over
    /// vanLeer-limited MUSCL reconstruction: low-Mach-consistent upwind
    /// dissipation whose mass-flux pressure-diffusion term keeps the
    /// collocated pressure–velocity coupling (no odd-even decoupling).
    /// Only the density-based compressible flux modules implement it; other
    /// models fall back to `SecondOrderUpwindVanLeer` behaviour.
    Slau2,
}

impl Scheme {
    pub fn gpu_id(self) -> u32 {
        match self {
            Scheme::Upwind => 0,
            Scheme::SecondOrderUpwind => 1,
            Scheme::QUICK => 2,
            Scheme::SecondOrderUpwindMinMod => 3,
            Scheme::SecondOrderUpwindVanLeer => 4,
            Scheme::QUICKMinMod => 5,
            Scheme::QUICKVanLeer => 6,
            Scheme::Kep => 7,
            Scheme::Slau2 => 8,
        }
    }

    pub fn from_gpu_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Scheme::Upwind),
            1 => Some(Scheme::SecondOrderUpwind),
            2 => Some(Scheme::QUICK),
            3 => Some(Scheme::SecondOrderUpwindMinMod),
            4 => Some(Scheme::SecondOrderUpwindVanLeer),
            5 => Some(Scheme::QUICKMinMod),
            6 => Some(Scheme::QUICKVanLeer),
            7 => Some(Scheme::Kep),
            8 => Some(Scheme::Slau2),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Scheme::Upwind => "upwind",
            Scheme::SecondOrderUpwind => "sou",
            Scheme::QUICK => "quick",
            Scheme::SecondOrderUpwindMinMod => "sou_minmod",
            Scheme::SecondOrderUpwindVanLeer => "sou_vanleer",
            Scheme::QUICKMinMod => "quick_minmod",
            Scheme::QUICKVanLeer => "quick_vanleer",
            Scheme::Kep => "kep",
            Scheme::Slau2 => "slau2",
        }
    }
}

impl std::str::FromStr for Scheme {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "upwind" => Ok(Scheme::Upwind),
            "sou" | "second_order_upwind" | "second-order-upwind" => Ok(Scheme::SecondOrderUpwind),
            "quick" => Ok(Scheme::QUICK),
            "sou_minmod" | "sou-minmod" => Ok(Scheme::SecondOrderUpwindMinMod),
            "sou_vanleer" | "sou-vanleer" => Ok(Scheme::SecondOrderUpwindVanLeer),
            "quick_minmod" | "quick-minmod" => Ok(Scheme::QUICKMinMod),
            "quick_vanleer" | "quick-vanleer" => Ok(Scheme::QUICKVanLeer),
            "kep" => Ok(Scheme::Kep),
            "slau2" => Ok(Scheme::Slau2),
            _ => Err(format!("unknown scheme: {}", value)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheme_as_str_matches_expected_names() {
        assert_eq!(Scheme::Upwind.as_str(), "upwind");
        assert_eq!(Scheme::SecondOrderUpwind.as_str(), "sou");
        assert_eq!(Scheme::QUICK.as_str(), "quick");
        assert_eq!(Scheme::SecondOrderUpwindMinMod.as_str(), "sou_minmod");
        assert_eq!(Scheme::SecondOrderUpwindVanLeer.as_str(), "sou_vanleer");
        assert_eq!(Scheme::QUICKMinMod.as_str(), "quick_minmod");
        assert_eq!(Scheme::QUICKVanLeer.as_str(), "quick_vanleer");
    }

    #[test]
    fn scheme_from_str_parses_aliases() {
        assert_eq!("upwind".parse::<Scheme>().unwrap(), Scheme::Upwind);
        assert_eq!("sou".parse::<Scheme>().unwrap(), Scheme::SecondOrderUpwind);
        assert_eq!(
            "second_order_upwind".parse::<Scheme>().unwrap(),
            Scheme::SecondOrderUpwind
        );
        assert_eq!(
            "second-order-upwind".parse::<Scheme>().unwrap(),
            Scheme::SecondOrderUpwind
        );
        assert_eq!("quick".parse::<Scheme>().unwrap(), Scheme::QUICK);
        assert_eq!(
            "sou_minmod".parse::<Scheme>().unwrap(),
            Scheme::SecondOrderUpwindMinMod
        );
        assert_eq!(
            "sou-minmod".parse::<Scheme>().unwrap(),
            Scheme::SecondOrderUpwindMinMod
        );
        assert_eq!(
            "sou_vanleer".parse::<Scheme>().unwrap(),
            Scheme::SecondOrderUpwindVanLeer
        );
        assert_eq!(
            "sou-vanleer".parse::<Scheme>().unwrap(),
            Scheme::SecondOrderUpwindVanLeer
        );
        assert_eq!(
            "quick_minmod".parse::<Scheme>().unwrap(),
            Scheme::QUICKMinMod
        );
        assert_eq!(
            "quick-minmod".parse::<Scheme>().unwrap(),
            Scheme::QUICKMinMod
        );
        assert_eq!(
            "quick_vanleer".parse::<Scheme>().unwrap(),
            Scheme::QUICKVanLeer
        );
        assert_eq!(
            "quick-vanleer".parse::<Scheme>().unwrap(),
            Scheme::QUICKVanLeer
        );
    }

    #[test]
    fn scheme_from_str_errors_on_unknown() {
        let err = "nope".parse::<Scheme>().unwrap_err();
        assert!(err.contains("unknown scheme"));
    }

    #[test]
    fn scheme_from_gpu_id_roundtrips() {
        for scheme in [
            Scheme::Upwind,
            Scheme::SecondOrderUpwind,
            Scheme::QUICK,
            Scheme::SecondOrderUpwindMinMod,
            Scheme::SecondOrderUpwindVanLeer,
            Scheme::QUICKMinMod,
            Scheme::QUICKVanLeer,
            Scheme::Kep,
            Scheme::Slau2,
        ] {
            assert_eq!(Scheme::from_gpu_id(scheme.gpu_id()), Some(scheme));
        }
        assert_eq!(Scheme::from_gpu_id(999), None);
    }

    #[test]
    fn scheme_flux_family_names_roundtrip() {
        assert_eq!(Scheme::Kep.as_str(), "kep");
        assert_eq!(Scheme::Slau2.as_str(), "slau2");
        assert_eq!("kep".parse::<Scheme>().unwrap(), Scheme::Kep);
        assert_eq!("slau2".parse::<Scheme>().unwrap(), Scheme::Slau2);
    }
}
