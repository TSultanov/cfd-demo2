#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scheme {
    #[default]
    Upwind,
    SecondOrderUpwind,
    QUICK,
}

impl Scheme {
    pub fn gpu_id(self) -> u32 {
        match self {
            Scheme::Upwind => 0,
            Scheme::SecondOrderUpwind => 1,
            Scheme::QUICK => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Scheme::Upwind => "upwind",
            Scheme::SecondOrderUpwind => "sou",
            Scheme::QUICK => "quick",
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
    }

    #[test]
    fn scheme_from_str_errors_on_unknown() {
        let err = "nope".parse::<Scheme>().unwrap_err();
        assert!(err.contains("unknown scheme"));
    }
}
