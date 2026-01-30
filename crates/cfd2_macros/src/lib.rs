//! Proc-macros for automatic port derivation.
//!
//! This crate provides derive macros for:
//! - `ModulePorts` - Derive port requirements for solver modules
//! - `PortSet` - Derive parameter/field port sets

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Expr, Lit, Meta};

/// Derive macro for module port requirements.
#[proc_macro_derive(ModulePorts, attributes(port))]
pub fn derive_module_ports(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let module_name = extract_module_name(&input.attrs);

    let expanded = quote! {
        #[automatically_derived]
        impl #impl_generics ::cfd2::solver::model::ports::ModulePortsTrait for #name #ty_generics #where_clause {
            type PortSet = #name;

            const MODULE_NAME: &'static str = #module_name;

            fn module_name(&self) -> &'static str {
                Self::MODULE_NAME
            }

            fn register_ports(
                &self,
                registry: &mut ::cfd2::solver::model::ports::PortRegistry,
            ) -> Result<(), ::cfd2::solver::model::ports::PortRegistryError> {
                <Self as ::cfd2::solver::model::ports::PortSetTrait>::register(self, registry)
            }

            fn port_set(&self) -> &Self::PortSet {
                self
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for port sets.
#[proc_macro_derive(PortSet, attributes(param))]
pub fn derive_port_set(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        #[automatically_derived]
        impl #impl_generics ::cfd2::solver::model::ports::PortSetTrait for #name #ty_generics #where_clause {
            const SET_NAME: &'static str = stringify!(#name);

            fn register(
                &self,
                _registry: &mut ::cfd2::solver::model::ports::PortRegistry,
            ) -> Result<(), ::cfd2::solver::model::ports::PortRegistryError> {
                // Registration logic will be added later
                Ok(())
            }

            fn from_registry(
                _registry: &mut ::cfd2::solver::model::ports::PortRegistry,
            ) -> Result<Self, ::cfd2::solver::model::ports::PortRegistryError>
            where
                Self: Sized
            {
                todo!("PortSet::from_registry not yet implemented")
            }
        }
    };

    TokenStream::from(expanded)
}

fn extract_module_name(attrs: &[syn::Attribute]) -> String {
    for attr in attrs {
        if attr.path().is_ident("port") {
            if let Meta::List(meta_list) = &attr.meta {
                let nested = meta_list.parse_args::<Meta>().expect("Expected meta");
                if let Meta::NameValue(nv) = nested {
                    if nv.path.is_ident("module") {
                        if let Expr::Lit(syn::ExprLit {
                            lit: Lit::Str(s), ..
                        }) = &nv.value
                        {
                            return s.value();
                        }
                    }
                }
            }
        }
    }
    panic!("#[port(module = \"name\")] attribute is required");
}
