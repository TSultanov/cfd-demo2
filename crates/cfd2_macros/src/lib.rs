//! Proc-macros for automatic port derivation.
//!
//! This crate provides derive macros for:
//! - `ModulePorts` - Derive port requirements for solver modules
//! - `PortSet` - Derive parameter/field port sets

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Expr, Lit, Meta, Type};

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
#[proc_macro_derive(PortSet, attributes(param, field, buffer))]
pub fn derive_port_set(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Parse struct fields and generate registration/lookup code
    let fields = match &input.data {
        syn::Data::Struct(data) => &data.fields,
        _ => {
            return syn::Error::new_spanned(input, "PortSet can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    // Generate registration code for each field
    let register_fields: Vec<_> = fields
        .iter()
        .filter_map(generate_register_field)
        .collect();



    // Generate field names for struct construction
    let field_names: Vec<_> = fields
        .iter()
        .map(|field| {
            field
                .ident
                .as_ref()
                .expect("PortSet only supports named structs")
        })
        .collect();

    // Generate compile-time validation
    let validations = generate_validations(fields);

    // Generate port manifest entries
    let param_specs: Vec<_> = fields
        .iter()
        .filter_map(generate_param_spec)
        .collect();

    let field_specs: Vec<_> = fields
        .iter()
        .filter_map(generate_field_spec)
        .collect();

    let buffer_specs: Vec<_> = fields
        .iter()
        .filter_map(generate_buffer_spec)
        .collect();

    // Check if we found any specs
    if param_specs.is_empty() && field_specs.is_empty() && buffer_specs.is_empty() {
        return syn::Error::new_spanned(
            &input.ident,
            "PortSet derive requires at least one field with #[param], #[field], or #[buffer] attribute",
        )
        .to_compile_error()
        .into();
    }

    let expanded = quote! {
        #validations

        #[automatically_derived]
        impl #impl_generics ::cfd2::solver::model::ports::PortSetTrait for #name #ty_generics #where_clause {
            const SET_NAME: &'static str = stringify!(#name);

            fn register(
                &self,
                registry: &mut ::cfd2::solver::model::ports::PortRegistry,
            ) -> Result<(), ::cfd2::solver::model::ports::PortRegistryError> {
                // Registration is idempotent; from_registry handles actual registration
                // This method is kept for backward compatibility but is a no-op
                Ok(())
            }

            fn from_registry(
                registry: &mut ::cfd2::solver::model::ports::PortRegistry,
            ) -> Result<Self, ::cfd2::solver::model::ports::PortRegistryError>
            where
                Self: Sized
            {
                #(#register_fields)*

                Ok(Self {
                    #(#field_names),*
                })
            }

            fn port_manifest() -> ::cfd2::solver::model::module::PortManifest
            where
                Self: Sized
            {
                ::cfd2::solver::model::module::PortManifest {
                    params: vec![#(#param_specs),*],
                    fields: vec![#(#field_specs),*],
                    buffers: vec![#(#buffer_specs),*],
                    gradient_targets: vec![],
                    resolved_state_slots: None,
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Generate registration code for a single field.
fn generate_register_field(field: &syn::Field) -> Option<proc_macro2::TokenStream> {
    let field_name = field.ident.as_ref()?;

    // Check for param attribute
    if let Some(_attr) = field.attrs.iter().find(|a| a.path().is_ident("param")) {
        return generate_param_registration(field, field_name);
    }

    // Check for field attribute
    if let Some(_attr) = field.attrs.iter().find(|a| a.path().is_ident("field")) {
        return generate_field_registration(field, field_name);
    }

    // Check for buffer attribute
    if let Some(_attr) = field.attrs.iter().find(|a| a.path().is_ident("buffer")) {
        return generate_buffer_registration(field, field_name);
    }

    None
}

/// Generate parameter registration code.
fn generate_param_registration(
    field: &syn::Field,
    field_name: &syn::Ident,
) -> Option<proc_macro2::TokenStream> {
    let args = field
        .attrs
        .iter()
        .find(|a| a.path().is_ident("param"))
        .and_then(|attr| parse_param_args(attr).ok())?;

    let name = args.name?;
    let wgsl = args.wgsl?;

    Some(quote! {
        let #field_name = registry.register_param(
            #name,
            #wgsl,
        )?;
    })
}

/// Generate field registration code.
fn generate_field_registration(
    field: &syn::Field,
    field_name: &syn::Ident,
) -> Option<proc_macro2::TokenStream> {
    let args = field
        .attrs
        .iter()
        .find(|a| a.path().is_ident("field"))
        .and_then(|attr| parse_field_args(attr).ok())?;
    let name = args.name?;

    // Extract dimension and kind from the field type
    // Expected: FieldPort<Dimension, Kind>
    let (dim, kind) = extract_field_port_types(&field.ty)?;

    Some(quote! {
        let #field_name = registry.register_field::<#dim, #kind>(#name)?;
    })
}

/// Generate buffer registration code.
fn generate_buffer_registration(
    field: &syn::Field,
    field_name: &syn::Ident,
) -> Option<proc_macro2::TokenStream> {
    let args = field
        .attrs
        .iter()
        .find(|a| a.path().is_ident("buffer"))
        .and_then(|attr| parse_buffer_args(attr).ok())?;

    let name = args.name?;
    let group = args.group.unwrap_or(0);
    let binding = args.binding.unwrap_or(0);

    // Extract type and access mode from the field type
    // Expected: BufferPort<Type, AccessMode>
    let (buf_type, access) = extract_buffer_port_types(&field.ty)?;

    Some(quote! {
        let #field_name = registry.register_buffer::<#buf_type, #access>(
            #name,
            #group,
            #binding,
        )?;
    })
}


/// Generate compile-time validations.
fn generate_validations(fields: &syn::Fields) -> proc_macro2::TokenStream {
    let mut validations = Vec::new();
    let mut seen_params: Vec<String> = Vec::new();
    let mut seen_fields: Vec<String> = Vec::new();
    let mut seen_wgsl: Vec<String> = Vec::new();

    for field in fields.iter() {
        let field_name = field
            .ident
            .as_ref()
            .map(|i| i.to_string())
            .unwrap_or_default();

        // Check param attributes
        if let Some(attr) = field.attrs.iter().find(|a| a.path().is_ident("param")) {
            if let Ok(args) = parse_param_args(attr) {
                // Check for duplicate param keys
                if let Some(ref name) = args.name {
                    if seen_params.contains(name) {
                        let msg = format!("Duplicate parameter key: {}", name);
                        validations.push(quote! {
                            compile_error!(#msg);
                        });
                    }
                    seen_params.push(name.clone());
                }

                // Check for duplicate wgsl names
                if let Some(ref wgsl) = args.wgsl {
                    if seen_wgsl.contains(wgsl) {
                        let msg = format!("Duplicate wgsl field name: {}", wgsl);
                        validations.push(quote! {
                            compile_error!(#msg);
                        });
                    }
                    seen_wgsl.push(wgsl.clone());
                }

                // Validate field type is ParamPort
                if !is_param_port_type(&field.ty) {
                    let msg = format!(
                        "Field '{}' has #[param] attribute but is not a ParamPort type",
                        field_name
                    );
                    validations.push(quote! {
                        compile_error!(#msg);
                    });
                }
            }
        }

        // Check field attributes
        if let Some(attr) = field.attrs.iter().find(|a| a.path().is_ident("field")) {
            if let Ok(args) = parse_field_args(attr) {
                // Check for duplicate field names
                if let Some(ref name) = args.name {
                    if seen_fields.contains(name) {
                        let msg = format!("Duplicate field name: {}", name);
                        validations.push(quote! {
                            compile_error!(#msg);
                        });
                    }
                    seen_fields.push(name.clone());
                }

                // Validate field type is FieldPort
                if !is_field_port_type(&field.ty) {
                    let msg = format!(
                        "Field '{}' has #[field] attribute but is not a FieldPort type",
                        field_name
                    );
                    validations.push(quote! {
                        compile_error!(#msg);
                    });
                }
            }
        }

        // Check buffer attributes
        if field.attrs.iter().any(|a| a.path().is_ident("buffer")) {
            // Validate field type is BufferPort
            if !is_buffer_port_type(&field.ty) {
                let msg = format!(
                    "Field '{}' has #[buffer] attribute but is not a BufferPort type",
                    field_name
                );
                validations.push(quote! {
                    compile_error!(#msg);
                });
            }
        }
    }

    quote! {
        #(#validations)*
    }
}

/// Parse param attribute arguments.
#[derive(Default)]
struct ParamArgs {
    name: Option<String>,
    wgsl: Option<String>,
}

fn parse_param_args(attr: &syn::Attribute) -> Result<ParamArgs, syn::Error> {
    let mut args = ParamArgs::default();

    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("name") {
            let value = meta.value()?;
            let lit: Lit = value.parse()?;
            if let Lit::Str(s) = lit {
                args.name = Some(s.value());
            }
        } else if meta.path.is_ident("wgsl") {
            let value = meta.value()?;
            let lit: Lit = value.parse()?;
            if let Lit::Str(s) = lit {
                args.wgsl = Some(s.value());
            }
        }
        Ok(())
    })?;

    Ok(args)
}

/// Parse field attribute arguments.
#[derive(Default)]
struct FieldArgs {
    name: Option<String>,
}

fn parse_field_args(attr: &syn::Attribute) -> Result<FieldArgs, syn::Error> {
    let mut args = FieldArgs::default();

    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("name") {
            let value = meta.value()?;
            let lit: Lit = value.parse()?;
            if let Lit::Str(s) = lit {
                args.name = Some(s.value());
            }
        }
        Ok(())
    })?;

    Ok(args)
}

/// Parse buffer attribute arguments.
#[derive(Default)]
struct BufferArgs {
    name: Option<String>,
    group: Option<u32>,
    binding: Option<u32>,
}

fn parse_buffer_args(attr: &syn::Attribute) -> Result<BufferArgs, syn::Error> {
    let mut args = BufferArgs::default();

    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("name") {
            let value = meta.value()?;
            let lit: Lit = value.parse()?;
            if let Lit::Str(s) = lit {
                args.name = Some(s.value());
            }
        } else if meta.path.is_ident("group") {
            let value = meta.value()?;
            let lit: Lit = value.parse()?;
            if let Lit::Int(i) = lit {
                args.group = Some(i.base10_parse()?);
            }
        } else if meta.path.is_ident("binding") {
            let value = meta.value()?;
            let lit: Lit = value.parse()?;
            if let Lit::Int(i) = lit {
                args.binding = Some(i.base10_parse()?);
            }
        }
        Ok(())
    })?;

    Ok(args)
}

/// Extract dimension and kind types from FieldPort<Dim, Kind>.
fn extract_field_port_types(ty: &Type) -> Option<(Type, Type)> {
    // This is a simplified version - in production you'd want more robust parsing
    if let Type::Path(type_path) = ty {
        let segment = type_path.path.segments.first()?;
        if segment.ident == "FieldPort" {
            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                let types: Vec<_> = args.args.iter().collect();
                if types.len() >= 2 {
                    if let (syn::GenericArgument::Type(dim), syn::GenericArgument::Type(kind)) =
                        (types[0], types[1])
                    {
                        return Some((dim.clone(), kind.clone()));
                    }
                }
            }
        }
    }
    None
}

/// Extract type and access mode from BufferPort<Type, Access>.
fn extract_buffer_port_types(ty: &Type) -> Option<(Type, Type)> {
    if let Type::Path(type_path) = ty {
        let segment = type_path.path.segments.first()?;
        if segment.ident == "BufferPort" {
            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                let types: Vec<_> = args.args.iter().collect();
                if types.len() >= 2 {
                    if let (
                        syn::GenericArgument::Type(buf_type),
                        syn::GenericArgument::Type(access),
                    ) = (types[0], types[1])
                    {
                        return Some((buf_type.clone(), access.clone()));
                    }
                }
            }
        }
    }
    None
}

/// Check if a type is ParamPort<_, _>.
fn is_param_port_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "ParamPort";
        }
    }
    false
}

/// Check if a type is FieldPort<_, _>.
fn is_field_port_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "FieldPort";
        }
    }
    false
}

/// Check if a type is BufferPort<_, _>.
fn is_buffer_port_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "BufferPort";
        }
    }
    false
}

/// Generate a ParamSpec for the port manifest.
fn generate_param_spec(field: &syn::Field) -> Option<proc_macro2::TokenStream> {
    let attr = field.attrs.iter().find(|a| a.path().is_ident("param"))?;
    let args = parse_param_args(attr).ok()?;

    let name = args.name?;
    let wgsl = args.wgsl?;

    // Extract ParamPort<T, D> to get T and D
    let (param_type, dim) = extract_param_port_types(&field.ty)?;

    // Map param type to WGSL type string
    let wgsl_type = map_param_type_to_wgsl(&param_type);

    Some(quote! {
        ::cfd2::solver::model::module::ParamSpec {
            key: #name,
            wgsl_field: #wgsl,
            wgsl_type: #wgsl_type,
            unit: <#dim as ::cfd2::solver::model::ports::UnitDimension>::to_runtime(),
        }
    })
}

/// Generate a FieldSpec for the port manifest.
fn generate_field_spec(field: &syn::Field) -> Option<proc_macro2::TokenStream> {
    let args = field
        .attrs
        .iter()
        .find(|a| a.path().is_ident("field"))
        .and_then(|attr| parse_field_args(attr).ok())?;

    let name = args.name?;

    // Extract FieldPort<D, K> to get D and K
    let (dim, kind) = extract_field_port_types(&field.ty)?;

    // Map kind to PortFieldKind
    let field_kind = map_field_kind(&kind);

    Some(quote! {
        ::cfd2::solver::model::module::FieldSpec {
            name: #name,
            kind: #field_kind,
            unit: <#dim as ::cfd2::solver::model::ports::UnitDimension>::to_runtime(),
        }
    })
}

/// Generate a BufferSpec for the port manifest.
fn generate_buffer_spec(field: &syn::Field) -> Option<proc_macro2::TokenStream> {
    let attr = field.attrs.iter().find(|a| a.path().is_ident("buffer"))?;
    let args = parse_buffer_args(attr).ok()?;

    let name = args.name?;
    let group = args.group.unwrap_or(0);
    let binding = args.binding.unwrap_or(0);

    // Extract BufferPort<T, A> to get T and A
    let (buf_type, access) = extract_buffer_port_types(&field.ty)?;

    // Map buffer type to element WGSL type
    let elem_wgsl = map_buffer_type_to_wgsl(&buf_type);

    // Map access mode to BufferAccess
    let access_mode = map_buffer_access(&access);

    Some(quote! {
        ::cfd2::solver::model::module::BufferSpec {
            name: #name,
            group: #group,
            binding: #binding,
            elem_wgsl_type: #elem_wgsl,
            access: #access_mode,
        }
    })
}

/// Extract T and D from ParamPort<T, D>.
fn extract_param_port_types(ty: &Type) -> Option<(Type, Type)> {
    if let Type::Path(type_path) = ty {
        // Check if the last segment is ParamPort (handles both `ParamPort` and `crate::...::ParamPort`)
        let segment = type_path.path.segments.last()?;
        if segment.ident == "ParamPort" {
            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                let types: Vec<_> = args.args.iter().collect();
                if types.len() >= 2 {
                    if let (
                        syn::GenericArgument::Type(param_type),
                        syn::GenericArgument::Type(dim),
                    ) = (types[0], types[1])
                    {
                        return Some((param_type.clone(), dim.clone()));
                    }
                }
            }
        }
    }
    None
}

/// Map a param type to its WGSL type string.
fn map_param_type_to_wgsl(ty: &Type) -> proc_macro2::TokenStream {
    // Check if it's a known type by looking at the type name
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            let ident = &segment.ident;
            let name = ident.to_string();
            match name.as_str() {
                "F32" => return quote!("f32"),
                "F64" => return quote!("f32"), // WGSL doesn't have f64, use f32
                "I32" => return quote!("i32"),
                "U32" => return quote!("u32"),
                _ => {}
            }
        }
    }
    // Default to f32 for unknown types
    quote!("f32")
}

/// Map a field kind type to PortFieldKind.
fn map_field_kind(kind: &Type) -> proc_macro2::TokenStream {
    if let Type::Path(type_path) = kind {
        if let Some(segment) = type_path.path.segments.last() {
            let ident = &segment.ident;
            let name = ident.to_string();
            match name.as_str() {
                "Scalar" => return quote!(::cfd2::solver::model::module::PortFieldKind::Scalar),
                "Vector2" => return quote!(::cfd2::solver::model::module::PortFieldKind::Vector2),
                "Vector3" => return quote!(::cfd2::solver::model::module::PortFieldKind::Vector3),
                _ => {}
            }
        }
    }
    // Default to Scalar
    quote!(::cfd2::solver::model::module::PortFieldKind::Scalar)
}

/// Map a buffer type to its element WGSL type string.
fn map_buffer_type_to_wgsl(ty: &Type) -> proc_macro2::TokenStream {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            let ident = &segment.ident;
            let name = ident.to_string();
            match name.as_str() {
                "BufferF32" => return quote!("f32"),
                "BufferU32" => return quote!("u32"),
                "BufferI32" => return quote!("i32"),
                "BufferVec2F32" => return quote!("vec2<f32>"),
                "BufferVec3F32" => return quote!("vec3<f32>"),
                _ => {}
            }
        }
    }
    // Default to f32
    quote!("f32")
}

/// Map an access mode type to BufferAccess.
fn map_buffer_access(access: &Type) -> proc_macro2::TokenStream {
    if let Type::Path(type_path) = access {
        if let Some(segment) = type_path.path.segments.last() {
            let ident = &segment.ident;
            let name = ident.to_string();
            match name.as_str() {
                "ReadOnly" => return quote!(::cfd2::solver::model::module::BufferAccess::ReadOnly),
                "ReadWrite" => {
                    return quote!(::cfd2::solver::model::module::BufferAccess::ReadWrite)
                }
                _ => {}
            }
        }
    }
    // Default to ReadWrite
    quote!(::cfd2::solver::model::module::BufferAccess::ReadWrite)
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
