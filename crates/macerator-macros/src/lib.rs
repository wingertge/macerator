use darling::FromMeta;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{parse_quote, LifetimeParam, Type};
use syn::{spanned::Spanned, FnArg, GenericParam, ItemFn, Pat};
use syn::{Expr, Lifetime};

#[derive(FromMeta, Default)]
#[darling(default)]
struct WithSimdOpts {
    #[darling(default)]
    arch: Option<Expr>,
}

#[proc_macro_attribute]
pub fn with_simd(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    match with_simd_impl(attr.into(), item.into()) {
        Ok(out) => out.into(),
        Err(e) => e.into_compile_error().into(),
    }
}

const ANON_LIFETIME: &str = "'__simd";

fn with_simd_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream, syn::Error> {
    let opts = match attr.is_empty() {
        true => WithSimdOpts::default(),
        false => {
            let meta = syn::parse2::<syn::Meta>(attr)?;
            WithSimdOpts::from_meta(&meta)?
        }
    };

    let arch = opts.arch.unwrap_or(parse_quote!(macerator::Arch::new()));
    let func = syn::parse2::<syn::ItemFn>(item)?;

    let ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = func.clone();

    let name = &sig.ident;

    let lifetimes = sig.generics.lifetimes();
    let type_params = sig.generics.type_params();
    let const_params = sig.generics.const_params();

    let mut outer_fn_sig = sig.clone();
    outer_fn_sig.generics.params = lifetimes
        .map(|l| GenericParam::Lifetime(l.clone()))
        .chain(type_params.skip(1).map(|t| GenericParam::Type(t.clone())))
        .chain(const_params.map(|c| GenericParam::Const(c.clone())))
        .collect();
    let mut inner_fn_sig = sig.clone();
    inner_fn_sig.ident = format_ident!("{}_impl", name);
    let struct_name = format_ident!("{}_struct", name);

    let fields = sig
        .inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Receiver(_) => Err(syn::Error::new(arg.span(), "Can't use macro on methods")),
            FnArg::Typed(pat_type) => {
                let ident = match &*pat_type.pat {
                    Pat::Ident(pat_ident) => &pat_ident.ident,
                    _ => todo!(),
                };
                let mut ty = *pat_type.ty.clone();
                let has_implicit_ref = add_named_lifetimes(&mut ty);
                Ok((ident, ty, has_implicit_ref))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let anon_lifetime = Lifetime::new(ANON_LIFETIME, Span::call_site());

    let output_ty = match sig.output.clone() {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, mut ty) => {
            add_named_lifetimes(&mut ty);
            quote! { #ty }
        }
    };

    let inner_name = &inner_fn_sig.ident;

    let mut struct_generics = outer_fn_sig.generics.clone();
    struct_generics.params.insert(
        0,
        GenericParam::Lifetime(LifetimeParam::new(anon_lifetime.clone())),
    );

    let (impl_generics, type_generics, where_clause) = struct_generics.split_for_impl();

    let field_decl = fields.iter().map(|(ident, ty, _)| quote![#ident: #ty]);
    let field_names = fields.iter().map(|it| it.0).collect::<Vec<_>>();

    let simd_generic_name = sig.generics.type_params().next().unwrap().ident.clone();

    let mut inner_generics_no_lifetime = inner_fn_sig.generics.clone();
    inner_generics_no_lifetime.params = inner_generics_no_lifetime
        .params
        .into_iter()
        .filter(|it| !matches!(it, GenericParam::Lifetime(_)))
        .collect();
    let (_, inner_generics, _) = inner_generics_no_lifetime.split_for_impl();

    let turbofish = inner_generics.as_turbofish();

    let mut struct_generics_no_lifetime = struct_generics.clone();
    struct_generics_no_lifetime.params = struct_generics_no_lifetime
        .params
        .into_iter()
        .filter(|it| !matches!(it, GenericParam::Lifetime(_)))
        .collect();
    let (_, struct_turbofish_generics, _) = struct_generics_no_lifetime.split_for_impl();
    let struct_turbofish = struct_turbofish_generics.as_turbofish();

    Ok(quote! {
        #(#attrs)*
        #vis #outer_fn_sig {
            #[allow(non_camel_case_types)]
            struct #struct_name #impl_generics #where_clause {
                #(#field_decl,)*
                __lifetime: ::core::marker::PhantomData<&#anon_lifetime ()>,
            };

            impl #impl_generics macerator::WithSimd for #struct_name #type_generics #where_clause {
                type Output = #output_ty;

                #[inline(always)]
                fn with_simd<#simd_generic_name: macerator::Simd>(self) -> <Self as macerator::WithSimd>::Output {
                    let Self {
                        #(#field_names,)*
                        ..
                    } = self;
                    #[allow(unused_unsafe)]
                    unsafe {
                        #inner_name #turbofish(#(#field_names,)*)
                    }
                }
            }

            (#arch).dispatch( #struct_name #struct_turbofish { __lifetime: core::marker::PhantomData, #(#field_names,)* } )
        }

        #(#attrs)*
        #inner_fn_sig #block
    })
}

fn add_named_lifetimes(ty: &mut Type) -> bool {
    match ty {
        Type::Array(type_array) => add_named_lifetimes(&mut type_array.elem),
        Type::Group(type_group) => add_named_lifetimes(&mut type_group.elem),
        Type::Paren(type_paren) => add_named_lifetimes(&mut type_paren.elem),
        Type::Ptr(type_ptr) => add_named_lifetimes(&mut type_ptr.elem),
        Type::Reference(type_reference) => {
            if type_reference.lifetime.is_none() {
                type_reference.lifetime = Some(Lifetime::new(
                    ANON_LIFETIME,
                    type_reference.and_token.span(),
                ));
                true
            } else {
                false
            }
        }
        Type::Slice(type_slice) => add_named_lifetimes(&mut type_slice.elem),
        Type::Tuple(type_tuple) => type_tuple.elems.iter_mut().any(add_named_lifetimes),
        _ => false,
    }
}
