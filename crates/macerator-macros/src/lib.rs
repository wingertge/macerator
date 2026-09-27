use darling::FromMeta;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{parse_quote, LifetimeParam, Token, Type};
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

    let unsafety = opts.arch.as_ref().map(|arch| Token![unsafe](arch.span()));
    let arch = opts
        .arch
        .unwrap_or(parse_quote!(macerator::AutoArch::new()));
    let func = syn::parse2::<syn::ItemFn>(item)?;

    let ItemFn {
        attrs,
        vis,
        sig,
        block,
        modifiers: _,
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
        .enumerate()
        .map(|(i, arg)| match arg {
            FnArg::Receiver(_) => Err(syn::Error::new(arg.span(), "Can't use macro on methods")),
            FnArg::Typed(pat_type) => {
                let ident = match &*pat_type.pat {
                    Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                    Pat::Wild(_) => format_ident!("__arg{i}"),
                    pat => {
                        return Err(syn::Error::new(
                            pat.span(),
                            "`with_simd` arguments must be plain identifiers",
                        ))
                    }
                };
                let mut ty = *pat_type.ty.clone();
                let has_implicit_ref = add_named_lifetimes(&mut ty);
                Ok((ident, ty, has_implicit_ref))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // The dispatcher moves its arguments into a struct, so `_` parameters need
    // a name there. The inner fn keeps the `_`.
    for (arg, (ident, ..)) in outer_fn_sig.inputs.iter_mut().zip(&fields) {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Wild(_) = *pat_type.pat {
                *pat_type.pat = parse_quote!(#ident);
            }
        }
    }

    let anon_lifetime = Lifetime::new(ANON_LIFETIME, Span::call_site());

    let output_ty = match sig.output.clone() {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, mut ty) => {
            add_named_lifetimes(&mut ty);
            quote! { #ty }
        }
    };

    let inner_name = &inner_fn_sig.ident;
    // The inner fn must inline into the target-feature trampoline to be compiled
    // with its features. Only attributes that describe the body carry over; the
    // rest (`inline`, `target_feature`, attribute macros, ...) stay on the
    // dispatcher.
    let inner_attrs = attrs.iter().filter(|attr| {
        [
            "doc", "allow", "warn", "deny", "forbid", "expect", "cfg", "cfg_attr",
        ]
        .iter()
        .any(|name| attr.path().is_ident(name))
    });

    let mut struct_generics = outer_fn_sig.generics.clone();
    struct_generics.params.insert(
        0,
        GenericParam::Lifetime(LifetimeParam::new(anon_lifetime.clone())),
    );

    let (impl_generics, type_generics, where_clause) = struct_generics.split_for_impl();

    let field_decl = fields.iter().map(|(ident, ty, _)| quote![#ident: #ty]);
    let field_names = fields.iter().map(|it| &it.0).collect::<Vec<_>>();

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
        #[allow(unused_mut, clippy::all)]
        #unsafety #vis #outer_fn_sig {
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

        #(#inner_attrs)*
        #[inline(always)]
        #inner_fn_sig #block
    })
}

fn add_named_lifetimes(ty: &mut Type) -> bool {
    match ty {
        Type::Array(type_array) => add_named_lifetimes(&mut type_array.elem),
        Type::Group(type_group) => add_named_lifetimes(&mut type_group.elem),
        Type::Paren(type_paren) => add_named_lifetimes(&mut type_paren.elem),
        Type::Ptr(type_ptr) => add_named_lifetimes(&mut type_ptr.elem),
        Type::Reference(type_reference) if type_reference.lifetime.is_none() => {
            type_reference.lifetime = Some(Lifetime::new(
                ANON_LIFETIME,
                type_reference.and_token.span(),
            ));
            true
        }
        Type::Slice(type_slice) => add_named_lifetimes(&mut type_slice.elem),
        Type::Tuple(type_tuple) => type_tuple.elems.iter_mut().any(add_named_lifetimes),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::with_simd_impl;
    use quote::{quote, ToTokens};

    fn generated_fn<'a>(expanded: &'a syn::File, name: &str) -> &'a syn::ItemFn {
        expanded
            .items
            .iter()
            .find_map(|item| match item {
                syn::Item::Fn(func) if func.sig.ident == name => Some(func),
                _ => None,
            })
            .unwrap()
    }

    fn attr_names(expanded: &syn::File, name: &str) -> Vec<String> {
        generated_fn(expanded, name)
            .attrs
            .iter()
            .map(|attr| attr.path().to_token_stream().to_string())
            .collect()
    }

    /// The `inline` attributes of the generated fn `name`, spaces removed.
    fn inline_attrs(expanded: &syn::File, name: &str) -> Vec<String> {
        generated_fn(expanded, name)
            .attrs
            .iter()
            .filter(|attr| attr.path().is_ident("inline"))
            .map(|attr| attr.meta.to_token_stream().to_string().replace(' ', ""))
            .collect()
    }

    #[test]
    fn user_inline_attribute_stays_on_the_dispatcher() {
        // The body has to inline into the target-feature trampoline to be
        // compiled with the right features, so `inline(never)` must not reach
        // the inner fn.
        let expanded = with_simd_impl(
            quote!(),
            quote! {
                /// Docs.
                #[inline(never)]
                #[target_feature(enable = "avx2")]
                #[allow(clippy::identity_op)]
                fn f<S: Simd>(x: u32) -> u32 { x }
            },
        )
        .unwrap();
        let expanded: syn::File = syn::parse2(expanded).unwrap();
        assert_eq!(inline_attrs(&expanded, "f"), ["inline(never)"]);
        assert_eq!(
            attr_names(&expanded, "f_impl"),
            ["doc", "allow", "inline"],
            "only doc and lint attributes reach the inner fn"
        );
        assert_eq!(inline_attrs(&expanded, "f_impl"), ["inline(always)"]);
    }

    #[test]
    fn destructuring_argument_is_a_compile_error() {
        let err = with_simd_impl(
            quote!(),
            quote! {
                fn f<S: Simd>((a, b): (u32, u32)) -> u32 { a + b }
            },
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "`with_simd` arguments must be plain identifiers"
        );
    }
}
