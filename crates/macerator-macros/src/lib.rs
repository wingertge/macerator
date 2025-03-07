use darling::FromMeta;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse_quote;
use syn::Expr;
use syn::{spanned::Spanned, FnArg, GenericParam, ItemFn, Pat};

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
                let ty = &*pat_type.ty;
                Ok((ident, ty))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let output_ty = match sig.output.clone() {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, ty) => quote! { #ty },
    };

    let inner_name = &inner_fn_sig.ident;
    let (impl_generics, type_generics, where_clause) = outer_fn_sig.generics.split_for_impl();
    let field_decl = fields.iter().map(|(ident, ty)| quote![#ident: #ty]);
    let field_names = fields.iter().map(|it| it.0).collect::<Vec<_>>();

    let simd_generic_name = sig.generics.type_params().next().unwrap().ident.clone();
    let (_, inner_generics, _) = inner_fn_sig.generics.split_for_impl();
    let turbofish = inner_generics.as_turbofish();
    let struct_turbofish = type_generics.as_turbofish();

    Ok(quote! {
        #(#attrs)*
        #vis #outer_fn_sig {
            #[allow(non_camel_case_types)]
            struct #struct_name #impl_generics #where_clause {
                #(#field_decl,)*
            };

            impl #impl_generics macerator::WithSimd for #struct_name #type_generics #where_clause {
                type Output = #output_ty;

                #[inline(always)]
                fn with_simd<#simd_generic_name: macerator::Simd>(self) -> <Self as macerator::WithSimd>::Output {
                    let Self {
                        #(#field_names,)*
                    } = self;
                    #[allow(unused_unsafe)]
                    unsafe {
                        #inner_name #turbofish(#(#field_names,)*)
                    }
                }
            }

            (#arch).dispatch( #struct_name #struct_turbofish { #(#field_names,)* } )
        }

        #(#attrs)*
        #inner_fn_sig #block
    })
}
