use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{parse_quote, LifetimeParam, Safety, Token, Type};
use syn::{spanned::Spanned, FnArg, GenericParam, ItemFn, Pat};
use syn::{visit_mut, visit_mut::VisitMut};
use syn::{Expr, Lifetime};
use syn::{Ident, TypeGroup, TypeParen, WherePredicate};

#[derive(FromMeta, Default)]
#[darling(default)]
struct WithSimdOpts {
    #[darling(default)]
    arch: Option<Expr>,
}

/// Turns a function generic over a SIMD backend into one that picks the best
/// backend for the CPU at runtime and runs its body compiled for that backend.
///
/// ```ignore
/// #[with_simd]
/// fn sum<S: Simd, T: ReduceAdd>(values: &[T]) -> T {
///     // ...
/// }
///
/// let total = sum::<f32>(&values);
/// ```
///
/// The first type parameter is the backend. The generated function drops it,
/// along with any `where` bounds on it, and the original body moves to a
/// separate `sum_impl` next to it. Arguments must be plain identifiers or `_`,
/// and methods are not supported. A type with a hidden lifetime needs it written
/// out, as in `Foo<'_>`, since the macro can't see that `Foo` has one.
///
/// The body always inlines into the code compiled for the backend, so
/// attributes like `#[inline]` go on the generated function, while docs, lints
/// and `cfg` go on both. Functions the body calls need to inline too in order
/// to get the backend's CPU features, so mark them `#[inline(always)]`.
///
/// `#[with_simd(arch = expr)]` dispatches on `expr`, usually an `Arch`, in place of
/// detecting the CPU. The generated function is `unsafe`, since `expr` may name a
/// backend the CPU doesn't have.
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
        false => WithSimdOpts::from_list(&NestedMeta::parse_meta_list(attr)?)?,
    };

    let arch_unsafe = opts.arch.as_ref().map(|arch| Token![unsafe](arch.span()));
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

    let Some(simd_param) = sig.generics.type_params().next() else {
        return Err(syn::Error::new(
            sig.ident.span(),
            "`with_simd` needs a SIMD type parameter, e.g. `<S: Simd>`",
        ));
    };
    let simd_generic_name = simd_param.ident.clone();

    // Only the SIMD parameter goes, so callers' generic arguments keep the
    // declared order.
    let mut outer_fn_sig = sig.clone();
    if let Some(arch_unsafe) = arch_unsafe {
        outer_fn_sig.safety = Safety::Unsafe(arch_unsafe);
    }
    outer_fn_sig.generics.params = sig
        .generics
        .params
        .iter()
        .filter(|param| !matches!(param, GenericParam::Type(ty) if ty.ident == simd_generic_name))
        .cloned()
        .collect();

    // The dispatcher doesn't have the SIMD parameter, so `where` bounds on it
    // only apply to the inner fn.
    if let Some(where_clause) = &mut outer_fn_sig.generics.where_clause {
        where_clause.predicates = core::mem::take(&mut where_clause.predicates)
            .into_iter()
            .filter(|predicate| !bounds_param(predicate, &simd_generic_name))
            .collect();
    }

    let mut inner_fn_sig = sig.clone();
    inner_fn_sig.ident = format_ident!("{}_impl", name);
    let struct_name = format_ident!("{}_struct", name);

    let mut input_lifetimes = NameElidedLifetimes::default();
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
                input_lifetimes.visit_type_mut(&mut ty);
                Ok((ident, ty))
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

    // The usual elision rule: an elided output lifetime is the one input lifetime.
    // With any other number the inner fn gets a compile error for it, so what it's
    // named here doesn't matter.
    let output_lifetime = match input_lifetimes.used.as_slice() {
        [lifetime] => lifetime.clone(),
        _ => anon_lifetime.clone(),
    };
    let output_ty = match sig.output.clone() {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, mut ty) => {
            NameElidedLifetimes {
                output: Some(output_lifetime),
                ..Default::default()
            }
            .visit_type_mut(&mut ty);
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
    for lifetime in input_lifetimes
        .generated
        .into_iter()
        .chain([anon_lifetime.clone()])
    {
        struct_generics
            .params
            .insert(0, GenericParam::Lifetime(LifetimeParam::new(lifetime)));
    }

    // Marks every parameter as used, since some may only be in the output type.
    let phantom_params = struct_generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Lifetime(param) => {
                let lifetime = &param.lifetime;
                Some(quote![&#lifetime ()])
            }
            GenericParam::Type(param) => {
                let ident = &param.ident;
                Some(quote![::core::marker::PhantomData<#ident>])
            }
            GenericParam::Const(_) => None,
        });

    let (impl_generics, type_generics, where_clause) = struct_generics.split_for_impl();

    let field_decl = fields.iter().map(|(ident, ty)| quote![#ident: #ty]);
    let field_names = fields.iter().map(|it| &it.0).collect::<Vec<_>>();

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
        #vis #outer_fn_sig {
            #[allow(non_camel_case_types)]
            struct #struct_name #impl_generics #where_clause {
                #(#field_decl,)*
                __generics: ::core::marker::PhantomData<(#(#phantom_params,)*)>,
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

            // `Arch::dispatch` is unsafe, `AutoArch::dispatch` isn't.
            #[allow(unused_unsafe)]
            unsafe {
                (#arch).dispatch( #struct_name #struct_turbofish { __generics: core::marker::PhantomData, #(#field_names,)* } )
            }
        }

        #(#inner_attrs)*
        #[inline(always)]
        #inner_fn_sig #block
    })
}

/// Whether `predicate` is a `where` bound on the type parameter `param`.
fn bounds_param(predicate: &WherePredicate, param: &Ident) -> bool {
    let WherePredicate::Type(predicate) = predicate else {
        return false;
    };
    let mut ty = &predicate.bounded_ty;
    // `macro_rules!` fragments arrive wrapped in invisible groups.
    while let Type::Group(TypeGroup { elem, .. }) | Type::Paren(TypeParen { elem, .. }) = ty {
        ty = elem;
    }
    matches!(ty, Type::Path(p) if p.qself.is_none() && p.path.is_ident(param))
}

/// Names the lifetimes a function signature can elide but a struct field can't.
///
/// Each one in an input gets its own name, because a `&mut` makes what's behind
/// it invariant, so `&mut &u32` wouldn't fit a single lifetime.
#[derive(Default)]
struct NameElidedLifetimes {
    /// When set, the name for all of them, in place of generating one each.
    output: Option<Lifetime>,
    generated: Vec<Lifetime>,
    /// Every lifetime position seen, named or generated.
    used: Vec<Lifetime>,
}

impl NameElidedLifetimes {
    fn name(&mut self, span: Span) -> Lifetime {
        if let Some(output) = &self.output {
            return output.clone();
        }
        let name = format!("{ANON_LIFETIME}{}", self.generated.len());
        let lifetime = Lifetime::new(&name, span);
        self.generated.push(lifetime.clone());
        lifetime
    }
}

impl VisitMut for NameElidedLifetimes {
    fn visit_type_reference_mut(&mut self, ty: &mut syn::TypeReference) {
        if ty.lifetime.is_none() {
            ty.lifetime = Some(self.name(ty.and_token.span()));
        }
        visit_mut::visit_type_reference_mut(self, ty);
    }

    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        if lifetime.ident == "_" {
            *lifetime = self.name(lifetime.span());
        }
        self.used.push(lifetime.clone());
    }

    // Elided lifetimes in `fn(&T)`, `Fn(&T)` and `for<'a>` bounds are higher-ranked,
    // so those are valid in a struct field as they are.
    fn visit_type_fn_ptr_mut(&mut self, _: &mut syn::TypeFnPtr) {}

    fn visit_parenthesized_generic_arguments_mut(
        &mut self,
        _: &mut syn::ParenthesizedGenericArguments,
    ) {
    }

    fn visit_trait_bound_mut(&mut self, bound: &mut syn::TraitBound) {
        if bound.lifetimes.is_none() {
            visit_mut::visit_trait_bound_mut(self, bound);
        }
    }

    // Array lengths and const generic arguments have their own scope.
    fn visit_expr_mut(&mut self, _: &mut Expr) {}
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
    fn where_bound_on_simd_param_stays_off_the_dispatcher() {
        let expanded = with_simd_impl(
            quote!(),
            quote! {
                fn f<S>(x: u32) -> u32 where S: Simd { x }
            },
        )
        .unwrap();
        let expanded: syn::File = syn::parse2(expanded).unwrap();
        let predicates = |name| {
            generated_fn(&expanded, name)
                .sig
                .generics
                .where_clause
                .as_ref()
                .map_or(0, |w| w.predicates.len())
        };
        assert_eq!(predicates("f"), 0);
        assert_eq!(predicates("f_impl"), 1);
    }

    #[test]
    fn where_bound_from_macro_rules_stays_off_the_dispatcher() {
        // A `$s:ty` fragment arrives as an invisible group.
        let param = proc_macro2::Group::new(proc_macro2::Delimiter::None, quote!(S));
        let expanded = with_simd_impl(
            quote!(),
            quote! {
                fn f<S>(x: u32) -> u32 where #param: Simd { x }
            },
        )
        .unwrap();
        let expanded: syn::File = syn::parse2(expanded).unwrap();
        let predicates = generated_fn(&expanded, "f")
            .sig
            .generics
            .where_clause
            .as_ref()
            .map_or(0, |w| w.predicates.len());
        assert_eq!(predicates, 0);
    }

    #[test]
    fn missing_simd_param_is_a_compile_error() {
        let err = with_simd_impl(quote!(), quote! { fn f(x: u32) -> u32 { x } }).unwrap_err();
        assert_eq!(
            err.to_string(),
            "`with_simd` needs a SIMD type parameter, e.g. `<S: Simd>`"
        );
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
