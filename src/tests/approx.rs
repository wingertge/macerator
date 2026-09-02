use half::f16;

/// Approximate equality using both the absolute difference and relative based
/// comparisons.
#[macro_export]
macro_rules! relative_eq {
    ($lhs:expr, $rhs:expr $(, $opt:ident = $val:expr)*) => {
        $crate::tests::approx::Relative::default()$(.$opt($val))*.eq(&$lhs, &$rhs)
    };
    ($lhs:expr, $rhs:expr $(, $opt:ident = $val:expr)*,) => {
        $crate::tests::approx::Relative::default()$(.$opt($val))*.eq(&$lhs, &$rhs)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __assert_approx {
    ($eq:ident, $given:expr, $expected:expr) => {{
        match (&($given), &($expected)) {
            (given, expected) => assert!(
                $eq!(*given, *expected),
"assert_{}!({}, {})

    left  = {:?}
    right = {:?}

",
                stringify!($eq),
                stringify!($given),
                stringify!($expected),
                given, expected,
            ),
        }
    }};
    ($eq:ident, $given:expr, $expected:expr, $($opt:ident = $val:expr),+) => {{
        match (&($given), &($expected)) {
            (given, expected) => assert!(
                $eq!(*given, *expected, $($opt = $val),+),
"assert_{}!({}, {}, {})

    left  = {:?}
    right = {:?}

",
                stringify!($eq),
                stringify!($given),
                stringify!($expected),
                stringify!($($opt = $val),+),
                given, expected,
            ),
        }
    }};
}

/// An assertion that delegates to [`relative_eq!`], and panics with a helpful
/// error on failure.
#[macro_export(local_inner_macros)]
macro_rules! assert_relative_eq {
    ($given:expr, $expected:expr $(, $opt:ident = $val:expr)*) => {
        __assert_approx!(relative_eq, $given, $expected $(, $opt = $val)*)
    };
    ($given:expr, $expected:expr $(, $opt:ident = $val:expr)*,) => {
        __assert_approx!(relative_eq, $given, $expected $(, $opt = $val)*)
    };
}

pub trait RelativeEq<Rhs = Self>
where
    Rhs: ?Sized,
{
    type Epsilon;

    /// The default epsilon for testing values that are close to zero
    ///
    /// This is used when no `epsilon` value is supplied to the
    /// [`relative_eq`](crate::relative_eq) macro.
    fn default_relative_epsilon() -> Self::Epsilon;

    /// The default relative tolerance for testing values that are far-apart.
    ///
    /// This is used when no `max_relative` value is supplied to the
    /// [`relative_eq`](crate::relative_eq) macro.
    fn default_max_relative() -> Self::Epsilon;

    /// A test for equality that uses a relative comparison if the values are
    /// far apart.
    fn relative_eq(&self, other: &Rhs, epsilon: Self::Epsilon, max_relative: Self::Epsilon)
        -> bool;
}

macro_rules! impl_relative_eq {
    ($T:ident) => {
        impl RelativeEq for $T {
            type Epsilon = $T;

            #[inline]
            fn default_relative_epsilon() -> $T {
                $T::MIN_POSITIVE
            }

            #[inline]
            fn default_max_relative() -> $T {
                $T::EPSILON
            }

            #[inline]
            #[allow(unused_imports)]
            fn relative_eq(&self, other: &$T, epsilon: $T, max_relative: $T) -> bool {
                use num_traits::float::FloatCore;
                // Handle same infinities
                if self == other {
                    return true;
                }

                // Handle remaining infinities
                if $T::is_infinite(*self) || $T::is_infinite(*other) {
                    return false;
                }

                let abs_diff = $T::abs(self - other);

                // For when the numbers are really close together
                if abs_diff <= epsilon {
                    return true;
                }

                let abs_self = $T::abs(*self);
                let abs_other = $T::abs(*other);

                let largest = if abs_other > abs_self {
                    abs_other
                } else {
                    abs_self
                };

                // Use a relative difference comparison
                abs_diff <= largest * max_relative
            }
        }
    };
}

impl_relative_eq!(f16);
impl_relative_eq!(f32);
impl_relative_eq!(f64);

pub struct Relative<A, B = A>
where
    A: RelativeEq<B> + ?Sized,
    B: ?Sized,
{
    /// The tolerance to use when testing values that are close together.
    pub epsilon: A::Epsilon,
    /// The relative tolerance for testing values that are far-apart.
    pub max_relative: A::Epsilon,
}

impl<A, B> Default for Relative<A, B>
where
    A: RelativeEq<B> + ?Sized,
    B: ?Sized,
{
    #[inline]
    fn default() -> Relative<A, B> {
        Relative {
            epsilon: A::default_relative_epsilon(),
            max_relative: A::default_max_relative(),
        }
    }
}

impl<A, B> Relative<A, B>
where
    A: RelativeEq<B> + ?Sized,
    B: ?Sized,
{
    /// Replace the epsilon value with the one specified.
    #[inline]
    pub fn epsilon(self, epsilon: A::Epsilon) -> Relative<A, B> {
        Relative { epsilon, ..self }
    }

    /// Replace the maximum relative value with the one specified.
    #[inline]
    pub fn max_relative(self, max_relative: A::Epsilon) -> Relative<A, B> {
        Relative {
            max_relative,
            ..self
        }
    }

    /// Perform the equality comparison
    #[inline]
    #[must_use]
    pub fn eq(self, lhs: &A, rhs: &B) -> bool {
        A::relative_eq(lhs, rhs, self.epsilon, self.max_relative)
    }
}
