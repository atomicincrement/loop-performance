//! In this section we look at the packed_simd crate and repr(SIMD)
//!
//! The original packed_simd crate was intended to develop a portable
//! model for SIMD arithmetic on Rust.
//!
//! However, the packed_simd maintainers seem to be out of touch
//! and there is a new version here: https://github.com/rust-lang/packed_simd
//! The new crate is packed_simd_2.
//!
//! This, in turn is deprecated for https://github.com/rust-lang/stdsimd
//!

use packed_simd_2::{f32x8, f32x16};

/// Although the `fast_add` implementation licks the sum problem
/// in nightly, there a lot of problems that are more complex than simple
/// sums.
///
/// The `packed_simd` crate was a prototype for a generic implmenetation
/// of SIMD in Rust, perhaps even for stable.
///
/// The `packed_simd` crate was a prototype for a generic implmenetation
/// of SIMD in Rust, perhaps even for stable.
///
/// The original packed_simd crate was intended to develop a portable
/// model for SIMD arithmetic on Rust.
///
/// However, the packed_simd maintainers seem to be out of touch
/// and there is a new version here: https://github.com/rust-lang/packed_simd
/// The new crate is packed_simd_2.
///
/// This, in turn is deprecated for https://github.com/rust-lang/stdsimd
///
/// Here is a packed_simd_2 horizontal add:
/// ```
/// pub fn a_packed_simd_fast_sum(numbers: &[f32]) -> f32 {
///     const CHUNK_SIZE: usize = 8;
///
///     let mut column_sums = numbers
///         .chunks_exact(8)
///         .fold(f32x8::splat(0.0), |p, v| p + f32x8::from_slice_unaligned(v));
///
///     let total = column_sums.sum();
///
///     let remainder = numbers
///         .chunks_exact(CHUNK_SIZE)
///         .remainder()
///         .iter()
///         .sum::<f32>();
///
///     total + remainder
/// }
/// ```
#[inline(never)]
pub fn a_packed_simd_fast_sum(numbers: &[f32]) -> f32 {
    const CHUNK_SIZE: usize = 8;

    let column_sums = numbers
        .chunks_exact(CHUNK_SIZE)
        .fold(f32x8::splat(0.0), |p, v| p + f32x8::from_slice_unaligned(v));

    let total = column_sums.sum();

    let remainder = numbers
        .chunks_exact(CHUNK_SIZE)
        .remainder()
        .iter()
        .sum::<f32>();

    total + remainder
}

#[inline(never)]
pub fn b_bigger_packed_simd_fast_sum(numbers: &[f32]) -> f32 {
    const CHUNK_SIZE: usize = 16;

    let column_sums = numbers
        .chunks_exact(CHUNK_SIZE)
        .fold(f32x16::splat(0.0), |p, v| p + f32x16::from_slice_unaligned(v));

    let total = column_sums.sum();

    let remainder = numbers
        .chunks_exact(CHUNK_SIZE)
        .remainder()
        .iter()
        .sum::<f32>();

    total + remainder
}
