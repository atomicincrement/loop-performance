//! In this section we improve the fold function to attempt
//! to make a stable fast vector sum.

use std::intrinsics::fadd_fast;

/// In nightly, we can use experimental features such as the fast_add intrinsic.
/// But with the stable compiler, this is not possible.
///
/// Lack of stability here is mostly down to lack of understanding
/// of floating point arithmetic in the community. The game industry
/// has been using fast floating point arithmetic for decades.
/// ```
/// pub fn a_fadd_fast_sum(numbers: &[f32]) -> f32 {
///     numbers
///         .iter()
///         .fold(0_f32, |p, v| unsafe { fadd_fast(p, *v) })
/// }
/// ```
pub fn a_fadd_fast_sum(numbers: &[f32]) -> f32 {
    numbers
        .iter()
        .fold(0_f32, |p, v| unsafe { fadd_fast(p, *v) })
}

/*
/// A macro that executes 16 operations in parallel.
macro_rules! do8 {
    ($f : expr) => {
        [
            $f(0),
            $f(1),
            $f(2),
            $f(3),
            $f(4),
            $f(5),
            $f(6),
            $f(7),
        ]
    };
}

/// A macro that executes 16 operations in parallel.
macro_rules! do16 {
    ($f : expr) => {
        [
            $f(0),
            $f(1),
            $f(2),
            $f(3),
            $f(4),
            $f(5),
            $f(6),
            $f(7),
            $f(8),
            $f(9),
            $f(10),
            $f(11),
            $f(12),
            $f(13),
            $f(14),
            $f(15),
        ]
    };
}
*/

/// A macro that executes 32 operations in parallel.
macro_rules! do32 {
    ($f : expr) => {
        [
            $f(0),
            $f(1),
            $f(2),
            $f(3),
            $f(4),
            $f(5),
            $f(6),
            $f(7),
            $f(8),
            $f(9),
            $f(10),
            $f(11),
            $f(12),
            $f(13),
            $f(14),
            $f(15),
            $f(16),
            $f(17),
            $f(18),
            $f(19),
            $f(20),
            $f(21),
            $f(22),
            $f(23),
            $f(24),
            $f(25),
            $f(26),
            $f(27),
            $f(28),
            $f(29),
            $f(30),
            $f(31),
        ]
    };
}

/// We can exploit the SLP vectoriser in LLVM to generate
/// code that looks optimal for a single function.
/// ```asm
/// example::a_sum32:
///         mov     rax, rdi
///         vmovups ymm0, ymmword ptr [rsi]
///         vmovups ymm1, ymmword ptr [rsi + 32]
///         vmovups ymm2, ymmword ptr [rsi + 64]
///         vmovups ymm3, ymmword ptr [rsi + 96]
///         vaddps  ymm0, ymm0, ymmword ptr [rdx]
///         vaddps  ymm1, ymm1, ymmword ptr [rdx + 32]
///         vaddps  ymm2, ymm2, ymmword ptr [rdx + 64]
///         vaddps  ymm3, ymm3, ymmword ptr [rdx + 96]
///         vmovups ymmword ptr [rdi], ymm0
///         vmovups ymmword ptr [rdi + 32], ymm1
///         vmovups ymmword ptr [rdi + 64], ymm2
///         vmovups ymmword ptr [rdi + 96], ymm3
///         vzeroupper
///         ret
/// 
/// ```
pub fn b_sum32(x: &[f32; 32], y: &[f32; 32]) -> [f32; 32] {
    do32!(
        |i| x[i] + y[i]
    )
}

/// If we treat the vector as a 32 x n matrix:
/// ```norun
/// [ v0,  v1, ... v31]
/// [v32, v33, ... v63]
/// ...
/// ```
///
/// We can sum the columns first which should result in vector adds.
/// ```norun
/// [v0+v32+..., v1+v33+..., ... v31+v63+...]
/// ```
///
/// We naively hoped that the SLP vectoriser would be run before the loop
/// vectoriser. But sadly not. It does produce faster code, but suboptimal.
///
/// But it seems that the SLP vectoriser is not run in loops.
/// It does vectorise but only to `xmm` registers and adds permutes
/// and so this seems to be the best we can do with **stable** Rust.
/// ```asm
/// .LBB1_3:
///         vpermilps       xmm2, xmmword ptr [rdi + 4*rcx], 27
///         vaddps  xmm7, xmm7, xmm2
///         vpermilps       xmm2, xmmword ptr [rdi + 4*rcx + 16], 27
///         vaddps  xmm0, xmm0, xmm2
///         vpermilps       xmm2, xmmword ptr [rdi + 4*rcx + 32], 27
///         vaddps  xmm6, xmm6, xmm2
///         vpermilps       xmm2, xmmword ptr [rdi + 4*rcx + 48], 27
///         vaddps  xmm5, xmm5, xmm2
///         vpermilps       xmm2, xmmword ptr [rdi + 4*rcx + 64], 27
///         vaddps  xmm4, xmm4, xmm2
///         vpermilps       xmm2, xmmword ptr [rdi + 4*rcx + 80], 27
///         vaddps  xmm3, xmm3, xmm2
///         vpermilps       xmm2, xmmword ptr [rdi + 4*rcx + 96], 27
///         vaddps  xmm1, xmm1, xmm2
///         vpermilps       xmm2, xmmword ptr [rdi + 4*rcx + 112], 27
///         vaddps  xmm8, xmm8, xmm2
///         add     rcx, 32
///         cmp     rax, rcx
///         jne     .LBB1_3
/// ```
///
/// The "27" mask or 00 01 10 11 seems to be a nop, bizzarely.
/// See https://www.felixcloutier.com/x86/vpermilps
///
#[inline(never)]
pub fn c_faster_sum(numbers: &[f32]) -> f32 {
    const CHUNK_SIZE: usize = 32;

    let column_sums = numbers
        .chunks_exact(CHUNK_SIZE)
        .fold([0.0; CHUNK_SIZE], |prev, chunk| {
            do32!(|i| prev[i] + chunk[i])
        });

    let total = column_sums.iter().sum::<f32>();

    let remainder = numbers
        .chunks_exact(CHUNK_SIZE)
        .remainder()
        .iter()
        .sum::<f32>();

    total + remainder
}

// use num_traits::Zero;

// pub fn b_fast_generic_fold<D: Zero + Copy, F: Fn(D, &D) -> D>(data: &[D], op: F) -> D {
//     const CHUNK_SIZE: usize = 8;
//     let column_sums = data
//         .chunks_exact(CHUNK_SIZE)
//         .fold([D::zero(); CHUNK_SIZE], |prev, chunk| {
//             do8!(|i| op(prev[i], &chunk[i]))
//         });

//     let total = column_sums.iter().fold(D::zero(), &op);

//     let remainder = data.chunks_exact(CHUNK_SIZE)
//         .remainder()
//         .iter()
//         .fold(D::zero(), op);
    
//     remainder + total
// }
