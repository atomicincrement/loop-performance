
//! In this section, we talk about the usual behaviour
//! of loops in rust.

/// `a.iter_mut().zip(b).for_each(|(a, b)| *a += b);`
///
/// Naive vector adds can be vectorised, but you must tell Rust
/// what target you are running on or it will assume something from the '90s (AVX or worse).
///
/// A good model is `-C target-cpu=native` or `-C target-feature=avx2+fma`
///
/// Also, Rust is unable to assume that the slices do not overlap
/// even though Rust's rules say that they do not. This results in the compiiler
/// generating two loops, one for the aliasing case and one for the non-aliasing case.
///
/// See.
/// https://stackoverflow.com/questions/57259126/why-does-the-rust-compiler-not-optimize-code-assuming-that-two-mutable-reference
///
/// With `-O -C target-cpu=skylake -C target-feature=avx2+fma`
/// Godbolt generates the following code:
/// ```asm
/// .LBB0_6:
///         # Load 128 bytes of data into ymm0-3 (32 numbers).
///         vmovups ymm0, ymmword ptr [rdx + 4*rcx]
///         vmovups ymm1, ymmword ptr [rdx + 4*rcx + 32]
///         vmovups ymm2, ymmword ptr [rdx + 4*rcx + 64]
///         vmovups ymm3, ymmword ptr [rdx + 4*rcx + 96]
///         # Add 32 numbers
///         vaddps  ymm0, ymm0, ymmword ptr [rdi + 4*rcx]
///         vaddps  ymm1, ymm1, ymmword ptr [rdi + 4*rcx + 32]
///         vaddps  ymm2, ymm2, ymmword ptr [rdi + 4*rcx + 64]
///         vaddps  ymm3, ymm3, ymmword ptr [rdi + 4*rcx + 96]
///         # Store 32 numbers
///         vmovups ymmword ptr [rdi + 4*rcx], ymm0
///         vmovups ymmword ptr [rdi + 4*rcx + 32], ymm1
///         vmovups ymmword ptr [rdi + 4*rcx + 64], ymm2
///         vmovups ymmword ptr [rdi + 4*rcx + 96], ymm3
///         # Do it again with 32 more numbers.
///         vmovups ymm0, ymmword ptr [rdx + 4*rcx + 128]
///         vmovups ymm1, ymmword ptr [rdx + 4*rcx + 160]
///         vmovups ymm2, ymmword ptr [rdx + 4*rcx + 192]
///         vmovups ymm3, ymmword ptr [rdx + 4*rcx + 224]
///         vaddps  ymm0, ymm0, ymmword ptr [rdi + 4*rcx + 128]
///         vaddps  ymm1, ymm1, ymmword ptr [rdi + 4*rcx + 160]
///         vaddps  ymm2, ymm2, ymmword ptr [rdi + 4*rcx + 192]
///         vaddps  ymm3, ymm3, ymmword ptr [rdi + 4*rcx + 224]
///         vmovups ymmword ptr [rdi + 4*rcx + 128], ymm0
///         vmovups ymmword ptr [rdi + 4*rcx + 160], ymm1
///         vmovups ymmword ptr [rdi + 4*rcx + 192], ymm2
///         vmovups ymmword ptr [rdi + 4*rcx + 224], ymm3
///         # Loop every 64 numbers.
///         add     rcx, 64
///         add     r9, 2
///         jne     .LBB0_6
/// ```
///
/// `ymm` registers hold eight 32 bit floats and for 64 bit floats.
/// so that each `vaddps` does eight work items.
///
/// The cpu can probably execute two simple instructions per cycle
/// plus the integer part in the background which means that this loop takes 12 cycles
/// and does 64 / 12 = 5.3 operations per cycle or about 16,000,000,000 ops/sec.
///
pub fn a_naive_vector_add(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
}

/// A naive sum evaluates expressions left to right.
/// [a, b, c, d] -> ((a + b) + c) + d.
///
/// Because rust respects the order of addition in fp numbers
/// it is unable to reorder the adds and so cannot vectorise
/// the loop.
///
/// With `-O -C target-cpu=skylake -C target-feature=avx2+fma`
/// Godbolt generates the following code:
/// ```asm
/// .LBB0_8:
///         vaddss  xmm0, xmm0, dword ptr [rax]
///         vaddss  xmm0, xmm0, dword ptr [rax + 4]
///         vaddss  xmm0, xmm0, dword ptr [rax + 8]
///         vaddss  xmm0, xmm0, dword ptr [rax + 12]
///         vaddss  xmm0, xmm0, dword ptr [rax + 16]
///         vaddss  xmm0, xmm0, dword ptr [rax + 20]
///         vaddss  xmm0, xmm0, dword ptr [rax + 24]
///         vaddss  xmm0, xmm0, dword ptr [rax + 28]
///         add     rax, 32
///         cmp     rax, rcx
///         jne     .LBB0_8
/// ```
///
/// Here the compiler has unrolled the loop eight times
/// which means that each time round the loop we do eight times
/// the work for one end of loop test.
///
/// We are doing about 2 operations per cycle or 6,000,000,000 ops/sec.
pub fn b_naive_sum(numbers: &[f32]) -> f32 {
    numbers.iter().fold(0_f32, |p, v| p + v)
}
