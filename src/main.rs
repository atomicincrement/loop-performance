//! # Rust loop performance.
//!
//! On linux, run `cat /proc/cpuinfo | grep "model name"`
//! to get your CPU information.


// Features needed for tests and nightly-only intrinsics.
#![feature(core_intrinsics)]
#![feature(test)]
#![allow(dead_code)]
mod section1;
mod section2;
mod section3;

use std::time::Instant;

// Number of elements in the test vectors.
//
// Note that the CPU is much faster than the memory, so
// once the data is larger than the cache, things will go south fast.
const NUM_ELEMS: usize = 0x1000000;

// Number of tests to run.
const NUM_TESTS: usize = 64;

// Take the minimum time of a number of runs.
// This is likely to be the true value as interrupts
// and thread preemption will make it longer.
#[inline(never)]
fn benchmark<T, F : FnMut() -> T>(name: &str, mut f: F) {
    let time = (0..NUM_TESTS)
        .map(|_| {
            let start = Instant::now();
            // black-box wraps the code in asm("volatile") to prevent code removal.
            std::hint::black_box((&mut f)());
            let time = start.elapsed().as_nanos() * 1000 / NUM_ELEMS as u128;
            time
        })
        .fold(u128::MAX, |p, v| p.min(v));

        println!("{} took {}ps/iteration.", name, time);
}

fn main() {
    let mut some_numbers: Vec<f32> = (0..NUM_ELEMS).map(|_| 1.0 as f32).collect();
    let some_other_numbers: Vec<f32> = (0..NUM_ELEMS).map(|_| 1.0 as f32).collect();

    benchmark(
        "section1::a_naive_vector_add",
        || section1::a_naive_vector_add(&mut *some_numbers, &some_other_numbers)
    );

    benchmark(
        "section1::b_naive_sum",
        || section1::b_naive_sum(&*some_numbers)
    );

    benchmark(
        "section2::a_fadd_fast_sum",
        || section2::a_fadd_fast_sum(&*some_numbers)
    );

    benchmark(
        "section2::a_faster_sum",
        || section2::c_faster_sum(&*some_numbers)
    );

    benchmark(
        "section3::a_packed_simd_fast_sum",
        || section3::a_packed_simd_fast_sum(&*some_numbers)
    );

    benchmark(
        "section3::b_bigger_packed_simd_fast_sum",
        || section3::b_bigger_packed_simd_fast_sum(&*some_numbers)
    );
}
