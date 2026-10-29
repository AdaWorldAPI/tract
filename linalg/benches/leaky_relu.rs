use criterion::*;
use tract_data::prelude::*;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use tract_linalg::element_wise::ElementWiseKer;

fn leaky_relu_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("leaky_relu_f16");
    group.throughput(Throughput::Elements(1024));
    let mut input = unsafe { Tensor::uninitialized_aligned::<f16>(&[1024], 16).unwrap() };
    let input = unsafe { input.as_slice_mut_unchecked::<f16>() };
    let alpha = f16::from_f32(0.1);
    group.bench_function("rust", |b| b.iter(|| rust_fp16(input, alpha)));
    group.bench_function("linalg", |b| b.iter(|| linalg16(input, alpha)));
    #[cfg(target_arch = "aarch64")]
    group.bench_function("linalg-asm", |b| {
        b.iter(|| tract_linalg::arm64::arm64fp16_leaky_relu_f16_16n::run(input, alpha))
    });
}

#[inline(never)]
fn rust_fp16(input: &mut [f16], alpha: f16) {
    for x in input {
        *x = if *x > f16::ZERO { *x } else { *x * alpha }
    }
}

#[inline(never)]
fn linalg16(input: &mut [f16], alpha: f16) {
    tract_linalg::routines::Func::LeakyRelu
        .ew_f16_param()
        .unwrap()
        .run_with_params(input, alpha)
        .unwrap();
}

fn leaky_relu_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("leaky_relu_f32");
    group.throughput(Throughput::Elements(1024));
    // 64-byte aligned: the AVX-512 kernels below load/store with `vmovaps`, which
    // requires zmm-width (64-byte) alignment.
    let mut input = unsafe { Tensor::uninitialized_aligned::<f32>(&[1024], 64).unwrap() };
    let input = unsafe { input.as_slice_mut_unchecked::<f32>() };
    let alpha = 0.1f32;
    group.bench_function("rust", |b| b.iter(|| rust_fp32(input, alpha)));
    group.bench_function("linalg", |b| b.iter(|| linalg32(input, alpha)));
    #[cfg(target_arch = "aarch64")]
    group.bench_function("linalg-asm", |b| {
        b.iter(|| tract_linalg::arm64::arm64simd_leaky_relu_f32_8n::run(input, alpha))
    });
    #[cfg(target_arch = "x86_64")]
    group.bench_function("linalg-avx512-asm", |b| {
        b.iter(|| tract_linalg::x86_64::act::x86_64_avx512_leaky_relu_f32_64n::run(input, alpha))
    });
    #[cfg(target_arch = "x86_64")]
    group.bench_function("linalg-ndarray-simd", |b| {
        b.iter(|| tract_linalg::x86_64::act::x86_64_ndarray_leaky_relu_f32_16n::run(input, alpha))
    });
}

#[inline(never)]
fn rust_fp32(input: &mut [f32], alpha: f32) {
    for x in input {
        *x = if *x > 0.0 { *x } else { *x * alpha }
    }
}

#[inline(never)]
fn linalg32(input: &mut [f32], alpha: f32) {
    tract_linalg::routines::Func::LeakyRelu
        .ew_f32_param()
        .unwrap()
        .run_with_params(input, alpha)
        .unwrap();
}

criterion_group!(benches, leaky_relu_f32, leaky_relu_f16);
criterion_main!(benches);
