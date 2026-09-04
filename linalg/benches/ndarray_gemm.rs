// Compares the hand-tuned AVX-512 asm 16x8 GEMM kernel against the additive
// ndarray-blas_gemm-backed candidate of the same tile geometry, on a full matrix multiply
// (not just one microkernel tile call): the kernel's own panel-walking machinery loops the
// tile many times over m/n/k, so this measures the whole GEMM each candidate produces.
use criterion::*;
use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec};

fn gemm_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f32_16x8");
    for &(m, k, n) in &[(512usize, 512usize, 512usize), (1024, 1024, 1024)] {
        group.throughput(Throughput::Elements((2 * m * k * n) as u64));
        for (label, mmm) in [
            ("asm", tract_linalg::x86_64::mmm::avx512_mmm_f32_16x8.mmm()),
            ("ndarray", tract_linalg::x86_64::mmm::ndarray_avx512_mmm_f32_16x8.mmm()),
        ] {
            group.bench_with_input(
                BenchmarkId::new(label, format!("{m}x{k}x{n}")),
                &(m, k, n),
                |be, &(m, k, n)| {
                    let packing = &mmm.packings()[0];
                    let a = Tensor::zero::<f32>(&[m, k]).unwrap();
                    let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
                    let b = Tensor::zero::<f32>(&[k, n]).unwrap();
                    let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
                    let mut cc = Tensor::zero::<f32>(&[n, m]).unwrap();
                    be.iter(|| unsafe {
                        mmm.run(
                            m,
                            n,
                            &[
                                FusedSpec::AddMatMul {
                                    a: AsInputValue::Borrowed(&*pa),
                                    b: AsInputValue::Borrowed(&*pb),
                                    packing: 0,
                                },
                                FusedSpec::Store(mmm.c_view(Some(1), Some(0)).wrap(&cc.view_mut())),
                            ],
                        )
                        .unwrap()
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, gemm_f32);
criterion_main!(benches);
