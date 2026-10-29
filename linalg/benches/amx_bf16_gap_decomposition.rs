// Diagnostic-only benchmark: decomposes the gap between the hand-tuned AVX-512 asm f32 GEMM
// kernel (`avx512_mmm_f32_16x8`) and the bf16-tile-based candidate added in PR #5
// (`linalg/src/x86_64/ndarray_bf16_gemm.rs`) into its two possible causes -- the raw AMX/VNNI
// tile arithmetic itself, versus the per-tile f32->bf16 conversion + VNNI packing that PR #5's
// kernel body performs on every `AddMatMul` call (once per 16x16 output tile).
//
// Four cases at the same m=k=n shapes as `ndarray_bf16_gemm.rs`:
//   B0 -- the existing asm kernel through the normal `MatMatMulKer` path (sanity baseline).
//   B1 -- the raw `bf16_tile_gemm_16x16_packed` primitive called directly in a hand-rolled
//         tiling loop over the whole m x k x n matmul, with A and B already fully converted to
//         bf16 and B already VNNI-packed *outside* the timed closure. Zero conversion, zero
//         allocation, zero packing inside the timed portion -- isolates whether the tile
//         arithmetic itself (AMX TDPBF16PS on this host) is fast.
//   B2 -- same tiling loop and same primitive, but A is converted from f32 to bf16 *inside* the
//         timed closure every iteration (simulating a runtime activation matrix), while B stays
//         pre-converted and pre-packed outside the loop (simulating a weight matrix packed once
//         at model-load time and reused across many activations).
//   B3 -- not implemented here: it is the existing `ndarray_bf16_16x16` case in
//         `ndarray_bf16_gemm.rs`, included in the PR comment report by re-running that bench.
//
// This file adds no new production kernel and touches no packing/plan code -- it is a
// standalone harness calling `ndarray::simd::*` directly, bypassing `MatMatMulKer` entirely for
// B1/B2.
//
// Two further cases measure the AMX-native packed kernel (`ndarray_bf16_native_gemm.rs` +
// `ndarray_amx_native_pack.rs`) through the real `MatMatMulKer::run` path, at two different
// operand-preparation lifetimes -- these are NOT the same measurement and are compared against
// different B-cases above:
//   P0 -- A and B both prepared (`prepare_one`) OUTSIDE the timed loop, matching B1's lifetime.
//         Isolates tract's packed-execution abstraction tax over the raw AMX ceiling, with
//         neither operand's preparation cost in the timed region. Compare against B1.
//   P1 -- B prepared ONCE outside the timed loop and reused every iteration (a persistent
//         weight, as in real inference); A's source stays f32 and is prepared (`prepare_one`)
//         fresh INSIDE every timed iteration, once per whole matrix (not per tile) -- the same
//         "runtime activation" lifetime B2 uses. This is the realistic inference-shaped
//         acceptance metric. Compare against B2.
use criterion::*;
use ndarray::simd::{PackedBf16B, bf16_tile_gemm_16x16_packed, f32_to_bf16_batch_rne};
use std::hint::black_box;
use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec};

const TILE: usize = 16;

/// Walks the m x k x n output in 16x16 tiles (m, n assumed multiples of 16; k padded to a
/// multiple of 32 by the caller) and accumulates each tile via `bf16_tile_gemm_16x16_packed`.
/// `a_bf16` is row-major `[m, k_padded]`; `b_packed` is one `PackedBf16B` per 16-column tile of
/// B, indexed `b_tiles[j_tile]`. No allocation, no conversion -- pure tile-primitive calls.
fn tiled_matmul_packed(
    m: usize,
    n: usize,
    k_padded: usize,
    a_bf16: &[u16],
    b_tiles: &[PackedBf16B],
    c: &mut [f32],
) {
    let m_tiles = m / TILE;
    let n_tiles = n / TILE;
    let mut tile_c = [0f32; TILE * TILE];
    for it in 0..m_tiles {
        let a_row_tile = &a_bf16[it * TILE * k_padded..(it + 1) * TILE * k_padded];
        for jt in 0..n_tiles {
            tile_c.fill(0.0);
            bf16_tile_gemm_16x16_packed(a_row_tile, &b_tiles[jt], &mut tile_c);
            for i in 0..TILE {
                for j in 0..TILE {
                    c[(it * TILE + i) * n + jt * TILE + j] = tile_c[i * TILE + j];
                }
            }
        }
    }
}

fn gap_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("amx_bf16_gap_decomposition");

    for &(m, k, n) in &[(512usize, 512usize, 512usize), (1024, 1024, 1024)] {
        group.throughput(Throughput::Elements((2 * m * k * n) as u64));
        let k_padded = k.next_multiple_of(32);
        let m_tiles = m / TILE;
        let n_tiles = n / TILE;

        // ---- B0: existing asm kernel through the normal MatMatMulKer path ----
        {
            let mmm = tract_linalg::x86_64::mmm::avx512_mmm_f32_16x8.mmm();
            group.bench_with_input(
                BenchmarkId::new("B0_asm_16x8", format!("{m}x{k}x{n}")),
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

        // ---- B1: raw AMX tile primitive, A and B fully pre-converted + pre-packed outside
        // the timed loop. Isolates the tile arithmetic itself. ----
        {
            let a_f32 = vec![0f32; m * k_padded];
            let mut a_bf16 = vec![0u16; m * k_padded];
            f32_to_bf16_batch_rne(&a_f32, &mut a_bf16);

            let b_f32 = vec![0f32; k_padded * n];
            let mut b_bf16_rm = vec![0u16; k_padded * n];
            f32_to_bf16_batch_rne(&b_f32, &mut b_bf16_rm);
            // One PackedBf16B per 16-column tile of B, row-major over k_padded.
            let mut b_tiles: Vec<PackedBf16B> = Vec::with_capacity(n_tiles);
            for jt in 0..n_tiles {
                let mut col_major = vec![0u16; k_padded * TILE];
                for kk in 0..k_padded {
                    for jj in 0..TILE {
                        col_major[kk * TILE + jj] = b_bf16_rm[kk * n + jt * TILE + jj];
                    }
                }
                b_tiles.push(PackedBf16B::pack(&col_major, k_padded));
            }
            let mut c_out = vec![0f32; m * n];

            group.bench_function(
                BenchmarkId::new("B1_raw_amx_prepacked", format!("{m}x{k}x{n}")),
                |be| {
                    be.iter(|| {
                        tiled_matmul_packed(m, n, k_padded, &a_bf16, &b_tiles, &mut c_out);
                        black_box(&c_out);
                    });
                },
            );
            let _ = m_tiles;
        }

        // ---- B2: A converted f32->bf16 inside the timed loop (runtime activation);
        // B pre-converted + pre-packed outside (weight matrix packed once at load time). ----
        {
            let a_f32 = vec![0f32; m * k_padded];

            let b_f32 = vec![0f32; k_padded * n];
            let mut b_bf16_rm = vec![0u16; k_padded * n];
            f32_to_bf16_batch_rne(&b_f32, &mut b_bf16_rm);
            let mut b_tiles: Vec<PackedBf16B> = Vec::with_capacity(n_tiles);
            for jt in 0..n_tiles {
                let mut col_major = vec![0u16; k_padded * TILE];
                for kk in 0..k_padded {
                    for jj in 0..TILE {
                        col_major[kk * TILE + jj] = b_bf16_rm[kk * n + jt * TILE + jj];
                    }
                }
                b_tiles.push(PackedBf16B::pack(&col_major, k_padded));
            }
            let mut c_out = vec![0f32; m * n];
            let mut a_bf16_scratch = vec![0u16; m * k_padded];

            group.bench_function(
                BenchmarkId::new("B2_activation_runtime_convert", format!("{m}x{k}x{n}")),
                |be| {
                    be.iter(|| {
                        f32_to_bf16_batch_rne(&a_f32, &mut a_bf16_scratch);
                        tiled_matmul_packed(m, n, k_padded, &a_bf16_scratch, &b_tiles, &mut c_out);
                        black_box(&c_out);
                    });
                },
            );
        }

        // ---- B3: existing PR #5 kernel through MatMatMulKer -- included for cross-reference;
        // see `ndarray_bf16_gemm.rs` for the primary measurement of this case. ----
        {
            let mmm = tract_linalg::x86_64::mmm::ndarray_avx512_bf16_mmm_f32_16x16.mmm();
            group.bench_with_input(
                BenchmarkId::new("B3_pr5_kernel_as_is", format!("{m}x{k}x{n}")),
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

        // ---- P0: AMX-native packed kernel through MatMatMulKer, A and B both prepared
        // outside the timed loop -- compare against B1 (raw AMX ceiling, same lifetime). ----
        {
            let mmm = tract_linalg::x86_64::mmm::ndarray_amx_native_bf16_mmm_f32_16x16.mmm();
            group.bench_with_input(
                BenchmarkId::new("P0_amx_native_prepacked", format!("{m}x{k}x{n}")),
                &(m, k, n),
                |be, &(m, k, n)| {
                    let packing = &mmm.packings()[1];
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
                                    packing: 1,
                                },
                                FusedSpec::Store(mmm.c_view(Some(1), Some(0)).wrap(&cc.view_mut())),
                            ],
                        )
                        .unwrap()
                    });
                },
            );
        }

        // ---- P1: AMX-native packed kernel through MatMatMulKer, B prepared once outside the
        // timed loop and reused (persistent weight); A re-prepared once per iteration inside
        // the timed loop, once for the whole matrix (runtime activation) -- compare against B2
        // (same lifetime split). This is the realistic inference-shaped acceptance metric. ----
        {
            let mmm = tract_linalg::x86_64::mmm::ndarray_amx_native_bf16_mmm_f32_16x16.mmm();
            group.bench_with_input(
                BenchmarkId::new("P1_amx_native_runtime_a", format!("{m}x{k}x{n}")),
                &(m, k, n),
                |be, &(m, k, n)| {
                    let packing = &mmm.packings()[1];
                    let a = Tensor::zero::<f32>(&[m, k]).unwrap();
                    let b = Tensor::zero::<f32>(&[k, n]).unwrap();
                    let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
                    let mut cc = Tensor::zero::<f32>(&[n, m]).unwrap();
                    be.iter(|| unsafe {
                        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
                        mmm.run(
                            m,
                            n,
                            &[
                                FusedSpec::AddMatMul {
                                    a: AsInputValue::Borrowed(&*pa),
                                    b: AsInputValue::Borrowed(&*pb),
                                    packing: 1,
                                },
                                FusedSpec::Store(mmm.c_view(Some(1), Some(0)).wrap(&cc.view_mut())),
                            ],
                        )
                        .unwrap();
                        black_box(&cc);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, gap_decomposition);
criterion_main!(benches);
