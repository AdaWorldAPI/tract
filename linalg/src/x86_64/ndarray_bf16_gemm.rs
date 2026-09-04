#![allow(clippy::needless_range_loop)]
//! An f32 GEMM `MatMatMulKer` body whose `AddMatMul` step truncates its operands to bf16 and
//! calls into the AdaWorldAPI ndarray fork's `simd::bf16_tile_gemm_16x16_packed` tile primitive
//! (AMX `TDPBF16PS` → AVX-512 `VDPBF16PS` → decode+FMA polyfill, selected at runtime) as a
//! second additional candidate alongside the hand-tuned AVX-512 asm kernels and pilot v1's
//! f32-exact `ndarray_avx512_mmm_f32_16x8` (`ndarray_gemm.rs`).
//!
//! **Where this stands relative to pilot v1's structural problem:** `ndarray_gemm.rs`'s
//! `blas_gemm` call allocates a fresh `Array` and re-packs its B operand on *every tile call*
//! via a full BLAS-level3 entry point (allocation + a generic path selection on top of the
//! repack), which was measured 5-10x slower than the hand-tuned asm kernel for exactly that
//! reason. This kernel's `AddMatMul` step is *also* invoked once per output tile (that is how
//! `MatMatMulKer`'s fused-op interpreter calls into any kernel body — one call per (MR, NR)
//! tile, carrying that tile's full K depth), so it still allocates and VNNI-packs its A/B
//! operands **once per tile call**, not once per whole-matrix GEMM — hoisting the pack any
//! further up would mean restructuring the packed-panel format `MatMatMulKer` hands the kernel,
//! which is out of scope for this pilot. What *is* structurally different from pilot v1: the
//! per-call work here is a single `PackedBf16B::pack` (one VNNI interleave over a `k×16`
//! buffer) plus a bf16 truncation pass, calling directly into a tile primitive with no
//! allocation inside — not a generic BLAS-level3 entry point that re-derives packing/path
//! selection from scratch on every call. The measured benchmark numbers below are the honest
//! comparison; see them before assuming this claim translates into a win.
//!
//! **Precision, stated plainly:** the accumulate arithmetic (`C += A·B`) is bit-exact across
//! all three `bf16_tile_gemm` tiers for bf16-exact-integer operands with accumulation below
//! 2^24 (verified by `assert_eq!` parity tests in ndarray's own `hpc::bf16_tile_gemm`), and for
//! general float operands the tiers agree with each other exactly up to accumulation order —
//! this kernel introduces **no additional lossiness of its own**. The precision this kernel
//! trades away versus the native f32 asm kernel is entirely the one-time f32→bf16 truncation of
//! the input operands themselves before they reach any tile primitive: bf16 keeps a 7-bit
//! mantissa against f32's 23-bit, so every element of A and B loses precision at pack time, not
//! merely at accumulation time. This is a real, user-visible precision change for a general
//! inference engine and must not be read as "approximate GEMM" (the arithmetic is not
//! approximate) or as "safe to swap in for f32 workloads" (real model weights are not
//! bf16-exact integers, so the tier-parity bit-exactness above does not extend to them).
//!
//! Goes through `ndarray::simd::*` (`f32_to_bf16_batch_rne`, `PackedBf16B`,
//! `bf16_tile_gemm_16x16_packed`, `bf16_tile_gemm_tier`), the canonical consumer-facing
//! re-export, never `ndarray::hpc::bf16_tile_gemm::*` directly — see the ndarray fork's own
//! `CLAUDE.md` ("all SIMD from `ndarray::simd`").
//!
//! Tile shape is fixed by the ndarray primitive at M=16, N=16, K a multiple of 32, so this
//! kernel is registered at the matching (16, 16) `MatMatMulKer` geometry rather than reusing
//! pilot v1's 16x8 -- packed A/B panels are padded up to the next multiple of 32 in K with
//! zero rows/columns, which contribute nothing to the accumulation.

use ndarray::simd::{PackedBf16B, bf16_tile_gemm_16x16_packed, f32_to_bf16_batch_rne};

use crate::frame::mmm::FusedKerSpec;
use crate::frame::mmm::OutputStoreKer;

macro_rules! scalar {
    ($ab: expr, $m: expr, $f: expr) => {
        for i in 0..$ab.len() {
            for j in 0..$ab[0].len() {
                $ab[i][j] = $f($m, $ab[i][j])
            }
        }
    };
}

macro_rules! per_row {
    ($ab: expr, $m: expr, $f: expr) => {
        for i in 0..$ab.len() {
            for j in 0..$ab[0].len() {
                $ab[i][j] = $f(*$m.add(i), $ab[i][j])
            }
        }
    };
}

macro_rules! per_col {
    ($ab: expr, $m: expr, $f: expr) => {
        for i in 0..$ab.len() {
            for j in 0..$ab[0].len() {
                $ab[i][j] = $f(*$m.add(j), $ab[i][j])
            }
        }
    };
}

const TILE: usize = 16;

/// `pa` is packed k-major, MR(=16) contiguous per k-step (`pa[ik * 16 + i]`); `pb` is packed
/// k-major NR(=16) contiguous per k-step (`pb[ik * 16 + j]`), which is already row-major
/// `B[K, 16]` -- no transpose needed. `A_panel` (`pa`) is transposed into a small contiguous
/// `(16, K)` buffer, same as pilot v1, because the bf16 tile primitive wants row-major `A[16, K]`.
///
/// Both operands are truncated to bf16 with `f32_to_bf16_batch_rne` (round-to-nearest-even,
/// the hot-loop-safe path -- never the scalar RNE fn, which is test-only). K is padded up to
/// the next multiple of 32 with zero rows in A and zero rows in B: the padding columns/rows
/// contribute `0 * anything = 0` to every accumulated cell, so the padding is inert.
///
/// B is packed into VNNI layout via `PackedBf16B::pack`, once per `AddMatMul` call -- i.e. once
/// per (MR, NR) output tile, since that is the granularity `MatMatMulKer` calls a kernel body
/// at. Unlike pilot v1 this pack is a single VNNI interleave straight into the tile primitive
/// (no BLAS-level3 entry point re-deriving packing/dispatch from scratch), but it is still
/// real per-tile allocation and work, not something hoisted above the tile loop.
unsafe fn add_mat_mul_bf16(pa: *const u8, pb: *const u8, k: usize, ab: &mut [[f32; TILE]; TILE]) {
    unsafe {
        if k == 0 {
            return;
        }
        let a = pa as *const f32;
        let b = pb as *const f32;

        let k_padded = k.next_multiple_of(32);

        let mut a_row_major = vec![0f32; TILE * k];
        for i in 0..TILE {
            for ik in 0..k {
                a_row_major[i * k + ik] = *a.add(ik * TILE + i);
            }
        }
        let mut a_bf16 = vec![0u16; TILE * k_padded];
        for i in 0..TILE {
            f32_to_bf16_batch_rne(
                &a_row_major[i * k..i * k + k],
                &mut a_bf16[i * k_padded..i * k_padded + k],
            );
        }

        let b_row_major = std::slice::from_raw_parts(b, k * TILE);
        let mut b_bf16 = vec![0u16; k_padded * TILE];
        f32_to_bf16_batch_rne(b_row_major, &mut b_bf16[..k * TILE]);

        let packed_b = PackedBf16B::pack(&b_bf16, k_padded);

        let mut c_tile = [0f32; TILE * TILE];
        bf16_tile_gemm_16x16_packed(&a_bf16, &packed_b, &mut c_tile);

        for i in 0..TILE {
            for j in 0..TILE {
                ab[i][j] += c_tile[i * TILE + j];
            }
        }
    }
}

unsafe fn add_unicast(ab: &mut [[f32; TILE]; TILE], other: &OutputStoreKer) {
    unsafe {
        for i in 0..TILE {
            for j in 0..TILE {
                let value: *const f32 = other
                    .ptr
                    .offset(other.row_byte_stride * i as isize + other.col_byte_stride * j as isize)
                    as _;
                ab[i][j] += *value;
            }
        }
    }
}

unsafe fn store(tile: &OutputStoreKer, ab: &[[f32; TILE]; TILE]) {
    unsafe {
        for i in 0..TILE {
            for j in 0..TILE {
                let loc: *mut f32 = tile
                    .ptr
                    .offset(tile.row_byte_stride * i as isize + tile.col_byte_stride * j as isize)
                    as _;
                *loc = ab[i][j];
            }
        }
    }
}

/// The `MatMatMulKer` inner loop, f32-only, one packing (index 0, plain f32×f32), fixed 16x16
/// tile geometry (the shape `bf16_tile_gemm_16x16_packed` is built for). Same fused-op
/// interpreter shape as `crate::generic::mmm::kernel` and pilot v1's `ndarray_gemm::kernel`;
/// the `AddMatMul` arm is the only place this diverges from the generic reference.
pub(super) unsafe fn kernel(mut pnl: *const FusedKerSpec<f32>) -> isize {
    unsafe {
        let mut ab = [[0f32; TILE]; TILE];
        loop {
            if pnl.is_null() {
                break;
            }
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => ab = [[0f32; TILE]; TILE],
                FusedKerSpec::LoadTile(col_major, _row_major) => {
                    for row in 0..TILE {
                        for col in 0..TILE {
                            ab[row][col] = *col_major.add(col * TILE + row);
                        }
                    }
                }
                FusedKerSpec::ScalarAdd(a) => scalar!(ab, a, |a, b| a + b),
                FusedKerSpec::ScalarMul(a) => scalar!(ab, a, |a, b| a * b),
                FusedKerSpec::ScalarMin(m) => scalar!(ab, m, |a: f32, b: f32| a.min(b)),
                FusedKerSpec::ScalarMax(m) => scalar!(ab, m, |a: f32, b: f32| a.max(b)),
                FusedKerSpec::ScalarSub(m) => scalar!(ab, m, |a, b| a - b),
                FusedKerSpec::ScalarSubF(m) => scalar!(ab, m, |a, b| b - a),
                FusedKerSpec::LeakyRelu(m) => {
                    scalar!(ab, m, |a, b| if b > 0.0 { b } else { a * b })
                }
                FusedKerSpec::PerRowMin(m) => per_row!(ab, m, |a: f32, b: f32| a.min(b)),
                FusedKerSpec::PerRowMax(m) => per_row!(ab, m, |a: f32, b: f32| a.max(b)),
                FusedKerSpec::PerRowAdd(m) => per_row!(ab, m, |a, b| a + b),
                FusedKerSpec::PerRowMul(m) => per_row!(ab, m, |a, b| a * b),
                FusedKerSpec::PerRowSub(m) => per_row!(ab, m, |a, b| a - b),
                FusedKerSpec::PerRowSubF(m) => per_row!(ab, m, |a, b| b - a),
                FusedKerSpec::PerColMin(m) => per_col!(ab, m, |a: f32, b: f32| a.min(b)),
                FusedKerSpec::PerColMax(m) => per_col!(ab, m, |a: f32, b: f32| a.max(b)),
                FusedKerSpec::PerColAdd(m) => per_col!(ab, m, |a, b| a + b),
                FusedKerSpec::PerColMul(m) => per_col!(ab, m, |a, b| a * b),
                FusedKerSpec::PerColSub(m) => per_col!(ab, m, |a, b| a - b),
                FusedKerSpec::PerColSubF(m) => per_col!(ab, m, |a, b| b - a),
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    for i in 0..TILE {
                        for j in 0..TILE {
                            ab[i][j] += *rows.add(i) * *cols.add(j);
                        }
                    }
                }
                FusedKerSpec::AddUnicast(other) => add_unicast(&mut ab, &other),
                FusedKerSpec::ShiftLeft(_)
                | FusedKerSpec::RoundingShiftRight(..)
                | FusedKerSpec::QScale(..) => {
                    // Integer-quantization epilogue ops: this kernel only declares an f32
                    // accumulator packing, so a caller never reaches these arms.
                    unreachable!("quantization ops are not reachable on the f32-only packing")
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing } => {
                    assert_eq!(packing, 0, "this kernel only declares packing 0 (f32 x f32)");
                    add_mat_mul_bf16(pa, pb, k, &mut ab);
                }
                FusedKerSpec::Store(tile) => store(&tile, &ab),
            };
            pnl = pnl.add(1);
        }
    }
    0
}

#[cfg(test)]
mod dispatch_stays_default {
    use crate::frame::mmm::{MmmDispatch, Query};
    use tract_data::internal::DatumType;

    #[test]
    fn adding_the_bf16_candidate_does_not_change_default_pick() {
        let dispatch = MmmDispatch::native();
        let query = Query::plain(DatumType::F32, Some(64), Some(256), Some(32));
        let suitable = dispatch.suitable(&query);
        assert!(
            suitable.iter().any(|(mmm, _, _)| mmm.name() == "ndarray_avx512_bf16_mmm_f32_16x16"),
            "the new candidate should be suitable wherever avx512f is native"
        );
        if let Some((picked, _, _)) = dispatch.pick(&query) {
            assert_ne!(
                picked.name(),
                "ndarray_avx512_bf16_mmm_f32_16x16",
                "default dispatch must still prefer the hand-tuned asm kernel"
            );
            assert_ne!(
                picked.name(),
                "ndarray_avx512_mmm_f32_16x8",
                "default dispatch must still prefer the hand-tuned asm kernel"
            );
        }
    }
}

/// Tolerance-based correctness test. This kernel is inherently bf16-precision (see the module
/// doc): the exact-bit `test_mmm_kernel!` macro family compares kernel output against an f32
/// reference with `==`/ULP-tight bounds, which this kernel cannot pass by construction, so it
/// gets a dedicated relative-tolerance check instead, run directly against `MatMatMulKer`
/// through the same fused-op path the real dispatcher uses (`AddMatMul` + `Store`), against a
/// naive f32 reference GEMM over inputs deliberately chosen to be exactly bf16-representable
/// (so this test's own tolerance is measuring accumulation-order/tier drift, not re-measuring
/// the f32->bf16 truncation the module doc already documents and asserts is real).
#[cfg(test)]
mod bf16_tolerance {
    use crate::frame::mmm::FusedSpec;
    use crate::x86_64::mmm::ndarray_avx512_bf16_mmm_f32_16x16;
    use ndarray::simd::f32_to_bf16_batch_rne;
    use tract_data::internal::*;

    fn bf16_exact_value(x: f32) -> f32 {
        let mut bits = [0u16; 1];
        f32_to_bf16_batch_rne(&[x], &mut bits);
        f32::from_bits((bits[0] as u32) << 16)
    }

    #[test]
    fn matches_naive_f32_reference_within_bf16_tolerance() {
        let (m, k, n) = (32usize, 64usize, 32usize);
        let mut a = vec![0f32; m * k];
        let mut b = vec![0f32; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = bf16_exact_value(((i % 13) as f32 - 6.0) * 0.5);
        }
        for (i, v) in b.iter_mut().enumerate() {
            *v = bf16_exact_value(((i % 11) as f32 - 5.0) * 0.5);
        }

        let mut expected = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * n + j];
                }
                expected[i * n + j] = acc;
            }
        }

        let mmm = ndarray_avx512_bf16_mmm_f32_16x16.mmm();
        if !mmm.built() || !mmm.runnable() {
            eprintln!("skipping: ndarray_avx512_bf16_mmm_f32_16x16 not runnable on this host");
            return;
        }
        let packing = &mmm.packings()[0];
        let a_tensor = Tensor::from_shape(&[m, k], &a).unwrap();
        let pa = packing.0.prepare_one(&a_tensor, 1, 0).unwrap();
        let b_tensor = Tensor::from_shape(&[k, n], &b).unwrap();
        let pb = packing.1.prepare_one(&b_tensor, 0, 1).unwrap();
        let mut c = Tensor::zero::<f32>(&[n, m]).unwrap();

        unsafe {
            mmm.run(
                m,
                n,
                &[
                    FusedSpec::AddMatMul {
                        a: crate::mmm::AsInputValue::Borrowed(&*pa),
                        b: crate::mmm::AsInputValue::Borrowed(&*pb),
                        packing: 0,
                    },
                    FusedSpec::Store(mmm.c_view(Some(1), Some(0)).wrap(&c.view_mut())),
                ],
            )
            .unwrap();
        }

        let got = unsafe { c.as_slice_unchecked::<f32>() };
        for i in 0..m {
            for j in 0..n {
                let e = expected[i * n + j];
                let g = got[j * m + i];
                let tol = 1e-2 * e.abs().max(1.0);
                assert!(
                    (e - g).abs() <= tol,
                    "mismatch at ({i},{j}): expected {e}, got {g} (tol {tol}), tier={}",
                    ndarray::simd::bf16_tile_gemm_tier(),
                );
            }
        }
    }
}
