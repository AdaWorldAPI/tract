#![allow(clippy::needless_range_loop)]
//! An f32 GEMM `MatMatMulKer` body whose `AddMatMul` step calls into the AdaWorldAPI
//! ndarray fork's `simd::BlasLevel3::blas_gemm` instead of a hand-written inner-product
//! loop, as an additional candidate alongside the hand-tuned AVX-512 asm kernels. Every
//! other fused op (bias, min/max, per-row/per-col, store) is the same scalar Rust the
//! generic reference kernel uses, so only the matmul accumulation itself is delegated.
//!
//! Goes through `ndarray::simd::BlasLevel3`, the canonical consumer-facing re-export,
//! never `ndarray::hpc::blas_level3` directly — see the ndarray fork's own `CLAUDE.md`
//! ("all SIMD from `ndarray::simd`").

#[cfg(target_arch = "x86_64")]
use ndarray::ArrayView2;
#[cfg(target_arch = "x86_64")]
use ndarray::simd::BlasLevel3;

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

/// `pa` is packed k-major, MR contiguous per k-step (`pa[ik * MR + i]`); `pb` likewise for
/// NR. That makes the panel-pair product `ab[i][j] += sum_ik pa[ik*MR+i] * pb[ik*NR+j]` the
/// matrix product `A_panel^T . B_panel` where `A_panel` is `(k, MR)` row-major and `B_panel`
/// is `(k, NR)` row-major. `A_panel` is transposed into a small contiguous `(MR, k)` buffer
/// (a copy, not a stride trick) so both operands reach `blas_gemm` as contiguous slices and
/// take the real backend path instead of ndarray's non-contiguous fallback loop.
#[cfg(target_arch = "x86_64")]
unsafe fn add_mat_mul_ndarray<const MR: usize, const NR: usize>(
    pa: *const u8,
    pb: *const u8,
    k: usize,
    ab: &mut [[f32; NR]; MR],
) {
    unsafe {
        if k == 0 {
            return;
        }
        let a = pa as *const f32;
        let b = pb as *const f32;

        let mut a_t = vec![0f32; MR * k];
        for i in 0..MR {
            for ik in 0..k {
                a_t[i * k + ik] = *a.add(ik * MR + i);
            }
        }
        let a_view = ArrayView2::from_shape((MR, k), &a_t).unwrap();
        let b_slice = std::slice::from_raw_parts(b, k * NR);
        let b_view = ArrayView2::from_shape((k, NR), b_slice).unwrap();

        let prod = a_view.blas_gemm(1.0f32, &b_view, 0.0f32);
        for i in 0..MR {
            for j in 0..NR {
                ab[i][j] += prod[[i, j]];
            }
        }
    }
}

// `linalg/src/lib.rs` compiles this module tree under `feature = "foreign-inventory"`
// on any host arch (to enumerate x86_64 kernel names as metadata for cross-compiled
// builds), but `ndarray` is only a dependency on x86_64 (`linalg/Cargo.toml`). This
// stub keeps the crate compiling there; `MMMRustKernel!(x86_64; ...)` marks the real
// kernel `built(cfg!(target_arch = "x86_64"))`, so `MmmDispatch` never selects it and
// this arm never runs off x86_64.
#[cfg(not(target_arch = "x86_64"))]
unsafe fn add_mat_mul_ndarray<const MR: usize, const NR: usize>(
    _pa: *const u8,
    _pb: *const u8,
    _k: usize,
    _ab: &mut [[f32; NR]; MR],
) {
    unreachable!("ndarray_gemm's kernel is x86_64-only and unbuilt elsewhere")
}

unsafe fn add_unicast<const MR: usize, const NR: usize>(
    ab: &mut [[f32; NR]; MR],
    other: &OutputStoreKer,
) {
    unsafe {
        for i in 0..MR {
            for j in 0..NR {
                let value: *const f32 = other
                    .ptr
                    .offset(other.row_byte_stride * i as isize + other.col_byte_stride * j as isize)
                    as _;
                ab[i][j] += *value;
            }
        }
    }
}

unsafe fn store<const MR: usize, const NR: usize>(tile: &OutputStoreKer, ab: &[[f32; NR]; MR]) {
    unsafe {
        for i in 0..MR {
            for j in 0..NR {
                let loc: *mut f32 = tile
                    .ptr
                    .offset(tile.row_byte_stride * i as isize + tile.col_byte_stride * j as isize)
                    as _;
                *loc = ab[i][j];
            }
        }
    }
}

/// The `MatMatMulKer` inner loop, f32-only, one packing (index 0, plain f32×f32). Same
/// fused-op interpreter shape as `crate::generic::mmm::kernel`; the `AddMatMul` arm is the
/// only place this diverges from it.
pub(super) unsafe fn kernel<const MR: usize, const NR: usize>(
    mut pnl: *const FusedKerSpec<f32>,
) -> isize {
    unsafe {
        let mut ab = [[0f32; NR]; MR];
        loop {
            if pnl.is_null() {
                break;
            }
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => ab = [[0f32; NR]; MR],
                FusedKerSpec::LoadTile(col_major, _row_major) => {
                    for row in 0..MR {
                        for col in 0..NR {
                            ab[row][col] = *col_major.add(col * MR + row);
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
                    for i in 0..MR {
                        for j in 0..NR {
                            ab[i][j] += *rows.add(i) * *cols.add(j);
                        }
                    }
                }
                FusedKerSpec::AddUnicast(other) => add_unicast::<MR, NR>(&mut ab, &other),
                FusedKerSpec::ShiftLeft(_)
                | FusedKerSpec::RoundingShiftRight(..)
                | FusedKerSpec::QScale(..) => {
                    // Integer-quantization epilogue ops: this kernel only declares an f32
                    // accumulator packing, so a caller never reaches these arms.
                    unreachable!("quantization ops are not reachable on the f32-only packing")
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing } => {
                    assert_eq!(packing, 0, "this kernel only declares packing 0 (f32 x f32)");
                    add_mat_mul_ndarray::<MR, NR>(pa, pb, k, &mut ab);
                }
                FusedKerSpec::Store(tile) => store::<MR, NR>(&tile, &ab),
            };
            pnl = pnl.add(1);
        }
    }
    0
}

#[cfg(all(test, target_arch = "x86_64"))]
mod dispatch_stays_default {
    use crate::frame::mmm::{MmmDispatch, Query};
    use tract_data::internal::DatumType;

    #[test]
    fn adding_the_ndarray_candidate_does_not_change_default_pick() {
        let dispatch = MmmDispatch::native();
        let query = Query::plain(DatumType::F32, Some(64), Some(256), Some(32));
        let suitable = dispatch.suitable(&query);
        assert!(
            suitable.iter().any(|(mmm, _, _)| mmm.name() == "ndarray_avx512_mmm_f32_16x8"),
            "the new candidate should be suitable wherever avx512f is native"
        );
        if let Some((picked, _, _)) = dispatch.pick(&query) {
            assert_ne!(
                picked.name(),
                "ndarray_avx512_mmm_f32_16x8",
                "default dispatch must still prefer the hand-tuned asm kernel"
            );
        }
    }
}
