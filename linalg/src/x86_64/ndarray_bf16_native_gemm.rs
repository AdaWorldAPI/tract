#![allow(clippy::needless_range_loop)]
//! An f32 GEMM `MatMatMulKer` body whose `AddMatMul` step consumes operands already packed
//! into `ndarray::simd::PackedBf16B` / panel-native bf16 form (`ndarray_amx_native_pack.rs`'s
//! `NdarrayAmxBf16A`/`NdarrayAmxBf16B` packing) and calls
//! `ndarray::simd::bf16_tile_gemm_16x16_packed` directly -- no allocation, no f32->bf16
//! conversion, and no VNNI packing inside `AddMatMul` itself, unlike `ndarray_bf16_gemm.rs`'s
//! kernel, which redoes all three on every tile call.
//!
//! Registered outside automatic dispatch (see `mmm.rs`'s registration comment for this kernel):
//! reachable only by direct construction, not through `MmmDispatch::native()`.
//!
//! Precision tradeoff is identical to `ndarray_bf16_gemm.rs`'s kernel (see its module doc):
//! operands are truncated to bf16 at pack time, one-time and lossy versus f32; the tile
//! arithmetic itself introduces no further lossiness beyond bf16-precision accumulation order.

use crate::frame::mmm::FusedKerSpec;
use crate::frame::mmm::OutputStoreKer;

#[cfg(target_arch = "x86_64")]
use ndarray::simd::{PackedBf16B, bf16_tile_gemm_16x16_packed};

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

/// `pa` points at an `NdarrayAmxBf16AValue` panel: row-major bf16 `[16, k_padded]`, ready to
/// feed `bf16_tile_gemm_16x16_packed` as `a_bf16` with no further work. `pb` points at one
/// `ndarray::simd::PackedBf16B` value (not a byte blob -- `NdarrayAmxBf16BValue::panel_bytes`
/// hands back a pointer to the `PackedBf16B` itself, cast to `*const u8`), reinterpreted back
/// in place. Both were built once at `prepare_one`/`prepare_one_view` time by
/// `ndarray_amx_native_pack.rs`; this function performs no allocation, no bf16 conversion, and
/// no VNNI packing.
#[cfg(target_arch = "x86_64")]
unsafe fn add_mat_mul_amx_native(
    pa: *const u8,
    pb: *const u8,
    k: usize,
    ab: &mut [[f32; TILE]; TILE],
) {
    unsafe {
        if k == 0 {
            return;
        }
        let k_padded = k.next_multiple_of(32);
        let a_bf16 = std::slice::from_raw_parts(pa as *const u16, TILE * k_padded);
        let b = &*(pb as *const PackedBf16B);
        debug_assert_eq!(b.k(), k_padded);

        let mut c_tile = [0f32; TILE * TILE];
        bf16_tile_gemm_16x16_packed(a_bf16, b, &mut c_tile);

        for i in 0..TILE {
            for j in 0..TILE {
                ab[i][j] += c_tile[i * TILE + j];
            }
        }
    }
}

// See `ndarray_bf16_gemm.rs`'s identical stub for why this exists: `linalg/src/lib.rs`
// compiles this module tree under `feature = "foreign-inventory"` on any host arch, but
// `ndarray` is only a dependency on x86_64.
#[cfg(not(target_arch = "x86_64"))]
unsafe fn add_mat_mul_amx_native(
    _pa: *const u8,
    _pb: *const u8,
    _k: usize,
    _ab: &mut [[f32; TILE]; TILE],
) {
    unreachable!("ndarray_bf16_native_gemm's kernel is x86_64-only and unbuilt elsewhere")
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

/// The `MatMatMulKer` inner loop. `AddMatMul` only accepts packing index 1 (this kernel's
/// `NdarrayAmxBf16A`/`NdarrayAmxBf16B` packing) -- packing 0 (the framework's default f32
/// packing) is never a valid call here, since this kernel is only ever reached by direct
/// construction with the native packing explicitly selected.
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
                    unreachable!("quantization ops are not reachable on the f32-only packing")
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing } => {
                    assert_eq!(packing, 1, "this kernel only declares packing 1 (AMX-native bf16)");
                    add_mat_mul_amx_native(pa, pb, k, &mut ab);
                }
                FusedKerSpec::Store(tile) => store(&tile, &ab),
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

    /// Same guard as `ndarray_bf16_gemm.rs`'s: this kernel is registered without
    /// `inventory::submit!`, so it must never surface through `MmmDispatch::native()`, for
    /// both a concrete and a symbolic (`None`) N.
    #[test]
    fn amx_native_candidate_is_not_reachable_through_automatic_dispatch() {
        let dispatch = MmmDispatch::native();
        for n in [Some(32), None] {
            let query = Query::plain(DatumType::F32, Some(64), Some(256), n);
            let suitable = dispatch.suitable(&query);
            assert!(
                suitable
                    .iter()
                    .all(|(mmm, _, _)| mmm.name() != "ndarray_amx_native_bf16_mmm_f32_16x16"),
                "the AMX-native candidate must never appear in automatic dispatch (n={n:?})"
            );
        }
    }
}

/// Tolerance-based correctness test, same shape as `ndarray_bf16_gemm.rs`'s `bf16_tolerance`
/// module: this kernel is bf16-precision by construction, run directly against `MatMatMulKer`
/// through the real fused-op path (`AddMatMul` + `Store`) with packing 1 explicitly selected.
#[cfg(all(test, target_arch = "x86_64"))]
mod bf16_tolerance {
    use crate::frame::mmm::FusedSpec;
    use crate::x86_64::mmm::ndarray_amx_native_bf16_mmm_f32_16x16;
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

        let mmm = ndarray_amx_native_bf16_mmm_f32_16x16.mmm();
        if !mmm.built() || !mmm.runnable() {
            eprintln!("skipping: ndarray_amx_native_bf16_mmm_f32_16x16 not runnable on this host");
            return;
        }
        let packing = &mmm.packings()[1];
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
                        packing: 1,
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
