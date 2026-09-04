#![allow(clippy::needless_range_loop)]
//! Packing formats that hold the ndarray AMX-native bf16 tile representation
//! built once at `prepare_one`/`prepare_one_view` time, so the kernel body in
//! `ndarray_bf16_native_gemm.rs` never converts or VNNI-packs anything inside
//! `AddMatMul`.
//!
//! Panel geometry mirrors `amx_bf16.rs`'s `PackedAmxBf16A`/`PackedBf16K2` --
//! r=16, K padded to a multiple of 32 -- but panels are stored as owned Rust
//! values (`Vec<u16>` row-major bf16 for A, `ndarray::simd::PackedBf16B` for
//! B) rather than raw bytes in a `Blob`: the consuming kernel calls
//! `ndarray::simd::bf16_tile_gemm_16x16_packed`'s typed API directly, so
//! there is nothing to gain from a byte-blob indirection and every extra
//! layer would be an extra copy.
//!
//! `AMX_BF16_A` is meant for activation-lifetime operands (converted once
//! per matrix, reused across every tile of that matmul); `AMX_BF16_B` is
//! meant for constant-weight-lifetime operands (converted and VNNI-packed
//! once, reused across every matmul that shares the weight).

use std::alloc::Layout;
use std::fmt::Display;
use std::hash::{Hash, Hasher};

use tract_data::internal::*;

use ndarray::simd::{PackedBf16B, f32_to_bf16_batch_rne};

use crate::WeightType;
use crate::frame::mmm::{
    EagerPackedInput, MMMInputFormat, MMMInputValue, PackedExoticFact, PackedMatrixStorage,
};

const R: usize = 16;

fn k_padded(k: usize) -> usize {
    k.next_multiple_of(32)
}

/// Round every f32 element of `tensor` through bf16 (round-to-nearest-even),
/// matching what the packers in this module do at pack time. Non-f32 tensors
/// pass through unchanged. Lets a reference f32 matmul reproduce the
/// kernel's bf16 rounding.
fn simulate_bf16_precision_loss(mut tensor: Tensor) -> TractResult<Tensor> {
    if tensor.datum_type() == f32::datum_type() {
        let mut plain = tensor.try_as_plain_mut()?;
        let slice = plain.as_slice_mut::<f32>()?;
        let mut bits = vec![0u16; slice.len()];
        f32_to_bf16_batch_rne(slice, &mut bits);
        for (v, b) in slice.iter_mut().zip(bits.iter()) {
            *v = f32::from_bits((*b as u32) << 16);
        }
    }
    Ok(tensor)
}

// ───────────────────────────── A: activation lifetime ──────────────────────

/// Packing format for the AMX-native bf16 kernel's A operand: bf16-converted,
/// panel-native (row-major `[16, k_padded]` per panel) data, built once per
/// `prepare_one`/`prepare_one_view` call. Never carries partially-converted
/// or per-tile state -- the whole matrix is converted in one
/// `f32_to_bf16_batch_rne` pass per panel row at pack time.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct NdarrayAmxBf16A {
    r: usize,
}

impl NdarrayAmxBf16A {
    pub fn new(r: usize) -> Self {
        assert_eq!(r, R, "ndarray's bf16 tile primitive is fixed at r=16");
        NdarrayAmxBf16A { r }
    }
}

impl Display for NdarrayAmxBf16A {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NdarrayAmxBf16A[{}]", self.r)
    }
}

impl MMMInputFormat for NdarrayAmxBf16A {
    fn prepare_tensor(&self, t: &Tensor, k_axis: usize, mn_axis: usize) -> TractResult<Tensor> {
        Ok(PackedMatrixStorage::new(self.prepare_one(t, k_axis, mn_axis)?)
            .into_tensor(t.datum_type()))
    }

    fn prepare_one_view(
        &self,
        t: &TensorView,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        let k = t.shape()[k_axis];
        let mn = t.shape()[mn_axis];
        let kp = k_padded(k);
        let panels_count = mn.div_ceil(self.r);
        let st = t.strides();
        let (ks, ms) = (st[k_axis], st[mn_axis]);

        let mut panels: Vec<Vec<u16>> = Vec::with_capacity(panels_count);
        let mut row_f32 = vec![0f32; k];
        unsafe {
            let src = t.as_ptr_unchecked::<f32>();
            for p in 0..panels_count {
                let pw = self.r.min(mn - p * self.r);
                let mn0 = (p * self.r) as isize;
                let mut panel = vec![0u16; self.r * kp];
                for lm in 0..pw {
                    let srow_base = src.offset((mn0 + lm as isize) * ms);
                    for kk in 0..k {
                        row_f32[kk] = *srow_base.offset(kk as isize * ks);
                    }
                    f32_to_bf16_batch_rne(&row_f32, &mut panel[lm * kp..lm * kp + k]);
                }
                panels.push(panel);
            }
        }

        Ok(Box::new(NdarrayAmxBf16AValue {
            fact: PackedExoticFact { format: Box::new(self.clone()), mn: mn.to_dim(), k },
            format: self.clone(),
            panels,
            k,
            mn,
        }))
    }

    fn k_alignment(&self) -> usize {
        32
    }

    fn r(&self) -> usize {
        self.r
    }

    fn precursor(&self) -> WeightType {
        WeightType::Plain(f32::datum_type())
    }

    fn simulate_precision_loss(&self, tensor: Tensor) -> TractResult<Tensor> {
        simulate_bf16_precision_loss(tensor)
    }

    fn merge_with<'o, 'a: 'o, 'b: 'o>(
        &'a self,
        o: &'b dyn MMMInputFormat,
    ) -> Option<&'o dyn MMMInputFormat> {
        o.downcast_ref::<NdarrayAmxBf16A>().filter(|x| x.r == self.r).map(|_| self as _)
    }

    fn mem_size(&self, k: TDim, mn: TDim) -> TDim {
        mn.divceil(self.r) * (self.r * k_padded(k.to_usize().unwrap_or(0)) * 2)
    }

    fn extract_at_mn_f16(&self, _: &EagerPackedInput, _: usize, _: &mut [f16]) -> TractResult<()> {
        bail!("no f16 extract")
    }

    fn extract_at_mn_f32(&self, _: &EagerPackedInput, _: usize, _: &mut [f32]) -> TractResult<()> {
        bail!("no f32 extract")
    }
}

/// One prepared A operand: `panels[p]` is row-major bf16 `[16, k_padded]`,
/// ready to feed `ndarray::simd::bf16_tile_gemm_16x16_packed` as `a_bf16`
/// with no further conversion.
#[derive(Clone, Debug)]
pub struct NdarrayAmxBf16AValue {
    fact: PackedExoticFact,
    format: NdarrayAmxBf16A,
    panels: Vec<Vec<u16>>,
    k: usize,
    mn: usize,
}

impl Hash for NdarrayAmxBf16AValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.format.hash(state);
        self.k.hash(state);
        self.mn.hash(state);
        self.panels.hash(state);
    }
}

impl PartialEq for NdarrayAmxBf16AValue {
    fn eq(&self, other: &Self) -> bool {
        self.format == other.format
            && self.k == other.k
            && self.mn == other.mn
            && self.panels == other.panels
    }
}
impl Eq for NdarrayAmxBf16AValue {}

impl Display for NdarrayAmxBf16AValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} value (mn={} k={})", self.format, self.mn, self.k)
    }
}

impl MMMInputValue for NdarrayAmxBf16AValue {
    fn format(&self) -> &dyn MMMInputFormat {
        &self.format
    }
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        None
    }
    fn panel_bytes(&self, i: usize, _buffer: Option<*mut u8>) -> TractResult<*const u8> {
        Ok(self.panels[i].as_ptr() as *const u8)
    }
    fn mn(&self) -> usize {
        self.mn
    }
    fn k(&self) -> usize {
        self.k
    }
    fn exotic_fact(&self) -> &dyn ExoticFact {
        &self.fact
    }
    fn extract_at_mn_f16(&self, _mn: usize, _slice: &mut [f16]) -> TractResult<()> {
        bail!("no f16 extract")
    }
    fn extract_at_mn_f32(&self, _mn: usize, _slice: &mut [f32]) -> TractResult<()> {
        bail!("no f32 extract")
    }
}

// ───────────────────────────── B: constant-weight lifetime ─────────────────

/// Packing format for the AMX-native bf16 kernel's B operand: bf16-converted
/// AND VNNI-packed (`ndarray::simd::PackedBf16B`) once per
/// `prepare_one`/`prepare_one_view` call, reusable across every matmul that
/// shares the packed weight -- the shape a constant weight tensor would be
/// packed into once at model-load time.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct NdarrayAmxBf16B {
    r: usize,
}

impl NdarrayAmxBf16B {
    pub fn new(r: usize) -> Self {
        assert_eq!(r, R, "ndarray's bf16 tile primitive is fixed at r=16");
        NdarrayAmxBf16B { r }
    }
}

impl Display for NdarrayAmxBf16B {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NdarrayAmxBf16B[{}]", self.r)
    }
}

impl MMMInputFormat for NdarrayAmxBf16B {
    fn prepare_tensor(&self, t: &Tensor, k_axis: usize, mn_axis: usize) -> TractResult<Tensor> {
        Ok(PackedMatrixStorage::new(self.prepare_one(t, k_axis, mn_axis)?)
            .into_tensor(t.datum_type()))
    }

    fn prepare_one_view(
        &self,
        t: &TensorView,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        let k = t.shape()[k_axis];
        let mn = t.shape()[mn_axis];
        let kp = k_padded(k);
        let panels_count = mn.div_ceil(self.r);
        let st = t.strides();
        let (ks, ms) = (st[k_axis], st[mn_axis]);

        let mut panels: Vec<PackedBf16B> = Vec::with_capacity(panels_count);
        let mut row_major_bf16 = vec![0u16; kp * self.r];
        let mut row_f32 = vec![0f32; self.r];
        unsafe {
            let src = t.as_ptr_unchecked::<f32>();
            for p in 0..panels_count {
                let pw = self.r.min(mn - p * self.r);
                let mn0 = (p * self.r) as isize;
                row_major_bf16.fill(0);
                for kk in 0..k {
                    let srow_base = src.offset(kk as isize * ks + mn0 * ms);
                    for lm in 0..pw {
                        row_f32[lm] = *srow_base.offset(lm as isize * ms);
                    }
                    let mut row_bf16 = [0u16; R];
                    f32_to_bf16_batch_rne(&row_f32[..pw], &mut row_bf16[..pw]);
                    row_major_bf16[kk * self.r..kk * self.r + pw].copy_from_slice(&row_bf16[..pw]);
                }
                panels.push(PackedBf16B::pack(&row_major_bf16, kp));
            }
        }

        Ok(Box::new(NdarrayAmxBf16BValue {
            fact: PackedExoticFact { format: Box::new(self.clone()), mn: mn.to_dim(), k },
            format: self.clone(),
            panels,
            k,
            mn,
        }))
    }

    fn k_alignment(&self) -> usize {
        32
    }

    fn r(&self) -> usize {
        self.r
    }

    fn precursor(&self) -> WeightType {
        WeightType::Plain(f32::datum_type())
    }

    fn simulate_precision_loss(&self, tensor: Tensor) -> TractResult<Tensor> {
        simulate_bf16_precision_loss(tensor)
    }

    fn merge_with<'o, 'a: 'o, 'b: 'o>(
        &'a self,
        o: &'b dyn MMMInputFormat,
    ) -> Option<&'o dyn MMMInputFormat> {
        o.downcast_ref::<NdarrayAmxBf16B>().filter(|x| x.r == self.r).map(|_| self as _)
    }

    fn mem_size(&self, k: TDim, mn: TDim) -> TDim {
        mn.divceil(self.r) * (k_padded(k.to_usize().unwrap_or(0)) * self.r * 2)
    }

    fn extract_at_mn_f16(&self, _: &EagerPackedInput, _: usize, _: &mut [f16]) -> TractResult<()> {
        bail!("no f16 extract")
    }

    fn extract_at_mn_f32(&self, _: &EagerPackedInput, _: usize, _: &mut [f32]) -> TractResult<()> {
        bail!("no f32 extract")
    }
}

/// One prepared B operand: `panels[p]` is an `ndarray::simd::PackedBf16B`
/// ready to feed `bf16_tile_gemm_16x16_packed` directly, with no further
/// conversion or VNNI packing.
pub struct NdarrayAmxBf16BValue {
    fact: PackedExoticFact,
    format: NdarrayAmxBf16B,
    panels: Vec<PackedBf16B>,
    k: usize,
    mn: usize,
}

impl Clone for NdarrayAmxBf16BValue {
    fn clone(&self) -> Self {
        NdarrayAmxBf16BValue {
            fact: self.fact.clone(),
            format: self.format.clone(),
            panels: self
                .panels
                .iter()
                .map(|p| PackedBf16B::from_le_bytes(p.as_le_bytes(), p.k()))
                .collect(),
            k: self.k,
            mn: self.mn,
        }
    }
}

impl std::fmt::Debug for NdarrayAmxBf16BValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} ({} panels)", self.format, self.panels.len())
    }
}

impl Hash for NdarrayAmxBf16BValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.format.hash(state);
        self.k.hash(state);
        self.mn.hash(state);
        for p in &self.panels {
            p.data().hash(state);
            p.k().hash(state);
        }
    }
}

impl PartialEq for NdarrayAmxBf16BValue {
    fn eq(&self, other: &Self) -> bool {
        self.format == other.format
            && self.k == other.k
            && self.mn == other.mn
            && self.panels.len() == other.panels.len()
            && self
                .panels
                .iter()
                .zip(other.panels.iter())
                .all(|(a, b)| a.k() == b.k() && a.data() == b.data())
    }
}
impl Eq for NdarrayAmxBf16BValue {}

impl Display for NdarrayAmxBf16BValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} value (mn={} k={})", self.format, self.mn, self.k)
    }
}

impl MMMInputValue for NdarrayAmxBf16BValue {
    fn format(&self) -> &dyn MMMInputFormat {
        &self.format
    }
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        None
    }
    fn panel_bytes(&self, i: usize, _buffer: Option<*mut u8>) -> TractResult<*const u8> {
        Ok(&self.panels[i] as *const PackedBf16B as *const u8)
    }
    fn mn(&self) -> usize {
        self.mn
    }
    fn k(&self) -> usize {
        self.k
    }
    fn exotic_fact(&self) -> &dyn ExoticFact {
        &self.fact
    }
    fn extract_at_mn_f16(&self, _mn: usize, _slice: &mut [f16]) -> TractResult<()> {
        bail!("no f16 extract")
    }
    fn extract_at_mn_f32(&self, _mn: usize, _slice: &mut [f32]) -> TractResult<()> {
        bail!("no f32 extract")
    }
}
