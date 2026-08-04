//! Push-constant layouts and helpers for the vendored ggml-vulkan kernels
//! moe-stream uses. Field order mirrors the GLSL exactly — do not reorder.

/// generic_binary_head push constants (rms_norm, add, get_rows, ...).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BinaryPush {
    pub ne: u32,
    /// ne00..ne03, nb00..nb03 (element strides)
    pub src0: [u32; 8],
    /// ne10..ne13, nb10..nb13
    pub src1: [u32; 8],
    /// ne20..ne23, nb20..nb23
    pub dst: [u32; 8],
    pub misalign_offsets: u32,
    pub param1: f32,
    pub param2: f32,
    pub param3: i32,
}

impl BinaryPush {
    /// Contiguous 2-D helper: shape (n0, n1), unit element strides.
    pub fn contig2d(a: (u32, u32), b: (u32, u32), d: (u32, u32)) -> BinaryPush {
        let dims = |(n0, n1): (u32, u32)| [n0, n1, 1, 1, 1, n0, n0 * n1, n0 * n1];
        BinaryPush {
            ne: d.0 * d.1,
            src0: dims(a),
            src1: dims(b),
            dst: dims(d),
            misalign_offsets: 0,
            param1: 0.0,
            param2: 0.0,
            param3: 0,
        }
    }

    /// Copy src (n0, n1) contiguous into dst rows of stride `dst_row_stride`
    /// (elements), for strided cpy/get_rows-style ops using this head.
    pub fn strided_rows(n0: u32, n1: u32, dst_row_stride: u32) -> BinaryPush {
        BinaryPush {
            ne: n0 * n1,
            src0: [n0, n1, 1, 1, 1, n0, n0 * n1, n0 * n1],
            src1: [n0, n1, 1, 1, 1, n0, n0 * n1, n0 * n1],
            dst: [
                n0,
                n1,
                1,
                1,
                1,
                dst_row_stride,
                dst_row_stride * n1,
                dst_row_stride * n1,
            ],
            misalign_offsets: 0,
            param1: 0.0,
            param2: 0.0,
            param3: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<BinaryPush>(),
            )
        }
    }
}

/// mul_mat_vec_base push constants, non-MUL_MAT_ID branch.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MatVecPush {
    pub ncols: u32,
    pub stride_a: u32,
    pub stride_b: u32,
    pub stride_d: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_d: u32,
    pub fusion_flags: u32,
    pub base_work_group_y: u32,
    pub ne02: u32,
    pub ne12: u32,
    pub broadcast2: u32,
    pub broadcast3: u32,
}

impl MatVecPush {
    /// Single matrix (nrows x ncols) times single vector.
    pub fn simple(ncols: u32, nrows: u32) -> MatVecPush {
        MatVecPush {
            ncols,
            stride_a: ncols, // element stride between rows (f32/f16 paths)
            stride_b: ncols,
            stride_d: nrows,
            batch_stride_a: nrows * ncols,
            batch_stride_b: ncols,
            batch_stride_d: nrows,
            fusion_flags: 0,
            base_work_group_y: 0,
            ne02: 1,
            ne12: 1,
            broadcast2: 1,
            broadcast3: 1,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<MatVecPush>(),
            )
        }
    }
}

/// mul_mat_vec_base push constants, MUL_MAT_ID branch (expert matvec).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MatVecIdPush {
    pub ncols: u32,
    pub stride_a: u32,
    pub stride_b: u32,
    pub stride_d: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_d: u32,
    pub fusion_flags: u32,
    pub nei0: u32,
    pub ne11: u32,
    pub expert_i1: u32,
    pub nbi1: u32,
}

impl MatVecIdPush {
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<MatVecIdPush>(),
            )
        }
    }
}

/// soft_max.comp push constants.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SoftMaxPush {
    pub kx: u32,
    pub ky: u32,
    pub ne00: u32,
    pub ne01: u32,
    pub ne02: u32,
    pub ne12: u32,
    pub ne13: u32,
    pub nb11: u32,
    pub nb12: u32,
    pub nb13: u32,
    pub scale: f32,
    pub max_bias: f32,
    pub m0: f32,
    pub m1: f32,
    pub n_head_log2: u32,
    pub nrows_x: u32,
    pub has_sinks: u32,
}

impl SoftMaxPush {
    /// Plain row softmax over `ncols`, `nrows` rows, no mask/bias/sinks.
    pub fn rows(ncols: u32, nrows: u32, scale: f32) -> SoftMaxPush {
        SoftMaxPush {
            kx: ncols,
            ky: 0,
            ne00: ncols,
            ne01: nrows,
            ne02: 1,
            ne12: 1,
            ne13: 1,
            nb11: ncols,
            nb12: ncols * nrows,
            nb13: ncols * nrows,
            scale,
            max_bias: 0.0,
            m0: 0.0,
            m1: 0.0,
            n_head_log2: 0,
            nrows_x: nrows,
            has_sinks: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<SoftMaxPush>(),
            )
        }
    }
}

/// glu_head.glsl push constants (swiglu etc.).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GluPush {
    pub n: u32,
    pub ne00: u32,
    pub ne20: u32,
    pub mode: u32,
    pub alpha: f32,
    pub limit: f32,
    pub nb00: u32,
    pub nb01: u32,
    pub nb02: u32,
    pub nb03: u32,
    pub nb10: u32,
    pub nb11: u32,
    pub nb12: u32,
    pub nb13: u32,
    pub nb20: u32,
    pub nb21: u32,
    pub nb22: u32,
    pub nb23: u32,
    pub ne21: u32,
    pub ne22: u32,
    pub misalign_offsets: u32,
    pub ne2_012mp: u32,
    pub ne2_012l: u32,
    pub ne2_01mp: u32,
    pub ne2_01l: u32,
    pub ne2_0mp: u32,
    pub ne2_0l: u32,
}

/// ggml-vulkan's init_fastdiv_values: magic multiplier for division by d.
pub fn fastdiv_magic(d: u32) -> (u32, u32) {
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < d as u64 {
        l += 1;
    }
    let mp = ((1u64 << 32) * ((1u64 << l) - d as u64) / d as u64 + 1) as u32;
    (mp, l)
}

/// generic_unary_head push constants (cpy, scale, ...). L values for the
/// fastdivs are packed 3-per-u32 as bytes (slot*8, 6 bits).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UnaryPush {
    pub ne: u32,
    /// ne00..ne03, nb00..nb03 (element strides)
    pub src0: [u32; 8],
    /// ne10..ne13, nb10..nb13
    pub dst: [u32; 8],
    /// (aoffset << 16) | doffset, element units, <= 65535 each.
    pub misalign_offsets: u32,
    pub param1: f32,
    pub param2: f32,
    pub param3: f32,
    pub param4: f32,
    pub ne0_012mp: u32,
    pub ne0_01mp: u32,
    pub ne0_0mp: u32,
    pub ne0_ls: u32,
    pub ne1_012mp: u32,
    pub ne1_01mp: u32,
    pub ne1_0mp: u32,
    pub ne1_ls: u32,
}

impl UnaryPush {
    fn pack_fastdiv(ne: [u32; 4]) -> (u32, u32, u32, u32) {
        let (mp012, l012) = fastdiv_magic(ne[0] * ne[1] * ne[2]);
        let (mp01, l01) = fastdiv_magic(ne[0] * ne[1]);
        let (mp0, l0) = fastdiv_magic(ne[0]);
        (mp012, mp01, mp0, l012 | (l01 << 8) | (l0 << 16))
    }

    /// Copy `n` contiguous elements (dst offset applied via descriptor range).
    pub fn contig_copy(n: u32) -> UnaryPush {
        Self::strided_copy(n, 1, 0)
    }

    /// Copy `n` contiguous src elements into dst with element stride
    /// `dst_stride` starting at element `dst_off` (< 65536).
    pub fn strided_copy(n: u32, dst_stride: u32, dst_off: u32) -> UnaryPush {
        assert!(dst_off <= 0xFFFF);
        let src0 = [n, 1, 1, 1, 1, n, n, n];
        let dst = [
            n,
            1,
            1,
            1,
            dst_stride,
            n * dst_stride,
            n * dst_stride,
            n * dst_stride,
        ];
        let (a012, a01, a0, als) = Self::pack_fastdiv([n, 1, 1, 1]);
        let (d012, d01, d0, dls) = Self::pack_fastdiv([n, 1, 1, 1]);
        UnaryPush {
            ne: n,
            src0,
            dst,
            misalign_offsets: dst_off,
            param1: 0.0,
            param2: 0.0,
            param3: 0.0,
            param4: 0.0,
            ne0_012mp: a012,
            ne0_01mp: a01,
            ne0_0mp: a0,
            ne0_ls: als,
            ne1_012mp: d012,
            ne1_01mp: d01,
            ne1_0mp: d0,
            ne1_ls: dls,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<UnaryPush>(),
            )
        }
    }
}

impl GluPush {
    /// Split-mode GLU: gate rows in A, up rows in B, both (n0, n1) contiguous.
    pub fn split(n0: u32, n1: u32) -> GluPush {
        let (mp012, l012) = fastdiv_magic(n0 * n1);
        let (mp01, l01) = fastdiv_magic(n0 * n1);
        let (mp0, l0) = fastdiv_magic(n0);
        GluPush {
            n: n0 * n1,
            ne00: n0,
            ne20: n0,
            mode: 2, // split: op(a, b)
            alpha: 0.0,
            limit: 0.0,
            nb00: 1,
            nb01: n0,
            nb02: n0 * n1,
            nb03: n0 * n1,
            nb10: 1,
            nb11: n0,
            nb12: n0 * n1,
            nb13: n0 * n1,
            nb20: 1,
            nb21: n0,
            nb22: n0 * n1,
            nb23: n0 * n1,
            ne21: n1,
            ne22: 1,
            misalign_offsets: 0,
            ne2_012mp: mp012,
            ne2_012l: l012,
            ne2_01mp: mp01,
            ne2_01l: l01,
            ne2_0mp: mp0,
            ne2_0l: l0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<GluPush>(),
            )
        }
    }
}

/// rope_params push constants (rope_head.glsl). Field order matters.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RopePush {
    pub rope_mode: u32,
    pub nrows: u32,
    pub n_dims: u32,
    pub freq_scale: f32,
    pub freq_base: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub corr_dims: [f32; 2],
    pub theta_scale: f32,
    pub has_ff: u32,
    pub sections: [i32; 4],
    pub is_imrope: u32,
    pub is_back: u32,
    pub set_rows_stride: u32,
    pub ne00: u32,
    pub ne01: u32,
    pub ne02: u32,
    pub nb01: u32,
    pub nb02: u32,
    pub nb03: u32,
    pub nb11: u32,
    pub nb12: u32,
    pub nb13: u32,
    pub a_offset: u32,
    pub d_offset: u32,
}

impl RopePush {
    /// NEOX rope over contiguous (head_dim, n_heads, n_tokens) with full
    /// rotation (n_dims == head_dim), no freq factors, no yarn.
    pub fn neox(head_dim: u32, n_heads: u32, n_tokens: u32, freq_base: f32) -> RopePush {
        RopePush {
            rope_mode: 2, // GGML_ROPE_TYPE_NEOX
            nrows: n_heads * n_tokens,
            n_dims: head_dim,
            freq_scale: 1.0,
            freq_base,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_dims: [0.0, 0.0],
            theta_scale: (freq_base as f64).powf(-2.0 / head_dim as f64) as f32,
            has_ff: 0,
            sections: [0; 4],
            is_imrope: 0,
            is_back: 0,
            set_rows_stride: 0,
            ne00: head_dim,
            ne01: n_heads,
            ne02: n_tokens,
            nb01: head_dim,
            nb02: head_dim * n_heads,
            nb03: head_dim * n_heads * n_tokens,
            nb11: head_dim,
            nb12: head_dim * n_heads,
            nb13: head_dim * n_heads * n_tokens,
            a_offset: 0,
            d_offset: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<RopePush>(),
            )
        }
    }
}
