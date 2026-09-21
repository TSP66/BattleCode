// The bot's hot loops, compiled to wasm so they can be priced against the
// judge's own cost table before anything is submitted. Written to mirror
// net.hpp instruction for instruction: same loop shapes, same constants, and
// unchecked indexing so there are no bounds checks C++ would not emit.
#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! { loop {} }

const PITCH: usize = 52;
const CELLS: usize = 49;
const PAD_W: usize = 9;
const W: usize = 96;
const K: usize = W * 9;

static mut A: [f32; W * K] = [0.0; W * K];
static mut B: [f32; K * PITCH] = [0.0; K * PITCH];
static mut C: [f32; W * PITCH] = [0.0; W * PITCH];
static mut PAD: [f32; W * PAD_W * PAD_W] = [0.0; W * PAD_W * PAD_W];
static mut GAMMA: [f32; W] = [0.0; W];
static mut BETA: [f32; W] = [0.0; W];

#[no_mangle]
pub extern "C" fn gemm(m: usize, k: usize) {
    unsafe {
        let a = &*core::ptr::addr_of!(A);
        let b = &*core::ptr::addr_of!(B);
        let c = &mut *core::ptr::addr_of_mut!(C);
        for i in 0..m {
            let mut acc = [0.0f32; PITCH];
            for kk in 0..k {
                let s = *a.get_unchecked(i * k + kk);
                let row = b.get_unchecked(kk * PITCH..kk * PITCH + PITCH);
                for j in 0..PITCH {
                    *acc.get_unchecked_mut(j) += s * *row.get_unchecked(j);
                }
            }
            for j in 0..PITCH {
                *c.get_unchecked_mut(i * PITCH + j) = *acc.get_unchecked(j);
            }
        }
    }
}

#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    if x > 88.0 { return 3.4e38; }
    if x < -88.0 { return 0.0; }
    let x = x * 1.442_695_f32;
    let xf = floorf(x);
    let f = x - xf;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * 0.009_618_12)));
    let bits = (((xf as i32) + 127) << 23) as u32;
    p * f32::from_bits(bits)
}

#[inline(always)]
fn floorf(x: f32) -> f32 {
    let t = x as i32 as f32;
    if t > x { t - 1.0 } else { t }
}

#[no_mangle]
pub extern "C" fn silu(n: usize) {
    unsafe {
        let c = &mut *core::ptr::addr_of_mut!(C);
        for i in 0..n {
            let v = *c.get_unchecked(i);
            *c.get_unchecked_mut(i) = v / (1.0 + fast_exp(-v));
        }
    }
}

#[no_mangle]
pub extern "C" fn group_norm(ch: usize, groups: usize) {
    unsafe {
        let x = &mut *core::ptr::addr_of_mut!(C);
        let gamma = &*core::ptr::addr_of!(GAMMA);
        let beta = &*core::ptr::addr_of!(BETA);
        let per = ch / groups;
        for g in 0..groups {
            let mut sum = 0.0f32;
            let mut sq = 0.0f32;
            for c in g * per..(g + 1) * per {
                for j in 0..CELLS {
                    let v = *x.get_unchecked(c * PITCH + j);
                    sum += v;
                    sq += v * v;
                }
            }
            let n = (per * CELLS) as f32;
            let mean = sum / n;
            let inv = 1.0 / sqrtf(sq / n - mean * mean + 1e-5);
            for c in g * per..(g + 1) * per {
                let a = *gamma.get_unchecked(c) * inv;
                let b = *beta.get_unchecked(c) - mean * a;
                for j in 0..PITCH {
                    let v = *x.get_unchecked(c * PITCH + j);
                    *x.get_unchecked_mut(c * PITCH + j) = v * a + b;
                }
            }
        }
    }
}

// core has no scalar sqrt in no_std; the wasm f32x4 one is a single
// instruction and is called eight times a layer, so the lane shuffle is free
#[inline(always)]
fn sqrtf(x: f32) -> f32 {
    use core::arch::wasm32::{f32x4_splat, f32x4_sqrt, f32x4_extract_lane};
    f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
}

#[no_mangle]
pub extern "C" fn im2col(ch: usize) {
    unsafe {
        let inp = &*core::ptr::addr_of!(C);
        let pad = &mut *core::ptr::addr_of_mut!(PAD);
        let col = &mut *core::ptr::addr_of_mut!(B);
        for c in 0..ch {
            let base = c * PAD_W * PAD_W;
            for i in 0..PAD_W * PAD_W { *pad.get_unchecked_mut(base + i) = 0.0; }
            for r in 0..7 {
                for j in 0..7 {
                    *pad.get_unchecked_mut(base + (r + 1) * PAD_W + 1 + j) =
                        *inp.get_unchecked(c * PITCH + r * 7 + j);
                }
            }
        }
        for c in 0..ch {
            let base = c * PAD_W * PAD_W;
            for k in 0..9 {
                let kh = k / 3;
                let kw = k % 3;
                let dst = (c * 9 + k) * PITCH;
                for r in 0..7 {
                    for j in 0..7 {
                        *col.get_unchecked_mut(dst + r * 7 + j) =
                            *pad.get_unchecked(base + (r + kh) * PAD_W + kw + j);
                    }
                }
                *col.get_unchecked_mut(dst + 49) = 0.0;
                *col.get_unchecked_mut(dst + 50) = 0.0;
                *col.get_unchecked_mut(dst + 51) = 0.0;
            }
        }
    }
}

// bf16 -> f32 widening, the bot's first-turn cost
#[no_mangle]
pub extern "C" fn widen(n: usize) {
    unsafe {
        let src = core::ptr::addr_of!(B) as *const u16;
        let dst = core::ptr::addr_of_mut!(A) as *mut f32;
        for i in 0..n {
            *dst.add(i) = f32::from_bits((*src.add(i) as u32) << 16);
        }
    }
}

// Same gemm, but with the 13 accumulator vectors held as locals across the k
// loop instead of being reloaded and restored from memory each iteration.
// That removes a v128.load and a v128.store -- 4 points -- from every four
// multiply-accumulates.
#[no_mangle]
pub extern "C" fn gemm_simd(m: usize, k: usize) {
    use core::arch::wasm32::*;
    unsafe {
        let a = core::ptr::addr_of!(A) as *const f32;
        let b = core::ptr::addr_of!(B) as *const f32;
        let c = core::ptr::addr_of_mut!(C) as *mut f32;
        for i in 0..m {
            let mut acc = [f32x4_splat(0.0); PITCH / 4];
            for kk in 0..k {
                let s = f32x4_splat(*a.add(i * k + kk));
                let row = b.add(kk * PITCH);
                for j in 0..PITCH / 4 {
                    let v = v128_load(row.add(j * 4) as *const v128);
                    acc[j] = f32x4_add(acc[j], f32x4_mul(s, v));
                }
            }
            for j in 0..PITCH / 4 {
                v128_store(c.add(i * PITCH + j * 4) as *mut v128, acc[j]);
            }
        }
    }
}

// Four output rows at a time, so each loaded B vector feeds four
// accumulators rather than one.
#[no_mangle]
pub extern "C" fn gemm_simd4(m: usize, k: usize) {
    use core::arch::wasm32::*;
    unsafe {
        let a = core::ptr::addr_of!(A) as *const f32;
        let b = core::ptr::addr_of!(B) as *const f32;
        let c = core::ptr::addr_of_mut!(C) as *mut f32;
        let mut i = 0;
        while i + 4 <= m {
            let mut acc = [[f32x4_splat(0.0); PITCH / 4]; 4];
            for kk in 0..k {
                let row = b.add(kk * PITCH);
                for r in 0..4 {
                    let s = f32x4_splat(*a.add((i + r) * k + kk));
                    for j in 0..PITCH / 4 {
                        let v = v128_load(row.add(j * 4) as *const v128);
                        acc[r][j] = f32x4_add(acc[r][j], f32x4_mul(s, v));
                    }
                }
            }
            for r in 0..4 {
                for j in 0..PITCH / 4 {
                    v128_store(c.add((i + r) * PITCH + j * 4) as *mut v128, acc[r][j]);
                }
            }
            i += 4;
        }
    }
}

// Four rows again, but each B vector is loaded once and fed to all four
// accumulators, rather than reloaded per row.
#[no_mangle]
pub extern "C" fn gemm_t4(m: usize, k: usize) {
    use core::arch::wasm32::*;
    unsafe {
        let a = core::ptr::addr_of!(A) as *const f32;
        let b = core::ptr::addr_of!(B) as *const f32;
        let c = core::ptr::addr_of_mut!(C) as *mut f32;
        let mut i = 0;
        while i + 4 <= m {
            let mut acc = [[f32x4_splat(0.0); PITCH / 4]; 4];
            for kk in 0..k {
                let row = b.add(kk * PITCH);
                let s0 = f32x4_splat(*a.add(i * k + kk));
                let s1 = f32x4_splat(*a.add((i + 1) * k + kk));
                let s2 = f32x4_splat(*a.add((i + 2) * k + kk));
                let s3 = f32x4_splat(*a.add((i + 3) * k + kk));
                for j in 0..PITCH / 4 {
                    let v = v128_load(row.add(j * 4) as *const v128);
                    acc[0][j] = f32x4_add(acc[0][j], f32x4_mul(s0, v));
                    acc[1][j] = f32x4_add(acc[1][j], f32x4_mul(s1, v));
                    acc[2][j] = f32x4_add(acc[2][j], f32x4_mul(s2, v));
                    acc[3][j] = f32x4_add(acc[3][j], f32x4_mul(s3, v));
                }
            }
            for r in 0..4 {
                for j in 0..PITCH / 4 {
                    v128_store(c.add((i + r) * PITCH + j * 4) as *mut v128, acc[r][j]);
                }
            }
            i += 4;
        }
    }
}

// Eight rows, with the lane axis blocked so the live accumulator count stays
// small enough to sit in locals instead of spilling.
#[no_mangle]
pub extern "C" fn gemm_t8(m: usize, k: usize) {
    use core::arch::wasm32::*;
    unsafe {
        let a = core::ptr::addr_of!(A) as *const f32;
        let b = core::ptr::addr_of!(B) as *const f32;
        let c = core::ptr::addr_of_mut!(C) as *mut f32;
        let mut i = 0;
        while i + 8 <= m {
            let mut jb = 0;
            while jb < PITCH / 4 {
                let mut acc = [f32x4_splat(0.0); 8];
                for kk in 0..k {
                    let v = v128_load(b.add(kk * PITCH + jb * 4) as *const v128);
                    for r in 0..8 {
                        let s = f32x4_splat(*a.add((i + r) * k + kk));
                        acc[r] = f32x4_add(acc[r], f32x4_mul(s, v));
                    }
                }
                for r in 0..8 {
                    v128_store(c.add((i + r) * PITCH + jb * 4) as *mut v128, acc[r]);
                }
                jb += 1;
            }
            i += 8;
        }
    }
}

// Four rows by four lane-vectors: sixteen live accumulators, which is few
// enough to stay in locals, while each loaded B vector still serves four
// rows and each splat four vectors.
#[no_mangle]
pub extern "C" fn gemm_t44(m: usize, k: usize) {
    use core::arch::wasm32::*;
    unsafe {
        let a = core::ptr::addr_of!(A) as *const f32;
        let b = core::ptr::addr_of!(B) as *const f32;
        let c = core::ptr::addr_of_mut!(C) as *mut f32;
        let mut i = 0;
        while i + 4 <= m {
            let mut jb = 0;
            while jb < PITCH / 4 {
                let n = if jb + 4 <= PITCH / 4 { 4 } else { PITCH / 4 - jb };
                let mut acc = [[f32x4_splat(0.0); 4]; 4];
                for kk in 0..k {
                    let row = b.add(kk * PITCH + jb * 4);
                    for r in 0..4 {
                        let s = f32x4_splat(*a.add((i + r) * k + kk));
                        for j in 0..n {
                            let v = v128_load(row.add(j * 4) as *const v128);
                            acc[r][j] = f32x4_add(acc[r][j], f32x4_mul(s, v));
                        }
                    }
                }
                for r in 0..4 {
                    for j in 0..n {
                        v128_store(c.add((i + r) * PITCH + (jb + j) * 4) as *mut v128, acc[r][j]);
                    }
                }
                jb += 4;
            }
            i += 4;
        }
    }
}

// Two rows, all thirteen lane-vectors: twenty-six accumulators.
#[no_mangle]
pub extern "C" fn gemm_t2(m: usize, k: usize) {
    use core::arch::wasm32::*;
    unsafe {
        let a = core::ptr::addr_of!(A) as *const f32;
        let b = core::ptr::addr_of!(B) as *const f32;
        let c = core::ptr::addr_of_mut!(C) as *mut f32;
        let mut i = 0;
        while i + 2 <= m {
            let mut acc = [[f32x4_splat(0.0); PITCH / 4]; 2];
            for kk in 0..k {
                let row = b.add(kk * PITCH);
                let s0 = f32x4_splat(*a.add(i * k + kk));
                let s1 = f32x4_splat(*a.add((i + 1) * k + kk));
                for j in 0..PITCH / 4 {
                    let v = v128_load(row.add(j * 4) as *const v128);
                    acc[0][j] = f32x4_add(acc[0][j], f32x4_mul(s0, v));
                    acc[1][j] = f32x4_add(acc[1][j], f32x4_mul(s1, v));
                }
            }
            for r in 0..2 {
                for j in 0..PITCH / 4 {
                    v128_store(c.add((i + r) * PITCH + j * 4) as *mut v128, acc[r][j]);
                }
            }
            i += 2;
        }
    }
}
