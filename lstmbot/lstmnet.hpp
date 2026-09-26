// The LSTM policy's kernels (bcsim/train/lstm_net.py), priced on the judge's own
// cost table. Numerics are bcsim/train/export_lstm.py's: int16 weights at 12
// bits per output row, int16 activations, int32 accumulators; from the
// accumulator on (bias, GroupNorm, SiLU, LayerNorm, the LSTM cell) in float.
//
// Two fixed-point scales. The grid is Q12 (every channel is in [0, 1], and Q12
// is what matches the simulator's float grid to the last step). Activations
// between layers are Q10, range +-32: trained activations reach ~18 in the
// residual blocks and ~9 after the LayerNorm, and Q12's +-8 clipped ~1 value
// a turn, which cost logits up to 3.4 (p99 1.5) against PyTorch; Q10 brings
// that to p99 0.08, the size of the bf16 noise the evaluations played under.
//
// One matrix kernel serves every conv and linear layer:
//     C[m][cell] = sum_k A[m][k] * B[k][cell],   B pair-interleaved [K/2][P][2]
// i32x4.dot_i16x8_s does 8 multiply-adds per SIMD op: 1.65 points a MAC against
// 3.29 for float (wasmprobe, 2026-09-24). A linear layer is the same kernel
// turned round (A = the input vector, M = 1; B = the weights).
//
// Built for wasm (the judge) the kernels use wasm_simd128; built natively (the
// parity checks, g++) the same integer arithmetic runs scalar, so the two give
// bit-identical integers.
#pragma once

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define LN_SIMD 1
#else
#define LN_SIMD 0
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "silu_table.hpp"

namespace ln {

using i16 = std::int16_t;
using i32 = std::int32_t;
constexpr float QG = 4096.0f;                // the grid, Q12
constexpr float QA = 1024.0f;                // activations between layers, Q10

#if LN_SIMD
template <int TV>
inline void gemm_t(i16 const* A, int M, int K, i16 const* B, int P, i32* C) {
    int const pairs = K / 2, stride = P / 4;
    for (int i = 0; i < M; i++) {
        i16 const* a = A + (std::size_t)i * K;
        for (int v0 = 0; v0 < stride; v0 += TV) {
            v128_t acc[TV];
            for (int j = 0; j < TV; j++) acc[j] = wasm_i32x4_splat(0);
            v128_t const* b = reinterpret_cast<v128_t const*>(B) + v0;
            for (int p = 0; p < pairs; p++, b += stride) {
                v128_t const s = wasm_v128_load32_splat(a + 2 * p);
                for (int j = 0; j < TV; j++)
                    acc[j] = wasm_i32x4_add(acc[j], wasm_i32x4_dot_i16x8(s, wasm_v128_load(b + j)));
            }
            for (int j = 0; j < TV; j++) wasm_v128_store(C + (std::size_t)i * P + (v0 + j) * 4, acc[j]);
        }
    }
}

// P must be a multiple of 4; the tile is the largest that divides it.
inline void gemm(i16 const* A, int M, int K, i16 const* B, int P, i32* C) {
    int const v = P / 4;
    if (v % 13 == 0) gemm_t<13>(A, M, K, B, P, C);
    else if (v % 8 == 0) gemm_t<8>(A, M, K, B, P, C);
    else if (v % 7 == 0) gemm_t<7>(A, M, K, B, P, C);
    else if (v % 4 == 0) gemm_t<4>(A, M, K, B, P, C);
    else if (v % 3 == 0) gemm_t<3>(A, M, K, B, P, C);
    else gemm_t<1>(A, M, K, B, P, C);
}
#else
// Native (parity) builds accumulate in int64 and record the largest |sum| any
// accumulator reached, partial sums included: the wasm kernel wraps silently
// past 2^31, so parity_lstm.py fails the check if this ever gets close.
inline std::int64_t acc_peak = 0;
inline void gemm(i16 const* A, int M, int K, i16 const* B, int P, i32* C) {
    for (int i = 0; i < M; i++) {
        i16 const* a = A + (std::size_t)i * K;
        i32* c = C + (std::size_t)i * P;
        for (int j = 0; j < P; j++) {
            std::int64_t s = 0;
            for (int p = 0; p < K / 2; p++) {
                i16 const* b = B + ((std::size_t)p * P + j) * 2;
                s += (std::int64_t)a[2 * p] * b[0] + (std::int64_t)a[2 * p + 1] * b[1];
                std::int64_t const m = s < 0 ? -s : s;
                if (m > acc_peak) acc_peak = m;
            }
            c[j] = (i32)s;
        }
    }
}
#endif

// Input [C][Pin] (h x w planes) -> B [C*9/2][Pout][2] for a 3x3 conv, pad 1.
inline void im2col(i16 const* in, int C, int h, int w, int Pin, int stride,
                   int ho, int wo, int Pout, i16* pad, i16* B) {
    int const pw = w + 2, ncell = ho * wo;
    std::memset(pad, 0, sizeof(i16) * (std::size_t)(h + 2) * pw);
    for (int c = 0; c < C; c++) {
        i16 const* src = in + (std::size_t)c * Pin;
        for (int r = 0; r < h; r++)
            std::memcpy(pad + (r + 1) * pw + 1, src + r * w, sizeof(i16) * (std::size_t)w);
        for (int t = 0; t < 9; t++) {
            int const k = c * 9 + t, ky = t / 3, kx = t % 3;
            i16* dst = B + (std::size_t)(k >> 1) * Pout * 2 + (k & 1);
            for (int oy = 0; oy < ho; oy++) {
                i16 const* row = pad + (oy * stride + ky) * pw + kx;
                i16* d = dst + (std::size_t)oy * wo * 2;
                for (int ox = 0; ox < wo; ox++) d[ox * 2] = row[ox * stride];
            }
            for (int j = ncell; j < Pout; j++) dst[j * 2] = 0;
        }
    }
}

#if LN_SIMD
// Stride-1 im2col, pairwise: tap k and tap k+1 of the same output row are two
// contiguous runs of the padded planes, so 8 cells of each are loaded and one
// shuffle interleaves them into B's [k/2][cell][2] layout. Rows are written in
// chunks of 8 and may run up to 7 cells past a row's end; the next row, pair or
// layer overwrites those, and cells past ncell are never read as outputs.
// `pads` holds every channel's (h+2)*(w+2) plane, with 16 cells of slack.
inline void im2col_s1(i16 const* in, int C, int h, int w, int Pin, int Pout, i16* pads, i16* B) {
    int const pw = w + 2, plane = (h + 2) * pw;
    std::memset(pads, 0, sizeof(i16) * ((std::size_t)C * plane + 16));
    for (int c = 0; c < C; c++) {
        i16 const* src = in + (std::size_t)c * Pin;
        i16* p = pads + (std::size_t)c * plane;
        for (int r = 0; r < h; r++)
            std::memcpy(p + (r + 1) * pw + 1, src + r * w, sizeof(i16) * (std::size_t)w);
    }
    int const K = C * 9;
    for (int k = 0; k < K; k += 2) {
        int const c0 = k / 9, t0 = k % 9, c1 = (k + 1) / 9, t1 = (k + 1) % 9;
        i16 const* p0 = pads + (std::size_t)c0 * plane + (t0 / 3) * pw + (t0 % 3);
        i16 const* p1 = pads + (std::size_t)c1 * plane + (t1 / 3) * pw + (t1 % 3);
        i16* dst = B + (std::size_t)(k >> 1) * Pout * 2;
        for (int oy = 0; oy < h; oy++) {
            i16 const* r0 = p0 + oy * pw;
            i16 const* r1 = p1 + oy * pw;
            i16* d = dst + (std::size_t)oy * w * 2;
            for (int ox = 0; ox < w; ox += 8) {
                v128_t const a = wasm_v128_load(r0 + ox), b = wasm_v128_load(r1 + ox);
                wasm_v128_store(d + 2 * ox, wasm_i16x8_shuffle(a, b, 0, 8, 1, 9, 2, 10, 3, 11));
                wasm_v128_store(d + 2 * ox + 8, wasm_i16x8_shuffle(a, b, 4, 12, 5, 13, 6, 14, 7, 15));
            }
        }
    }
}
#endif

// [C][P] -> [C/2][P][2], the B layout for a 1x1 conv.
inline void interleave(i16 const* in, int C, int P, i16* B) {
    for (int c = 0; c < C; c += 2) {
        i16 const* x = in + (std::size_t)c * P;
        i16 const* y = x + P;
        i16* d = B + (std::size_t)c * P;
        for (int j = 0; j < P; j++) {
            d[2 * j] = x[j];
            d[2 * j + 1] = y[j];
        }
    }
}

inline i16 to_q(float v, float q) {
    float const x = v * q;
    return (i16)(x >= 32767.0f ? 32767 : (x <= -32767.0f ? -32767 : (int)std::lrintf(x)));
}

// z[m][j] = acc[m][j] * s[m] / qin + b[m], the real pre-activation; qin is the
// scale of the layer's input (QG for the stem, QA after it).
inline void dequant(i32 const* acc, float const* s, float const* b, int M, int P, int ncell, float qin,
                    float* z) {
    for (int m = 0; m < M; m++) {
        float const k = s[m] / qin, bb = b[m];
        i32 const* a = acc + (std::size_t)m * P;
        float* o = z + (std::size_t)m * P;
        for (int j = 0; j < ncell; j++) o[j] = (float)a[j] * k + bb;
    }
}

// The linear layers' form (M = 1): the outputs run along the lanes, so each
// output j has its own scale and bias -- z[j] = acc[j] * s[j] / QA + b[j].
inline void dequant_vec(i32 const* acc, float const* s, float const* b, int n, float* z) {
    for (int j = 0; j < n; j++) z[j] = (float)acc[j] * (s[j] / QA) + b[j];
}

// SiLU from a 16384-entry table over [-16, 16), linearly interpolated (within
// ~1e-5 of x * sigmoid(x)), for a few points a value where expf costs ~100. The
// table is precomputed (silu_table.hpp, written by export_lstm.py), so turn 0
// does not pay 16k exp calls for it.
constexpr int SILU_N = 16384;
constexpr float SILU_LO = -16.0f, SILU_INV_STEP = SILU_N / 32.0f;
inline void build_silu() {}
inline float silu(float x) {
    if (x <= SILU_LO) return 0.0f;
    if (x >= -SILU_LO) return x;
    float const f = (x - SILU_LO) * SILU_INV_STEP;
    int const k = (int)f;
    float const t = f - (float)k;
    return SILU_T[k] + t * (SILU_T[k + 1] - SILU_T[k]);
}

// GroupNorm (8 groups, eps 1e-5, stats over the real cells) + optional skip + SiLU,
// from z, into Q10 `out`. `skip` is Q10.
inline void gn_silu(float const* z, int C, int P, int ncell, float const* gamma, float const* beta,
                    i16 const* skip, i16* out, int groups = 8) {
    int const per = C / groups;
    for (int g = 0; g < groups; g++) {
        double sum = 0.0, sq = 0.0;
        for (int c = g * per; c < (g + 1) * per; c++) {
            float const* row = z + (std::size_t)c * P;
            for (int j = 0; j < ncell; j++) { sum += row[j]; sq += (double)row[j] * row[j]; }
        }
        double const n = (double)per * ncell;
        double const mean = sum / n;
        double const var = sq / n - mean * mean;
        float const inv = (float)(1.0 / std::sqrt((var > 0 ? var : 0) + 1e-5));
        for (int c = g * per; c < (g + 1) * per; c++) {
            float const a = gamma[c] * inv, b = beta[c] - (float)mean * a;
            float const* row = z + (std::size_t)c * P;
            i16 const* sk = skip ? skip + (std::size_t)c * P : nullptr;
            i16* o = out + (std::size_t)c * P;
            for (int j = 0; j < ncell; j++) {
                float v = row[j] * a + b;
                if (sk) v += (float)sk[j] / QA;
                o[j] = to_q(silu(v), QA);
            }
            for (int j = ncell; j < P; j++) o[j] = 0;
        }
    }
}

inline void layer_norm(float* x, int n, float const* gamma, float const* beta) {
    double sum = 0.0, sq = 0.0;
    for (int i = 0; i < n; i++) { sum += x[i]; sq += (double)x[i] * x[i]; }
    double const mean = sum / n, var = sq / n - mean * mean;
    float const inv = (float)(1.0 / std::sqrt((var > 0 ? var : 0) + 1e-5));
    for (int i = 0; i < n; i++) x[i] = (float)(x[i] - mean) * inv * gamma[i] + beta[i];
}

inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// One LSTMCell from its real gate pre-activations z (i, f, g, o), cell state in
// float; h comes back in float and as the Q10 row the next layer reads.
inline void lstm_cell(float const* z, int H, float* c, float* h, i16* hq) {
    for (int u = 0; u < H; u++) {
        float const ig = sigm(z[u]), fg = sigm(z[H + u]), gg = std::tanh(z[2 * H + u]), og = sigm(z[3 * H + u]);
        c[u] = fg * c[u] + ig * gg;
        h[u] = og * std::tanh(c[u]);
        hq[u] = to_q(h[u], QA);
    }
}

// The two byte planes -> int16 values, little-endian.
inline void join_planes(char const* hi, char const* lo, std::uint32_t n, i16* out) {
    std::uint32_t i = 0;
#if LN_SIMD
    for (; i + 16 <= n; i += 16) {
        v128_t const h = wasm_v128_load(hi + i), l = wasm_v128_load(lo + i);
        wasm_v128_store(out + i, wasm_i8x16_shuffle(l, h, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23));
        wasm_v128_store(out + i + 8, wasm_i8x16_shuffle(l, h, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31));
    }
#endif
    for (; i < n; i++)
        out[i] = (i16)(((std::uint16_t)(unsigned char)hi[i] << 8) | (unsigned char)lo[i]);
}

}  // namespace ln
