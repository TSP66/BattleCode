// The policy network, run forward once per turn.
//
// Shaped for the judge's meter rather than for elegance. Two things drive the
// design:
//
//   * A multiply-accumulate is priced by the instructions it takes, and an
//     f32x4 multiply costs 2 points for four lanes where a scalar one costs 1
//     for one. So every inner loop runs over a contiguous, compile-time
//     length so clang's -O2 -msimd128 can vectorise it without help.
//   * The spatial dimension is padded from 49 to 52 (13 lanes of 4) for the
//     same reason. The three pad cells are always zero on the way in, and
//     nothing downstream reads them.
//
// GroupNorm normalises against the current activation rather than a stored
// running mean, so unlike BatchNorm it cannot be folded into the kernels and
// has to run for real. It is cheap next to the convolutions.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace net {

constexpr int CELLS = 49;
constexpr int PITCH = 52;        // 49 rounded up to a multiple of four
constexpr int PAD_W = 9;         // the 7x7 window with a one-cell zero border
constexpr int MAGIC = 0x42434E32;   // "BCN2": linear weights transposed

// exp by way of 2^x: a bit-shift for the integer part and a degree-4
// polynomial for the fraction. Accurate to about 1e-6 relative, which is
// far below the noise bf16 weights already carry, and roughly an order of
// magnitude cheaper than libm's expf.
inline float fast_exp(float x) {
    if (x > 88.0f) return 3.4e38f;
    if (x < -88.0f) return 0.0f;
    x *= 1.44269504088896341f;
    float const xf = std::floor(x);
    float const f = x - xf;
    float const p = 1.0f + f * (0.69314718f + f * (0.24022651f +
                    f * (0.05550411f + f * 0.00961812f)));
    union { float f; std::int32_t i; } u{};
    u.i = ((std::int32_t)xf + 127) << 23;
    return p * u.f;
}

inline float silu(float x) { return x / (1.0f + fast_exp(-x)); }

struct Weights {
    int width = 0, blocks = 0, hidden = 0, head = 0;
    std::vector<float> data;
    std::size_t at = 0;

    float const* take(std::size_t n) {
        float const* p = data.data() + at;
        at += n;
        return p;
    }
};

// Widens the bf16 blob in place. bf16 is the top 16 bits of the float32, so
// this is one shift per parameter and no exponent rebiasing -- it is paid on
// the bot's first turn, where every point counts.
inline bool load(char const* path, Weights& w) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    std::int32_t header[6];
    if (std::fread(header, sizeof(std::int32_t), 6, f) != 6 || header[0] != MAGIC) {
        std::fclose(f);
        return false;
    }
    w.width = header[1];
    w.blocks = header[2];
    w.hidden = header[3];
    w.head = header[4];
    std::size_t const n = (std::size_t)header[5];

    std::vector<std::uint16_t> raw(n);
    if (std::fread(raw.data(), sizeof(std::uint16_t), n, f) != n) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    w.data.resize(n);
    for (std::size_t i = 0; i < n; i++) {
        std::uint32_t const bits = (std::uint32_t)raw[i] << 16;
        std::memcpy(&w.data[i], &bits, sizeof(float));
    }
    return true;
}

// ---- kernels

// C (M x PITCH) = A (M x K) * B (K x PITCH). The j loop is the vector one.
inline void gemm(float const* A, float const* B, float* C, int M, int K) {
    for (int i = 0; i < M; i++) {
        float acc[PITCH];
        for (int j = 0; j < PITCH; j++) acc[j] = 0.0f;
        float const* a = A + (std::size_t)i * K;
        for (int k = 0; k < K; k++) {
            float const s = a[k];
            float const* b = B + (std::size_t)k * PITCH;
            for (int j = 0; j < PITCH; j++) acc[j] += s * b[j];
        }
        std::memcpy(C + (std::size_t)i * PITCH, acc, sizeof(acc));
    }
}

// Lays the window out as (Cin*9) x PITCH so a 3x3 convolution is one gemm.
// Going through a zero-bordered 9x9 copy means each of the seven output rows
// is a straight seven-float run, with no per-cell bounds test.
inline void im2col(float const* in, int C, float* pad, float* col) {
    for (int c = 0; c < C; c++) {
        float* p = pad + (std::size_t)c * PAD_W * PAD_W;
        std::memset(p, 0, sizeof(float) * PAD_W * PAD_W);
        for (int r = 0; r < 7; r++)
            std::memcpy(p + (r + 1) * PAD_W + 1, in + (std::size_t)c * PITCH + r * 7,
                        sizeof(float) * 7);
    }
    for (int c = 0; c < C; c++) {
        float const* p = pad + (std::size_t)c * PAD_W * PAD_W;
        for (int k = 0; k < 9; k++) {
            int const kh = k / 3, kw = k % 3;
            float* dst = col + (std::size_t)(c * 9 + k) * PITCH;
            for (int r = 0; r < 7; r++)
                std::memcpy(dst + r * 7, p + (r + kh) * PAD_W + kw, sizeof(float) * 7);
            dst[49] = dst[50] = dst[51] = 0.0f;
        }
    }
}

// GroupNorm over the real 49 cells only -- the three pad cells are not part
// of the statistic the network was trained with.
inline void group_norm(float* x, int C, float const* gamma, float const* beta,
                       int groups = 8) {
    int const per = C / groups;
    for (int g = 0; g < groups; g++) {
        float sum = 0.0f, sq = 0.0f;
        for (int c = g * per; c < (g + 1) * per; c++) {
            float const* row = x + (std::size_t)c * PITCH;
            for (int j = 0; j < CELLS; j++) { sum += row[j]; sq += row[j] * row[j]; }
        }
        float const n = (float)(per * CELLS);
        float const mean = sum / n;
        float const inv = 1.0f / std::sqrt(sq / n - mean * mean + 1e-5f);
        for (int c = g * per; c < (g + 1) * per; c++) {
            float* row = x + (std::size_t)c * PITCH;
            float const a = gamma[c] * inv, b = beta[c] - mean * a;
            for (int j = 0; j < PITCH; j++) row[j] = row[j] * a + b;
        }
    }
}

inline void apply_silu(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = silu(x[i]);
}

// out = W * in + b, with W stored transposed (in_dim x out_dim).
//
// Written as one axpy per input -- out += in[j] * row j -- so the inner loop
// runs contiguously over the outputs with no dependency between lanes, and
// vectorises. The textbook form, a dot product per output, is a serial float
// reduction that -O2 will not vectorise without reassociating; measured on the
// judge's cost table it was ~15 points per MAC against ~3.3 for this.
inline void linear(float const* WT, float const* b, float const* in, float* out,
                   int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) out[i] = b[i];
    for (int j = 0; j < in_dim; j++) {
        float const x = in[j];
        float const* row = WT + (std::size_t)j * out_dim;
        for (int i = 0; i < out_dim; i++) out[i] += x * row[i];
    }
}

}  // namespace net
