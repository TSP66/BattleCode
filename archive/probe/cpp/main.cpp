// Prices a multiply-accumulate in C++ the way train/budget.py prices one in
// NumPy: do a known number of them per turn, read the judge's points, divide.
// MODE picks the shape so the rate can be checked for linearity.
#include "helper.hpp"

#include <cstddef>
#include <vector>

#ifndef MODE
#define MODE 0
#endif

namespace {

// M x K times K x N, row-major, i-k-j so the inner loop is contiguous in both
// B and C and -O2 can vectorise it.
void gemm(float const* A, float const* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        float* c = C + (std::size_t)i * N;
        for (int j = 0; j < N; ++j) c[j] = 0.0f;
        for (int k = 0; k < K; ++k) {
            float const a = A[(std::size_t)i * K + k];
            float const* b = B + (std::size_t)k * N;
            for (int j = 0; j < N; ++j) c[j] += a * b[j];
        }
    }
}

#if MODE == 0
constexpr int M = 1, K = 1, N = 1;          // baseline: parse and output only
#elif MODE == 1
constexpr int M = 256, K = 1024, N = 49;    // 12,845,056 MACs
#elif MODE == 2
constexpr int M = 512, K = 1024, N = 1;     //    524,288 MACs
#elif MODE == 3
constexpr int M = 32, K = 288, N = 49;      //    451,584 MACs
#elif MODE == 4
constexpr int M = 96, K = 864, N = 49;      //  4,064,256 MACs, one 96ch conv
#elif MODE == 5
constexpr int M = 512, K = 1024, N = 49;    // 25,690,112 MACs
#endif

std::vector<float> A, B, C;

}  // namespace

int main() {
    auto [ct, game] = unswbc::init();
    A.assign((std::size_t)M * K, 0.0f);
    B.assign((std::size_t)K * N, 0.0f);
    C.assign((std::size_t)M * N, 0.0f);

    while (unswbc::update(ct, game)) {
        // make the input depend on the round so nothing can be hoisted out
        A[0] = (float)game.get_round_num();
        B[0] = 1.0f;
#if MODE != 0
        gemm(A.data(), B.data(), C.data(), M, K, N);
#endif
        auto dir = unswbc::Direction::NORTH;
        if (C[0] > 1e30f) dir = unswbc::Direction::SOUTH;   // never taken
        ct.make_move(dir);
        unswbc::end_turn();
    }
}
