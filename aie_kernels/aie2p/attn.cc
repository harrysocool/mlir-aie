//===- attn.cc - fused single-head attention; vectorized dot/AXPY GEMM -----===//
// O = softmax((Q @ K^T) * scale) @ V, one head, on-chip. Input = [Q|K|V] concat.
// GEMMs vectorized with aie::mac (row-major, no mmul pre-tiling). bf16.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef ATTN_S
#define ATTN_S 64
#endif
#ifndef ATTN_D
#define ATTN_D 64
#endif
#ifndef ATTN_SCALE
#define ATTN_SCALE 0.125f
#endif
#define VL 16
#define LOG2E 1.44269504089f
using namespace aie;

extern "C" {
void attention(bfloat16 *restrict qkv, bfloat16 *restrict O) {
  event0();
  constexpr int S = ATTN_S;
  constexpr int d = ATTN_D;
  constexpr int DV = d / VL;   // d-vectors per row
  const float scale = ATTN_SCALE;
  bfloat16 *restrict Q = qkv;
  bfloat16 *restrict K = qkv + S * d;
  bfloat16 *restrict V = qkv + 2 * S * d;

  alignas(32) bfloat16 srow[S];
  vector<bfloat16, VL> l2e = broadcast<bfloat16, VL>((bfloat16)LOG2E);

  for (int i = 0; i < S; i++) {
    const bfloat16 *Qi = Q + i * d;
    // ---- QK^T row: scores[j] = scale * dot(Q[i], K[j]) ----
    float rowmax = -3.0e38f;
    for (int j = 0; j < S; j++) {
      const bfloat16 *Kj = K + j * d;
      accum<accfloat, VL> acc = zeros<accfloat, VL>();
      for (int k = 0; k < d; k += VL)
        acc = aie::mac(acc, load_v<VL>(Qi + k), load_v<VL>(Kj + k));
      float s = aie::reduce_add(acc.to_vector<float>()) * scale;
      srow[j] = (bfloat16)s;
      if (s > rowmax) rowmax = s;
    }
    // ---- softmax: exp2((x-max)*log2e), accum-vector sum ----
    vector<bfloat16, VL> rmax_l2e =
        broadcast<bfloat16, VL>((bfloat16)(rowmax * LOG2E));
    accum<accfloat, VL> exp_sum = zeros<accfloat, VL>();
    for (int j = 0; j < S; j += VL) {
      vector<bfloat16, VL> x = load_v<VL>(srow + j);
      accum<accfloat, VL> scaled = aie::mul(x, l2e);
      accum<accfloat, VL> ein = aie::sub(scaled, rmax_l2e);
      vector<bfloat16, VL> ev = aie::exp2<bfloat16>(ein.to_vector<float>());
      store_v(srow + j, ev);
      exp_sum = add(exp_sum, ev);
    }
    float inv = 1.0f / aie::reduce_add(exp_sum.to_vector<float>());
    vector<bfloat16, VL> invv = broadcast<bfloat16, VL>((bfloat16)inv);
    // ---- PV: O[i,:] = inv * sum_j P[j] * V[j,:]  (AXPY over d) ----
    accum<accfloat, VL> oacc[DV];
    for (int n = 0; n < DV; n++) oacc[n] = zeros<accfloat, VL>();
    for (int j = 0; j < S; j++) {
      vector<bfloat16, VL> pj = broadcast<bfloat16, VL>(srow[j]);
      const bfloat16 *Vj = V + j * d;
      for (int n = 0; n < DV; n++)
        oacc[n] = aie::mac(oacc[n], load_v<VL>(Vj + n * VL), pj);
    }
    bfloat16 *Oi = O + i * d;
    for (int n = 0; n < DV; n++) {
      vector<bfloat16, VL> ov = oacc[n].to_vector<bfloat16>();
      accum<accfloat, VL> norm = aie::mul(ov, invv);
      store_v(Oi + n * VL, norm.to_vector<bfloat16>());
    }
  }
  event1();
}
}
