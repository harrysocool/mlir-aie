//===- attn.cc - fused single-head attention; input=[Q|K|V] concat ---------===//
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
  const float scale = ATTN_SCALE;
  bfloat16 *restrict Q = qkv;
  bfloat16 *restrict K = qkv + S * d;
  bfloat16 *restrict V = qkv + 2 * S * d;

  alignas(32) bfloat16 srow[S];
  vector<bfloat16, VL> l2e = broadcast<bfloat16, VL>((bfloat16)LOG2E);

  for (int i = 0; i < S; i++) {
    float rowmax = -3.0e38f;
    for (int j = 0; j < S; j++) {
      float acc = 0.0f;
      for (int k = 0; k < d; k++)
        acc += (float)Q[i * d + k] * (float)K[j * d + k];
      acc *= scale;
      srow[j] = (bfloat16)acc;
      if (acc > rowmax) rowmax = acc;
    }
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
    float sum = aie::reduce_add(exp_sum.to_vector<float>());
    float inv = 1.0f / sum;
    for (int n = 0; n < d; n++) {
      float acc = 0.0f;
      for (int j = 0; j < S; j++)
        acc += (float)srow[j] * (float)V[j * d + n];
      O[i * d + n] = (bfloat16)(acc * inv);
    }
  }
  event1();
}
}
