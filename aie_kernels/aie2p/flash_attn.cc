// Full flash attention: all S rows, all temp arrays top-level aligned
#include <aie_api/aie.hpp>
#include <stdint.h>
#define S 64
#define d 64
#define VL 16
#define LOG2E 1.44269504089f
#define SCALE 0.125f
using namespace aie;
extern "C" {
void flash_attention(bfloat16 *restrict QKV, bfloat16 *restrict O) {
  event0();
  bfloat16 *restrict Q = QKV;
  bfloat16 *restrict K = QKV + S*d;
  bfloat16 *restrict V = QKV + 2*S*d;
  vector<bfloat16,VL> l2e_v = broadcast<bfloat16,VL>((bfloat16)LOG2E);
  // All temp buffers at top-level scope with alignas(32)
  alignas(32) bfloat16 s_bf[S];
  alignas(32) bfloat16 p[S];
  alignas(32) float    accm[d];

  for (int r = 0; r < S; r++) {
    const bfloat16 *qi = Q + r*d;
    float rowmax = -3.0e38f;
    for (int c = 0; c < S; c++) {
      const bfloat16 *kc = K + c*d;
      float acc = 0.0f;
      for (int e = 0; e < d; e++) acc += (float)qi[e]*(float)kc[e];
      float s = acc*SCALE;
      s_bf[c] = (bfloat16)s;
      if (s > rowmax) rowmax = s;
    }
    vector<bfloat16,VL> maxv_l2 = broadcast<bfloat16,VL>((bfloat16)(rowmax*LOG2E));
    accum<accfloat,VL> psum = zeros<accfloat,VL>();
    for (int c = 0; c < S; c += VL) {
      vector<bfloat16,VL> sv = load_v<VL>(s_bf+c);
      accum<accfloat,VL> sc2 = aie::mul(sv, l2e_v);
      accum<accfloat,VL> sh  = aie::sub(sc2, maxv_l2);
      vector<bfloat16,VL> ev = aie::exp2<bfloat16>(sh.to_vector<float>());
      store_v(p+c, ev);
      psum = add(psum, ev);
    }
    float inv = 1.0f / aie::reduce_add(psum.to_vector<float>());
    for (int e = 0; e < d; e++) accm[e] = 0.0f;
    for (int c = 0; c < S; c++) {
      float pv = (float)p[c]*inv;
      const bfloat16 *vc = V+c*d;
      for (int e = 0; e < d; e++) accm[e] += pv*(float)vc[e];
    }
    bfloat16 *or_ = O+r*d;
    for (int e = 0; e < d; e++) or_[e] = (bfloat16)accm[e];
  }
  event1();
}
}
