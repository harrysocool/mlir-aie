// Flash attention step: no j parameter — internal static counter tracks position
#include <aie_api/aie.hpp>
#include <stdint.h>
#define VL 16
#define LOG2E 1.44269504089f
#ifndef FLASH_BR
#define FLASH_BR 16
#endif
#ifndef FLASH_BC
#define FLASH_BC 64
#endif
#ifndef FLASH_D
#define FLASH_D 64
#endif
#ifndef FLASH_NKV     // number of K/V chunks = S/Bc
#define FLASH_NKV 9
#endif
#define FLASH_SCALE 0.125f
using namespace aie;

extern "C" {
void flash_attn_step_concat(bfloat16 *restrict qkv_concat,
                            bfloat16 *restrict o_out) {
  event0();
  constexpr int Br  = FLASH_BR;
  constexpr int Bc  = FLASH_BC;
  constexpr int d   = FLASH_D;
  constexpr int Nkv = FLASH_NKV;
  // Slice concat input
  bfloat16 *restrict q_tile = qkv_concat;
  bfloat16 *restrict k_chunk = qkv_concat + Br*d;
  bfloat16 *restrict v_chunk = qkv_concat + Br*d + Bc*d;

  // Static state: accumulators + chunk counter
  static float o_accum[Br*d];
  static float m_i[Br];
  static float l_i[Br];
  static int   j_cnt = 0;   // current K/V chunk index

  alignas(32) bfloat16 s_bf[Bc];
  alignas(32) bfloat16 p[Bc];
  vector<bfloat16,VL> l2e_v = broadcast<bfloat16,VL>((bfloat16)LOG2E);

  // On first chunk: init
  if (j_cnt == 0) {
    for (int r = 0; r < Br; r++) {
      m_i[r] = -3.0e38f; l_i[r] = 0.0f;
      for (int e = 0; e < d; e++) o_accum[r*d+e] = 0.0f;
    }
  }

  for (int r = 0; r < Br; r++) {
    const bfloat16 *qi = q_tile + r*d;
    float rowmax_new = m_i[r];
    for (int c = 0; c < Bc; c++) {
      const bfloat16 *kc = k_chunk + c*d;
      float acc = 0.0f;
      for (int e = 0; e < d; e++) acc += (float)qi[e]*(float)kc[e];
      float s = acc*FLASH_SCALE;
      s_bf[c] = (bfloat16)s;
      if (s > rowmax_new) rowmax_new = s;
    }

    float corr;
    if (j_cnt == 0 && m_i[r] <= -3.0e37f) {
      corr = 1.0f;
    } else {
      float delta = (m_i[r] - rowmax_new) * LOG2E;
      accum<accfloat,VL> a = zeros<accfloat,VL>();
      vector<bfloat16,VL> dv = broadcast<bfloat16,VL>((bfloat16)delta);
      accum<accfloat,VL> da = aie::add(a, dv);
      vector<bfloat16,VL> ec = aie::exp2<bfloat16>(da.to_vector<float>());
      corr = aie::reduce_add(ec) / (float)VL;
    }
    float l_new = l_i[r]*corr;
    vector<bfloat16,VL> maxv_l2 = broadcast<bfloat16,VL>((bfloat16)(rowmax_new*LOG2E));
    accum<accfloat,VL> psum = zeros<accfloat,VL>();
    for (int c = 0; c < Bc; c += VL) {
      vector<bfloat16,VL> sv = load_v<VL>(s_bf+c);
      accum<accfloat,VL> sc2 = aie::mul(sv, l2e_v);
      accum<accfloat,VL> sh  = aie::sub(sc2, maxv_l2);
      vector<bfloat16,VL> ev = aie::exp2<bfloat16>(sh.to_vector<float>());
      store_v(p+c, ev);
      psum = add(psum, ev);
    }
    l_new += aie::reduce_add(psum.to_vector<float>());
    float *oi = o_accum + r*d;
    for (int e = 0; e < d; e++) oi[e] *= corr;
    for (int c = 0; c < Bc; c++) {
      float pv = (float)p[c];
      const bfloat16 *vc = v_chunk + c*d;
      for (int e = 0; e < d; e++) oi[e] += pv*(float)vc[e];
    }
    m_i[r] = rowmax_new; l_i[r] = l_new;
  }

  j_cnt++;
  if (j_cnt == Nkv) {
    // Last chunk: normalize + output
    for (int r = 0; r < Br; r++) {
      float inv = 1.0f / l_i[r];
      bfloat16 *or_ = o_out + r*d;
      for (int e = 0; e < d; e++) or_[e] = (bfloat16)(o_accum[r*d+e]*inv);
    }
    j_cnt = 0;   // reset for next Q tile
  }
  event1();
}
}

extern "C" {
void flash_attn_step(bfloat16*q,bfloat16*k,bfloat16*v,bfloat16*o){flash_attn_step_concat(q,o);}
}
