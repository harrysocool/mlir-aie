//===- softmax_vl16.cc - rowwise softmax, VL=16, bf16 ---------------------===//
// Handles row sizes divisible by 16 (576, 1344, etc.). Three-pass algorithm:
// 1) max, 2) exp (stored back to input buffer), 3) scale (input→output).
// Based on aie2p/softmax.cc pattern: input acts as temp buffer for exp values.
#include <aie_api/aie.hpp>
#include <stdint.h>
#define VL 16
#define LOG2E 1.4453125f
using namespace aie;

// input: [n] bf16 - modified in-place for exp tmp, output: [n] bf16 - softmax result
void softmax_vl16(bfloat16 *restrict input, bfloat16 *restrict output,
                  const int32_t vector_size) {
  event0();
  const int iters = vector_size / VL;
  vector<bfloat16,VL> l2e = broadcast<bfloat16,VL>((bfloat16)LOG2E);

  // pass1: find max (over log2e-scaled values, matching softmax.cc)
  vector<bfloat16,VL> maxv = broadcast<bfloat16,VL>((bfloat16)-32768.f);
  {
    auto it = aie::cbegin_restrict_vector<VL>(input);
    for (int i=0;i<iters;i++) {
      vector<bfloat16,VL> x = *it++;
      accum<accfloat,VL> sc = aie::mul(x, l2e);
      maxv = aie::max(maxv, sc.to_vector<bfloat16>());
    }
  }
  vector<bfloat16,VL> maxv_bc = broadcast<bfloat16,VL>((bfloat16)aie::reduce_max(maxv));

  // pass2: compute exp(x*log2e - max), store back to input, accumulate sum
  accum<accfloat,VL> expsum = zeros<accfloat,VL>();
  {
    auto it_in  = aie::cbegin_restrict_vector<VL>(input);
    auto it_out = aie::begin_restrict_vector<VL>(input);   // write exp back to input
    for (int i=0;i<iters;i++) {
      vector<bfloat16,VL> x = *it_in++;
      accum<accfloat,VL> sc = aie::mul(x, l2e);
      accum<accfloat,VL> sh = aie::sub(sc, maxv_bc);
      vector<bfloat16,VL> ev = aie::exp2<bfloat16>(sh.to_vector<float>());
      *it_out++ = ev;
      expsum = add(expsum, ev);
    }
  }
  bfloat16 inv = (bfloat16)aie::inv(aie::reduce_add(expsum.to_vector<float>()));

  // pass3: scale exp values (read from input) -> write to output
  {
    auto it_in  = aie::cbegin_restrict_vector<VL>(input);
    auto it_out = aie::begin_restrict_vector<VL>(output);
    for (int i=0;i<iters;i++) {
      vector<bfloat16,VL> e = *it_in++;
      accum<accfloat,VL> o = aie::mul(e, inv);
      *it_out++ = o.to_vector<bfloat16>();
    }
  }
  event1();
}

extern "C" {
void softmax_bf16(bfloat16 *restrict in, bfloat16 *restrict out, int32_t n) {
  softmax_vl16(in, out, n);
}
}
