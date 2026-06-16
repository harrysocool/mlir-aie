<!---//===- README.md ------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------===//-->

# Fused single-head attention (SAM3 RoPE-attention, aie2p)

A single-core, single-dispatch fused attention block for the Strix Halo NPU
(aie2p / npu2):

```
O = softmax( (Q @ K^T) * scale ) @ V      # one head, on-chip, no DDR round-trips
```

Q, K, V are concatenated into one input stream `[Q|K|V]` of shape `[3*S, d]`
(the AIE compute tile exposes only 2 input DMA channels, so the three matrices
share one channel and the kernel slices them by `S*d` offset). Output `O` is
`[S, d]`. All `bfloat16`.

This targets SAM3's ViT-L backbone attention (`head_dim = 64`, `rope_theta =
1e4`). RoPE itself is a separate, already-validated kernel
(`programming_examples/ml/rope`); its paired-rotation is position-independent, so
SAM3's 2D axial RoPE only changes the host-side cos/sin LUT, not the kernel.

## Status

- Correctness: `cos = 0.99978` vs PyTorch SDPA (bf16), `S = d = 64`.
- Performance: the QK^T / PV GEMMs are **scalar** (correctness-first) → ~4.5 ms.
  Next step is vectorizing them with `aie::mmul` (expected 10-50x).

## Files

- `attn.py` : IRON design (1 input fifo `[Q|K|V]`, 1 output fifo, single Worker).
- `attn.cc` : the fused kernel — `aie_kernels/aie2p/attn.cc`.

## Build

Set up the toolchain env first (Peano + mlir_aie + XRT on PATH). See the sam3
repo `npu/iron/README.md` for the exact env (or `npu_iron/recover_iron.sh`).

```shell
export PEANO_INSTALL_DIR=<venv>/lib/python3.12/site-packages/llvm-aie
export PATH=/opt/xilinx/xrt/bin:<venv>/lib/python3.12/site-packages/mlir_aie/bin:$PATH
make NPU2=1 seq=64 dim=64        # -> build/final.xclbin, build/insts.bin
```

## Run

The host runner lives in the sam3 repo (`npu/iron/run/run_attn.py`, pyxrt).
It loads `build/final.xclbin` + `build/insts.bin`, kernel name `MLIR_AIE`,
opcode 3.
