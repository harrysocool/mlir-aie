"""Flash attention S=576: concat [Q_tile|K_chunk|V_chunk] → 2 DMA channels."""
import numpy as np, argparse, sys
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from ml_dtypes import bfloat16

def flash_attn_S576(dev, S=576, d=64, Br=16, Bc=64):
    n_q  = S // Br   # 36
    n_kv = S // Bc   # 9
    concat_size = (Br + 2*Bc) * d   # 16+128)*64=9216 elements

    concat_ty  = np.ndarray[(concat_size,), np.dtype[bfloat16]]  # [Q|K|V] per step
    o_tile_ty  = np.ndarray[(Br*d,),        np.dtype[bfloat16]]
    q_full_ty  = np.ndarray[(S*d,),         np.dtype[bfloat16]]
    o_full_ty  = np.ndarray[(S*d,),         np.dtype[bfloat16]]

    inf = ObjectFifo(concat_ty, depth=1, name="inf")   # 1 in chan
    of_ = ObjectFifo(o_tile_ty, depth=1, name="of")    # 1 out chan

    k = Kernel("flash_attn_step_concat", "flash_attn_step.o",
               [concat_ty, o_tile_ty])

    total_steps = n_q * n_kv   # 324

    def core_body(inf, of_, kern):
        for _ in range_(total_steps):
            ei = inf.acquire(1)
            eo = of_.acquire(1)
            kern(ei, eo)          # writes o_out only on last chunk of each Q tile
            inf.release(1)
            of_.release(1)

    w = Worker(core_body, [inf.cons(), of_.prod(), k])

    # Build TAP sequence: for each of 36 Q-tiles, 9 K/V chunks
    # Each concat transfer = [Q_tile_i | K_chunk_j | V_chunk_j]
    # But TensorTiler2D can't easily describe non-contiguous concat from 3 tensors...
    # Workaround: pass Q, K, V as separate tensors and do host-side concat
    # Alternative: build 3-tensor concat via DMA scatter (not easily expressible in IRON)
    
    # Simplest approach: use a PRE-BUILT concat buffer as single input
    # Pass Q+K+V as ONE tensor (pre-concatenated by host per step)
    # Host prepares: for each step, concat [Q_tile_i, K_chunk_j, V_chunk_j]
    # This is feasible but loses the "single dispatch" benefit since host must iterate
    
    # For now: demonstrate the IRON design compiles, validate with correct data later
    concat_full_ty = np.ndarray[(total_steps*concat_size,), np.dtype[bfloat16]]
    
    rt = Runtime()
    with rt.sequence(concat_full_ty, o_full_ty) as (QKV, O):
        rt.start(w)
        step_taps = TensorTiler2D.simple_tiler((total_steps, concat_size), (1, concat_size))
        o_taps    = TensorTiler2D.simple_tiler((n_q, Br*d), (1, Br*d))
        for step in range(total_steps):
            rt.fill(inf.prod(), QKV, step_taps[step])
        for i in range(n_q):
            rt.drain(of_.cons(), O, o_taps[i], wait=(i==n_q-1))

    return Program(dev, rt).resolve_program()

p = argparse.ArgumentParser()
p.add_argument("-d","--dev",required=True,dest="device"); o=p.parse_args(sys.argv[1:])
dev = NPU2() if o.device=="npu2" else NPU1()
print(flash_attn_S576(dev))
