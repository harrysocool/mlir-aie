import numpy as np, sys
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16

S=576; d=64; Br=16; Bc=64; n_q=36; n_kv=9
chunk_size = (Br + 2*Bc) * d   # 9216
total_steps = n_q * n_kv        # 324

step_ty    = np.ndarray[(chunk_size,),           np.dtype[bfloat16]]
o_tile_ty  = np.ndarray[(Br*d,),                 np.dtype[bfloat16]]
full_in_ty = np.ndarray[(total_steps*chunk_size,), np.dtype[bfloat16]]  # 5.97M
full_out_ty= np.ndarray[(n_q*Br*d,),             np.dtype[bfloat16]]   # 36.9K

inf = ObjectFifo(step_ty,   depth=2, name="inf")
of_ = ObjectFifo(o_tile_ty, depth=2, name="of")
k   = Kernel("flash_attn_step_concat", "flash_attn_step.o", [step_ty, o_tile_ty])

def core_body(inf, of_, kern):
    for _ in range_(n_q):
        eo = of_.acquire(1)
        for _ in range_(n_kv):
            ei = inf.acquire(1)
            kern(ei, eo)
            inf.release(1)
        of_.release(1)

w = Worker(core_body, [inf.cons(), of_.prod(), k])
rt = Runtime()
with rt.sequence(full_in_ty, full_out_ty) as (qkv, o):
    rt.start(w)
    rt.fill(inf.prod(), qkv)
    rt.drain(of_.cons(), o, wait=True)

m = Program(NPU2(), rt).resolve_program()
print(m)
