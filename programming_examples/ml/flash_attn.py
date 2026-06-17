# Flash attention IRON design: one dispatch per head, data stays on chip.
# Input: concat [Q|K|V] = [3*S*d] bf16. Output: O [S*d] bf16.
import numpy as np, argparse, sys
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from ml_dtypes import bfloat16

def flash_attn(dev, S, d):
    vol    = S * d
    in_ty  = np.ndarray[(3*vol,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(vol,),   np.dtype[bfloat16]]
    inf = ObjectFifo(in_ty,  depth=1, name="inf")
    of  = ObjectFifo(out_ty, depth=1, name="of")
    k   = Kernel("flash_attention", "flash_attn.o", [in_ty, out_ty])
    def core_body(i_in, o_out, kern):
        ei = i_in.acquire(1); eo = o_out.acquire(1)
        kern(ei, eo)
        i_in.release(1); o_out.release(1)
    w = Worker(core_body, fn_args=[inf.cons(), of.prod(), k])
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (qkv, o):
        rt.start(w)
        rt.fill(inf.prod(), qkv)
        rt.drain(of.cons(), o, wait=True)
    return Program(dev, rt).resolve_program()

p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-s", "--seq", type=int, default=576)
p.add_argument("-e", "--dim", type=int, default=64)
o = p.parse_args(sys.argv[1:])
dev = NPU2() if o.device == "npu2" else NPU1()
print(flash_attn(dev, o.seq, o.dim))
