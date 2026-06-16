# SAM3 rowwise softmax IRON design — mirror softmax.py structure, parameterized n.
# Uses softmax_vl16.cc (VL=16). n must be divisible by 16.
import numpy as np, argparse, sys
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16

def rowwise_softmax(dev, total_size, row_size, n_cores=2):
    assert total_size % row_size == 0, f"{total_size} not divisible by {row_size}"
    assert row_size % 16 == 0, f"row_size {row_size} not divisible by 16"
    N          = total_size
    n          = row_size
    N_div_n    = N // n             # total number of row-tiles
    tiles      = N_div_n // n_cores # tiles per core

    tensor_ty  = np.ndarray[(N,),           np.dtype[bfloat16]]
    mtile_ty   = np.ndarray[(n*n_cores,),   np.dtype[bfloat16]]
    tile_ty    = np.ndarray[(n,),           np.dtype[bfloat16]]

    k = Kernel("softmax_bf16", "softmax_vl16.o", [tile_ty, tile_ty, np.int32])

    inA  = ObjectFifo(mtile_ty, name="inA")
    outC = ObjectFifo(mtile_ty, name="outC")

    inA_fifos = inA.cons().split(
        offsets=[n*i for i in range(n_cores)],
        obj_types=[tile_ty]*n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
    )
    outC_fifos = outC.prod().join(
        offsets=[n*i for i in range(n_cores)],
        obj_types=[tile_ty]*n_cores,
        names=[f"memC{i}" for i in range(n_cores)],
    )

    def core_fn(fi, fo, kern):
        for _ in range_(tiles):
            ei = fi.acquire(1); eo = fo.acquire(1)
            kern(ei, eo, n)
            fi.release(1); fo.release(1)

    workers = [Worker(core_fn, [inA_fifos[i].cons(), outC_fifos[i].prod(), k])
               for i in range(n_cores)]

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        rt.start(*workers)
        rt.fill(inA.prod(), A)
        rt.drain(outC.cons(), C, wait=True)
    return Program(dev, rt).resolve_program()

p = argparse.ArgumentParser()
p.add_argument("-d","--dev",   required=True)
p.add_argument("-t","--total", type=int, required=True)
p.add_argument("-n","--row",   type=int, required=True)
p.add_argument("-c","--cores", type=int, default=2)
o = p.parse_args(sys.argv[1:])
dev = NPU2() if o.dev=="npu2" else NPU1()
print(rowwise_softmax(dev, o.total, o.row, o.cores))
