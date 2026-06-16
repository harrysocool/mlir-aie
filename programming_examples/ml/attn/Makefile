##===- Makefile - fused single-head attention (SAM3 RoPE-attn) ------------===##
# Builds a fused single-head attention design (QK^T -> scale -> softmax -> PV)
# for the Strix Halo NPU (aie2p / npu2). Correctness-first; GEMM is scalar.
#
# Requires (see ../../../README or the sam3 npu/iron/README):
#   PEANO_INSTALL_DIR   -> llvm-aie (Peano) install (site-packages/llvm-aie)
#   PATH                -> must include xclbinutil (XRT bin) and mlir_aie/bin
##===---------------------------------------------------------------------===##
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common
VPATH := ${srcdir}/../../../aie_kernels/aie2p

all: build/final.xclbin

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = attn
seq ?= 64
dim ?= 64

aie_py_src = attn.py

build/attn.o: attn.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DATTN_S=${seq} -DATTN_D=${dim} -c $< -o ${@F}
else
	echo "Only npu2 (aie2p / Strix) is supported for this design"; false
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -s ${seq} -e ${dim} > $@

build/final.xclbin: build/aie.mlir build/attn.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-compile-host --no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin ${<F}

clean:
	rm -rf build
