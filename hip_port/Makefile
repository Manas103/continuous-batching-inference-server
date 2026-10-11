# HIP port build. Two independent targets:
#
#   make check        builds and links correctness_check against HIP's
#                      CUDA backend (hipcc dispatches to nvcc under the
#                      hood) -- this is the only target that produces a
#                      runnable binary on this machine (one consumer GPU,
#                      no AMD hardware).
#   make gfx942        compile-only (--cuda-device-only, no link, no run)
#                      for gfx942 (CDNA 3 / MI300-class) via ROCm clang.
#                      This is what .github/workflows/gfx942-compile.yml
#                      runs in CI; it never runs on an AMD GPU because
#                      there isn't one in CI either.
#
# Both read CUDA_PATH/HIP_PLATFORM/HIP_RUNTIME from the environment so the
# nvidia-backend build can be pointed at a specific CUDA install (this repo
# uses CUDA 12.4, matching the CUDA original's CMakeLists.txt).

HIPCC ?= /opt/rocm/bin/hipcc
CLANG ?= /opt/rocm/llvm/bin/clang++
CUDA_PATH ?= /usr/local/cuda-12.4
INCLUDE = -I include
CXXFLAGS = -std=c++17 -O2

SRCS = src/model_weights.cpp src/paged_kv_cache.cpp src/transformer.cpp
KERNELS = kernels/attention.hip.cpp kernels/elementwise.hip.cpp kernels/linear.hip.cpp
MAIN = src/correctness_check.cpp

OBJDIR = build_obj

.PHONY: check bench gfx942 clean

check: $(OBJDIR)/.stamp
	CUDA_PATH=$(CUDA_PATH) HIP_PLATFORM=nvidia HIP_RUNTIME=nvidia $(HIPCC) \
		$(OBJDIR)/attention.o $(OBJDIR)/elementwise.o $(OBJDIR)/linear.o \
		$(OBJDIR)/model_weights.o $(OBJDIR)/paged_kv_cache.o $(OBJDIR)/transformer.o \
		$(OBJDIR)/correctness_check.o -o correctness_check

$(OBJDIR)/.stamp:
	mkdir -p $(OBJDIR)
	CUDA_PATH=$(CUDA_PATH) HIP_PLATFORM=nvidia HIP_RUNTIME=nvidia $(HIPCC) $(INCLUDE) $(CXXFLAGS) -c kernels/attention.hip.cpp -o $(OBJDIR)/attention.o
	CUDA_PATH=$(CUDA_PATH) HIP_PLATFORM=nvidia HIP_RUNTIME=nvidia $(HIPCC) $(INCLUDE) $(CXXFLAGS) -c kernels/elementwise.hip.cpp -o $(OBJDIR)/elementwise.o
	CUDA_PATH=$(CUDA_PATH) HIP_PLATFORM=nvidia HIP_RUNTIME=nvidia $(HIPCC) $(INCLUDE) $(CXXFLAGS) -c kernels/linear.hip.cpp -o $(OBJDIR)/linear.o
	CUDA_PATH=$(CUDA_PATH) HIP_PLATFORM=nvidia HIP_RUNTIME=nvidia $(HIPCC) $(INCLUDE) $(CXXFLAGS) -c src/model_weights.cpp -o $(OBJDIR)/model_weights.o
	CUDA_PATH=$(CUDA_PATH) HIP_PLATFORM=nvidia HIP_RUNTIME=nvidia $(HIPCC) $(INCLUDE) $(CXXFLAGS) -c src/paged_kv_cache.cpp -o $(OBJDIR)/paged_kv_cache.o
	CUDA_PATH=$(CUDA_PATH) HIP_PLATFORM=nvidia HIP_RUNTIME=nvidia $(HIPCC) $(INCLUDE) $(CXXFLAGS) -c src/transformer.cpp -o $(OBJDIR)/transformer.o
	CUDA_PATH=$(CUDA_PATH) HIP_PLATFORM=nvidia HIP_RUNTIME=nvidia $(HIPCC) $(INCLUDE) $(CXXFLAGS) -c src/correctness_check.cpp -o $(OBJDIR)/correctness_check.o
	touch $(OBJDIR)/.stamp

bench: $(OBJDIR)/.stamp
	CUDA_PATH=$(CUDA_PATH) HIP_PLATFORM=nvidia HIP_RUNTIME=nvidia $(HIPCC) $(INCLUDE) $(CXXFLAGS) -c src/bench_linear.cpp -o $(OBJDIR)/bench_linear.o
	CUDA_PATH=$(CUDA_PATH) HIP_PLATFORM=nvidia HIP_RUNTIME=nvidia $(HIPCC) \
		$(OBJDIR)/linear.o $(OBJDIR)/bench_linear.o -o bench_linear

gfx942:
	$(HIPCC) --offload-arch=gfx942 --cuda-device-only $(INCLUDE) -c kernels/attention.hip.cpp -o /tmp/attention_gfx942.o
	$(HIPCC) --offload-arch=gfx942 --cuda-device-only $(INCLUDE) -c kernels/elementwise.hip.cpp -o /tmp/elementwise_gfx942.o
	$(HIPCC) --offload-arch=gfx942 --cuda-device-only $(INCLUDE) -c kernels/linear.hip.cpp -o /tmp/linear_gfx942.o

clean:
	rm -rf $(OBJDIR) correctness_check
