
#
# Copyright 2014-2021 NVIDIA Corporation. All rights reserved
#
ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

CUDA_INSTALL_PATH ?= /usr/local/cuda-12.3
PROFILER_HOST_UTILS_SRC ?= /usr/local/cuda-12.3/extras/CUPTI/samples/extensions/src/profilerhost_util
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I/usr/local/cuda-12.3/extras/CUPTI/samples/extensions/include/profilerhost_util -I/usr/local/cuda-12.3/include -I/usr/local/cuda-12.3/extras/CUPTI/samples/extensions/include/c_util -I./ -I/usr/local/cuda-12.3/extras/CUPTI/include/



TARGET_ARCH ?= $(HOST_ARCH)
TARGET_OS ?= $(shell uname | tr A-Z a-z)

ifeq ($(OS),Windows_NT)
    LIB_PATH ?= ..\..\lib64
else
    ifneq ($(TARGET_ARCH), $(HOST_ARCH))
        INCLUDES += -I$(CUDA_INSTALL_PATH)/targets/$(HOST_ARCH)-$(shell uname | tr A-Z a-z)/include
        INCLUDES += -I$(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/include
        LIB_PATH ?= $(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/lib
        TARGET_CUDA_PATH = -L $(LIB_PATH)/stubs
    else
        EXTRAS_LIB_PATH := /usr/local/cuda-12.3/extras/CUPTI/lib64
        LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64
    endif
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):$(LIB_PATH)
    LIBS = -lcuda -L $(LIB_PATH) -lcupti -lnvperf_host -lnvperf_target -L ..\extensions\src\profilerhost_util -lprofilerHostUtil
    OBJ = obj
    LIBEXT = lib
    LIBPREFIX =
    BINEXT = .exe
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS = -Xlinker -framework -Xlinker cuda -L $(LIB_PATH) -lcupti -lnvperf_host -lnvperf_target -L /usr/local/cuda-12.3/extras/CUPTI/samples/extensions/src/profilerhost_util -lprofilerHostUtil
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
        LIBS :=
        ifeq ($(HOST_ARCH), $(TARGET_ARCH))
            export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
            LIBS = -L $(EXTRAS_LIB_PATH)
        endif
        LIBS += $(TARGET_CUDA_PATH) -lcuda -L $(LIB_PATH) -lcupti -lnvperf_host -lnvperf_target -L /usr/local/cuda-12.3/extras/CUPTI/samples/extensions/src/profilerhost_util -lprofilerHostUtil
    endif
    OBJ = o
    LIBEXT = a
    LIBPREFIX = lib
    BINEXT =
endif

# Point to the necessary cross-compiler.
# NVCCFLAGS :=
ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    ifeq ($(TARGET_ARCH), aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/q++
            NVCCFLAGS += --qpp-config 8.3.0,gcc_ntoaarch64le -lsocket
        endif
    endif

    ifdef HOST_COMPILER
        NVCC_COMPILER := -ccbin $(HOST_COMPILER)
    endif
endif



# Gencode arguments
SMS ?= 52 53 60 61 62 70 72 75 80 86 87 89 90
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))


# include /scratch/user/siweicui/llama.cpp/Makefile


.DEFAULT: all
.PHONY: all

all: profiler_host_util tensor_usage_collector main


profiler_host_util:
	cd $(PROFILER_HOST_UTILS_SRC) && $(MAKE)

tensor_usage_collector: tensor_usage_collector.$(OBJ)
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) -o $@ $^ $(LIBS) $(INCLUDES)

tensor_usage_collector.$(OBJ): tensor_usage_collector.cpp
	$(NVCC) $(NVCC_COMPILER)  $(NVCCFLAGS) $(GENCODE_FLAGS) -c $(INCLUDES) $<

main: main.$(OBJ)
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) -o $@ main.$(OBJ) $(LIBS)

main.$(OBJ): main.cu
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) -arch=sm_75 -lineinfo $(GENCODE_FLAGS) -c $(INCLUDES) $<

run: tensor_usage_collector
	./$<

clean:
	rm -f main main.$(OBJ) tensor_usage_collector tensor_usage_collector.$(OBJ)
