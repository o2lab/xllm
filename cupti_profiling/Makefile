ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

CUDA_INSTALL_PATH ?= ../../../usr/local/cuda-12.3
PROFILER_HOST_UTILS_SRC ?= extensions/src/profilerhost_util
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I"$(CUDA_INSTALL_PATH)/extras/CUPTI/include" -I extensions/include/profilerhost_util -I extensions/include/c_util  

TARGET_ARCH ?= $(HOST_ARCH)
TARGET_OS ?= $(shell uname | tr A-Z a-z)

# Set required library paths.
# In the case of cross-compilation, set the libs to the correct ones under /usr/local/cuda/targets/<TARGET_ARCH>-<TARGET_OS>/lib

ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    INCLUDES += -I$(CUDA_INSTALL_PATH)/targets/$(HOST_ARCH)-$(shell uname | tr A-Z a-z)/include
    INCLUDES += -I$(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/include
    LIB_PATH ?= $(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/lib
    TARGET_CUDA_PATH = -L $(LIB_PATH)/stubs
else
    EXTRAS_LIB_PATH := ../../lib64
    LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64
endif


LIBS :=
ifeq ($(HOST_ARCH), $(TARGET_ARCH))
    export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
    LIBS = -L $(EXTRAS_LIB_PATH)
endif
LIBS += $(TARGET_CUDA_PATH) -lcuda -L $(LIB_PATH) -lcupti -lcudart -lnvperf_host -lnvperf_target -L extensions/src/profilerhost_util -lprofilerHostUtil 
OBJ = o
LIBEXT = a
LIBPREFIX = lib
BINEXT =


# Point to the necessary cross-compiler.
NVCCFLAGS :=
ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    ifeq ($(TARGET_ARCH), aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        endif
    endif

    ifdef HOST_COMPILER
        NVCC_COMPILER := -ccbin $(HOST_COMPILER)
    endif
endif

# Gencode arguments
SMS ?= 70 72 75 80 86 87 89 90
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

.DEFAULT: all
.PHONY: all

all : profiler_host_util main

profiler_host_util:
	cd $(PROFILER_HOST_UTILS_SRC) && $(MAKE)

main: main.$(OBJ) profiler.$(OBJ)
	g++ -o $@ $^  $(LIBS) $(INCLUDES)

main.$(OBJ): main.cu profiler.h 
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) -c $@ $(LIBS)  main.cu


profiler.$(OBJ): profiler.cc profiler.h helper_cupti.h 
	g++ -c $(INCLUDES) $(LIBS)  profiler.cc

run: main
	./$<

clean:
ifeq ($(OS), Windows_NT)
	del callback_profiling.exe callback_profiling.lib callback_profiling.exp callback_profiling.$(OBJ)
else
	rm -f main main.$(OBJ) profiler.$(OBJ)
endif
