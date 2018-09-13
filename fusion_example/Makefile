ROCM_PATH?= $(wildcard /opt/rocm)
HIP_PATH?= $(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc
MIOPEN_INSTALL_PATH=/home/jehandad/MLOpen/install.dir
INCLUDE_DIRS=-I$(HIP_PATH)/include -I$(ROCM_PATH)/include -I$(ROCM_PATH)/hipblas/include -I$(ROCM_PATH)/opencl/include -I$(MIOPEN_INSTALL_PATH)/include
LD_FLAGS=-L$(ROCM_PATH)/lib -L$(ROCM_PATH)/opencl/lib/x86_64 -L$(MIOPEN_INSTALL_PATH)/lib -lMIOpen -lOpenCL -lmiopengemm -lhipblas -lrocblas
TARGET=--amdgpu-target=gfx900
LAYER_TIMING=1

#HIPCC_FLAGS=-g -Wall $(CXXFLAGS) $(TARGET) $(INCLUDE_DIRS)
HIPCC_FLAGS=-g -O3 -Wall $(CXXFLAGS) $(TARGET) $(INCLUDE_DIRS)


all: main

HEADERS= miopen.hpp

main: main.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) main.cpp $(LD_FLAGS) -o fusion_example

clean:
	rm -f *.o *.out fusion_example
