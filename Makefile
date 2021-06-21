CC=g++
LINKER_DIRS=-L/usr/local/cuda/lib
LINKER_FLAGS=-lcudart -lcuda
NVCC=nvcc
CUDA_ARCHITECTURE=20
OCELOT=`OcelotConfig -l`

all: main

main.o: main.cu
	$(NVCC) main.cu -c -I . 

main1: main.o device1.o 
	$(CC) main.o device1.o -o main1 $(LINKER_DIRS) $(OCELOT)

device1.o: cuda1.cu
	$(NVCC) -c cuda1.cu -arch=sm_$(CUDA_ARCHITECTURE) -I .

main2: main.o device2.o 
	$(CC) main.o device2.o -o main2 $(LINKER_DIRS) $(OCELOT)

device2.o: cuda2.cu
	$(NVCC) -c cuda2.cu -arch=sm_$(CUDA_ARCHITECTURE) -I .

clean:
	rm -f main.o device1.o device2.o main1 main2 kernel-times.json
