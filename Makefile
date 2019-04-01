all: shiftor wumanber_cpu

shiftor: shiftor.cu
	nvcc shiftor.cu -o shiftor

wumanber_cpu: wumanber_cpu.cpp
	g++ wumanber_cpu.cpp -o wumanber_cpu

clean:
	rm shiftor
	rm wumanber_cpu
