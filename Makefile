all: shiftor_cpu

shiftor_cpu: shiftor_cpu.cpp
	g++ shiftor_cpu.cpp -o shiftor_cpu

clean:
	rm shiftor_cpu
