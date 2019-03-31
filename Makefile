all: shiftor_cpu wumanber_cpu

shiftor_cpu: shiftor_cpu.cpp
	g++ shiftor_cpu.cpp -o shiftor_cpu

wumanber_cpu: wumanber_cpu.cpp
	g++ wumanber_cpu.cpp -o wumanber_cpu

clean:
	rm shiftor_cpu
	rm wumanber_cpu
