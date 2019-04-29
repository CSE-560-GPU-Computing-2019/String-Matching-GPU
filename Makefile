all: shiftor shiftor_scan wumanber_cpu wumanber wumanber_scan

shiftor: shiftor.cu
	nvcc shiftor.cu -o shiftor

shiftor_scan: shiftor_scan.cu
	nvcc shiftor_scan.cu -o shiftor_scan

wumanber_cpu: wumanber_cpu.cpp
	g++ wumanber_cpu.cpp -o wumanber_cpu

wumanber: wumanber.cu
	nvcc wumanber.cu -o wumanber

wumanber_scan: wumanber_scan.cu
	nvcc wumanber_scan.cu -o wumanber_scan

clean:
	rm shiftor shiftor_scan wumanber_cpu wumanber wumanber_scan
