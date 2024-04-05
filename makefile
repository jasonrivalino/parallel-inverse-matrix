OUTPUT_FOLDER = bin

all: serial parallelMPI parallelMP

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial
	./bin/serial < test_cases/32.txt > test_cases/outputSerial.txt

parallelMPI:
	mpic++ src/open-mpi/parallel.cpp -o $(OUTPUT_FOLDER)/parallel -lm
	mpiexec -n 4 ./bin/parallel < test_cases/32.txt > test_cases/outputParallel.txt

parallelMP:
	gcc src/open-mp/parallel.cpp --openmp -g -Wall -o $(OUTPUT_FOLDER)/parallel -lm

parallelMPMacbook:
	g++-13 src/open-mp/parallel.cpp -fopenmp -o $(OUTPUT_FOLDER)/parallel -lm
	./test_cases/parallel < test_cases/32.txt > test_cases/outputParallel.txt