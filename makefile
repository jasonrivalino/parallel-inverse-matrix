OUTPUT_FOLDER = bin

all: serial parallel

parallel:
	mpic++ src/open-mpi/parallel.cpp -o $(OUTPUT_FOLDER)/parallel -lm
	bin/parallel < test_cases/32.txt

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial
	bin/serial < test_cases/4.txt

dev:
	mpic++ -g -Wall -o paralel ./src/open-mpi/parallel.cpp
	mpiexec -n 4 ./paralel < test_cases/32.txt > test_cases/output32new.txt

dev2:
	mpic++ -g -Wall -o tes ./src/open-mpi/tes.cpp
	mpiexec -n 4 ./tes < test_cases/32.txt > test_cases/output32.txt