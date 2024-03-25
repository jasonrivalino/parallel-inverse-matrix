OUTPUT_FOLDER = bin

all: serial parallel dev

parallel:
	mpic++ src/open-mpi/parallel.cpp -o $(OUTPUT_FOLDER)/parallel -lm
	bin/parallel < test_cases/32.txt

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial
	bin/serial < test_cases/32.txt

dev:
	mpic++ -o paralel ./src/open-mpi/parallel.cpp
	mpiexec -n 2 ./paralel < test_cases/32.txt > test_cases/output.txt