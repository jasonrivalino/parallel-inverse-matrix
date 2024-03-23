OUTPUT_FOLDER = bin

all: serial parallel

parallel:
	mpic++ src/open-mpi/parallel.cpp -o $(OUTPUT_FOLDER)/parallel -lm
	bin/parallel < test_cases/32.txt

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial
	bin/serial < test_cases/32.txt