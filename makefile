OUTPUT_FOLDER = bin

all: serial parallel

parallel:
	mpic++ src/open-mpi/parallel.cpp -o $(OUTPUT_FOLDER)/parallel -lm

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial