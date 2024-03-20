OUTPUT_FOLDER = bin

all: serial parallel

parallel:
	mpic++ src/serial/parallel.cpp -o $(OUTPUT_FOLDER)/parallel -lm

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial