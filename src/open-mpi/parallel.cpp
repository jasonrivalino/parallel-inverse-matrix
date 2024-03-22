#include <iostream>
#include <mpi.h>
#include <cmath> 

using namespace std;

int main(int argc, char *argv[]) {
    int world_rank, world_size;
    double start_time, finish_time, total_time, avg_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = 0, i = 0, j = 0, k = 0;
    double **mat = NULL;
    double d = 0.0;

    // MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        cin >> n;
    }
    
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine local rows for each process
    int local_rows = n / world_size;
    int remainder = n % world_size;
    int start_row = world_rank * local_rows;
    int end_row = start_row + local_rows;
    if (world_rank == world_size - 1) {
        end_row += remainder;
    }

    // Allocating memory for matrix array
    mat = new double*[2*n];
    for (i = 0; i < 2*n; ++i)
    {
        mat[i] = new double[2*n]();
    }

    // MPI_Scatter(&mat[0][0], local_rows*2*n, MPI_DOUBLE, &mat[0][0], local_rows * 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Inputs the coefficients of the matrix
    for(i = start_row; i < end_row; ++i) {
        for(j = 0; j < n; ++j) {
            if (world_rank == 0) {
                cin >> mat[i][j];
            }
            // Broadcast the input values to all processes
            // MPI_Bcast(&mat[i][j], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    
    start_time = MPI_Wtime();

    // Initializing Right-hand side to identity matrix
    for(i = start_row; i < end_row; ++i) {
        for(j = 0; j < 2*n; ++j) {
            if(j == (i+n)) {
                mat[i][j] = 1;
            }
        }
    }
    
    // Partial pivoting in parallelized form
    for(i = n; i > 1; --i) {
        if(mat[i-1][1] < mat[i][1]) {
            for(j = 0; j < 2*n; ++j) {
                d = mat[i][j];
                mat[i][j] = mat[i-1][j];
                mat[i-1][j] = d;
            }
        }
    }

    // Reducing To Diagonal Matrix
    for(i = start_row; i < end_row; ++i) {
        for(j = 0; j < n; ++j) {
            if(j != i) {
                if(mat[i][i] != 0) {
                    d = mat[j][i] / mat[i][i];
                    for(k = 0; k < 2*n; ++k) {
                        mat[j][k] -= mat[i][k] * d;
                    }
                }
            }
        }
    }
    
    // MPI_Barrier(MPI_COMM_WORLD);

    // Gather the results from all processes
    // MPI_Gather(&mat[0][0], local_rows*2*n, MPI_DOUBLE, &mat[0][0], local_rows*2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reducing to unit matrix
    for(i = 0; i < n; ++i) {
        d = mat[i][i];
        for(j = 0; j < 2*n; ++j) {
            mat[i][j] = mat[i][j] / d;
        }
    }

    cout << n << endl;

    cout << "" << endl;

    // Hitung waktu selesai
    finish_time = MPI_Wtime();

    // Hitung total waktu yang dibutuhkan
    total_time = finish_time - start_time;

    // Hitung rata-rata waktu yang dibutuhkan
    avg_time = total_time / world_size;

    if (world_rank == 0) {
        std::cout << "Time taken: " << total_time << " seconds" << std::endl;
        std::cout << "Average time taken: " << avg_time << " seconds" << std::endl;
        cout << "" << endl;


        // Output the original matrix followed by the augmented matrix
        for(i = 0; i < n; ++i)
        {
            for(j = 0; j < 2*n; ++j)
            {
                if (abs(mat[i][j]) < 1e-5) {
                    cout << "0 ";
                } else {
                    cout << mat[i][j] << " ";
                }
            }
            cout << endl;
        }
    }

    // Deleting the memory allocated
    for (i = 0; i < n; ++i)
    {
        delete[] mat[i];
    }
    delete[] mat;


    MPI_Finalize();

    return 0;
}
