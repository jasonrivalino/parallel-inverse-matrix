#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, finish_time, total_time, avg_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0, i = 0, j = 0, k = 0;
    double **mat = NULL;
    double d = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    if (rank == 0) {
        cin >> n;
    }

    // Allocating memory for matrix array
    mat = new double*[2*n];
    for (i = 0; i < 2*n; ++i)
    {
        mat[i] = new double[2*n]();
    }
    
    //Inputs the coefficients of the matrix
    for(i = 0; i < n; ++i)
    {
        for(j = 0; j < n; ++j)
        {
            cin >> mat[i][j];
        }
    }
    
    // Initializing Right-hand side to identity matrix
    for(i = 0; i < n; ++i)
    {
        for(j = 0; j < 2*n; ++j)
        {
            if(j == (i+n))
            {
                mat[i][j] = 1;
            }
        }
    }
    
    // Partial pivoting
    for(i = n; i > 1; --i)
    {
        if(mat[i-1][1] < mat[i][1])
        {
            for(j = 0; j < 2*n; ++j)
            {
                d = mat[i][j];
                mat[i][j] = mat[i-1][j];
                mat[i-1][j] = d;
            }
        }
    }

    // Reducing To Diagonal Matrix
    for(i = 0; i < n; ++i)
    {
        for(j = 0; j < 2*n; ++j)
        {
            if(j != i)
            {
                d = mat[j][i] / mat[i][i];
                for(k = 0; k < n*2; ++k)
                {
                    mat[j][k] -= mat[i][k]*d;
                }
            }
        }
    }
    
    // Reducing To Unit Matrix
    for(i = 0; i < n; ++i)
    {
        d = mat[i][i];
        for(j = 0; j < 2*n; ++j)
        {
            mat[i][j] = mat[i][j]/d;
        }
    }
    
    cout << n << endl;
    for(i=0; i < n; ++i)
    {
        for(j = n; j < 2*n; ++j)
        {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    
    // Deleting the memory allocated
    for (i = 0; i < n; ++i)
    {
        delete[] mat[i];
    }
    delete[] mat;

    // Hitung waktu selesai
    finish_time = MPI_Wtime();

    // Hitung total waktu yang dibutuhkan
    total_time = finish_time - start_time;
    MPI_Reduce(&total_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Hitung rata-rata waktu yang dibutuhkan
    avg_time = total_time / size;

    if (rank == 0) {
        std::cout << "Time taken: " << total_time << " seconds" << std::endl;
        std::cout << "Average time taken: " << avg_time << " seconds" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
