#include <iostream>
#include <mpi.h>
#include <cmath>

using namespace std;

int *distributeRow(int totalElement, int colSize, int commSize) {
    int rowSize = totalElement / colSize;
    int base = rowSize / commSize;
    int remainder = rowSize % commSize;

    int *result = new int[commSize];
    for (int i = 0; i < commSize; i++) {
        result[i] = base;
        if (i < remainder) {
            result[i]++;
        }
    }
    return result;
}

double *flattenMatrix(double **matrix, int size) {
    double *flatMatrix = new double[size * size * 2];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < 2 * size; j++) {
            flatMatrix[i * (2 * size) + j] = matrix[i][j];
        }
    }
    return flatMatrix;
}

double **shapeMatrix(double *matrix, int size) {
    double **shapedMatrix = new double *[size];
    for (int i = 0; i < size; i++) {
        shapedMatrix[i] = new double[2 * size];
        for (int j = 0; j < 2 * size; j++) {
            shapedMatrix[i][j] = matrix[i * (2 * size) + j];
        }
    }
    return shapedMatrix;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int i, j;
    double d;
    double start_time, finish_time, total_time, avg_time;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    double **mat = NULL;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        cin >> n;
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *rowCount = distributeRow(n * n * 2, n * 2, size);
    double *recvbuf = new double[rowCount[rank] * n * 2];
    int *displs = new int[size];
    displs[0] = 0;
    for (int i = 0; i < size; i++) {
        rowCount[i] *= n * 2;
        if (i != 0) {
            displs[i] = displs[i - 1] + rowCount[i - 1];
        }
    }

    if (rank == 0) {
        mat = new double *[n];
        for (int i = 0; i < n; ++i) {
            mat[i] = new double[2 * n];
            for (int j = 0; j < 2 * n; ++j) {
                mat[i][j] = 0.0;
            }
        }

        MPI_Scatterv(&mat[0][0], rowCount, displs, MPI_DOUBLE, recvbuf, rowCount[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Inputs the coefficients of the matrix
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < n; ++j)
            {
                cin >> mat[i][j];
            }
        }

        // Send the matrix to all processes
        MPI_Scatterv(&flattenMatrix(mat, n)[0], rowCount, displs, MPI_DOUBLE, recvbuf, rowCount[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Mulai waktu yang dibutuhkan
        start_time = MPI_Wtime();

        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < 2 * n; ++j)
            {
                if (j == (i + n))
                {
                    mat[i][j] = 1;
                }
            }
        }
        
        // Partial pivoting
        for (i = n - 1; i > 1; --i)
        {
            if (mat[i - 1][1] < mat[i][1])
            {

                for (j = 0; j < 2 * n; ++j)
                {

                    d = mat[i][j];
                    mat[i][j] = mat[i - 1][j];
                    mat[i - 1][j] = d;
                }
            }
        }

        // Reducing To Diagonal Matrix
        for(i = 0; i < n; ++i) {
            for(j = 0; j < n; ++j) {
                if(j != i) {
                    if(mat[i][i] != 0) {
                        d = mat[j][i] / mat[i][i];
                        for(int k = 0; k < 2*n; ++k) {
                            mat[j][k] -= mat[i][k] * d;
                        }
                    }
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Reducing to unit matrix
        for(i = 0; i < n; ++i) {
            d = mat[i][i];
            for(j = 0; j < 2*n; ++j) {
                mat[i][j] = mat[i][j] / d;
            }
        }        

        // Gather the results from all processes
        MPI_Gather(&mat[0][0], rowCount[rank], MPI_DOUBLE, &mat[0][0], rowCount[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Scatterv(&flattenMatrix(mat, n)[0], rowCount, displs, MPI_DOUBLE, recvbuf, rowCount[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatterv(NULL, rowCount, displs, MPI_DOUBLE, recvbuf, rowCount[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (i = 0; i < rowCount[rank]; ++i)
        {
            cout << "Process " << rank << ", element " << i << ": " << recvbuf[i] << endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        finish_time = MPI_Wtime();

        cout << n << endl;

        cout << "" << endl;

        // Hitung total waktu yang dibutuhkan
        total_time = finish_time - start_time;
        // MPI_Reduce(&total_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Hitung rata-rata waktu yang dibutuhkan
        avg_time = total_time / size;
        std::cout << "Time taken: " << total_time << " seconds" << std::endl;
        std::cout << "Average time taken: " << avg_time << " seconds" << std::endl;
        cout << "" << endl;

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

        for (i = 0; i < n; ++i)
        {
            delete[] mat[i];
        }
        delete[] mat;
    }

    // Deleting the memory allocated

    MPI_Finalize();

    return 0;
}
