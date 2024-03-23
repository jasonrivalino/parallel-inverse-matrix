#include <iostream>
#include <mpi.h>

using namespace std;

int *distributeRow(int totalElement, int colSize, int commSize)
{
    int rowSize = totalElement / colSize;
    int base = rowSize / commSize;
    int remainder = rowSize % commSize;

    int *result = new int[commSize];
    for (int i = 0; i < commSize; i++)
    {
        result[i] = base;
        if (i < remainder)
        {
            result[i]++;
        }
    }
    return result;
}

double *flattenMatrix(double **matrix, int size)
{
    double *flatMatrix = new double[size * size * 2];
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < 2 * size; j++)
        {
            flatMatrix[2*size*i+j] = matrix[i][j];
        }
    }
    return flatMatrix;
}

double **shapeMatrix(double *matrix, int size)
{
    double **shapedMatrix = new double*[size];
    for (int i = 0; i < size*size*2; i++) {
        if (i<size)
        {
            shapedMatrix[i] = new double[2*size];
        }
        shapedMatrix[i/(2*size)][i%(2*size)] = matrix[i];
    }
    return shapedMatrix;
}

int main(int argc, char *argv[])
{
    int rank, size;
    double start_time, finish_time, total_time, avg_time;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int i = 0, j = 0;
    int n;
    double **mat = NULL;
    double d = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        cin >> n;
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *rowCount = distributeRow(n * n * 2, n * 2, size);
    double *recvbuf = new double[rowCount[rank] * n * 2];
    int *displs = new int[size];
    displs[0] = 0;
    for (i = 0; i < size; i++)
    {
        rowCount[i] *= n * 2;
        if (i != 0)
        {
            displs[i] = displs[i - 1] + rowCount[i - 1];
        }
    }
    if (rank == 0)
    {
        mat = new double *[n];
        for (i = 0; i < n; ++i)
        {
            mat[i] = new double[2 * n];
        }
        // Inputs the coefficients of the matrix
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < n; ++j)
            {
                cin >> mat[i][j];
            }
        }
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
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < 2 * n; ++j)
            {
                cout << mat[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
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

        // Hitung total waktu yang dibutuhkan
        total_time = finish_time - start_time;
        // MPI_Reduce(&total_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Hitung rata-rata waktu yang dibutuhkan
        avg_time = total_time / size;
        std::cout << "Time taken: " << total_time << " seconds" << std::endl;
        std::cout << "Average time taken: " << avg_time << " seconds" << std::endl;
        cout << "" << endl;

        for (i = 0; i < n; ++i)
        {
            for (j = n; j < 2 * n; ++j)
            {
                cout << mat[i][j] << " ";
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
