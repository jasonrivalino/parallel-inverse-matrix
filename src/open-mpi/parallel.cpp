#include <iostream>
#include <mpi.h>
#include <cmath>

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

int index1d(int row, int col, int size)
{
    return row * (2 * size) + col;
}

int main(int argc, char *argv[])
{
    int rank, size;
    int i, j;
    double d;
    int startLocal, endLocal;
    double start_time, finish_time, total_time, avg_time;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    double *flatMat = NULL;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        cin >> n;
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

    startLocal = displs[rank] / (2 * n);
    endLocal = (displs[rank] + rowCount[rank]) / (2 * n);

    flatMat = new double[n * n * 2];

    if (rank == 0)
    {

        // Inputs the coefficients of the matrix
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < 2 * n; ++j)
            {
                if (j < n)
                {
                    cin >> flatMat[index1d(i, j, n)];
                }
                else if (j == (i + n))
                {
                    flatMat[index1d(i, j, n)] = 1;
                }
                else
                {
                    flatMat[index1d(i, j, n)] = 0;
                }
            }
        }

        start_time = MPI_Wtime();

    }

    MPI_Bcast(flatMat, 2 * n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //
    for (i = 0; i < n; ++i)
    {
        for (j = startLocal; j < endLocal; ++j)
        {
            if (j != i)
            {
                if (flatMat[index1d(i, i, n)] != 0)
                {
                    d = flatMat[index1d(j, i, n)] / flatMat[index1d(i, i, n)];
                    for (int k = 0; k < 2 * n; ++k)
                    {
                        flatMat[index1d(j, k, n)] -= flatMat[index1d(i, k, n)] * d;
                    }
                }
            }
        }
        MPI_Allgather(&flatMat[displs[rank]], rowCount[rank], MPI_DOUBLE, flatMat, rowCount[rank], MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // Reducing to unit matrix
    for (i = startLocal; i < endLocal; ++i)
    {
        d = flatMat[index1d(i, i, n)];
        for (j = 0; j < 2 * n; ++j)
        {
            flatMat[index1d(i, j, n)] = flatMat[index1d(i, j, n)] / d;
        }
    }

    MPI_Allgather(MPI_IN_PLACE, rowCount[rank], MPI_DOUBLE, flatMat, rowCount[rank], MPI_DOUBLE, MPI_COMM_WORLD);

    // print the result
    if (rank == 0)
    {
        finish_time = MPI_Wtime();

        cout << n << endl;

        cout << "" << endl;

        // Hitung total waktu yang dibutuhkan
        total_time = finish_time - start_time;

        // Hitung rata-rata waktu yang dibutuhkan
        avg_time = total_time / size;
        std::cout << "Time taken: " << total_time << " seconds" << std::endl;
        std::cout << "Average time taken: " << avg_time << " seconds" << std::endl;
        cout << "" << endl;

        for (i = 0; i < n; ++i)
        {
            for (j = n; j < 2 * n; ++j)
            {
                cout << flatMat[index1d(i, j, n)] << " ";
            }
            cout << endl;
        }
    }

    // Deleting the memory allocated
    delete[] flatMat;
    delete[] recvbuf;
    delete[] rowCount;
    delete[] displs;
    MPI_Finalize();

    return 0;
}