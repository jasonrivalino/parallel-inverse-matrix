/**
 * Inverse of a Matrix
 * Gauss-Jordan Elimination
 * Source: https://github.com/peterabraham/Gauss-Jordan-Elimination/blob/master/GaussJordanElimination.cpp
 **/

#include <iostream>
#include <ctime>
#include <cmath>
using namespace std;

int main()
{
    clock_t start, end;
    int i = 0, j = 0, k = 0, n = 0;
    double **mat = NULL;
    double d = 0.0;


    cin >> n;

    // Allocating memory for matrix array
    mat = new double *[2 * n];
    for (i = 0; i < 2 * n; ++i)
    {
        mat[i] = new double[2 * n]();
    }

    // Inputs the coefficients of the matrix
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            cin >> mat[i][j];
        }
    }

    // Mulai menghitung waktu
    start = clock();

    // Initializing Right-hand side to identity matrix
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
    for (i = n; i > 1; --i)
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
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            if (j != i)
            {
                if (mat[i][i] != 0)
                {
                    d = mat[j][i] / mat[i][i];
                    for (k = 0; k < 2 * n; ++k)
                    {
                        mat[j][k] -= mat[i][k] * d;
                    }
                }
            }
        }
    }

    // Reducing To Unit Matrix
    for (i = 0; i < n; ++i)
    {
        if (mat[i][i] != 0)
        {
            d = mat[i][i];
            for (j = 0; j < 2 * n; ++j)
            {
                mat[i][j] = mat[i][j] / d;
            }
        }
    }

    end = clock();
    
    cout << n << endl;

    cout << "" << endl;


    // Hitung total waktu yang dibutuhkan
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Hitung rata-rata waktu yang dibutuhkan
    double avg_time = total_time;

    cout << "Total time: " << total_time << " seconds" << endl;
    cout << "Average time: " << avg_time << " seconds" << endl;
    cout << "" << endl;

    for(i = 0; i < n; ++i)
    {
        for(j = n; j < 2*n; ++j)
        {
            if (abs(mat[i][j]) < 1e-5) {
                cout << "0 ";
            } else {
                cout << mat[i][j] << " ";
            }
        }
        cout << endl;
    }

    // Deleting the memory allocated
    for (i = 0; i < n; ++i)
    {
        delete[] mat[i];
    }
    delete[] mat;

    return 0;
}