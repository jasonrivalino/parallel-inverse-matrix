
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip> // for std::setw

__global__ void partialPivoting(double *mat, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n - 1) {
        if (mat[tid * (2 * n) + 1] < mat[(tid + 1) * (2 * n) + 1]) {
            for (int j = 0; j < 2 * n; ++j) {
                double temp = mat[tid * (2 * n) + j];
                mat[tid * (2 * n) + j] = mat[(tid + 1) * (2 * n) + j];
                mat[(tid + 1) * (2 * n) + j] = temp;
            }
        }
    }
}

__global__ void reduceToDiagonal(double *mat, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (mat[tid * (2 * n) + tid] != 0) {
            for (int i = tid + 1; i < n; ++i) {
                double d = mat[i * (2 * n) + tid] / mat[tid * (2 * n) + tid]; // Use d here
                for (int k = tid; k < 2 * n; ++k) {
                    mat[i * (2 * n) + k] -= mat[tid * (2 * n) + k] * d;
                }
            }
        }
    }
}

__global__ void reduceToUnitMatrix(double *mat, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (mat[tid * (2 * n) + tid] != 0) {
            double d = mat[tid * (2 * n) + tid]; // Use d here
            for (int j = tid; j < 2 * n; ++j) {
                mat[tid * (2 * n) + j] = mat[tid * (2 * n) + j] / d;
            }
        }
    }
}

int main() {
    // Host code
    int n;
    double *mat = nullptr;

    std::cin >> n;

    // Allocate memory for matrix array on CPU
    mat = new double[2 * n * 2 * n];

    // Inputs the coefficients of the matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> mat[i * (2 * n) + j];
        }
    }

    // Print the input matrix
    std::cout << "Input matrix:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 2 * n; ++j) {
            std::cout << std::setw(8) << mat[i * (2 * n) + j] << " ";
        }
        std::cout << std::endl;
    }

    // CUDA memory allocation
    double *d_mat;
    cudaMalloc((void **)&d_mat, (2 * n) * (2 * n) * sizeof(double));
    cudaMemcpy(d_mat, mat, (2 * n) * (2 * n) * sizeof(double), cudaMemcpyHostToDevice);

    // Print the content of d_mat after memory copy
    std::cout << "Content of d_mat after memory copy:" << std::endl;
    for (int i = 0; i < 2 * n; ++i) {
        for (int j = 0; j < 2 * n; ++j) {
            double val;
            cudaMemcpy(&val, &d_mat[i * (2 * n) + j], sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << std::setw(8) << val << " ";
        }
        std::cout << std::endl;
    }

    // Launch CUDA kernels
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

    // Call CUDA kernels for each step of the algorithm
    partialPivoting<<<numBlocks, threadsPerBlock>>>(d_mat, n);
    cudaDeviceSynchronize();

    // Print matrix after partial pivoting
    std::cout << "Matrix after partial pivoting:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << mat[i * (2 * n) + j] << " ";
        }
        std::cout << std::endl;
    }

    reduceToDiagonal<<<numBlocks, threadsPerBlock>>>(d_mat, n);
    cudaDeviceSynchronize();

    // Print matrix after reducing to diagonal
    std::cout << "Matrix after reducing to diagonal:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << mat[i * (2 * n) + j] << " ";
        }
        std::cout << std::endl;
    }

    reduceToUnitMatrix<<<numBlocks, threadsPerBlock>>>(d_mat, n);
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(mat, d_mat, (2 * n) * (2 * n) * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the output matrix
    std::cout << "Output matrix:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << mat[i * (2 * n) + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    delete[] mat;
    cudaFree(d_mat);

    return 0;
}
