#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>

__global__ void makeRightHandSideIdentity(double *mat, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        for (int j = n; j < 2 * n; ++j) {
            if (tid == j - n) {
                mat[tid * (2 * n) + j] = 1.0;
            } else {
                mat[tid * (2 * n) + j] = 0.0;
            }
        }
    }
}

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

void printMatrix(double *mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 2 * n; ++j) {
            std::cout << std::setw(8) << mat[i * (2 * n) + j] << " ";
        }
        std::cout << std::endl;
    }
}

void printDeviceMatrix(double *d_mat, int n) {
    double *temp_mat = new double[2 * n * 2 * n];
    cudaMemcpy(temp_mat, d_mat, (2 * n) * (2 * n) * sizeof(double), cudaMemcpyDeviceToHost);
    printMatrix(temp_mat, n);
    delete[] temp_mat;
}

int main() {
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
    // std::cout << "Input matrix:" << std::endl;
    // printMatrix(mat, n);

    // CUDA memory allocation
    double *d_mat;
    cudaMalloc((void **)&d_mat, (2 * n) * (2 * n) * sizeof(double));
    cudaMemcpy(d_mat, mat, (2 * n) * (2 * n) * sizeof(double), cudaMemcpyHostToDevice);

    // Print the content of d_mat after memory copy
    // std::cout << "Content of d_mat after memory copy:" << std::endl;
    // printDeviceMatrix(d_mat, n);

    // Launch CUDA kernels
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

    // Call CUDA kernel to make right hand side identity
    makeRightHandSideIdentity<<<numBlocks, threadsPerBlock>>>(d_mat, n);
    cudaDeviceSynchronize();

    std::cout << "Content of d_mat after making right hand side identity:" << std::endl;
    printDeviceMatrix(d_mat, n);

    partialPivoting<<<numBlocks, threadsPerBlock>>>(d_mat, n);
    cudaDeviceSynchronize();

    // std::cout << "Content of d_mat after partial pivoting:" << std::endl;
    // printDeviceMatrix(d_mat, n);

    reduceToDiagonal<<<numBlocks, threadsPerBlock>>>(d_mat, n);
    cudaDeviceSynchronize();

    // std::cout << "Content of d_mat after reduce to diagonal:" << std::endl;
    // printDeviceMatrix(d_mat, n);

    reduceToUnitMatrix<<<numBlocks, threadsPerBlock>>>(d_mat, n);
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(mat, d_mat, (2 * n) * (2 * n) * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the output matrix
    // std::cout << "Output matrix:" << std::endl;
    // printMatrix(mat, n);

    // Free memory
    delete[] mat;
    cudaFree(d_mat);

    return 0;
}