#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip> // for std::setw


__constant__ int sqrtThreadsPerBlock;

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
        if (mat[tid * (2 * n) + 1] > mat[(tid + 1) * (2 * n) + 1]) {
            for (int j = 0; j < 2 * n; ++j) {
                double temp = mat[tid * (2 * n) + j];
                mat[tid * (2 * n) + j] = mat[(tid + 1) * (2 * n) + j];
                mat[(tid + 1) * (2 * n) + j] = temp;
            }
        }
    }
}

__global__ void reduceToDiagonal(double *mat, int n, int currentRow) {
    __shared__ double d;
    int tid = (blockIdx.y * sqrtThreadsPerBlock * sqrtThreadsPerBlock) + (threadIdx.x * sqrtThreadsPerBlock + threadIdx.y);
    if (blockIdx.x != currentRow) {
        if (threadIdx.x + threadIdx.y == 0)
        {
            d = mat[blockIdx.x * 2*n + currentRow] / mat[currentRow * 2*n + currentRow];
        }
        __syncthreads();
        mat[blockIdx.x * 2*n + tid] -= mat[currentRow * 2*n + tid] * d;
    }
}

__global__ void reduceToUnitMatrix(double *mat, int n) {
    __shared__ double d;
    int tid = (blockIdx.y * sqrtThreadsPerBlock * sqrtThreadsPerBlock) + (threadIdx.x * sqrtThreadsPerBlock + threadIdx.y);
    if (mat[blockIdx.x * 2*n + blockIdx.x] != 0) {
        if (threadIdx.x + threadIdx.y == 0)
        {
            d = mat[blockIdx.x * 2*n + blockIdx.x];
        }
        __syncthreads();
        mat[blockIdx.x * 2*n + tid] /= d;
    }
}

void printMatrix(double *mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 2 * n; ++j) {
            if (std::abs(mat[i * (2 * n) + j]) < std::exp(-5))
                std::cout << std::setw(8) << 0 << " ";
            else
                std::cout << std::setw(8) << mat[i * (2 * n) + j] << " ";
        }
        std::cout << std::endl;
    }
}

void printResultMatrix(double *mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = n; j < 2 * n; ++j) {
            if (std::abs(mat[i * (2 * n) + j]) < std::exp(-5))
                std::cout << std::setw(8) << 0 << " ";
            else
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

    // CUDA memory allocation
    double *d_mat;
    cudaMalloc((void **)&d_mat, (2 * n) * (2 * n) * sizeof(double));
    cudaMemcpy(d_mat, mat, (2 * n) * (2 * n) * sizeof(double), cudaMemcpyHostToDevice);

    // Launch CUDA kernels
    int tpb = static_cast<int>(sqrt(n/2));
    cudaMemcpyToSymbol(sqrtThreadsPerBlock, &tpb, sizeof(tpb));
    dim3 threadsPerBlock(tpb, tpb);
    dim3 numBlocks(n, 4);

    // Call CUDA kernel to make right hand side identity
    makeRightHandSideIdentity<<<numBlocks, threadsPerBlock>>>(d_mat, n);
    cudaDeviceSynchronize();

    // Partial Pivoting
    for (int i = n; i > 1; --i) {
        if (mat[2*n*(i - 1)+1] < mat[2*n*i+1]) {
            for (int j = 0; j < 2 * n; ++j) {
                double d = mat[2*n*i+j];
                mat[2*n*i+j] = mat[2*n*(i - 1)+j];
                mat[2*n*(i - 1)+j] = d;
            }
        }
    }

    // Reduce to Diagonal Matrix
    for (int i = 0; i < n; ++i){
        if (mat[i*2*n + i] != 0){
            reduceToDiagonal<<<numBlocks, threadsPerBlock>>>(d_mat, n, i);
        }
        cudaDeviceSynchronize();
    }

    // Reduce to Unit Matrix
    reduceToUnitMatrix<<<numBlocks, threadsPerBlock>>>(d_mat, n);
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(mat, d_mat, (2 * n) * (2 * n) * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the output matrix
    std::cout << "Output matrix:" << std::endl;
    printResultMatrix(mat, n);

    // Free memory
    delete[] mat;
    cudaFree(d_mat);

    return 0;
}