#include <iostream>

__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = threadIdx.x;
	// printf("Thread ID is %d\n", i);
    if (i < N)
        C[i] = A[i] + B[i] + i*4 - i/5 + i*i/1000;
}

int main()
{
    int N = 10000000;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; i++)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }


    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy host arrays to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel invocation with N threads
    VecAdd<<<1000, 10000>>>(d_A, d_B, d_C, N);

    // Copy the result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; i++)
    {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}