#include <iostream>

void VecAdd(float* A, float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
    {

        C[i] = A[i] + B[i] + i*4 - i/5 + i*i/1000;
    }
}

int main()
{
    int N = 10000000;

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

    // Perform vector addition on the CPU
    VecAdd(h_A, h_B, h_C, N);

    // Print the result
    for (int i = 0; i < N; i++)
    {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
