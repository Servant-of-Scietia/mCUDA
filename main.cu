#include <iostream>
#include <math.h>
#include <chrono>

void add(int n, float* a, float* b)
{
    for (int i = 0; i < n; i++)
    {
        b[i] = a[i] + b[i];
    }
}

int main()
{
    int N = 1<<30;

    float* x = new float[N];
    float* y = new float[N];

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    add(N, x, y);

    auto end = std::chrono::high_resolution_clock::now();
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Duration: " << duration.count() << " ms" << std::endl;

    delete [] x;
    delete [] y;

    return 0;
}


