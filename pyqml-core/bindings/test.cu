// #include <cuda_runtime.h>
#include <stdio.h>

// This CUDA kernel is a simple smoke test that prints a message from the GPU device to confirm
// that a basic kernel launch path is available in this environment.
__global__ void hello_kernel()
{
    printf("Hello from GPU\n");
}

// This entry point launches the test kernel and reports any launch or synchronization errors so
// the CUDA setup can be checked quickly when developing GPU support.
int main()
{
    printf("Starting...\n");

    hello_kernel<<<1, 1>>>();

    // 🔥 check launch error FIRST
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess)
    {
        printf("Launch error: %s\n", cudaGetErrorString(err1));
    }

    // 🔥 then sync
    cudaError_t err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess)
    {
        printf("Sync error: %s\n", cudaGetErrorString(err2));
    }

    printf("CPU done\n");
    return 0;
}
