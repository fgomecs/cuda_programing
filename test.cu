// The following include brings in the standard C++ library that helps us with input/output (e.g., printing messages)
#include <iostream>

// This header file is needed for CUDA-specific functionality, like launching kernels (GPU functions) and using GPU resources.
#include <cuda_runtime.h>

// This is the **CUDA kernel**. CUDA code runs on the GPU. Any function declared with `__global__` is a CUDA kernel.
// `__global__` tells the compiler that this function should run on the GPU, not the CPU.
__global__ void helloWorld() {
    // This line uses the `printf` function to print a message from the GPU.
    // `threadIdx.x` is a special variable in CUDA that gives each thread its unique ID within a block.
    // In this case, each thread prints its own ID (0, 1, 2, 3, etc.).
    printf("Hello, World from thread %d!\n", threadIdx.x);
}

// This is the **main function** that runs on the **CPU** (not on the GPU).
int main() {
    // Here, we're launching the kernel (GPU function) we defined earlier.
    // `helloWorld<<<1, 5>>>();` launches the kernel with 1 block of 5 threads.
    // In CUDA, you define the number of blocks and threads for your kernel execution.
    // `<<<1, 5>>>` means 1 block with 5 threads. Each thread runs `helloWorld`.
    helloWorld<<<1, 5>>>();

    // `cudaDeviceSynchronize()` makes sure that the CPU waits until all threads on the GPU have finished.
    // Without this line, the CPU might finish running the program before the GPU is done printing messages.
    // This is important because the GPU runs in parallel, and we need to synchronize the CPU and GPU.
    cudaDeviceSynchronize();

    // The program ends here, returning 0 to indicate that everything ran successfully.
    return 0;
}
