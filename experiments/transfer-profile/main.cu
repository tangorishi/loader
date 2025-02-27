#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

// Function to perform the memory transfer and measure bandwidth
double measureTransfer(size_t dataSize, int numStreams) {
    float *h_data; // host data
    float *d_data; // device data
    std::vector<cudaStream_t> streams(numStreams); // CUDA streams

    // Allocate host memory
    cudaMallocHost((void**)&h_data, dataSize * sizeof(float));

    // Initialize host data (optional, for realistic data transfer)
    for(size_t i = 0; i < dataSize; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_data, dataSize * sizeof(float));

    // Initialize streams
    for(int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Calculate chunk size per stream
    size_t chunkSize = dataSize / numStreams;

    // Copy data from host to device using streams
    for(int i = 0; i < numStreams; ++i) {
        cudaMemcpyAsync(d_data + i * chunkSize, h_data + i * chunkSize, chunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    }

    // Synchronize streams
    for(int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Stop timer
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = stop - start;
    double bandwidth = dataSize * sizeof(float) / (1 << 30) / elapsed.count(); // Bandwidth in GB/s

    // Clean up
    for(int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_data);
    cudaFree(d_data);

    return bandwidth;
}

int main() {
    // Example data sizes (256MB, 512MB, 1GB, 2GB, 4GB)
    std::vector<size_t> dataSizes = {1 << 28, 1 << 29, 1 << 30, ((size_t) 1) << 31, ((size_t) 1) << 32};
    std::vector<int> streamCounts = {1, 2, 4, 8, 16}; // Example stream counts

    for (auto& dataSize : dataSizes) {
        double maxBandwidth = 0;
        int optimalStreams = 0;

        for (auto& numStreams : streamCounts) {
            double bandwidth = measureTransfer(dataSize, numStreams);
            std::cout << "Data Size: " << dataSize << " Bytes, Streams: " << numStreams << ", Bandwidth: " << bandwidth << " GB/s\n";

            if (bandwidth > maxBandwidth) {
                maxBandwidth = bandwidth;
                optimalStreams = numStreams;
            }
        }

        std::cout << "Optimal for " << dataSize << " Bytes: " << optimalStreams << " streams, " << maxBandwidth << " GB/s\n";
    }

    return 0;
}
