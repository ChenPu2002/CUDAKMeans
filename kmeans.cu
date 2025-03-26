#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <fstream> // 用于文件操作
#include <cmath>
#include <cstdlib>
#include <float.h> // 用于 FLT_MAX

// 数据点和簇数的定义
#define NUM_POINTS 1024    // 数据点数量
#define DIMENSIONS 2       // 数据点的维度
#define NUM_CLUSTERS 3     // 簇的数量
#define MAX_ITERATIONS 100 // 最大迭代次数

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                         \
    {                                                                            \
        const cudaError_t error = call;                                          \
        if (error != cudaSuccess)                                                \
        {                                                                        \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "        \
                      << cudaGetErrorString(error) << std::endl;                 \
            exit(1);                                                             \
        }                                                                        \
    }

// 计算点到簇中心的距离的核函数
__device__ float calculateDistance(const float *point, const float *center, int dimensions)
{
    float distance = 0.0f;
    for (int i = 0; i < dimensions; i++)
    {
        float diff = point[i] - center[i];
        distance += diff * diff;
    }
    return sqrtf(distance);
}

// 分配数据点到最近的簇
__global__ void assignClusters(const float *data, const float *centroids, int *clusterAssignments, int numPoints, int dimensions, int numClusters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints)
    {
        float minDistance = FLT_MAX;
        int closestCluster = -1;

        // 遍历所有簇中心，找到最近的
        for (int c = 0; c < numClusters; c++)
        {
            float distance = calculateDistance(&data[idx * dimensions], &centroids[c * dimensions], dimensions);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestCluster = c;
            }
        }

        // 更新簇分配
        clusterAssignments[idx] = closestCluster;
    }
}

// 更新簇中心的核函数
__global__ void updateCentroids(const float *data, float *centroids, const int *clusterAssignments, int *clusterSizes, int numPoints, int dimensions, int numClusters)
{
    extern __shared__ float sharedCentroids[]; // 使用共享内存存储簇中心

    int clusterIdx = blockIdx.x; // 每个簇对应一个块
    int tid = threadIdx.x;

    // 初始化共享内存
    for (int i = tid; i < dimensions; i += blockDim.x)
    {
        sharedCentroids[clusterIdx * dimensions + i] = 0.0f;
    }
    __syncthreads();

    // 累加属于当前簇的数据点
    for (int i = tid; i < numPoints; i += blockDim.x)
    {
        if (clusterAssignments[i] == clusterIdx)
        {
            for (int d = 0; d < dimensions; d++)
            {
                atomicAdd(&sharedCentroids[clusterIdx * dimensions + d], data[i * dimensions + d]);
            }
            atomicAdd(&clusterSizes[clusterIdx], 1);
        }
    }
    __syncthreads();

    // 计算新的簇中心
    for (int i = tid; i < dimensions; i += blockDim.x)
    {
        if (clusterSizes[clusterIdx] > 0)
        {
            centroids[clusterIdx * dimensions + i] = sharedCentroids[clusterIdx * dimensions + i] / clusterSizes[clusterIdx];
        }
    }
}

int main()
{
    std::vector<float> h_data(NUM_POINTS * DIMENSIONS);   // 主机上的数据
    std::vector<float> h_centroids(NUM_CLUSTERS * DIMENSIONS); // 簇中心
    std::vector<int> h_clusterAssignments(NUM_POINTS, 0); // 每个点所属的簇
    std::vector<int> h_clusterSizes(NUM_CLUSTERS, 0);     // 每个簇的数据量

    // 随机初始化数据点
    for (int i = 0; i < NUM_POINTS * DIMENSIONS; i++)
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;

    // 随机初始化簇中心
    for (int i = 0; i < NUM_CLUSTERS * DIMENSIONS; i++)
        h_centroids[i] = static_cast<float>(rand()) / RAND_MAX;

    // 保存随机生成的初始数据点到文件
    std::ofstream dataFile("log/initial_data_points.txt");
    for (int i = 0; i < NUM_POINTS; i++)
    {
        for (int d = 0; d < DIMENSIONS; d++)
        {
            dataFile << h_data[i * DIMENSIONS + d] << " ";
        }
        dataFile << "\n";
    }
    dataFile.close();

    // 保存初始簇中心到文件
    std::ofstream centroidsFile("log/initial_centroids.txt");
    for (int c = 0; c < NUM_CLUSTERS; c++)
    {
        for (int d = 0; d < DIMENSIONS; d++)
        {
            centroidsFile << h_centroids[c * DIMENSIONS + d] << " ";
        }
        centroidsFile << "\n";
    }
    centroidsFile.close();

    // 分配设备内存
    float *d_data, *d_centroids;
    int *d_clusterAssignments, *d_clusterSizes;
    CUDA_CHECK(cudaMalloc(&d_data, NUM_POINTS * DIMENSIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, NUM_CLUSTERS * DIMENSIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clusterAssignments, NUM_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clusterSizes, NUM_CLUSTERS * sizeof(int)));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), NUM_POINTS * DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), NUM_CLUSTERS * DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice));

    // 定义线程和块的大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_POINTS + threadsPerBlock - 1) / threadsPerBlock;

    // 运行 K-Means 算法
    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        assignClusters<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_centroids, d_clusterAssignments, NUM_POINTS, DIMENSIONS, NUM_CLUSTERS);
        CUDA_CHECK(cudaMemset(d_clusterSizes, 0, NUM_CLUSTERS * sizeof(int))); // 重置簇大小
        int sharedMemorySize = NUM_CLUSTERS * DIMENSIONS * sizeof(float);
        updateCentroids<<<NUM_CLUSTERS, threadsPerBlock, sharedMemorySize>>>(d_data, d_centroids, d_clusterAssignments, d_clusterSizes, NUM_POINTS, DIMENSIONS, NUM_CLUSTERS);
    }

    // 将结果复制回主机
    CUDA_CHECK(cudaMemcpy(h_centroids.data(), d_centroids, NUM_CLUSTERS * DIMENSIONS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_clusterAssignments.data(), d_clusterAssignments, NUM_POINTS * sizeof(int), cudaMemcpyDeviceToHost));

    // 保存最终簇中心到文件
    std::ofstream finalCentroidsFile("log/final_centroids.txt");
    for (int c = 0; c < NUM_CLUSTERS; c++)
    {
        for (int d = 0; d < DIMENSIONS; d++)
        {
            finalCentroidsFile << h_centroids[c * DIMENSIONS + d] << " ";
        }
        finalCentroidsFile << "\n";
    }
    finalCentroidsFile.close();

    // 保存每个点的簇分配结果到文件
    std::ofstream clusterAssignmentsFile("log/cluster_assignments.txt");
    for (int i = 0; i < NUM_POINTS; i++)
    {
        clusterAssignmentsFile << h_clusterAssignments[i] << "\n";
    }
    clusterAssignmentsFile.close();

    // 打印最终簇中心
    std::cout << "Final cluster centroids:" << std::endl;
    for (int c = 0; c < NUM_CLUSTERS; c++)
    {
        std::cout << "Cluster " << c << ": ";
        for (int d = 0; d < DIMENSIONS; d++)
            std::cout << h_centroids[c * DIMENSIONS + d] << " ";
        std::cout << std::endl;
    }

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_clusterAssignments));
    CUDA_CHECK(cudaFree(d_clusterSizes));

    return 0;
}