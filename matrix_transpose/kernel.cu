
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include<stdlib.h>
#include<time.h>
#include<chrono>
#include<string.h>

#define MATRIX_SCALE 1024

int g_tile_size = 32;
int g_handle_size = 4;

__constant__ int constant_tile_size;
int src_matrix[MATRIX_SCALE][MATRIX_SCALE];
int dest_matrix[MATRIX_SCALE][MATRIX_SCALE];
int dest_matrix_cpu_transpose[MATRIX_SCALE][MATRIX_SCALE];
/// <summary>
/// 矩阵拷贝的kernel
/// </summary>
/// <param name="input_dest">目的矩阵</param>
/// <param name="input_src">源矩阵</param>
/// <returns></returns>
__global__ void cuda_copy_kernel(int* input_dest,int* input_src) {//矩阵拷贝（baseline）
    int x = blockIdx.x * constant_tile_size + threadIdx.x;
    int y = blockIdx.y * constant_tile_size + threadIdx.y;
    int matrix_width = constant_tile_size * blockDim.x;
    for (int i = 0; i < constant_tile_size; i+=blockDim.y) {
        input_dest[(y + i) * matrix_width + x]= input_src[(y + i) * matrix_width + x] ;
    }
    
}
/// <summary>
/// cuda实现矩阵的拷贝
/// </summary>
/// <param name="input_dest">目的矩阵</param>
/// <param name="input_src">源矩阵</param>
/// <param name="number">总共的矩阵元素的个数</param>
void cuda_copy_matrix(int* input_dest,int* input_src,unsigned int number){
    int* matrix_input = nullptr;
    int* matrix_output = nullptr;
    /*分配内存*/
    cudaMalloc((void**)&matrix_input, sizeof(int) * number);
    cudaMalloc((void**)&matrix_output, sizeof(int) * number);
    /*拷贝数据*/
    cudaMemcpy(matrix_input, input_src, sizeof(int) * number, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constant_tile_size, &g_tile_size, sizeof(int),0,cudaMemcpyHostToDevice);
    /*插入cude_event计时*/
    cudaEvent_t start,stop;
    float ElpausedTime=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    /*准备使用kernel*/
    dim3 gridDim(32, 32, 1);
    dim3 blockDim(g_tile_size, g_tile_size/g_handle_size, 1);
    cuda_copy_kernel << <gridDim, blockDim>> > (matrix_output, matrix_input);
    /*计算时间*/
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ElpausedTime, start, stop);
    std::cout << "copy消耗时间为" << ElpausedTime <<"ms" << std::endl;
    /*数据同步*/
    cudaDeviceSynchronize();
    /*返回数据*/
    cudaMemcpy(input_dest, matrix_output, sizeof(int) * number, cudaMemcpyDeviceToHost);
    cudaFree(matrix_input);
    cudaFree(matrix_output);

}
/// <summary>
/// 随机生成元素
/// </summary>
/// <param name="input_matrix">目的矩阵</param>
/// <param name="number">矩阵的元素个数</param>
void random_generate(int * input_matrix,unsigned int number) {
    srand(time(NULL));
    for (int i = 0; i < number; i++) {
        input_matrix[i] = rand() % 100;
    }
}
/// <summary>
/// 打印矩阵
/// </summary>
/// <param name="input_matrix">目的矩阵</param>
/// <param name="number">矩阵元素的个数</param>
void print_matrix(const int* input_matrix, unsigned int number) {
    srand(time(NULL));
    for (int i = 0; i < number; i++) {
        std::cout << input_matrix[i]<<std::endl;
    }
}
/// <summary>
/// cpu版本的矩阵转置，并计时
/// </summary>
/// <param name="input_dest_matrix">目标矩阵</param>
/// <param name="input_src_matrix">源矩阵</param>
/// <param name="rows">行数</param>
/// <param name="columns">列数</param>
void transpose_matrix_cpu(int* input_dest_matrix, const int* input_src_matrix, unsigned int rows,unsigned int columns) {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < columns; k++) {
            input_dest_matrix[i*rows+k] = input_src_matrix[k*columns+i];
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "cpu transpose consume:" << elapsed_time << "ms" << std::endl;
}
/// <summary>
/// 检查矩阵是不是一样
/// </summary>
/// <param name="input_dest_matrix">目标矩阵</param>
/// <param name="input_src_matrix">源矩阵</param>
/// <param name="number">总个数</param>
void check_matrix(const int* input_dest_matrix, const int* input_src_matrix,const unsigned int number) {
    //比较
    if (std::memcmp(input_dest_matrix, input_src_matrix, number) == 0) {
        std::cout << "matrix check sucessful" << std::endl;
        return;
    }
    //纠错
    for (int i = 0; i < number; i++) {
        if (input_dest_matrix[i] != input_src_matrix[i]) {
            std::cout << "erro in " << i << std::endl;
            return;
        }
    }
}

int main()
{
    random_generate(src_matrix[0], MATRIX_SCALE * MATRIX_SCALE);
    cuda_copy_matrix(dest_matrix[0], src_matrix[0], MATRIX_SCALE * MATRIX_SCALE);//cuda矩阵拷贝
    check_matrix(dest_matrix[0], src_matrix[0], MATRIX_SCALE * MATRIX_SCALE);
    transpose_matrix_cpu(dest_matrix_cpu_transpose[0], src_matrix[0], MATRIX_SCALE, MATRIX_SCALE);//cpu版本transpose

    return 0;
}


