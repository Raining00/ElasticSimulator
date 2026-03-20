//#include "math/matrix.hpp"
//#include "math/Vector.hpp"
//#include <iostream>
//
///**
// * All about dDs / dx
// * vertex1 of tet:
// * dDs / dx0 = [-1, -1, -1,      dDs / dx1 = [ 0,  0,  0       dDs / dx2 = [ 0,  0,  0
// *                0,  0,  0,                    -1, -1, -1                      0,  0,  0,
// *                0,  0,  0]                     0,  0,  0]                    -1, -1, -1]
// * vertex2 of tet:
// * dDs / dx3 = [ 1, 0, 0,        dDs / dx4 = [0, 0, 0,         dDs / dx5 = [0, 0, 0,
// *               0, 0, 0,                     1, 0, 0,                      0, 0, 0,
// *               0, 0, 0]                     0, 0, 0]                      1, 0, 0]
// * vertex3 of tet:
// * dDs_dx6 = [0, 1, 0,           dDs / dx7 = [0, 0, 0,         dDs / dx8 = [0, 0, 0,
// *            0, 0, 0,                        0, 1, 0,                      0 ,0, 0,
// *            0, 0, 0]                        0, 0, 0]                      0, 1, 0]
// * vertex4 of tet:
// * dDs / dx9 = [0, 0, 1          dDs / dx10 = [0, 0, 0,        dDs / dx11 = [0, 0, 0,
// *              0, 0, 0,                       0, 0, 1,                      0, 0, 0,
// *              0, 0, 0]                       0, 0, 0]                      0, 0, 1]
// * 
// */
//
///**
//*-8.42231 -2.64442 -1.49155
//3.38562 1.98684 7.69219
//3.77723 7.33149 0.81639
//*/
//
//int main()
//{
//	mat3<double> dDs_dx[12];
//	mat3<double> dF_dx[12];
//	mat3<double> Dm_inv(-8.42231, -2.64442, -1.49155,
//						3.38562, 1.98684, 7.69219,
//						3.77723, 7.33149, 0.81639);
//	mat3<double> Dm_invT = mat3<double>::transpose(Dm_inv);
//
//	#pragma unroll
//	for (int i = 0; i < 12; i++)
//	{
//		dDs_dx[i] = mat3<double>(double(0));
//		dF_dx[i] = mat3<double>(double(0));
//	}
//
//	#pragma unroll
//	for (int i = 1; i < 4; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			dDs_dx[i * 3 + j](j, (i - 1)) = 1;
//	}
//
//	#pragma unroll
//	for (int i = 0; i < 3; i++)
//		dDs_dx[i] = static_cast<double>(-1) * (dDs_dx[i + 3] + dDs_dx[i + 6] + dDs_dx[i + 9]);
//
//	#pragma unroll
//	for (int i = 0; i < 12; i++)
//	{
//		dF_dx[i] = dDs_dx[i] * Dm_inv;
//	}
//
//	#pragma unroll
//	for (int i = 0; i < 12; i++)
//	{
//		mat3<double> tmp_mat = dDs_dx[i];
//		printf("dDs_dx%i \n", i);
//		for (int j = 0; j < 3; j++)
//		{
//			for (int k = 0; k < 3; k++)
//				std::cout << tmp_mat(j, k) << ' ';
//			std::cout << std::endl;
//		}
//	}
//
//	#pragma unroll
//	for (int i = 0; i < 12; i++)
//	{
//		mat3<double> tmp_mat = dF_dx[i];
//		printf("dF_dx%i \n", i);
//		for (int j = 0; j < 3; j++)
//		{
//			for (int k = 0; k < 3; k++)
//				std::cout << tmp_mat(j, k) << ' ';
//			std::cout << std::endl;
//		}
//	}
//
//	std::cout << std::endl << "method2: " << std::endl;
//	Vec3d g[4];
//	g[1] = Dm_invT.column(0);
//	g[2] = Dm_invT.column(1);
//	g[3] = Dm_invT.column(2);
//	g[0] = -(g[1] + g[2] + g[3]);
//
//	#pragma unroll
//	for (int a = 0; a < 4; ++a) {
//	#pragma unroll
//		for (int c = 0; c < 3; ++c) {
//			mat3<double> dF(double(0));
//			for (int j = 0; j < 3; ++j) {
//				dF(c, j) = g[a][j];
//			}
//			dF_dx[a * 3 + c] = dF;
//		}
//	}
//
//	#pragma unroll
//	for (int i = 0; i < 12; i++)
//	{
//		mat3<double> tmp_mat = dF_dx[i];
//		printf("dF_dx%i \n", i);
//		for (int j = 0; j < 3; j++)
//		{
//			for (int k = 0; k < 3; k++)
//				std::cout << tmp_mat(j, k) << ' ';
//			std::cout << std::endl;
//		}
//	}
//	return 0;
//}

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <vector>

int main() {
    // 初始化 cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // 矩阵大小和非零元素
    int rows = 3, cols = 3, nnz = 4;

    // CSR 格式数据（GPU 内存）
    int h_csrOffsets[] = { 0, 1, 3, 4 };   // 行偏移
    int h_csrColumns[] = { 0, 0, 2, 1 };   // 列索引
    float h_csrValues[] = { 1.0f, 2.0f, 3.0f, 4.0f }; // 非零值

    int* d_csrOffsets, * d_csrColumns;
    float* d_csrValues;
    cudaMalloc((void**)&d_csrOffsets, (rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_csrColumns, nnz * sizeof(int));
    cudaMalloc((void**)&d_csrValues, nnz * sizeof(float));

    cudaMemcpy(d_csrOffsets, h_csrOffsets, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColumns, h_csrColumns, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValues, h_csrValues, nnz * sizeof(float), cudaMemcpyHostToDevice);

    // 创建 CSR 矩阵描述符
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA,
        rows, cols, nnz,
        d_csrOffsets, d_csrColumns, d_csrValues,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // 创建稠密向量 x 和 y
    float h_x[] = { 1.0f, 2.0f, 3.0f };
    float h_y[3] = { 0.0f, 0.0f, 0.0f };
    float* d_x, * d_y;
    cudaMalloc((void**)&d_x, cols * sizeof(float));
    cudaMalloc((void**)&d_y, rows * sizeof(float));
    cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F);

    // SpMV 参数
    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    void* dBuffer = NULL;

    cusparseSpMV_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // 第一次 SpMV
    cusparseSpMV(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        dBuffer);

    cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "第一次结果: ";
    for (int i = 0; i < rows; i++) std::cout << h_y[i] << " ";
    std::cout << std::endl;

    // 修改矩阵数值（例如把所有值加 10）
    for (int i = 0; i < nnz; i++) h_csrValues[i] += 10.0f;
    cudaMemcpy(d_csrValues, h_csrValues, nnz * sizeof(float), cudaMemcpyHostToDevice);

    // 第二次 SpMV
    cusparseSpMV(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        dBuffer);

    cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "修改后结果: ";
    for (int i = 0; i < rows; i++) std::cout << h_y[i] << " ";
    std::cout << std::endl;

    // 清理资源
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaFree(d_csrOffsets);
    cudaFree(d_csrColumns);
    cudaFree(d_csrValues);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(dBuffer);

    return 0;
}
