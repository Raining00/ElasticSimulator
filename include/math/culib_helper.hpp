#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#ifndef CHECK_CUDA(func)
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
#endif

#ifndef CHECK_CUSPARSE(func)
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
#endif

#ifndef CHECK_CUBLAS(func)
#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
#endif

template <typename T>
struct CudaTypeTraits;

template <>
struct CudaTypeTraits<float>
{
    static constexpr cudaDataType value = CUDA_R_32F;
};

template <>
struct CudaTypeTraits<double>
{
    static constexpr cudaDataType value = CUDA_R_64F;
};

template <typename T>
struct CublasApi;

template <>
struct CublasApi<float>
{
    static cublasStatus_t dot(
        cublasHandle_t handle,
        int n,
        const float* x,
        int incx,
        const float* y,
        int incy,
        float* result)
    {
        return cublasSdot(handle, n, x, incx, y, incy, result);
    }

    static cublasStatus_t axpy(
        cublasHandle_t handle,
        int n,
        const float* alpha,
        const float* x,
        int incx,
        float* y,
        int incy)
    {
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
    }

    static cublasStatus_t axbp(
        cublasHandle_t handle,
        int n,
        const float* alpha,
        const float* x,
        int incx,
        float* y,
        int incy)
    {
        return axpy(handle, n, alpha, x, incx, y, incy);
    }

    static cublasStatus_t scale(
        cublasHandle_t handle,
        int n,
        const float* alpha,
        float* x,
        int incx)
    {
        return cublasSscal(handle, n, alpha, x, incx);
    }

    static cublasStatus_t copy(
        cublasHandle_t handle,
        int n,
        const float* x,
        int incx,
        float* y,
        int incy)
    {
        return cublasScopy(handle, n, x, incx, y, incy);
    }
};

template <>
struct CublasApi<double>
{
    static cublasStatus_t dot(
        cublasHandle_t handle,
        int n,
        const double* x,
        int incx,
        const double* y,
        int incy,
        double* result)
    {
        return cublasDdot(handle, n, x, incx, y, incy, result);
    }

    static cublasStatus_t axpy(
        cublasHandle_t handle,
        int n,
        const double* alpha,
        const double* x,
        int incx,
        double* y,
        int incy)
    {
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
    }

    static cublasStatus_t axbp(
        cublasHandle_t handle,
        int n,
        const double* alpha,
        const double* x,
        int incx,
        double* y,
        int incy)
    {
        return axpy(handle, n, alpha, x, incx, y, incy);
    }

    static cublasStatus_t scale(
        cublasHandle_t handle,
        int n,
        const double* alpha,
        double* x,
        int incx)
    {
        return cublasDscal(handle, n, alpha, x, incx);
    }

    static cublasStatus_t copy(
        cublasHandle_t handle,
        int n,
        const double* x,
        int incx,
        double* y,
        int incy)
    {
        return cublasDcopy(handle, n, x, incx, y, incy);
    }
};

template <typename T>
struct CusparseApi;

template <>
struct CusparseApi<float>
{
    static constexpr cudaDataType value_type = CudaTypeTraits<float>::value;

    static cusparseStatus_t createCsr(
        cusparseSpMatDescr_t* matA,
        int64_t rows,
        int64_t cols,
        int64_t nnz,
        int* rowOffsets,
        int* colIndices,
        float* values)
    {
        return cusparseCreateCsr(
            matA,
            rows,
            cols,
            nnz,
            rowOffsets,
            colIndices,
            values,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            value_type);
    }

    static cusparseStatus_t CreateCsr(
        cusparseSpMatDescr_t* matA,
        int64_t rows,
        int64_t cols,
        int64_t nnz,
        int* rowOffsets,
        int* colIndices,
        float* values)
    {
        return createCsr(matA, rows, cols, nnz, rowOffsets, colIndices, values);
    }

    static cusparseStatus_t CreateDnVec(
        cusparseDnVecDescr_t* vec,
        int64_t size,
        float* values)
    {
        return cusparseCreateDnVec(vec, size, values, value_type);
    }

    static cusparseStatus_t CreateDnMat(
        cusparseDnMatDescr_t* mat,
        int64_t rows,
        int64_t cols,
        int64_t ld,
        float* values,
        cusparseOrder_t order = CUSPARSE_ORDER_ROW)
    {
        return cusparseCreateDnMat(mat, rows, cols, ld, values, value_type, order);
    }

    static cusparseStatus_t spmvBufferSize(
        cusparseHandle_t handle,
        cusparseOperation_t opA,
        const float* alpha,
        cusparseConstSpMatDescr_t matA,
        cusparseConstDnVecDescr_t vecX,
        const float* beta,
        cusparseDnVecDescr_t vecY,
        cusparseSpMVAlg_t alg,
        size_t* bufferSize)
    {
        return cusparseSpMV_bufferSize(
            handle,
            opA,
            alpha,
            matA,
            vecX,
            beta,
            vecY,
            value_type,
            alg,
            bufferSize);
    }

    static cusparseStatus_t SpMV_bufferSize(
        cusparseHandle_t handle,
        cusparseOperation_t opA,
        const float* alpha,
        cusparseConstSpMatDescr_t matA,
        cusparseConstDnVecDescr_t vecX,
        const float* beta,
        cusparseDnVecDescr_t vecY,
        cusparseSpMVAlg_t alg,
        size_t* bufferSize)
    {
        return spmvBufferSize(handle, opA, alpha, matA, vecX, beta, vecY, alg, bufferSize);
    }

    static cusparseStatus_t SpMV(
        cusparseHandle_t handle,
        cusparseOperation_t opA,
        const float* alpha,
        cusparseConstSpMatDescr_t matA,
        cusparseConstDnVecDescr_t vecX,
        const float* beta,
        cusparseDnVecDescr_t vecY,
        cusparseSpMVAlg_t alg,
        void* externalBuffer)
    {
        return cusparseSpMV(
            handle,
            opA,
            alpha,
            matA,
            vecX,
            beta,
            vecY,
            value_type,
            alg,
            externalBuffer);
    }
};

template <>
struct CusparseApi<double>
{
    static constexpr cudaDataType value_type = CudaTypeTraits<double>::value;

    static cusparseStatus_t createCsr(
        cusparseSpMatDescr_t* matA,
        int64_t rows,
        int64_t cols,
        int64_t nnz,
        int* rowOffsets,
        int* colIndices,
        double* values)
    {
        return cusparseCreateCsr(
            matA,
            rows,
            cols,
            nnz,
            rowOffsets,
            colIndices,
            values,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            value_type);
    }

    static cusparseStatus_t CreateCsr(
        cusparseSpMatDescr_t* matA,
        int64_t rows,
        int64_t cols,
        int64_t nnz,
        int* rowOffsets,
        int* colIndices,
        double* values)
    {
        return createCsr(matA, rows, cols, nnz, rowOffsets, colIndices, values);
    }

    static cusparseStatus_t createDnVec(
        cusparseDnVecDescr_t* vec,
        int64_t size,
        double* values)
    {
        return cusparseCreateDnVec(vec, size, values, value_type);
    }

    static cusparseStatus_t CreateDnVec(
        cusparseDnVecDescr_t* vec,
        int64_t size,
        double* values)
    {
        return createDnVec(vec, size, values);
    }

    static cusparseStatus_t createDnMat(
        cusparseDnMatDescr_t* mat,
        int64_t rows,
        int64_t cols,
        int64_t ld,
        double* values,
        cusparseOrder_t order = CUSPARSE_ORDER_ROW)
    {
        return cusparseCreateDnMat(mat, rows, cols, ld, values, value_type, order);
    }

    static cusparseStatus_t CreateDnMat(
        cusparseDnMatDescr_t* mat,
        int64_t rows,
        int64_t cols,
        int64_t ld,
        double* values,
        cusparseOrder_t order = CUSPARSE_ORDER_ROW)
    {
        return createDnMat(mat, rows, cols, ld, values, order);
    }

    static cusparseStatus_t spmvBufferSize(
        cusparseHandle_t handle,
        cusparseOperation_t opA,
        const double* alpha,
        cusparseConstSpMatDescr_t matA,
        cusparseConstDnVecDescr_t vecX,
        const double* beta,
        cusparseDnVecDescr_t vecY,
        cusparseSpMVAlg_t alg,
        size_t* bufferSize)
    {
        return cusparseSpMV_bufferSize(
            handle,
            opA,
            alpha,
            matA,
            vecX,
            beta,
            vecY,
            value_type,
            alg,
            bufferSize);
    }

    static cusparseStatus_t SpMV_bufferSize(
        cusparseHandle_t handle,
        cusparseOperation_t opA,
        const double* alpha,
        cusparseConstSpMatDescr_t matA,
        cusparseConstDnVecDescr_t vecX,
        const double* beta,
        cusparseDnVecDescr_t vecY,
        cusparseSpMVAlg_t alg,
        size_t* bufferSize)
    {
        return spmvBufferSize(handle, opA, alpha, matA, vecX, beta, vecY, alg, bufferSize);
    }

    static cusparseStatus_t SpMV(
        cusparseHandle_t handle,
        cusparseOperation_t opA,
        const double* alpha,
        cusparseConstSpMatDescr_t matA,
        cusparseConstDnVecDescr_t vecX,
        const double* beta,
        cusparseDnVecDescr_t vecY,
        cusparseSpMVAlg_t alg,
        void* externalBuffer)
    {
        return cusparseSpMV(
            handle,
            opA,
            alpha,
            matA,
            vecX,
            beta,
            vecY,
            value_type,
            alg,
            externalBuffer);
    }
};

template <typename T>
using CublasOps = CublasApi<T>;

template <typename T>
using CusparseOps = CusparseApi<T>;
