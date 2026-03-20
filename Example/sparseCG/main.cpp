#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using Scalar = double;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void print_matrix(const std::vector<Scalar>& mat, const int row, const int col)
{
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < col; j++)
        {
            std::cout << mat[i * row + j] << ' ';
        }
        std::cout << std::endl;
    }
}

int main()
{
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    cusparseHandle_t  cusparseH;
    cusparseCreate(&cusparseH);

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB;
    void* dBuffer = NULL;
    void* dBuffer_sp2dn = NULL;
    size_t               bufferSize = 0;
    
    std::vector<Scalar> h_A;
    h_A.assign(9 * 9, 0);
    std::vector<Scalar> diag_vals = { 5, 6, 7, 8, 9, 8, 7, 6, 5 };
    std::vector<Scalar> lower_diag = { 1, 1, 1, 1, 1, 1, 1, 1 };
    std::vector<Scalar> upper_diag = { 2, 2, 2, 2, 2, 2, 2, 2 };

    std::vector<Scalar> h_b = { 7, 9, 10, 11, 12, 10, 9, 6};
    Scalar* d_b;
    cudaMalloc((void**)&d_b, sizeof(Scalar) * h_b.size());
    cudaMemcpy(d_b, h_b.data(), sizeof(Scalar) * h_b.size(), cudaMemcpyHostToDevice);

    Scalar* d_x;
    cudaMalloc((void**)&d_x, sizeof(Scalar) * h_b.size());
    cudaMemset(d_x, 5, sizeof(Scalar) * h_b.size());
    
    Scalar* r;
    cudaMalloc((void**)&r, sizeof(Scalar) * h_b.size());
    cudaMemset(r, 0, sizeof(Scalar) * h_b.size());

    Scalar* p;
    cudaMalloc((void**)&p, sizeof(Scalar) * h_b.size());
    cudaMemset(p, 0, sizeof(Scalar) * h_b.size());

    for(int i = 0; i < 9; i++)
    {
        h_A[i * 9 + i] = diag_vals[i];
    }

    for(int i = 1; i < 9; i++)
    {
        h_A[i * 9 + (i-1)] = lower_diag[i-1];
    }

    for(int i = 1; i < 9; i++)
    {
        h_A[(i-1) * 9 + i] = upper_diag[i-1];
    }

    print_matrix(h_A, 9, 9);

    int   num_rows = 9;
    int   num_cols = 9;
    int   ld = num_cols;

    int* d_csr_offsets, * d_csr_columns;
    Scalar* d_csr_values, * d_dense;
    int dense_size = num_rows * num_cols;
    CHECK_CUDA(cudaMalloc((void**)&d_dense, dense_size * sizeof(Scalar)))
    CHECK_CUDA(cudaMalloc((void**)&d_csr_offsets,
        (num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMemcpy(d_dense, h_A.data(), dense_size * sizeof(Scalar),
        cudaMemcpyHostToDevice))

    cusparseCreateDnMat(&matB, num_rows, num_cols, num_rows, d_dense, CUDA_R_64F, CUSPARSE_ORDER_ROW);
    cusparseCreateCsr(&matA, num_rows, num_cols, 0,
        d_csr_offsets, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
        cusparseH, matB, matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
   
    // execute Dense to sparse conversion
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(cusparseH, matB, matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        dBuffer));

    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matA, &num_rows_tmp, &num_cols_tmp,
        &nnz));

    std::cout << "row num:" << num_rows_tmp << std::endl
        << "cols num:" << num_cols_tmp << std::endl
        << "nnz = " << nnz << std::endl;

    // allocate CSR column indices and values
    CHECK_CUDA(cudaMalloc((void**)&d_csr_columns, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_csr_values, nnz * sizeof(Scalar)));
        // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(matA, d_csr_offsets, d_csr_columns,
        d_csr_values));

    // execute Dense to Sparse conversion
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(cusparseH, matB, matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        dBuffer));

    // Dense to Sparse conversion
    CHECK_CUDA(cudaMemset(d_dense, 0, sizeof(Scalar) * dense_size));
    cusparseSparseToDense_bufferSize(
        cusparseH, matA, matB,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        &bufferSize);
    cudaMalloc(&dBuffer_sp2dn, bufferSize);

    cusparseSparseToDense(cusparseH, matA, matB,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        dBuffer);

    std::vector<Scalar> h_dense;
    h_dense.assign(num_rows * num_cols, 0);
    cudaMemcpy(h_dense.data(), d_dense, dense_size * sizeof(Scalar),
        cudaMemcpyDeviceToHost);

    print_matrix(h_dense, num_rows, num_cols);


    // CG Process
    


    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroy(cusparseH));
    cublasDestroy(cublasH);
    return 0;
}