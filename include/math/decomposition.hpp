#pragma once

#include "math/matrix.hpp"
#include "math/helper_math.h"

#include "math/svd3_cuda.h"

template<typename Real>
__host__ __device__ __forceinline__ void computeSVD(const mat3<Real>& A, mat3<Real>& W, mat3<Real>& S, mat3<Real>& V) {

	svd(A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8],
		W[0], W[3], W[6], W[1], W[4], W[7], W[2], W[5], W[8],
		S[0], S[3], W[6], S[1], S[4], S[7], S[2], S[5], S[8],
		V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
}

/*
* Returns polar decomposition of 3x3 matrix M where
* M = Fe = Re * Se = U * P
* U is an orthonormal matrix
* S is symmetric positive semidefinite
* Can get Polar Decomposition from SVD, see first section of http://en.wikipedia.org/wiki/Polar_decomposition
*/
template<typename Real>
__host__ __device__ void computePD(const mat3<Real>& A, mat3<Real>& R) {
	// U is unitary matrix (i.e. orthogonal/orthonormal)
	// P is positive semidefinite Hermitian matrix
	mat3<Real> W, S, V;
	computeSVD(A, W, S, V);
	R = mat3<Real>::multiplyABt(W, V);
}


/*
* Returns polar decomposition of 3x3 matrix M where
* M = Fe = Re * Se = U * P
* U is an orthonormal matrix
* S is symmetric positive semidefinite
* Can get Polar Decomposition from SVD, see first section of http://en.wikipedia.org/wiki/Polar_decomposition
*/
template <typename Real>
__host__ __device__ void computePD(const mat3<Real>& A, mat3<Real>& R, mat3<Real>& P) {
	// U is unitary matrix (i.e. orthogonal/orthonormal)
	// P is positive semidefinite Hermitian matrix
	mat3<Real> W, S, V;
	computeSVD(A, W, S, V);
	R = mat3<Real>::multiplyABt(W, V);
	P = mat3<Real>::multiplyADBt(V, S, V);
}

/*
* In snow we desire both SVD and polar decompositions simultaneously without
* re-computing USV for polar.
* here is a function that returns all the relevant values
* SVD : A = W * S * V'
* PD : A = R * E
*/
template <typename Real>
__host__ __device__ void computeSVDandPD(const mat3<Real>& A, mat3<Real>& W, mat3<Real>& S, mat3<Real>& V, mat3<Real>& R) {
	computeSVD(A, W, S, V);
	R = mat3<Real>::multiplyABt(W, V);
}
