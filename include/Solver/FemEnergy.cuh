#pragma once

#include "math/decomposition.hpp"
#include "math/helper_math.h"
#include "math/matrix.hpp"

template<typename Real>
__inline__ __device__ void setColumnFromMat3(Mat9x9<Real>& Q, int column, const mat3<Real>& A)
{
    int index = 0;
    #pragma unroll 3
    for (int j = 0; j < 3; ++j) {
        #pragma unroll 3
        for (int i = 0; i < 3; ++i, ++index) {
            Q(index, column) = A(i, j);
        }
    }
}

template<typename Real>
__inline__ __device__ void buildTwistAndFlipEigenvectors(
    const mat3<Real>& U,
    const mat3<Real>& V,
    Mat9x9<Real>& Q)
{
    const Real invSqrt2 = static_cast<Real>(0.70710678118654752440);
    const mat3<Real> Vt = mat3<Real>::transpose(V);

    mat3<Real> T0(static_cast<Real>(0));
    T0(1, 2) = static_cast<Real>(-1);
    T0(2, 1) = static_cast<Real>(1);

    mat3<Real> T1(static_cast<Real>(0));
    T1(0, 2) = static_cast<Real>(1);
    T1(2, 0) = static_cast<Real>(-1);

    mat3<Real> T2(static_cast<Real>(0));
    T2(0, 1) = static_cast<Real>(1);
    T2(1, 0) = static_cast<Real>(-1);

    mat3<Real> L0(static_cast<Real>(0));
    L0(1, 2) = static_cast<Real>(1);
    L0(2, 1) = static_cast<Real>(1);

    mat3<Real> L1(static_cast<Real>(0));
    L1(0, 2) = static_cast<Real>(1);
    L1(2, 0) = static_cast<Real>(1);

    mat3<Real> L2(static_cast<Real>(0));
    L2(0, 1) = static_cast<Real>(1);
    L2(1, 0) = static_cast<Real>(1);

    setColumnFromMat3(Q, 0, (U * T0 * Vt) * invSqrt2);
    setColumnFromMat3(Q, 1, (U * T1 * Vt) * invSqrt2);
    setColumnFromMat3(Q, 2, (U * T2 * Vt) * invSqrt2);
    setColumnFromMat3(Q, 3, (U * L0 * Vt) * invSqrt2);
    setColumnFromMat3(Q, 4, (U * L1 * Vt) * invSqrt2);
    setColumnFromMat3(Q, 5, (U * L2 * Vt) * invSqrt2);
}

template<typename Real>
__inline__ __device__ void buildScalingEigenvectors(
    const mat3<Real>& U,
    const mat3<Real>& scalingQ,
    const mat3<Real>& V,
    Mat9x9<Real>& Q)
{
    const mat3<Real> Vt = mat3<Real>::transpose(V);

    #pragma unroll 3
    for (int mode = 0; mode < 3; ++mode) {
        mat3<Real> D(static_cast<Real>(0));
        D(0, 0) = scalingQ(0, mode);
        D(1, 1) = scalingQ(1, mode);
        D(2, 2) = scalingQ(2, mode);
        setColumnFromMat3(Q, mode + 6, U * D * Vt);
    }
}

template<typename Real>
__inline__ __device__ void buildScalingEigenvectors(
    const mat3<Real>& U,
    const mat3<Real>& V,
    Mat9x9<Real>& Q)
{
    buildScalingEigenvectors(U, mat3<Real>(static_cast<Real>(1)), V, Q);
}

template<typename Real>
__inline__ __device__ void symmetricEigenDecomposition(
    mat3<Real> A,
    mat3<Real>& eigenvectors,
    Vector<Real, 3>& eigenvalues)
{
    Real q[4];
    jacobiEigenanlysis(A(0, 0), A(1, 0), A(1, 1), A(2, 0), A(2, 1), A(2, 2), q);
    quatToMat3(
        q,
        eigenvectors[0], eigenvectors[3], eigenvectors[6],
        eigenvectors[1], eigenvectors[4], eigenvectors[7],
        eigenvectors[2], eigenvectors[5], eigenvectors[8]);

    eigenvalues[0] = A(0, 0);
    eigenvalues[1] = A(1, 1);
    eigenvalues[2] = A(2, 2);
}

template<typename Real>
__inline__ __device__ Mat9x9<Real> reconstructHessianFromEigenSystem(
    const Mat9x9<Real>& eigenvectors,
    const Real eigenvalues[9],
    bool clampEigenvalues)
{
    Mat9x9<Real> hessian(static_cast<Real>(0));
    #pragma unroll 9
    for (int mode = 0; mode < 9; ++mode) {
        Real eigenvalue = eigenvalues[mode];
        if (clampEigenvalues && eigenvalue < static_cast<Real>(0)) {
            eigenvalue = static_cast<Real>(0);
        }

        #pragma unroll 9
        for (int i = 0; i < 9; ++i) {
            #pragma unroll 9
            for (int j = 0; j < 9; ++j) {
                hessian(i, j) += eigenvalue * eigenvectors(i, mode) * eigenvectors(j, mode);
            }
        }
    }

    return hessian;
}

template<typename Real>
__inline__ __device__ Vec9<Real> partialJpartialF(const mat3<Real>& F)
{
    mat3<Real> A(cross(F.column(1), F.column(2)),
                 cross(F.column(2), F.column(0)),
                 cross(F.column(0), F.column(1)));

    Vec9<Real> pJpF(static_cast<Real>(0));
    unsigned int index = 0;
    #pragma unroll 3
    for (unsigned int j = 0; j < 3; j++) {
        #pragma unroll 3
        for (unsigned int i = 0; i < 3; i++, index++) {
            pJpF(index, 0) = A(i, j);
        }
    }

    return pJpF;
}

template <typename Real>
__inline__ __device__ mat3<Real> P_STVK(const mat3<Real>& F, Real mu, Real lambda)
{
    using Mat3 = mat3<Real>;
    Mat3 FtF = Mat3::multiplyAtB(F, F);
    Mat3 E = (FtF - Mat3(static_cast<Real>(1))) * static_cast<Real>(0.5);
    Real trE = Mat3::trace(E);
    Mat3 S = E * (static_cast<Real>(2) * mu) + Mat3(lambda * trE);
    return F * S;
}

template <typename Real>
__inline__ __device__ mat3<Real> P_Corotated(const mat3<Real>& F, Real mu, Real lambda)
{
    using Mat3 = mat3<Real>;
    Mat3 R;
    computePD<Real>(F, R);

    Mat3 RtF = Mat3::multiplyAtB(R, F);
    Real tr = Mat3::trace(RtF) - static_cast<Real>(3);

    return (F - R) * (static_cast<Real>(2) * mu) + R * (lambda * tr);
}

template <typename Real>
__inline__ __device__ mat3<Real> P_ARAP(const mat3<Real>& F, Real mu)
{
    using Mat3 = mat3<Real>;
    Mat3 U;
    Mat3 Sigma;
    Mat3 V;
    computeSVD(F, U, Sigma, V);

    Mat3 R = Mat3::multiplyABt(U, V);
    Mat3 S = Mat3::multiplyADBt(V, Sigma, V);
    return R * ((S - Mat3(static_cast<Real>(1))) * (static_cast<Real>(2) * mu));
}

template <typename Real>
__inline__ __device__ mat3<Real> P_NeoHookean(const mat3<Real>& F, Real mu, Real lambda, Real alpha)
{
    using Mat3 = mat3<Real>;
    using Vec3 = Vector<Real, 3>;
    Real J = Mat3::determinant(F);
    J = (J > static_cast<Real>(1e-4)) ? J : static_cast<Real>(1e-4);

    Vec3 col0 = F.column(0);
    Vec3 col1 = F.column(1);
    Vec3 col2 = F.column(2);
    Mat3 pJpF = Mat3(cross(col1, col2), cross(col2, col0), cross(col0, col1));

    return mu * F + lambda * (J - alpha) * pJpF;
}

template <typename Real>
__inline__ __device__ mat3<Real> computeStressForEnergy(
    const mat3<Real>& F,
    Real mu,
    Real lambda,
    Real alpha,
    int energyType)
{
    if (energyType == 0) {
        return P_STVK(F, mu, lambda);
    }
    if (energyType == 1) {
        return P_Corotated(F, mu, lambda);
    }
    if (energyType == 3) {
        return P_ARAP(F, mu);
    }
    return P_NeoHookean(F, mu, lambda, alpha);
}

template<typename Real>
__inline__ __device__ Mat9x9<Real> computeNeoHookeanHessian(
    const mat3<Real>& F,
    Real mu,
    Real lambda,
    Real alpha)
{
    Vec9<Real> pjpf = partialJpartialF(F);
    const Real I3 = mat3<Real>::determinant(F);
    const Real scale = lambda * (I3 - alpha);
    const mat3<Real> f0hat = crossProductMatrix(F.column(0)) * scale;
    const mat3<Real> f1hat = crossProductMatrix(F.column(1)) * scale;
    const mat3<Real> f2hat = crossProductMatrix(F.column(2)) * scale;

    Mat9x9<Real> hessJ(static_cast<Real>(0));
    #pragma unroll 3
    for (int j = 0; j < 3; j++) {
        #pragma unroll 3
        for (int i = 0; i < 3; i++) {
            hessJ(i, j + 3) = Real(-1) * f2hat(i,j);
            hessJ(i + 3, j) =  f2hat(i,j);

            hessJ(i, j + 6) =  f1hat(i,j);
            hessJ(i + 6, j) = Real(-1) * f1hat(i, j);

            hessJ(i + 3, j + 6) = Real(-1) * f0hat(i,j);
            hessJ(i + 6, j + 3) =  f0hat(i,j);
        }
    }

    return mu * Mat9x9<Real>::Identity() + lambda * pjpf * transpose(pjpf) + hessJ;
}

template<typename Real>
__inline__ __device__ void buildStableNeoHookeanEigensystem(
    const mat3<Real>& U,
    const mat3<Real>& Sigma,
    const mat3<Real>& V,
    Real mu,
    Real lambda,
    Real alpha,
    Real eigenvalues[9],
    Mat9x9<Real>& eigenvectors)
{
    const Real s0 = Sigma(0, 0);
    const Real s1 = Sigma(1, 1);
    const Real s2 = Sigma(2, 2);
    const Real J = s0 * s1 * s2;

    const Real front = lambda * (J - alpha);
    eigenvalues[0] = front * s0 + mu;
    eigenvalues[1] = front * s1 + mu;
    eigenvalues[2] = front * s2 + mu;
    eigenvalues[3] = -front * s0 + mu;
    eigenvalues[4] = -front * s1 + mu;
    eigenvalues[5] = -front * s2 + mu;

    mat3<Real> A(static_cast<Real>(0));
    const Real s0s0 = s0 * s0;
    const Real s1s1 = s1 * s1;
    const Real s2s2 = s2 * s2;
    A(0, 0) = mu + lambda * s1s1 * s2s2;
    A(1, 1) = mu + lambda * s0s0 * s2s2;
    A(2, 2) = mu + lambda * s0s0 * s1s1;

    const Real frontOffDiag = lambda * (static_cast<Real>(2) * J - alpha);
    A(0, 1) = frontOffDiag * s2;
    A(0, 2) = frontOffDiag * s1;
    A(1, 2) = frontOffDiag * s0;
    A(1, 0) = A(0, 1);
    A(2, 0) = A(0, 2);
    A(2, 1) = A(1, 2);

    mat3<Real> scalingQ;
    Vector<Real, 3> scalingEigenvalues;
    symmetricEigenDecomposition(A, scalingQ, scalingEigenvalues);
    eigenvalues[6] = scalingEigenvalues[0];
    eigenvalues[7] = scalingEigenvalues[1];
    eigenvalues[8] = scalingEigenvalues[2];

    eigenvectors = Mat9x9<Real>(static_cast<Real>(0));
    buildTwistAndFlipEigenvectors(U, V, eigenvectors);
    buildScalingEigenvectors(U, scalingQ, V, eigenvectors);
}

template<typename Real>
__inline__ __device__ void buildSTVKEigensystem(
    const mat3<Real>& U,
    const mat3<Real>& Sigma,
    const mat3<Real>& V,
    Real mu,
    Real lambda,
    Real eigenvalues[9],
    Mat9x9<Real>& eigenvectors)
{
    const Real s0 = Sigma(0, 0);
    const Real s1 = Sigma(1, 1);
    const Real s2 = Sigma(2, 2);
    const Real I2 = s0 * s0 + s1 * s1 + s2 * s2;
    const Real front = -mu + lambda * static_cast<Real>(0.5) * (I2 - static_cast<Real>(3));
    const Real s0Sq = s0 * s0;
    const Real s1Sq = s1 * s1;
    const Real s2Sq = s2 * s2;
    const Real s0s1 = s0 * s1;
    const Real s0s2 = s0 * s2;
    const Real s1s2 = s1 * s2;

    eigenvalues[0] = front + mu * (s1Sq + s2Sq - s1s2);
    eigenvalues[1] = front + mu * (s0Sq + s2Sq - s0s2);
    eigenvalues[2] = front + mu * (s0Sq + s1Sq - s0s1);
    eigenvalues[3] = front + mu * (s1Sq + s2Sq + s1s2);
    eigenvalues[4] = front + mu * (s0Sq + s2Sq + s0s2);
    eigenvalues[5] = front + mu * (s0Sq + s1Sq + s0s1);

    mat3<Real> A(static_cast<Real>(0));
    A(0, 0) = front + (lambda + static_cast<Real>(3) * mu) * s0Sq;
    A(1, 1) = front + (lambda + static_cast<Real>(3) * mu) * s1Sq;
    A(2, 2) = front + (lambda + static_cast<Real>(3) * mu) * s2Sq;
    A(0, 1) = lambda * s0s1;
    A(1, 0) = A(0, 1);
    A(0, 2) = lambda * s0s2;
    A(2, 0) = A(0, 2);
    A(1, 2) = lambda * s1s2;
    A(2, 1) = A(1, 2);

    mat3<Real> scalingQ;
    Vector<Real, 3> scalingEigenvalues;
    symmetricEigenDecomposition(A, scalingQ, scalingEigenvalues);
    eigenvalues[6] = scalingEigenvalues[0];
    eigenvalues[7] = scalingEigenvalues[1];
    eigenvalues[8] = scalingEigenvalues[2];

    eigenvectors = Mat9x9<Real>(static_cast<Real>(0));
    buildTwistAndFlipEigenvectors(U, V, eigenvectors);
    buildScalingEigenvectors(U, scalingQ, V, eigenvectors);
}

template<typename Real>
__inline__ __device__ Mat9x9<Real> computeARAPHessian(
    const mat3<Real>& U,
    const mat3<Real>& Sigma,
    const mat3<Real>& V,
    Real mu)
{
    const Real s0 = Sigma(0, 0);
    const Real s1 = Sigma(1, 1);
    const Real s2 = Sigma(2, 2);
    const Real invSqrt2 = static_cast<Real>(0.70710678118654752440);
    const mat3<Real> Vt = mat3<Real>::transpose(V);

    mat3<Real> twist0(static_cast<Real>(0));
    twist0(1, 2) = static_cast<Real>(-1);
    twist0(2, 1) = static_cast<Real>(1);

    mat3<Real> twist1(static_cast<Real>(0));
    twist1(0, 2) = static_cast<Real>(1);
    twist1(2, 0) = static_cast<Real>(-1);

    mat3<Real> twist2(static_cast<Real>(0));
    twist2(0, 1) = static_cast<Real>(1);
    twist2(1, 0) = static_cast<Real>(-1);

    Mat9x9<Real> q(static_cast<Real>(0));
    setColumnFromMat3(q, 0, (U * twist0 * Vt) * invSqrt2);
    setColumnFromMat3(q, 1, (U * twist1 * Vt) * invSqrt2);
    setColumnFromMat3(q, 2, (U * twist2 * Vt) * invSqrt2);

    const Real eps = static_cast<Real>(1e-8);
    const Real denom0 = (s1 + s2 > eps) ? (s1 + s2) : eps;
    const Real denom1 = (s0 + s2 > eps) ? (s0 + s2) : eps;
    const Real denom2 = (s0 + s1 > eps) ? (s0 + s1) : eps;
    const Real lambda0 = mu * static_cast<Real>(2) / denom0;
    const Real lambda1 = mu * static_cast<Real>(2) / denom1;
    const Real lambda2 = mu * static_cast<Real>(2) / denom2;

    Mat9x9<Real> pPpF = mu * Mat9x9<Real>::Identity();
    #pragma unroll 3
    for (int mode = 0; mode < 3; ++mode) {
        const Real coeff = (mode == 0) ? lambda0 : ((mode == 1) ? lambda1 : lambda2);
        #pragma unroll 9
        for (int i = 0; i < 9; ++i) {
            #pragma unroll 9
            for (int j = 0; j < 9; ++j) {
                pPpF(i, j) -= coeff * q(i, mode) * q(j, mode);
            }
        }
    }

    return pPpF * static_cast<Real>(2);
}

template<typename Real>
__inline__ __device__ Mat9x9<Real> computeClampedARAPHessian(
    const mat3<Real>& U,
    const mat3<Real>& Sigma,
    const mat3<Real>& V,
    Real mu)
{
    Mat9x9<Real> eigenvectors(static_cast<Real>(0));
    buildTwistAndFlipEigenvectors(U, V, eigenvectors);
    buildScalingEigenvectors(U, V, eigenvectors);

    const Real s0 = Sigma(0, 0);
    const Real s1 = Sigma(1, 1);
    const Real s2 = Sigma(2, 2);
    const Real eps = static_cast<Real>(1e-8);
    const Real denom0 = (s1 + s2 > eps) ? (s1 + s2) : eps;
    const Real denom1 = (s0 + s2 > eps) ? (s0 + s2) : eps;
    const Real denom2 = (s0 + s1 > eps) ? (s0 + s1) : eps;

    Real eigenvalues[9];
    #pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        eigenvalues[i] = static_cast<Real>(2) * mu;
    }
    eigenvalues[0] = mu * (static_cast<Real>(2) - static_cast<Real>(4) / denom0);
    eigenvalues[1] = mu * (static_cast<Real>(2) - static_cast<Real>(4) / denom1);
    eigenvalues[2] = mu * (static_cast<Real>(2) - static_cast<Real>(4) / denom2);

    return reconstructHessianFromEigenSystem(eigenvectors, eigenvalues, true);
}

template<typename Real>
__inline__ __device__ Mat9x9<Real> computeEnergyHessian(
    const mat3<Real>& F,
    Real mu,
    Real lambda,
    Real alpha,
    int energyType,
    bool clampEigenvalues)
{
    if (energyType == 0 || energyType == 3) {
        mat3<Real> U;
        mat3<Real> Sigma;
        mat3<Real> V;
        computeSVD(F, U, Sigma, V);

        if (energyType == 0) {
            Real eigenvalues[9];
            Mat9x9<Real> eigenvectors(static_cast<Real>(0));
            buildSTVKEigensystem(U, Sigma, V, mu, lambda, eigenvalues, eigenvectors);
            return reconstructHessianFromEigenSystem(eigenvectors, eigenvalues, clampEigenvalues);
        }

        return clampEigenvalues
            ? computeClampedARAPHessian(U, Sigma, V, mu)
            : computeARAPHessian(U, Sigma, V, mu);
    }

    if (energyType == 2 && clampEigenvalues) {
        mat3<Real> U;
        mat3<Real> Sigma;
        mat3<Real> V;
        computeSVD(F, U, Sigma, V);
        Real eigenvalues[9];
        Mat9x9<Real> eigenvectors(static_cast<Real>(0));
        buildStableNeoHookeanEigensystem(U, Sigma, V, mu, lambda, alpha, eigenvalues, eigenvectors);
        return reconstructHessianFromEigenSystem(eigenvectors, eigenvalues, true);
    }

    return computeNeoHookeanHessian(F, mu, lambda, alpha);
}
