#include "BaseStructure.hpp"
#include "math/Vector.hpp"
#include "math/helper_math.h"
#include "math/matrix.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

template <typename Real>
using Vec3 = Vector<Real, 3>;

template <typename Real>
struct TetExample
{
    std::string name;
    std::array<Vec3<Real>, 4> rest;
    std::array<Vec3<Real>, 4> current;
};

template <typename Real>
Mat9x12<Real> BuildBMatrixFromdFdxCpu(const mat3<Real> dF_dx[12])
{
    Mat9x12<Real> B(Real(0));

    for (int a = 0; a < 12; ++a) {
        const mat3<Real>& M = dF_dx[a];

        B(0, a) = M(0, 0);
        B(1, a) = M(1, 0);
        B(2, a) = M(2, 0);
        B(3, a) = M(0, 1);
        B(4, a) = M(1, 1);
        B(5, a) = M(2, 1);
        B(6, a) = M(0, 2);
        B(7, a) = M(1, 2);
        B(8, a) = M(2, 2);
    }

    return B;
}

template <typename Real>
Mat9x9<Real> BuildMatrixGFromdJdFCpu(const mat3<Real>& dJdF)
{
    Mat9x9<Real> G(Real(0));
    Real g[9];

    g[0] = dJdF(0, 0);
    g[1] = dJdF(1, 0);
    g[2] = dJdF(2, 0);
    g[3] = dJdF(0, 1);
    g[4] = dJdF(1, 1);
    g[5] = dJdF(2, 1);
    g[6] = dJdF(0, 2);
    g[7] = dJdF(1, 2);
    g[8] = dJdF(2, 2);

    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 9; ++c) {
            G(r, c) = g[r] * g[c];
        }
    }

    return G;
}

template <typename Real>
Mat9x9<Real> BuildHessianMatrixCpu(
    const Vec3<Real>& f0,
    const Vec3<Real>& f1,
    const Vec3<Real>& f2)
{
    mat3<Real> f0hat = crossProductMatrix(f0);
    mat3<Real> f1hat = crossProductMatrix(f1);
    mat3<Real> f2hat = crossProductMatrix(f2);
    mat3<Real> zeroMat(Real(0));
    const mat3<Real> blocks[3][3] = {
        {zeroMat, Real(-1) * f2hat, f1hat},
        {f2hat, zeroMat, Real(-1) * f0hat},
        {Real(-1) * f1hat, f0hat, zeroMat},
    };

    Mat9x9<Real> H(Real(0));
    for (int bi = 0; bi < 3; ++bi) {
        for (int bj = 0; bj < 3; ++bj) {
            const mat3<Real>& M = blocks[bi][bj];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    H(bi * 3 + r, bj * 3 + c) = M(r, c);
                }
            }
        }
    }

    return H;
}

template <typename Real>
mat3<Real> ComputeF(
    const Tetrahedron<Real>& tet,
    const std::array<Vec3<Real>, 4>& current)
{
    const Vec4i ids = tet.verticesIndex;
    const Vec3<Real>& x0 = current[ids.x];
    const Vec3<Real>& x1 = current[ids.y];
    const Vec3<Real>& x2 = current[ids.z];
    const Vec3<Real>& x3 = current[ids.w];

    mat3<Real> Ds(x1 - x0, x2 - x0, x3 - x0);
    return Ds * tet.Dm_inv;
}

template <typename Real>
Tetrahedron<Real> MakeTetFromRest(const std::array<Vec3<Real>, 4>& rest)
{
    Tetrahedron<Real> tet{};
    tet.verticesIndex = make_vec4i(0, 1, 2, 3);

    const Vec3<Real>& x0 = rest[0];
    const Vec3<Real>& x1 = rest[1];
    const Vec3<Real>& x2 = rest[2];
    const Vec3<Real>& x3 = rest[3];

    mat3<Real> Dm(x1 - x0, x2 - x0, x3 - x0);
    tet.Dm_inv = mat3<Real>::inverse(Dm);
    tet.volume = std::abs(mat3<Real>::determinant(Dm) / Real(6));
    tet.K = Mat12x12<Real>(Real(0));
    return tet;
}

template <typename Real>
Mat12x12<Real> ComputeKCpu(
    const Tetrahedron<Real>& tet,
    const mat3<Real>& F,
    Real mu,
    Real lambda)
{
    using Mat3 = mat3<Real>;

    Real J = Mat3::determinant(F);
    J = (J > Real(1e-4)) ? J : Real(1e-4);

    Mat3 dF_dx[12];
    Mat3 Dm_inv = tet.Dm_inv;
    Mat3 Dm_invT = Mat3::transpose(Dm_inv);
    Vec3<Real> g[4];

    g[1] = Dm_invT.column(0);
    g[2] = Dm_invT.column(1);
    g[3] = Dm_invT.column(2);
    g[0] = -(g[1] + g[2] + g[3]);

    for (int a = 0; a < 4; ++a) {
        for (int c = 0; c < 3; ++c) {
            Mat3 dF(Real(0));
            for (int j = 0; j < 3; ++j) {
                dF(c, j) = g[a][j];
            }
            dF_dx[a * 3 + c] = dF;
        }
    }

    Mat9x12<Real> B = BuildBMatrixFromdFdxCpu(dF_dx);

    Vec3<Real> f0 = F.column(0);
    Vec3<Real> f1 = F.column(1);
    Vec3<Real> f2 = F.column(2);

    Mat3 dJ_dF(
        cross(f1, f2),
        cross(f2, f0),
        cross(f0, f1));

    Mat9x9<Real> G = BuildMatrixGFromdJdFCpu(dJ_dF);
    Mat9x9<Real> hessianJ = BuildHessianMatrixCpu(f0, f1, f2);
    Mat9x9<Real> I9 = Mat9x9<Real>::Identity();

    Mat9x9<Real> dP_dF =
        mu * I9
        + lambda * G
        + (lambda * (J - Real(1)) - mu) * hessianJ;

    return tet.volume * (transpose(B) * dP_dF * B);
}

template <typename Real>
Real MaxAbs(const Mat12x12<Real>& M)
{
    Real maxValue = Real(0);
    for (int r = 0; r < 12; ++r) {
        for (int c = 0; c < 12; ++c) {
            maxValue = std::max(maxValue, std::abs(M(r, c)));
        }
    }
    return maxValue;
}

template <typename Real>
Real MaxAbsRowSum(const Mat12x12<Real>& M)
{
    Real maxValue = Real(0);
    for (int r = 0; r < 12; ++r) {
        Real sum = Real(0);
        for (int c = 0; c < 12; ++c) {
            sum += M(r, c);
        }
        maxValue = std::max(maxValue, std::abs(sum));
    }
    return maxValue;
}

template <typename Real>
Real MaxAsymmetry(const Mat12x12<Real>& M)
{
    Real maxValue = Real(0);
    for (int r = 0; r < 12; ++r) {
        for (int c = 0; c < 12; ++c) {
            maxValue = std::max(maxValue, std::abs(M(r, c) - M(c, r)));
        }
    }
    return maxValue;
}

template <typename Real>
void PrintVec3(const Vec3<Real>& v)
{
    std::cout << "[" << v.x << ", " << v.y << ", " << v.z << "]";
}

template <typename Real>
void PrintMat3(const mat3<Real>& M, const char* label)
{
    std::cout << label << ":\n";
    for (int r = 0; r < 3; ++r) {
        std::cout << "  ";
        for (int c = 0; c < 3; ++c) {
            std::cout << std::setw(12) << M(r, c) << ' ';
        }
        std::cout << '\n';
    }
}

template <typename Real>
void PrintMat12x12(const Mat12x12<Real>& M, const char* label)
{
    std::cout << label << ":\n";
    for (int r = 0; r < 12; ++r) {
        std::cout << "  ";
        for (int c = 0; c < 12; ++c) {
            std::cout << std::setw(11) << M(r, c) << ' ';
        }
        std::cout << '\n';
    }
}

int main()
{
    using Real = double;

    const Real youngsModulus = 1.0e6;
    const Real poissonRatio = 0.30;
    const Real mu = youngsModulus / (Real(2) * (Real(1) + poissonRatio));
    const Real lambda =
        youngsModulus * poissonRatio /
        ((Real(1) + poissonRatio) * (Real(1) - Real(2) * poissonRatio));

    const std::vector<TetExample<Real>> examples = {
        {
            "unit_rest_small_shear",
            {Vec3<Real>{0.0, 0.0, 0.0}, Vec3<Real>{1.0, 0.0, 0.0}, Vec3<Real>{0.0, 1.0, 0.0}, Vec3<Real>{0.0, 0.0, 1.0}},
            {Vec3<Real>{0.0, 0.0, 0.0}, Vec3<Real>{1.08, 0.05, 0.00}, Vec3<Real>{0.02, 0.97, 0.04}, Vec3<Real>{0.01, 0.03, 1.12}},
        },
        {
            "translated_rest_anisotropic_stretch",
            {Vec3<Real>{0.2, 0.1, 0.3}, Vec3<Real>{1.4, 0.2, 0.4}, Vec3<Real>{0.1, 1.5, 0.5}, Vec3<Real>{0.3, 0.2, 1.6}},
            {Vec3<Real>{0.2, 0.1, 0.3}, Vec3<Real>{1.52, 0.24, 0.42}, Vec3<Real>{0.18, 1.44, 0.56}, Vec3<Real>{0.31, 0.16, 1.73}},
        },
        {
            "compressed_and_sheared",
            {Vec3<Real>{-0.1, 0.0, 0.0}, Vec3<Real>{0.9, 0.1, 0.0}, Vec3<Real>{0.0, 1.1, 0.1}, Vec3<Real>{0.0, 0.2, 1.2}},
            {Vec3<Real>{-0.1, 0.0, 0.0}, Vec3<Real>{0.78, 0.12, 0.05}, Vec3<Real>{0.05, 0.96, 0.14}, Vec3<Real>{0.07, 0.26, 1.03}},
        },
    };

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CPU reproduction of Solver.cu::K_ComputeK\n";
    std::cout << "Material: E=" << youngsModulus
              << ", nu=" << poissonRatio
              << ", mu=" << mu
              << ", lambda=" << lambda << "\n\n";

    for (const TetExample<Real>& example : examples) {
        Tetrahedron<Real> tet = MakeTetFromRest(example.rest);
        mat3<Real> F = ComputeF(tet, example.current);
        Mat12x12<Real> K = ComputeKCpu(tet, F, mu, lambda);

        std::cout << "Example: " << example.name << '\n';
        std::cout << "  Rest volume: " << tet.volume << '\n';
        std::cout << "  Current vertices:\n";
        for (int i = 0; i < 4; ++i) {
            std::cout << "    v" << i << " = ";
            PrintVec3(example.current[i]);
            std::cout << '\n';
        }

        PrintMat3(F, "  F");
        PrintMat12x12(K, "  K");

        std::cout << "  Checks:\n";
        std::cout << "    max|K| = " << MaxAbs(K) << '\n';
        std::cout << "    max asymmetry = " << MaxAsymmetry(K) << '\n';
        std::cout << "    max |row sum| = " << MaxAbsRowSum(K) << '\n';
        std::cout << '\n';
    }

    return 0;
}
