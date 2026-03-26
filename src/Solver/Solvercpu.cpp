#include "Solver.h"

#include <algorithm>
#include <cmath>

inline float rsqrt(float x)
{
    return 1.0f / std::sqrt(x);
}

inline double rsqrt(double x)
{
    return 1.0 / std::sqrt(x);
}

#include "math/decomposition.hpp"

template<typename Real>
mat3<Real> computeF(const Tetrahedron<Real>& tet, std::vector<Vector<Real, 3>>& vertices)
{
    int i0 = tet.verticesIndex.x;
    int i1 = tet.verticesIndex.y;
    int i2 = tet.verticesIndex.z;
    int i3 = tet.verticesIndex.w;
    
    Vector<Real, 3> x0 = vertices[i0];
    Vector<Real, 3> x1 = vertices[i1];
    Vector<Real, 3> x2 = vertices[i2];
    Vector<Real, 3> x3 = vertices[i3];

    mat3<Real> Ds(x1 - x0, x2 - x0, x3 - x0);
    return Ds * tet.Dm_inv;
}

template<typename Real>
mat3<Real> computeStress(const mat3<Real>& F, const typename ElasticitySolverT<Real>::Parameters& params)
{
    using Mat3 = mat3<Real>;
    using Vec3 = Vector<Real, 3>;

    if (params.energyType == STVK)
    {
        Mat3 FtF = Mat3::multiplyAtB(F, F);
        Mat3 E = (FtF - Mat3(static_cast<Real>(1))) * static_cast<Real>(0.5);
        Real trE = Mat3::trace(E);
        Mat3 S = E * (static_cast<Real>(2) * params.mu) + Mat3(params.lambda * trE);
        return F * S;
    }

    if (params.energyType == COROTATED)
    {
        Mat3 R;
        computePD(F, R);
        Mat3 RtF = Mat3::multiplyAtB(R, F);
        Real tr = Mat3::trace(RtF) - static_cast<Real>(3);
        return (F - R) * (static_cast<Real>(2) * params.mu) + R * (params.lambda * tr);
    }

    Real J = Mat3::determinant(F);
    J = (J > static_cast<Real>(1e-4)) ? J : static_cast<Real>(1e-4);

    Vec3 col0 = F.column(0);
    Vec3 col1 = F.column(1);
    Vec3 col2 = F.column(2);
    Mat3 pJpF(cross(col1, col2), cross(col2, col0), cross(col0, col1));
    return params.mu * F + params.lambda * (J - params.alpha) * pJpF;
}

template<typename Real>
void preCompute(std::vector<Tetrahedron<Real>>& tetrahedron, std::vector<Vector<Real, 3>>& vertices, std::vector<Real>& mass)
{
    std::fill(mass.begin(), mass.end(), static_cast<Real>(0));
    for (int i = 0; i < tetrahedron.size(); i++)
    {
        Tetrahedron<Real>& tet = tetrahedron[i];
        Vec4i ids = tet.verticesIndex;
        Vector<Real, 3> x[4];
        for (int j = 0; j < 4; j++)
            x[j] = vertices[ids[j]];
        mat3<Real> Dm(x[1] - x[0], x[2] - x[0], x[3] - x[0]);
        Real det = mat3<Real>::determinant(Dm);
        tet.volume = fabs(det / 6.0);
        Real tetMass = tet.volume * static_cast<Real>(0.25);
        for (int j = 0; j < 4; j++)
            mass[ids[j]] += tetMass;
        tet.Dm_inv = mat3<Real>::inverse(Dm);
    }
}

template<typename Real>
void ElasticitySolverT<Real>::SimulateCPU(bool export_results, int frameId)
{
    if (frameId == 0)
    {
        h_params.mu = h_params.youngs_modulus / (static_cast<Real>(2.0) * (static_cast<Real>(1.0) + h_params.poisson_ratio));
        h_params.lambda = h_params.youngs_modulus * h_params.poisson_ratio / ((static_cast<Real>(1.0) + h_params.poisson_ratio) * (static_cast<Real>(1.0) - static_cast<Real>(2.0) * h_params.poisson_ratio));
        h_params.alpha = (h_params.energyType == NEOHOOKEAN && std::abs(h_params.lambda) > static_cast<Real>(1e-12))
            ? Real(1) + h_params.mu / h_params.lambda
            : static_cast<Real>(0);
        preCompute(h_tet, h_vertex, h_mass);
        for (int i = 0; i < h_mass.size(); i++)
            h_mass[i] *= h_params.density;
        for (int i = 0; i < h_vertex.size(); i++)
            h_velocity[i] = Vec3({ Real(0), Real(0), Real(0) });
    }
    for(int i = 0; i< 10; i++)
        StepCPU();
    if (export_results)
        ExportMesh(frameId / 10);
}

template<typename Real>
void ElasticitySolverT<Real>:: StepCPU()
{
    std::vector<Vec3> force(h_vertex.size(), Vec3{ Real(0), Real(0), Real(0) });
    for (int i = 0; i < h_vertex.size(); i++)
        force[i] = h_params.gravity * h_mass[i];

    for(int i = 0; i < h_tet.size(); i++)
    {
        Tetrahedron<Real>& tet = h_tet[i];
        Vec4i ids = tet.verticesIndex;

        mat3<Real> F = computeF(tet, h_vertex);
        mat3<Real> P = computeStress(F, h_params);
        const mat3<Real>& DmInv = tet.Dm_inv;
        mat3<Real> DmInvT = mat3<Real>::transpose(DmInv);
        mat3<Real> H = P * DmInvT * (-tet.volume);
        Vec3 f1 = H.column(0);
        Vec3 f2 = H.column(1);
        Vec3 f3 = H.column(2);
        Vec3 f0{ -f1.x - f2.x - f3.x,
                -f1.y - f2.y - f3.y,
                -f1.z - f2.z - f3.z };
        force[ids[0]] += f0;
        force[ids[1]] += f1;
        force[ids[2]] += f2;
        force[ids[3]] += f3;
    }

    for(int i = 0; i < h_vertex.size(); i++)
    {
        Vec3& vn = h_velocity[i];
        Vec3& xn = h_vertex[i];
        if (h_mass[i] <= static_cast<Real>(1e-12))
            continue;
        Real massInv = Real(1) / h_mass[i];
        vn = vn * (static_cast<Real>(1) - h_params.damping) + force[i] * massInv * h_params.dt;

        xn += vn * h_params.dt;

        if(xn[1] < h_params.boundary_min[1])
        {
            xn[1] = h_params.boundary_min[1];
            vn = -Real(0.2) * vn;
        }
    }
}



template void ElasticitySolverT<float>::SimulateCPU(bool export_results, int frameId);
template void ElasticitySolverT<double>::SimulateCPU(bool export_results, int frameId);

template void ElasticitySolverT<float>::StepCPU();
template void ElasticitySolverT<double>::StepCPU();

template mat3<float> computeF(const Tetrahedron<float>& tet, std::vector<Vector<float, 3>>& vertices);
template mat3<double> computeF(const Tetrahedron<double>& tet, std::vector<Vector<double, 3>>& vertices);
