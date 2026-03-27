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
Mat9x12<Real> computepFpx(const mat3<Real>& DmInv)
{
  const Real m = DmInv(0, 0);
  const Real n = DmInv(0, 1);
  const Real o = DmInv(0, 2);
  const Real p = DmInv(1, 0);
  const Real q = DmInv(1, 1);
  const Real r = DmInv(1, 2);
  const Real s = DmInv(2, 0);
  const Real t = DmInv(2, 1);
  const Real u = DmInv(2, 2);

  const Real t1 = -m - p - s;
  const Real t2 = -n - q - t;
  const Real t3 = -o - r - u;
  Mat9x12<Real> PFPu(static_cast<Real>(0));
  PFPu(0, 0)  = t1;
  PFPu(0, 3)  = m;
  PFPu(0, 6)  = p;
  PFPu(0, 9)  = s;
  PFPu(1, 1)  = t1;
  PFPu(1, 4)  = m;
  PFPu(1, 7)  = p;
  PFPu(1, 10) = s;
  PFPu(2, 2)  = t1;
  PFPu(2, 5)  = m;
  PFPu(2, 8)  = p;
  PFPu(2, 11) = s;
  PFPu(3, 0)  = t2;
  PFPu(3, 3)  = n;
  PFPu(3, 6)  = q;
  PFPu(3, 9)  = t;
  PFPu(4, 1)  = t2;
  PFPu(4, 4)  = n;
  PFPu(4, 7)  = q;
  PFPu(4, 10) = t;
  PFPu(5, 2)  = t2;
  PFPu(5, 5)  = n;
  PFPu(5, 8)  = q;
  PFPu(5, 11) = t;
  PFPu(6, 0)  = t3;
  PFPu(6, 3)  = o;
  PFPu(6, 6)  = r;
  PFPu(6, 9)  = u;
  PFPu(7, 1)  = t3;
  PFPu(7, 4)  = o;
  PFPu(7, 7)  = r;
  PFPu(7, 10) = u;
  PFPu(8, 2)  = t3;
  PFPu(8, 5)  = o;
  PFPu(8, 8)  = r;
  PFPu(8, 11) = u;

  return PFPu;
}

template<typename Real>
void multiplyDenseMatrixVector(
    const std::vector<Real>& A,
    const std::vector<Real>& x,
    std::vector<Real>& y)
{
    const size_t n = x.size();
    y.assign(n, static_cast<Real>(0));
    for (size_t row = 0; row < n; ++row)
    {
        Real sum = static_cast<Real>(0);
        const size_t rowOffset = row * n;
        for (size_t col = 0; col < n; ++col)
            sum += A[rowOffset + col] * x[col];
        y[row] = sum;
    }
}

template<typename Real>
Real dotStdVector(const std::vector<Real>& a, const std::vector<Real>& b)
{
    Real sum = static_cast<Real>(0);
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

template<typename Real>
bool PCG(
    const std::vector<Real>& A,
    const std::vector<Real>& b,
    std::vector<Real>& x,
    int maxIterations,
    Real tolerance)
{
    const size_t n = b.size();
    if (A.size() != n * n || x.size() != n)
        return false;

    std::vector<Real> diagonalInv(n, static_cast<Real>(1));
    for (size_t i = 0; i < n; ++i)
    {
        const Real diag = A[i * n + i];
        if (std::abs(diag) > static_cast<Real>(1e-12))
            diagonalInv[i] = static_cast<Real>(1) / diag;
    }

    std::vector<Real> r(n, static_cast<Real>(0));
    std::vector<Real> z(n, static_cast<Real>(0));
    std::vector<Real> p(n, static_cast<Real>(0));
    std::vector<Real> Ap(n, static_cast<Real>(0));

    multiplyDenseMatrixVector(A, x, Ap);
    for (size_t i = 0; i < n; ++i)
    {
        r[i] = b[i] - Ap[i];
        z[i] = diagonalInv[i] * r[i];
        p[i] = z[i];
    }

    Real rz = dotStdVector(r, z);
    const Real tolerance2 = tolerance * tolerance;
    if (rz <= tolerance2)
        return true;

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        multiplyDenseMatrixVector(A, p, Ap);
        const Real denom = dotStdVector(p, Ap);
        if (std::abs(denom) <= static_cast<Real>(1e-20))
            return false;

        const Real alpha = rz / denom;
        for (size_t i = 0; i < n; ++i)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            z[i] = diagonalInv[i] * r[i];
        }

        const Real rzNew = dotStdVector(r, z);
        if (rzNew <= tolerance2)
        {
            printf("PCG Converged in %d iterations.\n", iter);
            return true;
        }

        const Real beta = rzNew / rz;
        for (size_t i = 0; i < n; ++i)
            p[i] = z[i] + beta * p[i];
        rz = rzNew;
    }
    return false;
}

template<typename Real>
StaticMatrix<Real, 9, 1> partialJpartialF(const mat3<Real>& F)
{
    mat3<Real> A(cross(F.column(1), F.column(2)),
                 cross(F.column(2), F.column(0)),
                 cross(F.column(0), F.column(1)));

    StaticMatrix<Real, 9, 1> pJpF(static_cast<Real>(0));
    unsigned int index = 0;
    for (unsigned int j = 0; j < 3; j++)
    for (unsigned int i = 0; i < 3; i++, index++)
        pJpF(index, 0) = A(i,j);

    return pJpF;
}

template<typename Real>
Mat9x9<Real> computeHessian(const mat3<Real>& F, Real mu, Real lambda, Real alpha)
{
    StaticMatrix<Real, 9, 1> pjpf = partialJpartialF(F);
    const Real I3 = mat3<Real>::determinant(F);
    const Real scale = lambda * (I3 - alpha);
    const mat3<Real> f0hat = crossProductMatrix(F.column(0)) * scale;
    const mat3<Real> f1hat = crossProductMatrix(F.column(1)) * scale;
    const mat3<Real> f2hat = crossProductMatrix(F.column(2)) * scale;

    Mat9x9<Real> hessJ(static_cast<Real>(0));

    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            hessJ(i, j + 3) = -f2hat(i,j);
            hessJ(i + 3, j) =  f2hat(i,j);

            hessJ(i, j + 6) =  f1hat(i,j);
            hessJ(i + 6, j) = -f1hat(i,j);

            hessJ(i + 3, j + 6) = -f0hat(i,j);
            hessJ(i + 6, j + 3) =  f0hat(i,j);
        }
    }
    Mat9x9<Real> I9 = Mat9x9<Real>::Identity();

    return mu * I9 + lambda * pjpf * transpose(pjpf) + hessJ;
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
  
    if (h_params.solverType == EXPLICIT)
    {
        for (int i = 0; i < 10; i++)
        {
            StepCPUExplicit();
        }
        ExportMesh(frameId / 10);
    }
    else
    {
        StepCPUImplicit();
        ExportMesh(frameId);
    }
}

template<typename Real>
void ElasticitySolverT<Real>:: StepCPUExplicit()
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

template<typename Real>
void ElasticitySolverT<Real>:: StepCPUImplicit()
{
    std::vector<Vec3> force(h_vertex.size(), Vec3{ Real(0), Real(0), Real(0) });
    std::vector<Mat12x12<Real>> perElementHessians(h_tet.size(), Mat12x12<Real>(Real(0)));
    Real invDt = Real(1) / h_params.dt;
    Real invDt2 = invDt * invDt;
    unsigned int numVerts = h_vertex.size();
    std::vector<Real> h_DnA(3 * numVerts * 3 * numVerts, 0);
    std::vector<Real> h_b(3 * numVerts, 0);
    std::vector<Real> h_deltaX(3 * numVerts, 0);

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

        Mat9x12<Real> pFpx = computepFpx(DmInv);
        Mat9x9<Real>  hessian = -tet.volume * computeHessian(F, h_params.mu, h_params.lambda, h_params.alpha);
        perElementHessians[i] = (transpose(pFpx) * hessian) * pFpx;
    }   

    for (unsigned int i = 0; i < h_tet.size(); i++)
    {
        const Vec4i& tet = h_tet[i].verticesIndex;
        const Mat12x12<Real>& H = perElementHessians[i];
        for (int y = 0; y < 4; y++)
        {
            int yVertex = tet[y];
            for (int x = 0; x < 4; x++)
            {
                int xVertex = tet[x];
                for (int b = 0; b < 3; b++)
                    for (int a = 0; a < 3; a++)
                    {
                        const Real entry = -H(3 * x + a, 3 * y + b);
                        const size_t row = static_cast<size_t>(3 * xVertex + a);
                        const size_t col = static_cast<size_t>(3 * yVertex + b);
                        h_DnA[row * (3 * numVerts) + col] += entry;
                    }
            }
        }
    }

    for(unsigned int i = 0; i < h_vertex.size(); i++)
    {
        Real m = h_mass[i];
        for(int j = 0; j < 3; j++)
        {
            Real entry = invDt2  * m;
            const size_t diag = static_cast<size_t>(3 * i + j);
            h_DnA[diag * (3 * numVerts) + diag] += entry;
            h_b[diag] = invDt * m * h_velocity[i][j] + force[i][j];
        }
    }

    const int dofs = static_cast<int>(3 * numVerts);
    bool converged = PCG(h_DnA, h_b, h_deltaX, std::max(64, dofs), static_cast<Real>(1e-8));

    for (unsigned int i = 0; i < h_vertex.size(); ++i)
    {
        Vec3 dx{
            h_deltaX[3 * i + 0],
            h_deltaX[3 * i + 1],
            h_deltaX[3 * i + 2]
        };
        h_vertex[i] += dx;
        h_velocity[i] = dx / h_params.dt;

        if (h_vertex[i][1] < h_params.boundary_min[1])
        {
            h_vertex[i][1] = h_params.boundary_min[1];
            h_velocity[i] = -Real(0.2) * h_velocity[i];
        }
    }
}

template void ElasticitySolverT<float>::SimulateCPU(bool export_results, int frameId);
template void ElasticitySolverT<double>::SimulateCPU(bool export_results, int frameId);

template void ElasticitySolverT<float>::StepCPUExplicit();
template void ElasticitySolverT<double>::StepCPUExplicit();

template void ElasticitySolverT<float>::StepCPUImplicit();
template void ElasticitySolverT<double>::StepCPUImplicit();

template mat3<float> computeF(const Tetrahedron<float>& tet, std::vector<Vector<float, 3>>& vertices);
template mat3<double> computeF(const Tetrahedron<double>& tet, std::vector<Vector<double, 3>>& vertices);
