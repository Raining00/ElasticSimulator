#include "../include/MeshToTet.hpp"
#include <fstream>
#include <sstream>
#include <map>
#include <array>
#include <algorithm>

#define TETLIBRARY
#include "tetgen/tetgen.h"

bool loadOBJ(const std::string& filename, Mesh& mesh) {
    std::ifstream in(filename);
    if (!in.is_open()) return false;

    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;

        if (tag == "v") {
            float3 v;
            ss >> v.x >> v.y >> v.z;
            mesh.vertices.push_back(v);
        }
        else if (tag == "f") {
            Triangle f;
            ss >> f.verticesIndex.x >> f.verticesIndex.y >> f.verticesIndex.z;
            f.verticesIndex.x -= 1; // OBJ is 1-based
            f.verticesIndex.y -= 1;
            f.verticesIndex.z -= 1;
            mesh.faces.push_back(f);
        }
    }
    return true;
}

void saveOBJ(const std::string& filename, const Mesh& mesh) {
    std::ofstream out(filename);
    if (!out.is_open()) return;

    for (const auto& v : mesh.vertices) {
        out << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    for (const auto& f : mesh.faces) {
        out << "f " 
            << (f.verticesIndex.x + 1) << " " 
            << (f.verticesIndex.y + 1) << " " 
            << (f.verticesIndex.z + 1) << "\n";
    }
}

void buildTetgenInput(const Mesh& mesh, tetgenio& in) {
    in.firstnumber = 0; // 0-based indexing 

    // vertices
    in.numberofpoints = static_cast<int>(mesh.vertices.size());
    in.pointlist = new REAL[mesh.vertices.size() * 3];
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        in.pointlist[i * 3 + 0] = mesh.vertices[i].x;
        in.pointlist[i * 3 + 1] = mesh.vertices[i].y;
        in.pointlist[i * 3 + 2] = mesh.vertices[i].z;
    }
    // faces
    in.numberoffacets = static_cast<int>(mesh.faces.size());
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];

    for (int i = 0; i < in.numberoffacets; ++i) {
        tetgenio::facet& f = in.facetlist[i];
        f.numberofpolygons = 1;
        f.polygonlist = new tetgenio::polygon[1];
        f.numberofholes = 0;
        f.holelist = nullptr;

        f.polygonlist[0].numberofvertices = 3;
        f.polygonlist[0].vertexlist = new int[3];
        f.polygonlist[0].vertexlist[0] = mesh.faces[i].verticesIndex.x;
        f.polygonlist[0].vertexlist[1] = mesh.faces[i].verticesIndex.y;
        f.polygonlist[0].vertexlist[2] = mesh.faces[i].verticesIndex.z;
    }
}

void tetrahedralizeMesh(const Mesh& mesh, 
    std::vector<Tetrahedron>& tets, 
    std::vector<float3>& vertices)
{
    tetgenio in, out;

    buildTetgenInput(mesh, in);

    char tetgenOptions[] = "pq1.2a0.1"; // quality mesh with max volume 0.1
    tetrahedralize(tetgenOptions, &in, &out);

    // vertices
    vertices.resize(out.numberofpoints);
    for(int i = 0; i < out.numberofpoints; ++i) {
        vertices[i] = make_float3(
            out.pointlist[i * 3 + 0],
            out.pointlist[i * 3 + 1],
            out.pointlist[i * 3 + 2]
        );
    }

    // tets
    tets.resize(out.numberoftetrahedra);
    for(int i = 0; i < out.numberoftetrahedra; ++i) {
        tets[i].verticesIndex = make_int4(
            out.tetrahedronlist[i * 4 + 0],
            out.tetrahedronlist[i * 4 + 1],
            out.tetrahedronlist[i * 4 + 2],
            out.tetrahedronlist[i * 4 + 3]
        );
    }
}

void extractSurfaceTriangles(
    const std::vector<Tetrahedron>& tets,
    const std::vector<float3>& vertices)
{
    struct FaceKey {
        int v[3];

        FaceKey(int a, int b, int c) {
            v[0] = a;
            v[1] = b;
            v[2] = c;
            std::sort(v, v + 3);
        }

        bool operator<(const FaceKey& other) const {
            for (int i = 0; i < 3; i++) {
                if (v[i] != other.v[i])
                    return v[i] < other.v[i];
            }
            return false;
        }
    };

    struct FaceInfo {
        int a, b, c;   // 原始顺序
        int opposite;  // 四面体中的对点
    };

	Mesh surfaceMesh;
    std::map<FaceKey, int> faceCount;
    std::map<FaceKey, FaceInfo> faceMap;

    for (const auto& tet : tets) {
        int a = tet.verticesIndex.x;
        int b = tet.verticesIndex.y;
        int c = tet.verticesIndex.z;
        int d = tet.verticesIndex.w;

        std::array<std::tuple<int,int,int,int>,4> faces = {
            std::make_tuple(a,b,c,d),
            std::make_tuple(a,b,d,c),
            std::make_tuple(a,c,d,b),
            std::make_tuple(b,c,d,a)
        };


        for (auto& f : faces) {
            int i,j,k,op;
            std::tie(i,j,k,op) = f;

            FaceKey key(i,j,k);
            faceCount[key]++;

            if (faceCount[key] == 1) {
                faceMap[key] = { i, j, k, op };
            }
        }
    }

    // The face that only appears once is a surface face
    for (auto& [key, count] : faceCount) {
        if (count != 1) continue;

        FaceInfo info = faceMap[key];

        float3 xa = vertices[info.a];
        float3 xb = vertices[info.b];
        float3 xc = vertices[info.c];
        float3 xd = vertices[info.opposite];

        // calculate normal
        float3 ab{xb.x - xa.x, xb.y - xa.y, xb.z - xa.z};
        float3 ac{xc.x - xa.x, xc.y - xa.y, xc.z - xa.z};

        float3 normal = normalize(make_float3(
            ab.y * ac.z - ab.z * ac.y,
            ab.z * ac.x - ab.x * ac.z,
            ab.x * ac.y - ab.y * ac.x
        ));

        float3 ad{xd.x - xa.x, xd.y - xa.y, xd.z - xa.z};
        double dot = normal.x * ad.x + normal.y * ad.y + normal.z * ad.z;

        if (dot > 0) {
            surfaceMesh.faces.push_back({ make_int3(info.a, info.c, info.b) });
        } else {
            surfaceMesh.faces.push_back({ make_int3(info.a, info.b, info.c) });
        }
    }

    surfaceMesh.vertices = vertices;
    
    printf("Extrace surface %d triangles\n", surfaceMesh.faces.size());
   
	saveOBJ("D:/Code/ElasticSimulator/surface_mesh.obj", surfaceMesh);
}