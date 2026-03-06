#include "MeshToTet.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <array>
#include <algorithm>

#define TETLIBRARY
#include "tetgen/tetgen.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

bool loadOBJ(const std::string& filename, Mesh& mesh) {
    std::ifstream in(filename);
    if (!in.is_open()) return false;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str())) {
        std::cerr << warn << err << std::endl;
        return false;
    }
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        mesh.vertices.push_back(make_vec3f(
            attrib.vertices[i + 0],
            attrib.vertices[i + 1],
            attrib.vertices[i + 2]
        ));
    }
    // faces
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                std::cerr << "Non-triangle face found. Skipping." << std::endl;
                index_offset += fv;
                continue;
            }
            tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];
            mesh.faces.push_back({ make_vec3i(idx0.vertex_index, idx1.vertex_index, idx2.vertex_index) });
            index_offset += fv;
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
    std::vector<Vec3f>& vertices)
{
    tetgenio in, out;

    buildTetgenInput(mesh, in);

    char tetgenOptions[] = "pq1.5/20Y"; 
    tetrahedralize(tetgenOptions, &in, &out);

    // vertices
    vertices.resize(out.numberofpoints);
    for(int i = 0; i < out.numberofpoints; ++i) {
        vertices[i] = make_vec3f(
            out.pointlist[i * 3 + 0],
            out.pointlist[i * 3 + 1],
            out.pointlist[i * 3 + 2]
        );
    }

    // tets
    tets.resize(out.numberoftetrahedra);
    for(int i = 0; i < out.numberoftetrahedra; ++i) {
        tets[i].verticesIndex = make_vec4i(
            out.tetrahedronlist[i * 4 + 0],
            out.tetrahedronlist[i * 4 + 1],
            out.tetrahedronlist[i * 4 + 2],
            out.tetrahedronlist[i * 4 + 3]
        );
    }
}

void extractSurfaceTriangles(
    const std::vector<Tetrahedron>& tets,
    const std::vector<Vec3f>& vertices,
    Mesh& surfaceMesh)
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
        int a, b, c;   // original vertex indices of the face
        int opposite;  // corresponding opposite vertex in the tet (the one not in the face)
    };

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

        Vec3f xa = vertices[info.a];
        Vec3f xb = vertices[info.b];
        Vec3f xc = vertices[info.c];
        Vec3f xd = vertices[info.opposite];

        // calculate normal
        Vec3f ab{xb.x - xa.x, xb.y - xa.y, xb.z - xa.z};
        Vec3f ac{xc.x - xa.x, xc.y - xa.y, xc.z - xa.z};

        Vec3f normal = normalize(make_vec3f(
            ab.y * ac.z - ab.z * ac.y,
            ab.z * ac.x - ab.x * ac.z,
            ab.x * ac.y - ab.y * ac.x
        ));

        Vec3f ad{xd.x - xa.x, xd.y - xa.y, xd.z - xa.z};
        double dot = normal.x * ad.x + normal.y * ad.y + normal.z * ad.z;

        if (dot > 0) {
            surfaceMesh.faces.push_back({ make_vec3i(info.a, info.b, info.c) });
        } else {
            surfaceMesh.faces.push_back({ make_vec3i(info.a, info.c, info.b) });
        }
    }

    surfaceMesh.vertices = vertices;
    
    printf("Extrace surface %d triangles\n", surfaceMesh.faces.size());
   
	// saveOBJ("D:/Code/ElasticSimulator/surface_mesh.obj", surfaceMesh);
}

