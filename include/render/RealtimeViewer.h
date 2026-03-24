#pragma once

#include "BaseStructure.hpp"
#include <cstddef>
#include <cstdint>

struct GLFWwindow;
struct cudaGraphicsResource;

template<typename Real>
class RealtimeViewer
{
public:
    enum class CameraMode
    {
        TPS = 0,
        FPS = 1
    };

    RealtimeViewer();
    ~RealtimeViewer();

    bool Initialize(
        const Mesh<Real>& surface_mesh,
        const Vector<Real, 3>& boundary_min,
        const Vector<Real, 3>& boundary_max,
        int width = 1280,
        int height = 720);
    void Shutdown();

    bool ShouldClose() const;
    void PollEvents();
    void UpdateFromCuda(const Vec3f* d_vertices, size_t vertex_count);
    void UpdateFromCuda(const Vec3d* d_vertices, size_t vertex_count);
    void RenderFrame();

    CameraMode GetCameraMode() const;

private:
    struct Impl;
    Impl* impl_;
};


