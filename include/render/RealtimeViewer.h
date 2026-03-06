#pragma once

#include "BaseStructure.hpp"
#include <cstddef>
#include <cstdint>

struct GLFWwindow;
struct cudaGraphicsResource;

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

    bool Initialize(const Mesh& surface_mesh, int width = 1280, int height = 720);
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


