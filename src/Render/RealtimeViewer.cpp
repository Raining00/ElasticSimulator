#include "render/RealtimeViewer.h"
#include "render/RealtimeViewerCuda.h"

#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void CheckCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

GLuint CompileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLchar log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        std::cerr << "[OpenGL] Shader compile error: " << log << std::endl;
        glDeleteShader(shader);
        throw std::runtime_error("shader compilation failed");
    }
    return shader;
}

GLuint CreateProgram(const char* vs_src, const char* fs_src)
{
    GLuint vs = CompileShader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, fs_src);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLchar log[1024];
        glGetProgramInfoLog(program, 1024, nullptr, log);
        std::cerr << "[OpenGL] Program link error: " << log << std::endl;
        glDeleteProgram(program);
        throw std::runtime_error("shader linking failed");
    }
    return program;
}
} // namespace
template <typename Real>
struct RealtimeViewer<Real>::Impl
{
    GLFWwindow* window = nullptr;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLuint program = 0;
    GLint u_mvp = -1;

    cudaGraphicsResource* cuda_vbo = nullptr;

    size_t vertex_count = 0;
    size_t index_count = 0;

    int width = 1280;
    int height = 720;

    CameraMode mode = CameraMode::TPS;

    bool lmb_down = false;
    bool mmb_down = false;
    bool first_mouse = true;
    double last_x = 0.0;
    double last_y = 0.0;

    glm::vec3 tps_target = glm::vec3(0.0f);
    float tps_distance = 3.0f;
    float tps_yaw = -90.0f;
    float tps_pitch = 20.0f;

    glm::vec3 fps_position = glm::vec3(0.0f, 0.5f, 3.0f);
    float fps_yaw = -90.0f;
    float fps_pitch = 0.0f;

    float mouse_sensitivity = 0.2f;
    float tps_pan_sensitivity = 0.002f;
    float move_speed = 2.5f;
    bool key1_prev = false;
    bool key2_prev = false;

    std::chrono::steady_clock::time_point last_tick = std::chrono::steady_clock::now();

    glm::vec3 GetTPSPosition() const
    {
        const float yaw = glm::radians(tps_yaw);
        const float pitch = glm::radians(tps_pitch);
        glm::vec3 offset;
        offset.x = tps_distance * std::cos(pitch) * std::cos(yaw);
        offset.y = tps_distance * std::sin(pitch);
        offset.z = tps_distance * std::cos(pitch) * std::sin(yaw);
        return tps_target + offset;
    }

    glm::vec3 GetFPSForward() const
    {
        const float yaw = glm::radians(fps_yaw);
        const float pitch = glm::radians(fps_pitch);
        glm::vec3 forward;
        forward.x = std::cos(yaw) * std::cos(pitch);
        forward.y = std::sin(pitch);
        forward.z = std::sin(yaw) * std::cos(pitch);
        return glm::normalize(forward);
    }

    glm::mat4 GetViewMatrix() const
    {
        if (mode == CameraMode::TPS) {
            return glm::lookAt(GetTPSPosition(), tps_target, glm::vec3(0.0f, 1.0f, 0.0f));
        }
        return glm::lookAt(fps_position, fps_position + GetFPSForward(), glm::vec3(0.0f, 1.0f, 0.0f));
    }

    void HandleCameraSwitch()
    {
        const bool key1 = glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS;
        const bool key2 = glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS;

        if (key1 && !key1_prev) {
            mode = CameraMode::TPS;
        }
        if (key2 && !key2_prev) {
            mode = CameraMode::FPS;
            fps_position = GetTPSPosition();
            const glm::vec3 to_target = glm::normalize(tps_target - fps_position);
            fps_pitch = glm::degrees(std::asin(to_target.y));
            fps_yaw = glm::degrees(std::atan2(to_target.z, to_target.x));
        }

        key1_prev = key1;
        key2_prev = key2;
    }

    void HandleFPSMovement(float dt)
    {
        if (mode != CameraMode::FPS) {
            return;
        }

        const glm::vec3 forward = GetFPSForward();
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        const glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
        const float step = move_speed * dt;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) fps_position += forward * step;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) fps_position -= forward * step;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) fps_position -= right * step;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) fps_position += right * step;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) fps_position += up * step;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) fps_position -= up * step;
    }

    void ProcessMouseDelta(double xpos, double ypos)
    {
        if (first_mouse) {
            last_x = xpos;
            last_y = ypos;
            first_mouse = false;
            return;
        }

        const float dx_raw = static_cast<float>(xpos - last_x);
        const float dy_raw = static_cast<float>(ypos - last_y);
        const float dx = dx_raw * mouse_sensitivity;
        const float dy = -dy_raw * mouse_sensitivity;
        last_x = xpos;
        last_y = ypos;

        if (!lmb_down && !mmb_down) {
            return;
        }

        if (mode == CameraMode::TPS && mmb_down) {
            const glm::vec3 camera_pos = GetTPSPosition();
            const glm::vec3 view_dir = glm::normalize(tps_target - camera_pos);
            const glm::vec3 right = glm::normalize(glm::cross(view_dir, glm::vec3(0.0f, 1.0f, 0.0f)));
            const glm::vec3 up = glm::normalize(glm::cross(right, view_dir));
            const float pan_scale = std::max(tps_distance, 0.2f) * tps_pan_sensitivity;
            tps_target += (-dx_raw * pan_scale) * right + (dy_raw * pan_scale) * up;
            return;
        }

        if (!lmb_down) {
            return;
        }

        if (mode == CameraMode::TPS) {
            tps_yaw += dx;
            tps_pitch = std::clamp(tps_pitch + dy, -85.0f, 85.0f);
        } else {
            fps_yaw += dx;
            fps_pitch = std::clamp(fps_pitch + dy, -85.0f, 85.0f);
        }
    }

    void TickInput()
    {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_tick).count();
        last_tick = now;
        dt = std::min(dt, 0.05f);

        HandleCameraSwitch();
        HandleFPSMovement(dt);
    }

    static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos)
    {
        auto* self = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        if (self != nullptr) {
            self->ProcessMouseDelta(xpos, ypos);
        }
    }

    static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
    {
        (void)mods;
        auto* self = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        if (self == nullptr) {
            return;
        }

        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            self->lmb_down = (action == GLFW_PRESS);
        } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
            self->mmb_down = (action == GLFW_PRESS);
        }
    }

    static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
    {
        (void)xoffset;
        auto* self = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        if (self == nullptr || self->mode != CameraMode::TPS) {
            return;
        }
        self->tps_distance = std::clamp(self->tps_distance - static_cast<float>(yoffset) * 0.2f, 0.2f, 50.0f);
    }

    static void FramebufferSizeCallback(GLFWwindow* window, int width, int height)
    {
        auto* self = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        if (self == nullptr) {
            return;
        }
        self->width = std::max(width, 1);
        self->height = std::max(height, 1);
        glViewport(0, 0, self->width, self->height);
    }
};

template <typename Real>
RealtimeViewer<Real>::RealtimeViewer()
    : impl_(new Impl())
{
}

template <typename Real>
RealtimeViewer<Real>::~RealtimeViewer()
{
    Shutdown();
    delete impl_;
    impl_ = nullptr;
}

template <typename Real>
bool RealtimeViewer<Real>::Initialize(const Mesh<Real>& surface_mesh, int width, int height)
{
    impl_->width = std::max(width, 1);
    impl_->height = std::max(height, 1);

    if (!glfwInit()) {
        std::cerr << "[GLFW] Failed to initialize." << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    impl_->window = glfwCreateWindow(impl_->width, impl_->height, "ElasticSimulator Realtime Viewer", nullptr, nullptr);
    if (impl_->window == nullptr) {
        std::cerr << "[GLFW] Failed to create window." << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(impl_->window);
    glfwSwapInterval(1);

    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cerr << "[OpenGL] gladLoadGL failed." << std::endl;
        Shutdown();
        return false;
    }

    glfwSetWindowUserPointer(impl_->window, impl_);
    glfwSetCursorPosCallback(impl_->window, Impl::CursorPosCallback);
    glfwSetMouseButtonCallback(impl_->window, Impl::MouseButtonCallback);
    glfwSetScrollCallback(impl_->window, Impl::ScrollCallback);
    glfwSetFramebufferSizeCallback(impl_->window, Impl::FramebufferSizeCallback);

    const char* vs = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        uniform mat4 uMVP;
        void main() {
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
    )";

    const char* fs = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(0.85, 0.88, 0.92, 1.0);
        }
    )";

    try {
        impl_->program = CreateProgram(vs, fs);
    } catch (const std::exception&) {
        Shutdown();
        return false;
    }

    impl_->u_mvp = glGetUniformLocation(impl_->program, "uMVP");

    impl_->vertex_count = surface_mesh.vertices.size();
    impl_->index_count = surface_mesh.faces.size() * 3;

    std::vector<uint32_t> indices;
    indices.reserve(impl_->index_count);
    for (const auto& f : surface_mesh.faces) {
        indices.push_back(static_cast<uint32_t>(f.verticesIndex.x));
        indices.push_back(static_cast<uint32_t>(f.verticesIndex.y));
        indices.push_back(static_cast<uint32_t>(f.verticesIndex.z));
    }

    glGenVertexArrays(1, &impl_->vao);
    glGenBuffers(1, &impl_->vbo);
    glGenBuffers(1, &impl_->ebo);

    glBindVertexArray(impl_->vao);

    glBindBuffer(GL_ARRAY_BUFFER, impl_->vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(impl_->vertex_count * sizeof(Vec3f)), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3f), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, impl_->ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(indices.size() * sizeof(uint32_t)),
        indices.data(),
        GL_STATIC_DRAW);

    glBindVertexArray(0);

    try {
        CheckCuda(cudaGraphicsGLRegisterBuffer(&impl_->cuda_vbo, impl_->vbo, cudaGraphicsMapFlagsWriteDiscard),
            "cudaGraphicsGLRegisterBuffer");
    } catch (const std::exception&) {
        Shutdown();
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glViewport(0, 0, impl_->width, impl_->height);

    return true;
}

template <typename Real>
void RealtimeViewer<Real>::Shutdown()
{
    if (impl_ == nullptr) {
        return;
    }

    if (impl_->cuda_vbo != nullptr) {
        cudaGraphicsUnregisterResource(impl_->cuda_vbo);
        impl_->cuda_vbo = nullptr;
    }

    if (impl_->program != 0) {
        glDeleteProgram(impl_->program);
        impl_->program = 0;
    }
    if (impl_->ebo != 0) {
        glDeleteBuffers(1, &impl_->ebo);
        impl_->ebo = 0;
    }
    if (impl_->vbo != 0) {
        glDeleteBuffers(1, &impl_->vbo);
        impl_->vbo = 0;
    }
    if (impl_->vao != 0) {
        glDeleteVertexArrays(1, &impl_->vao);
        impl_->vao = 0;
    }

    if (impl_->window != nullptr) {
        glfwDestroyWindow(impl_->window);
        impl_->window = nullptr;
        glfwTerminate();
    }
}

template <typename Real>
bool RealtimeViewer<Real>::ShouldClose() const
{
    return (impl_->window == nullptr) || (glfwWindowShouldClose(impl_->window) != 0);
}

template <typename Real>
void RealtimeViewer<Real>::PollEvents()
{
    glfwPollEvents();
    impl_->TickInput();
}

template <typename Real>
void RealtimeViewer<Real>::UpdateFromCuda(const Vec3f* d_vertices, size_t vertex_count)
{
    if (impl_->cuda_vbo == nullptr || d_vertices == nullptr || vertex_count != impl_->vertex_count) {
        return;
    }

    try {
        CheckCuda(cudaGraphicsMapResources(1, &impl_->cuda_vbo, 0), "cudaGraphicsMapResources");
        Vec3f* vbo_ptr = nullptr;
        size_t num_bytes = 0;
        CheckCuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vbo_ptr), &num_bytes, impl_->cuda_vbo),
            "cudaGraphicsResourceGetMappedPointer");
        CheckCuda(cudaMemcpy(vbo_ptr, d_vertices, impl_->vertex_count * sizeof(Vec3f), cudaMemcpyDeviceToDevice),
            "cudaMemcpyDeviceToDevice");
        CheckCuda(cudaGraphicsUnmapResources(1, &impl_->cuda_vbo, 0), "cudaGraphicsUnmapResources");
    } catch (const std::exception&) {
        glfwSetWindowShouldClose(impl_->window, GLFW_TRUE);
    }
}

template <typename Real>
void RealtimeViewer<Real>::UpdateFromCuda(const Vec3d* d_vertices, size_t vertex_count)
{
    if (impl_->cuda_vbo == nullptr || d_vertices == nullptr || vertex_count != impl_->vertex_count) {
        return;
    }

    try {
        CheckCuda(cudaGraphicsMapResources(1, &impl_->cuda_vbo, 0), "cudaGraphicsMapResources");
        Vec3f* vbo_ptr = nullptr;
        size_t num_bytes = 0;
        CheckCuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vbo_ptr), &num_bytes, impl_->cuda_vbo),
            "cudaGraphicsResourceGetMappedPointer");
        CheckCuda(ConvertDeviceVerticesD2F(d_vertices, vbo_ptr, impl_->vertex_count),
            "ConvertDeviceVerticesD2F");
        CheckCuda(cudaGraphicsUnmapResources(1, &impl_->cuda_vbo, 0), "cudaGraphicsUnmapResources");
    } catch (const std::exception&) {
        glfwSetWindowShouldClose(impl_->window, GLFW_TRUE);
    }
}

template <typename Real>
void RealtimeViewer<Real>::RenderFrame()
{
    if (impl_->window == nullptr) {
        return;
    }

    const glm::mat4 view = impl_->GetViewMatrix();
    const float aspect = static_cast<float>(impl_->width) / static_cast<float>(impl_->height);
    const glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 200.0f);
    const glm::mat4 mvp = proj * view;

    glClearColor(0.07f, 0.08f, 0.11f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(impl_->program);
    glUniformMatrix4fv(impl_->u_mvp, 1, GL_FALSE, glm::value_ptr(mvp));

    glBindVertexArray(impl_->vao);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(impl_->index_count), GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);

    glfwSwapBuffers(impl_->window);
}

template <typename Real>
typename RealtimeViewer<Real>::CameraMode RealtimeViewer<Real>::GetCameraMode() const
{
    return impl_->mode;
}

template class RealtimeViewer<float>;
template class RealtimeViewer<double>;


