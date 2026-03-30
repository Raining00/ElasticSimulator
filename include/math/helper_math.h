#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#include "math/Vector.hpp"
#include "math/matrix.hpp"
#include <cmath>

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifndef __CUDACC__
inline __host__ __device__ float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif

inline __host__ __device__ Vec2f make_vec2f(float x, float y) { return Vec2f{x, y}; }
inline __host__ __device__ Vec2f make_vec2f(float s) { return Vec2f{s}; }
inline __host__ __device__ Vec2f make_vec2f(Vec3f a) { return Vec2f{a.x, a.y}; }
inline __host__ __device__ Vec2f make_vec2f(Vec2i a) { return Vec2f{static_cast<float>(a.x), static_cast<float>(a.y)}; }
inline __host__ __device__ Vec2f make_vec2f(Vec2u a) { return Vec2f{static_cast<float>(a.x), static_cast<float>(a.y)}; }

inline __host__ __device__ Vec2d make_vec2d(double x, double y) { return Vec2d{x, y}; }
inline __host__ __device__ Vec2d make_vec2d(double s) { return Vec2d{s}; }
inline __host__ __device__ Vec2d make_vec2d(Vec3d a) { return Vec2d{a.x, a.y}; }
inline __host__ __device__ Vec2d make_vec2d(Vec2i a) { return Vec2d{static_cast<double>(a.x), static_cast<double>(a.y)}; }
inline __host__ __device__ Vec2d make_vec2d(Vec2u a) { return Vec2d{static_cast<double>(a.x), static_cast<double>(a.y)}; }

inline __host__ __device__ Vec2i make_vec2i(int x, int y) { return Vec2i{x, y}; }
inline __host__ __device__ Vec2i make_vec2i(int s) { return Vec2i{s}; }
inline __host__ __device__ Vec2i make_vec2i(Vec3i a) { return Vec2i{a.x, a.y}; }
inline __host__ __device__ Vec2i make_vec2i(Vec2u a) { return Vec2i{static_cast<int>(a.x), static_cast<int>(a.y)}; }
inline __host__ __device__ Vec2i make_vec2i(Vec2f a) { return Vec2i{static_cast<int>(a.x), static_cast<int>(a.y)}; }

inline __host__ __device__ Vec2u make_vec2u(uint x, uint y) { return Vec2u{x, y}; }
inline __host__ __device__ Vec2u make_vec2u(uint s) { return Vec2u{s}; }
inline __host__ __device__ Vec2u make_vec2u(Vec3u a) { return Vec2u{a.x, a.y}; }
inline __host__ __device__ Vec2u make_vec2u(Vec2i a) { return Vec2u{static_cast<uint>(a.x), static_cast<uint>(a.y)}; }

inline __host__ __device__ Vec3f make_vec3f(float x, float y, float z) { return Vec3f{x, y, z}; }
inline __host__ __device__ Vec3f make_vec3f(float s) { return Vec3f{s}; }
inline __host__ __device__ Vec3f make_vec3f(Vec2f a) { return Vec3f{a.x, a.y, 0.0f}; }
inline __host__ __device__ Vec3f make_vec3f(Vec2f a, float s) { return Vec3f{a.x, a.y, s}; }
inline __host__ __device__ Vec3f make_vec3f(Vec4f a) { return Vec3f{a.x, a.y, a.z}; }
inline __host__ __device__ Vec3f make_vec3f(Vec3i a) { return Vec3f{static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z)}; }
inline __host__ __device__ Vec3f make_vec3f(Vec3u a) { return Vec3f{static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z)}; }

inline __host__ __device__ Vec3d make_vec3d(double x, double y, double z) { return Vec3d{x, y, z}; }
inline __host__ __device__ Vec3d make_vec3d(double s) { return Vec3d{s}; }
inline __host__ __device__ Vec3d make_vec3d(Vec2d a) { return Vec3d{a.x, a.y, 0.0f}; }
inline __host__ __device__ Vec3d make_vec3d(Vec2d a, double s) { return Vec3d{a.x, a.y, s}; }
inline __host__ __device__ Vec3d make_vec3d(Vec4d a) { return Vec3d{a.x, a.y, a.z}; }
inline __host__ __device__ Vec3d make_vec3d(Vec3i a) { return Vec3d{static_cast<double>(a.x), static_cast<double>(a.y), static_cast<float>(a.z)}; }
inline __host__ __device__ Vec3d make_vec3d(Vec3u a) { return Vec3d{static_cast<double>(a.x), static_cast<double>(a.y), static_cast<float>(a.z)}; }

inline __host__ __device__ Vec3i make_vec3i(int x, int y, int z) { return Vec3i{x, y, z}; }
inline __host__ __device__ Vec3i make_vec3i(int s) { return Vec3i{s}; }
inline __host__ __device__ Vec3i make_vec3i(Vec2i a) { return Vec3i{a.x, a.y, 0}; }
inline __host__ __device__ Vec3i make_vec3i(Vec2i a, int s) { return Vec3i{a.x, a.y, s}; }
inline __host__ __device__ Vec3i make_vec3i(Vec3u a) { return Vec3i{static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z)}; }
inline __host__ __device__ Vec3i make_vec3i(Vec3f a) { return Vec3i{static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z)}; }

inline __host__ __device__ Vec3u make_vec3u(uint x, uint y, uint z) { return Vec3u{x, y, z}; }
inline __host__ __device__ Vec3u make_vec3u(uint s) { return Vec3u{s}; }
inline __host__ __device__ Vec3u make_vec3u(Vec2u a) { return Vec3u{a.x, a.y, 0}; }
inline __host__ __device__ Vec3u make_vec3u(Vec2u a, uint s) { return Vec3u{a.x, a.y, s}; }
inline __host__ __device__ Vec3u make_vec3u(Vec4u a) { return Vec3u{a.x, a.y, a.z}; }
inline __host__ __device__ Vec3u make_vec3u(Vec3i a) { return Vec3u{static_cast<uint>(a.x), static_cast<uint>(a.y), static_cast<uint>(a.z)}; }

inline __host__ __device__ Vec4f make_vec4f(float x, float y, float z, float w) { return Vec4f{x, y, z, w}; }
inline __host__ __device__ Vec4f make_vec4f(float s) { return Vec4f{s}; }
inline __host__ __device__ Vec4f make_vec4f(Vec3f a) { return Vec4f{a.x, a.y, a.z, 0.0f}; }
inline __host__ __device__ Vec4f make_vec4f(Vec3f a, float w) { return Vec4f{a.x, a.y, a.z, w}; }
inline __host__ __device__ Vec4f make_vec4f(Vec4i a) { return Vec4f{static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z), static_cast<float>(a.w)}; }
inline __host__ __device__ Vec4f make_vec4f(Vec4u a) { return Vec4f{static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z), static_cast<float>(a.w)}; }

inline __host__ __device__ Vec4i make_vec4i(int x, int y, int z, int w) { return Vec4i{x, y, z, w}; }
inline __host__ __device__ Vec4i make_vec4i(int s) { return Vec4i{s}; }
inline __host__ __device__ Vec4i make_vec4i(Vec3i a) { return Vec4i{a.x, a.y, a.z, 0}; }
inline __host__ __device__ Vec4i make_vec4i(Vec3i a, int w) { return Vec4i{a.x, a.y, a.z, w}; }
inline __host__ __device__ Vec4i make_vec4i(Vec4u a) { return Vec4i{static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z), static_cast<int>(a.w)}; }
inline __host__ __device__ Vec4i make_vec4i(Vec4f a) { return Vec4i{static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z), static_cast<int>(a.w)}; }

inline __host__ __device__ Vec4u make_vec4u(uint x, uint y, uint z, uint w) { return Vec4u{x, y, z, w}; }
inline __host__ __device__ Vec4u make_vec4u(uint s) { return Vec4u{s}; }
inline __host__ __device__ Vec4u make_vec4u(Vec3u a) { return Vec4u{a.x, a.y, a.z, 0}; }
inline __host__ __device__ Vec4u make_vec4u(Vec3u a, uint w) { return Vec4u{a.x, a.y, a.z, w}; }
inline __host__ __device__ Vec4u make_vec4u(Vec4i a) { return Vec4u{static_cast<uint>(a.x), static_cast<uint>(a.y), static_cast<uint>(a.z), static_cast<uint>(a.w)}; }

template <typename T, int N>
inline __host__ __device__ Vector<T, N> vmin(const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = (a[i] < b[i]) ? a[i] : b[i];
    return r;
}

template <typename T, int N>
inline __host__ __device__ Vector<T, N> vmax(const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = (a[i] > b[i]) ? a[i] : b[i];
    return r;
}

inline __host__ __device__ Vec2f fminf(Vec2f a, Vec2f b) { return vmin(a, b); }
inline __host__ __device__ Vec3f fminf(Vec3f a, Vec3f b) { return vmin(a, b); }
inline __host__ __device__ Vec4f fminf(Vec4f a, Vec4f b) { return vmin(a, b); }
inline __host__ __device__ Vec2f fmaxf(Vec2f a, Vec2f b) { return vmax(a, b); }
inline __host__ __device__ Vec3f fmaxf(Vec3f a, Vec3f b) { return vmax(a, b); }
inline __host__ __device__ Vec4f fmaxf(Vec4f a, Vec4f b) { return vmax(a, b); }

inline __host__ __device__ Vec2i min(Vec2i a, Vec2i b) { return vmin(a, b); }
inline __host__ __device__ Vec3i min(Vec3i a, Vec3i b) { return vmin(a, b); }
inline __host__ __device__ Vec4i min(Vec4i a, Vec4i b) { return vmin(a, b); }
inline __host__ __device__ Vec2u min(Vec2u a, Vec2u b) { return vmin(a, b); }
inline __host__ __device__ Vec3u min(Vec3u a, Vec3u b) { return vmin(a, b); }
inline __host__ __device__ Vec4u min(Vec4u a, Vec4u b) { return vmin(a, b); }

inline __host__ __device__ Vec2i max(Vec2i a, Vec2i b) { return vmax(a, b); }
inline __host__ __device__ Vec3i max(Vec3i a, Vec3i b) { return vmax(a, b); }
inline __host__ __device__ Vec4i max(Vec4i a, Vec4i b) { return vmax(a, b); }
inline __host__ __device__ Vec2u max(Vec2u a, Vec2u b) { return vmax(a, b); }
inline __host__ __device__ Vec3u max(Vec3u a, Vec3u b) { return vmax(a, b); }
inline __host__ __device__ Vec4u max(Vec4u a, Vec4u b) { return vmax(a, b); }

template <typename T, int N>
inline __host__ __device__ T dot(const Vector<T, N>& a, const Vector<T, N>& b)
{
    T s = T(0);
    for (int i = 0; i < N; ++i) s += a[i] * b[i];
    return s;
}

template <typename T, int N>
inline __host__ __device__ float length(const Vector<T, N>& v)
{
    return sqrtf(static_cast<float>(dot(v, v)));
}

template <typename T, int N>
inline __host__ __device__ Vector<T, N> normalize(const Vector<T, N>& v)
{
    const float inv_len = rsqrtf(static_cast<float>(dot(v, v)));
    return v * static_cast<T>(inv_len);
}

inline __host__ __device__ Vec3f cross(Vec3f a, Vec3f b)
{
    return make_vec3f(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

inline __host__ __device__ Vec3d cross(Vec3d a, Vec3d b)
{
    return make_vec3d(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

inline __host__ __device__ float clamp(float v, float a, float b) { return fmaxf(a, fminf(v, b)); }
inline __host__ __device__ int clamp(int v, int a, int b) { return v < a ? a : (v > b ? b : v); }
inline __host__ __device__ uint clamp(uint v, uint a, uint b) { return v < a ? a : (v > b ? b : v); }

template <typename T, int N>
inline __host__ __device__ Vector<T, N> clamp(const Vector<T, N>& v, T a, T b)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = clamp(v[i], a, b);
    return r;
}

template <typename T, int N>
inline __host__ __device__ Vector<T, N> clamp(const Vector<T, N>& v, const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = clamp(v[i], a[i], b[i]);
    return r;
}

inline __host__ __device__ Vec2f floorf(Vec2f v) { return make_vec2f(floorf(v.x), floorf(v.y)); }
inline __host__ __device__ Vec3f floorf(Vec3f v) { return make_vec3f(floorf(v.x), floorf(v.y), floorf(v.z)); }
inline __host__ __device__ Vec4f floorf(Vec4f v) { return make_vec4f(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w)); }

inline __host__ __device__ float fracf(float v) { return v - floorf(v); }
inline __host__ __device__ Vec2f fracf(Vec2f v) { return make_vec2f(fracf(v.x), fracf(v.y)); }
inline __host__ __device__ Vec3f fracf(Vec3f v) { return make_vec3f(fracf(v.x), fracf(v.y), fracf(v.z)); }
inline __host__ __device__ Vec4f fracf(Vec4f v) { return make_vec4f(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w)); }

inline __host__ __device__ Vec2f fmodf(Vec2f a, Vec2f b) { return make_vec2f(fmodf(a.x, b.x), fmodf(a.y, b.y)); }
inline __host__ __device__ Vec3f fmodf(Vec3f a, Vec3f b) { return make_vec3f(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z)); }
inline __host__ __device__ Vec4f fmodf(Vec4f a, Vec4f b) { return make_vec4f(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w)); }

inline __host__ __device__ Vec2f fabs(Vec2f v) { return make_vec2f(fabsf(v.x), fabsf(v.y)); }
inline __host__ __device__ Vec3f fabs(Vec3f v) { return make_vec3f(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
inline __host__ __device__ Vec4f fabs(Vec4f v) { return make_vec4f(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w)); }

inline __host__ __device__ Vec2i abs(Vec2i v) { return make_vec2i(::abs(v.x), ::abs(v.y)); }
inline __host__ __device__ Vec3i abs(Vec3i v) { return make_vec3i(::abs(v.x), ::abs(v.y), ::abs(v.z)); }
inline __host__ __device__ Vec4i abs(Vec4i v) { return make_vec4i(::abs(v.x), ::abs(v.y), ::abs(v.z), ::abs(v.w)); }

inline __host__ __device__ Vec3f reflect(Vec3f i, Vec3f n) { return i - 2.0f * n * dot(n, i); }

inline __host__ __device__ float lerp(float a, float b, float t) { return a + t * (b - a); }
inline __host__ __device__ Vec2f lerp(Vec2f a, Vec2f b, float t) { return a + t * (b - a); }
inline __host__ __device__ Vec3f lerp(Vec3f a, Vec3f b, float t) { return a + t * (b - a); }
inline __host__ __device__ Vec4f lerp(Vec4f a, Vec4f b, float t) { return a + t * (b - a); }

inline __host__ __device__ float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return y * y * (3.0f - 2.0f * y);
}

inline __host__ __device__ Vec2f smoothstep(Vec2f a, Vec2f b, Vec2f x)
{
    Vec2f y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return y * y * (make_vec2f(3.0f) - make_vec2f(2.0f) * y);
}

inline __host__ __device__ Vec3f smoothstep(Vec3f a, Vec3f b, Vec3f x)
{
    Vec3f y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return y * y * (make_vec3f(3.0f) - make_vec3f(2.0f) * y);
}

inline __host__ __device__ Vec4f smoothstep(Vec4f a, Vec4f b, Vec4f x)
{
    Vec4f y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return y * y * (make_vec4f(3.0f) - make_vec4f(2.0f) * y);
}

template <typename Real>
inline __host__ __device__ mat3<Real> crossProductMatrix(Vector<Real, 3> vec)
{
    mat3<Real> crossProductMat( 0,     -vec[2], vec[1],
                         vec[2], 0,     -vec[0],
                        -vec[1], vec[0], 0);
    return crossProductMat;
}

#endif

