#pragma once

#include <cstddef>
#include <type_traits>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __forceinline__
#define __forceinline__ inline
#endif

template <typename T, int N>
struct Vector;

template <typename T>
struct Vector<T, 2> {
    using value_type = T;
    static constexpr int dimension = 2;

    T x, y;

    __host__ __device__ __forceinline__ T& operator[](int i) { return (&x)[i]; }
    __host__ __device__ __forceinline__ const T& operator[](int i) const { return (&x)[i]; }
};

template <typename T>
struct Vector<T, 3> {
    using value_type = T;
    static constexpr int dimension = 3;

    T x, y, z;

    __host__ __device__ __forceinline__ T& operator[](int i) { return (&x)[i]; }
    __host__ __device__ __forceinline__ const T& operator[](int i) const { return (&x)[i]; }
};

template <typename T>
struct Vector<T, 4> {
    using value_type = T;
    static constexpr int dimension = 4;

    T x, y, z, w;

    __host__ __device__ __forceinline__ T& operator[](int i) { return (&x)[i]; }
    __host__ __device__ __forceinline__ const T& operator[](int i) const { return (&x)[i]; }
};

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator-(const Vector<T, N>& v)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = -v[i];
    return r;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator+(const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = a[i] + b[i];
    return r;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N>& operator+=(Vector<T, N>& a, const Vector<T, N>& b)
{
    for (int i = 0; i < N; ++i) a[i] += b[i];
    return a;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator+(const Vector<T, N>& a, T s)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = a[i] + s;
    return r;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator+(T s, const Vector<T, N>& a)
{
    return a + s;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N>& operator+=(Vector<T, N>& a, T s)
{
    for (int i = 0; i < N; ++i) a[i] += s;
    return a;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator-(const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = a[i] - b[i];
    return r;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N>& operator-=(Vector<T, N>& a, const Vector<T, N>& b)
{
    for (int i = 0; i < N; ++i) a[i] -= b[i];
    return a;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator-(const Vector<T, N>& a, T s)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = a[i] - s;
    return r;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator-(T s, const Vector<T, N>& a)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = s - a[i];
    return r;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N>& operator-=(Vector<T, N>& a, T s)
{
    for (int i = 0; i < N; ++i) a[i] -= s;
    return a;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator*(const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = a[i] * b[i];
    return r;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N>& operator*=(Vector<T, N>& a, const Vector<T, N>& b)
{
    for (int i = 0; i < N; ++i) a[i] *= b[i];
    return a;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator*(const Vector<T, N>& a, T s)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = a[i] * s;
    return r;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator*(T s, const Vector<T, N>& a)
{
    return a * s;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N>& operator*=(Vector<T, N>& a, T s)
{
    for (int i = 0; i < N; ++i) a[i] *= s;
    return a;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator/(const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = a[i] / b[i];
    return r;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N>& operator/=(Vector<T, N>& a, const Vector<T, N>& b)
{
    for (int i = 0; i < N; ++i) a[i] /= b[i];
    return a;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator/(const Vector<T, N>& a, T s)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = a[i] / s;
    return r;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N>& operator/=(Vector<T, N>& a, T s)
{
    for (int i = 0; i < N; ++i) a[i] /= s;
    return a;
}

template <typename T, int N>
__host__ __device__ __forceinline__ Vector<T, N> operator/(T s, const Vector<T, N>& a)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = s / a[i];
    return r;
}

using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;

using Vec2d = Vector<double, 2>;
using Vec3d = Vector<double, 3>;
using Vec4d = Vector<double, 4>;

using Vec2i = Vector<int, 2>;
using Vec3i = Vector<int, 3>;
using Vec4i = Vector<int, 4>;

using Vec2u = Vector<unsigned int, 2>;
using Vec3u = Vector<unsigned int, 3>;
using Vec4u = Vector<unsigned int, 4>;
