#include "render/RealtimeViewerCuda.h"

__global__ void kConvertDeviceVerticesD2F(const Vec3d* src, Vec3f* dst, int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    const Vec3d v = src[idx];
    dst[idx] = Vec3f{
        static_cast<float>(v.x),
        static_cast<float>(v.y),
        static_cast<float>(v.z)
    };
}

cudaError_t ConvertDeviceVerticesD2F(const Vec3d* src, Vec3f* dst, size_t count)
{
    if (src == nullptr || dst == nullptr) {
        return cudaErrorInvalidValue;
    }

    const int n = static_cast<int>(count);
    if (n <= 0) {
        return cudaSuccess;
    }

    constexpr int kBlockSize = 256;
    const int grid = (n + kBlockSize - 1) / kBlockSize;
    kConvertDeviceVerticesD2F<<<grid, kBlockSize>>>(src, dst, n);
    return cudaGetLastError();
}
