#pragma once

#include "math/Vector.hpp"
#include <cstddef>
#include <cuda_runtime_api.h>

cudaError_t ConvertDeviceVerticesD2F(const Vec3d* src, Vec3f* dst, size_t count);
