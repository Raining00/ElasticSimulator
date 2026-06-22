#pragma once

#include "BaseStructure.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

template <typename Real>
struct SkeletonBonePoseT
{
    using Vec3 = Vector<Real, 3>;

    std::string name;
    int parent = -1;
    std::string parent_name;
    Vec3 head = { Real(0), Real(0), Real(0) };
    Vec3 tail = { Real(0), Real(0), Real(0) };
    std::array<Real, 16> world_matrix{};
};

template <typename Real>
struct SkeletonFrameT
{
    int source_frame = 0;
    int output_index = 0;
    Real time_seconds = Real(0);
    Real fps = Real(30);
    std::vector<SkeletonBonePoseT<Real>> bones;
};

template <typename Real>
class SkeletonAnimationT
{
public:
    bool LoadFromDirectory(const std::string& directory);

    size_t FrameCount() const { return frames_.size(); }
    bool Empty() const { return frames_.empty(); }
    const SkeletonFrameT<Real>& Frame(size_t index) const { return frames_.at(index); }
    const std::vector<SkeletonFrameT<Real>>& Frames() const { return frames_; }

private:
    std::vector<SkeletonFrameT<Real>> frames_;
};

using SkeletonAnimationf = SkeletonAnimationT<float>;
using SkeletonAnimationd = SkeletonAnimationT<double>;
using SkeletonAnimation = SkeletonAnimationf;
