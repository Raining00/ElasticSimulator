#pragma once

#include "BaseStructure.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

enum PhysicsObjectType
{
    PHYSICS_OBJECT_UNKNOWN = 0,
    PHYSICS_OBJECT_ELASTIC = 1,
};

template <typename Real>
struct PhysicsAABBT
{
    using Vec3 = Vector<Real, 3>;

    Vec3 min = {
        std::numeric_limits<Real>::max(),
        std::numeric_limits<Real>::max(),
        std::numeric_limits<Real>::max()
    };
    Vec3 max = {
        std::numeric_limits<Real>::lowest(),
        std::numeric_limits<Real>::lowest(),
        std::numeric_limits<Real>::lowest()
    };
    bool valid = false;
};

template <typename Real>
struct WorldCollisionSettingsT
{
    using Vec3 = Vector<Real, 3>;

    bool enable_boundary = true;
    bool enable_barrier = true;
    Real barrier_distance = static_cast<Real>(0.02);
    Real barrier_stiffness = static_cast<Real>(5e3);
    Vec3 boundary_min = {
        static_cast<Real>(-10.0),
        static_cast<Real>(0.0),
        static_cast<Real>(-10.0)
    };
    Vec3 boundary_max = {
        static_cast<Real>(10.0),
        static_cast<Real>(10.0),
        static_cast<Real>(10.0)
    };
};

template <typename Real>
struct PhysicsContactT
{
    size_t object_a = 0;
    size_t object_b = 0;
    PhysicsAABBT<Real> overlap;
};

template <typename Real>
class PhysicsObjectT
{
public:
    using Vec3 = Vector<Real, 3>;
    using AABB = PhysicsAABBT<Real>;
    using CollisionSettings = WorldCollisionSettingsT<Real>;

    virtual ~PhysicsObjectT() = default;

    virtual void AdvanceFrame(bool export_result = false) = 0;
    virtual const Mesh<Real>& GetSurfaceMesh() const = 0;
    virtual const Vec3* GetDeviceVertices() const = 0;
    virtual size_t GetVertexCount() const = 0;
    virtual PhysicsObjectType GetObjectType() const = 0;
    virtual AABB GetWorldBounds() const = 0;
    virtual void SetWorldCollisionSettings(const CollisionSettings& settings) = 0;
    virtual bool EnableWorldCoupling() const { return true; }
};

template <typename Real>
class PhysicsWorldT
{
public:
    using Object = PhysicsObjectT<Real>;
    using AABB = PhysicsAABBT<Real>;
    using CollisionSettings = WorldCollisionSettingsT<Real>;
    using Contact = PhysicsContactT<Real>;

    void AddObject(Object& object)
    {
        objects.push_back(&object);
    }

    bool RemoveObject(Object& object)
    {
        auto it = std::find(objects.begin(), objects.end(), &object);
        if (it == objects.end())
            return false;

        objects.erase(it);
        return true;
    }

    void ClearObjects()
    {
        objects.clear();
    }

    CollisionSettings& GetCollisionSettings()
    {
        return collision_settings;
    }

    const CollisionSettings& GetCollisionSettings() const
    {
        return collision_settings;
    }

    void AdvanceFrame(bool export_result = false)
    {
        contacts.clear();

        for (Object* object : objects) {
            if (object != nullptr) {
                object->SetWorldCollisionSettings(collision_settings);
                object->AdvanceFrame(export_result);
            }
        }

        BuildContactPairs();
    }

    size_t GetObjectCount() const
    {
        return objects.size();
    }

    const std::vector<Object*>& GetObjects() const
    {
        return objects;
    }

    const std::vector<Contact>& GetContacts() const
    {
        return contacts;
    }

private:
    static AABB ComputeAABBOverlap(const AABB& a, const AABB& b)
    {
        AABB overlap;
        if (!a.valid || !b.valid)
            return overlap;

        overlap.min = {
            std::max(a.min.x, b.min.x),
            std::max(a.min.y, b.min.y),
            std::max(a.min.z, b.min.z)
        };
        overlap.max = {
            std::min(a.max.x, b.max.x),
            std::min(a.max.y, b.max.y),
            std::min(a.max.z, b.max.z)
        };
        overlap.valid =
            overlap.min.x <= overlap.max.x &&
            overlap.min.y <= overlap.max.y &&
            overlap.min.z <= overlap.max.z;
        return overlap;
    }

    void BuildContactPairs()
    {
        for (size_t i = 0; i < objects.size(); ++i) {
            Object* object_a = objects[i];
            if (object_a == nullptr || !object_a->EnableWorldCoupling())
                continue;

            const AABB bounds_a = object_a->GetWorldBounds();
            for (size_t j = i + 1; j < objects.size(); ++j) {
                Object* object_b = objects[j];
                if (object_b == nullptr || !object_b->EnableWorldCoupling())
                    continue;

                const AABB overlap = ComputeAABBOverlap(bounds_a, object_b->GetWorldBounds());
                if (!overlap.valid)
                    continue;

                contacts.push_back(Contact{ i, j, overlap });
            }
        }
    }

    CollisionSettings collision_settings;
    std::vector<Object*> objects;
    std::vector<Contact> contacts;
};

using PhysicsObjectf = PhysicsObjectT<float>;
using PhysicsObjectd = PhysicsObjectT<double>;
using PhysicsWorldf = PhysicsWorldT<float>;
using PhysicsWorldd = PhysicsWorldT<double>;
using PhysicsWorld = PhysicsWorldf;
using PhysicsAABBf = PhysicsAABBT<float>;
using PhysicsAABBd = PhysicsAABBT<double>;
using WorldCollisionSettingsf = WorldCollisionSettingsT<float>;
using WorldCollisionSettingsd = WorldCollisionSettingsT<double>;
