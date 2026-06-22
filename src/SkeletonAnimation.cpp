#include "SkeletonAnimation.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

class JsonCursor
{
public:
    explicit JsonCursor(std::string text)
        : text_(std::move(text))
    {
    }

    bool End()
    {
        SkipWhitespace();
        return pos_ >= text_.size();
    }

    bool Consume(char expected)
    {
        SkipWhitespace();
        if (pos_ >= text_.size() || text_[pos_] != expected) {
            return false;
        }
        ++pos_;
        return true;
    }

    void Expect(char expected)
    {
        if (!Consume(expected)) {
            throw std::runtime_error(std::string("Expected JSON character '") + expected + "'");
        }
    }

    std::string ParseString()
    {
        SkipWhitespace();
        Expect('"');
        std::string result;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') {
                return result;
            }
            if (c == '\\') {
                if (pos_ >= text_.size()) {
                    throw std::runtime_error("Invalid JSON string escape");
                }
                const char escaped = text_[pos_++];
                if (escaped == '"' || escaped == '\\' || escaped == '/') {
                    result.push_back(escaped);
                } else if (escaped == 'b') {
                    result.push_back('\b');
                } else if (escaped == 'f') {
                    result.push_back('\f');
                } else if (escaped == 'n') {
                    result.push_back('\n');
                } else if (escaped == 'r') {
                    result.push_back('\r');
                } else if (escaped == 't') {
                    result.push_back('\t');
                } else {
                    throw std::runtime_error("Unsupported JSON string escape");
                }
            } else {
                result.push_back(c);
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    double ParseNumber()
    {
        SkipWhitespace();
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin) {
            throw std::runtime_error("Expected JSON number");
        }
        pos_ = static_cast<size_t>(end - text_.c_str());
        return value;
    }

    bool ConsumeNull()
    {
        SkipWhitespace();
        constexpr const char* kNull = "null";
        if (text_.compare(pos_, 4, kNull) == 0) {
            pos_ += 4;
            return true;
        }
        return false;
    }

    void SkipValue()
    {
        SkipWhitespace();
        if (pos_ >= text_.size()) {
            throw std::runtime_error("Unexpected JSON end");
        }

        const char c = text_[pos_];
        if (c == '{') {
            Expect('{');
            if (Consume('}')) {
                return;
            }
            while (true) {
                ParseString();
                Expect(':');
                SkipValue();
                if (Consume('}')) {
                    return;
                }
                Expect(',');
            }
        }

        if (c == '[') {
            Expect('[');
            if (Consume(']')) {
                return;
            }
            while (true) {
                SkipValue();
                if (Consume(']')) {
                    return;
                }
                Expect(',');
            }
        }

        if (c == '"') {
            ParseString();
            return;
        }

        if (ConsumeNull()) {
            return;
        }

        if (text_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return;
        }
        if (text_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return;
        }

        ParseNumber();
    }

    template <typename Real>
    Vector<Real, 3> ParseVec3()
    {
        Expect('[');
        Vector<Real, 3> result{
            static_cast<Real>(ParseNumber()),
            Real(0),
            Real(0)
        };
        Expect(',');
        result[1] = static_cast<Real>(ParseNumber());
        Expect(',');
        result[2] = static_cast<Real>(ParseNumber());
        Expect(']');
        return result;
    }

    template <typename Real>
    std::array<Real, 16> ParseMatrix4()
    {
        std::array<Real, 16> result{};
        Expect('[');
        for (int row = 0; row < 4; ++row) {
            if (row > 0) {
                Expect(',');
            }
            Expect('[');
            for (int col = 0; col < 4; ++col) {
                if (col > 0) {
                    Expect(',');
                }
                result[static_cast<size_t>(row * 4 + col)] =
                    static_cast<Real>(ParseNumber());
            }
            Expect(']');
        }
        Expect(']');
        return result;
    }

private:
    void SkipWhitespace()
    {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    std::string text_;
    size_t pos_ = 0;
};

std::string ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open skeleton file: " + path.string());
    }
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

template <typename Real>
SkeletonBonePoseT<Real> ParseBone(JsonCursor& json)
{
    SkeletonBonePoseT<Real> bone;
    json.Expect('{');
    if (json.Consume('}')) {
        return bone;
    }

    while (true) {
        const std::string key = json.ParseString();
        json.Expect(':');

        if (key == "name") {
            bone.name = json.ParseString();
        } else if (key == "parent") {
            bone.parent = static_cast<int>(json.ParseNumber());
        } else if (key == "parent_name") {
            if (!json.ConsumeNull()) {
                bone.parent_name = json.ParseString();
            }
        } else if (key == "head") {
            bone.head = json.ParseVec3<Real>();
        } else if (key == "tail") {
            bone.tail = json.ParseVec3<Real>();
        } else if (key == "world_matrix") {
            bone.world_matrix = json.ParseMatrix4<Real>();
        } else {
            json.SkipValue();
        }

        if (json.Consume('}')) {
            return bone;
        }
        json.Expect(',');
    }
}

template <typename Real>
void ParseBonesArray(JsonCursor& json, std::vector<SkeletonBonePoseT<Real>>& bones)
{
    json.Expect('[');
    if (json.Consume(']')) {
        return;
    }

    while (true) {
        bones.push_back(ParseBone<Real>(json));
        if (json.Consume(']')) {
            return;
        }
        json.Expect(',');
    }
}

template <typename Real>
void ParseArmaturesArray(JsonCursor& json, SkeletonFrameT<Real>& frame)
{
    json.Expect('[');
    if (json.Consume(']')) {
        return;
    }

    bool loadedFirstArmature = false;
    while (true) {
        json.Expect('{');
        if (!json.Consume('}')) {
            while (true) {
                const std::string key = json.ParseString();
                json.Expect(':');

                if (key == "bones" && !loadedFirstArmature) {
                    ParseBonesArray<Real>(json, frame.bones);
                    loadedFirstArmature = true;
                } else {
                    json.SkipValue();
                }

                if (json.Consume('}')) {
                    break;
                }
                json.Expect(',');
            }
        }

        if (json.Consume(']')) {
            return;
        }
        json.Expect(',');
    }
}

template <typename Real>
SkeletonFrameT<Real> ParseSkeletonFrame(const std::filesystem::path& path)
{
    JsonCursor json(ReadTextFile(path));
    SkeletonFrameT<Real> frame;

    json.Expect('{');
    if (json.Consume('}')) {
        return frame;
    }

    while (true) {
        const std::string key = json.ParseString();
        json.Expect(':');

        if (key == "source_frame") {
            frame.source_frame = static_cast<int>(json.ParseNumber());
        } else if (key == "output_index") {
            frame.output_index = static_cast<int>(json.ParseNumber());
        } else if (key == "time_seconds") {
            frame.time_seconds = static_cast<Real>(json.ParseNumber());
        } else if (key == "fps") {
            frame.fps = static_cast<Real>(json.ParseNumber());
        } else if (key == "armatures") {
            ParseArmaturesArray<Real>(json, frame);
        } else {
            json.SkipValue();
        }

        if (json.Consume('}')) {
            break;
        }
        json.Expect(',');
    }

    if (frame.bones.empty()) {
        throw std::runtime_error("Skeleton frame has no bones: " + path.string());
    }
    if (!json.End()) {
        throw std::runtime_error("Unexpected trailing JSON content: " + path.string());
    }
    return frame;
}

} // namespace

template <typename Real>
bool SkeletonAnimationT<Real>::LoadFromDirectory(const std::string& directory)
{
    frames_.clear();

    const std::filesystem::path dir(directory);
    if (!std::filesystem::exists(dir)) {
        std::cerr << "Skeleton directory does not exist: " << directory << std::endl;
        return false;
    }

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    frames_.reserve(files.size());

    try {
        for (const auto& file : files) {
            frames_.push_back(ParseSkeletonFrame<Real>(file));
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        frames_.clear();
        return false;
    }

    std::cout << "Loaded " << frames_.size()
              << " skeleton frames from " << directory << std::endl;
    return !frames_.empty();
}

template class SkeletonAnimationT<float>;
template class SkeletonAnimationT<double>;
