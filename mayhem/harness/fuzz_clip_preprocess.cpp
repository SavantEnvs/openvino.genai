// LLaVA image preprocessing (bicubic resize, center crop, normalize) over an attacker-sized image and
// processor config. Input: 3 x u16 (shortest edge, crop height, crop width), u16 width, u16 height, pixels.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>

#include "visual_language/clip.hpp"
#include "visual_language/processor_config.hpp"

namespace ov::genai {
clip_image_f32 preprocess_clip_image_llava(const clip_image_u8& image, const ProcessorConfig& config);
}

namespace {
uint16_t u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    constexpr size_t kHeader = 10;
    if (size < kHeader) {
        return -1;
    }
    const int shortest_edge = u16(data) % 1025;
    const int crop_h = u16(data + 2) % 1025;
    const int crop_w = u16(data + 4) % 1025;
    const int nx = u16(data + 6) % 257;
    const int ny = u16(data + 8) % 257;
    if (shortest_edge == 0 || crop_h == 0 || crop_w == 0 || nx == 0 || ny == 0) {
        return -1;
    }
    clip_image_u8 image;
    image.nx = nx;
    image.ny = ny;
    image.buf.assign(static_cast<size_t>(nx) * ny * 3, 0);
    std::memcpy(image.buf.data(), data + kHeader, std::min(image.buf.size(), size - kHeader));

    ov::genai::ProcessorConfig config;
    config.size_shortest_edge = shortest_edge;
    config.crop_size_height = crop_h;
    config.crop_size_width = crop_w;
    try {
        clip_image_f32 out = ov::genai::preprocess_clip_image_llava(image, config);
        (void)out.buf.size();
    } catch (const std::exception&) {
    }
    return 0;
}
