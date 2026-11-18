// safetensors LoRA adapter held in memory: ov::genai::Adapter(ov::Tensor) -> safetensors.h header/JSON
// parsing, tensor extraction, LoRA name matching and alpha/rank grouping.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>

#include "openvino/genai/lora_adapter.hpp"
#include "openvino/runtime/tensor.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 8) {
        return -1;
    }
    try {
        ov::Tensor blob(ov::element::u8, ov::Shape{size});
        std::memcpy(blob.data<uint8_t>(), data, size);
        ov::genai::Adapter adapter(blob);
    } catch (const std::exception&) {
    }
    return 0;
}
