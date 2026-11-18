// GGUF LoRA adapter: ov::genai::Adapter(path.gguf) -> GGUFAdapterImpl (gguflib tensor loading + LoRA grouping).
#include <cstddef>
#include <cstdint>
#include <exception>

#include "openvino/genai/lora_adapter.hpp"
#include "temp_file.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 24) {
        return -1;
    }
    TempFile file(data, size, ".gguf");
    if (!file.ok()) {
        return -1;
    }
    try {
        ov::genai::Adapter adapter(file.path());
    } catch (const std::exception&) {
    }
    return 0;
}
