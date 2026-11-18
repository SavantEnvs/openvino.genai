// GGUF file -> ov::Model through the legacy (gguflib) reader: metadata + tensor parsing, quantized
// dequantization (Q4_0/Q4_1/Q8_0/Q4_K/Q6_K) and the llama/qwen graph builder.
#include <cstddef>
#include <cstdint>
#include <exception>

#include "gguf_utils/gguf_modeling.hpp"
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
        create_from_gguf(file.path(), false, true);
    } catch (const std::exception&) {
    }
    return 0;
}
