// generation_config.json parsing: ov::genai::GenerationConfig(path) and the validation that follows.
#include <cstddef>
#include <cstdint>
#include <exception>

#include "openvino/genai/generation_config.hpp"
#include "temp_file.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0) {
        return -1;
    }
    TempFile file(data, size, ".json");
    if (!file.ok()) {
        return -1;
    }
    try {
        ov::genai::GenerationConfig config(file.path());
        config.validate();
        (void)config.is_greedy_decoding();
        (void)config.is_beam_search();
        (void)config.is_multinomial();
        (void)config.is_assisting_generation();
        (void)config.is_prompt_lookup();
        (void)config.get_max_new_tokens(0);
    } catch (const std::exception&) {
    }
    return 0;
}
