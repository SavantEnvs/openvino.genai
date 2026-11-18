// Output parsers applied to model text: reasoning (<think>) parsers and Llama 3 JSON / pythonic tool-call
// parsers, both as one-shot parsers and as incremental (streaming) parsers fed in fuzz-chosen chunks.
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

#include "openvino/genai/json_container.hpp"
#include "openvino/genai/parsers.hpp"

namespace {

void parse_whole(uint8_t selector, const std::string& text) {
    ov::genai::JsonContainer message = ov::genai::JsonContainer::object();
    message["content"] = text;
    switch (selector % 6) {
    case 0: ov::genai::ReasoningParser(true, true).parse(message); break;
    case 1: ov::genai::ReasoningParser(false, false, "<a>", "</a>").parse(message); break;
    case 2: ov::genai::DeepSeekR1ReasoningParser().parse(message); break;
    case 3: ov::genai::Phi4ReasoningParser().parse(message); break;
    case 4: ov::genai::Llama3JsonToolParser().parse(message); break;
    default: ov::genai::Llama3PythonicToolParser().parse(message); break;
    }
    (void)message.to_json_string();
}

void parse_stream(uint8_t selector, const uint8_t* data, size_t size) {
    ov::genai::ReasoningIncrementalParser plain(selector & 1, selector & 2);
    ov::genai::DeepSeekR1ReasoningIncrementalParser r1;
    ov::genai::IncrementalParser* parser = (selector & 4) ? static_cast<ov::genai::IncrementalParser*>(&r1) : &plain;
    // Every other byte is a chunk length (1..8); the chunk bytes follow it.
    size_t pos = 0;
    while (pos < size) {
        size_t len = 1 + (data[pos] % 8);
        ++pos;
        len = len < size - pos ? len : size - pos;
        std::string delta(reinterpret_cast<const char*>(data + pos), len);
        pos += len;
        ov::genai::JsonContainer delta_message = ov::genai::JsonContainer::object();
        std::string filtered = parser->parse(delta_message, delta);
        (void)filtered;
        (void)delta_message.to_json_string();
    }
    parser->reset();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) {
        return -1;
    }
    uint8_t selector = data[0];
    try {
        if (selector & 0x80) {
            parse_stream(selector, data + 1, size - 1);
        } else {
            parse_whole(selector, std::string(reinterpret_cast<const char*>(data + 1), size - 1));
        }
    } catch (const std::exception&) {
    }
    return 0;
}
