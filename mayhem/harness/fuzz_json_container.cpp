// ov::genai::JsonContainer (the JSON type behind chat messages and parser output): parse, serialize,
// navigate and mutate, then build a ChatHistory from it.
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/json_container.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0) {
        return -1;
    }
    using ov::genai::JsonContainer;
    try {
        JsonContainer json = JsonContainer::from_json_string(std::string(reinterpret_cast<const char*>(data), size));
        std::string dumped = json.to_json_string();
        JsonContainer again = JsonContainer::from_json_string(dumped);
        if (!(again == json)) {
            __builtin_trap();
        }
        (void)json.to_json_string(2);
        (void)json.type_name();
        (void)json.size();
        (void)json.empty();
        if (json.is_array()) {
            for (size_t i = 0; i < json.size() && i < 8; ++i) {
                JsonContainer item = json[i];
                (void)item.to_json_string();
                (void)item.as_string();
                (void)item.as_int();
                (void)item.as_double();
                (void)item.as_bool();
            }
            ov::genai::ChatHistory history(json);
            (void)history.size();
            (void)history.get_messages().to_json_string();
        }
        if (json.is_object()) {
            for (const char* key : {"role", "content", "tool_calls", "reasoning_content", "name"}) {
                if (json.contains(key)) {
                    JsonContainer value = json[key];
                    (void)value.to_json_string();
                    (void)value.as_string();
                }
            }
            json["fuzz"] = std::string("x");
            json.erase("fuzz");
        }
    } catch (const std::exception&) {
    }
    return 0;
}
