// Chat templates: the Jinja template a model ships in tokenizer_config.json / chat_template.jinja, rendered
// by minja the way Tokenizer::TokenizerImpl::apply_chat_template (src/cpp/src/tokenizer/tokenizer_impl.cpp)
// does it: optional model patcher (utils::patch_chat_template_*), minja::chat_template(src, bos, eos), then
// apply() over a ChatHistory's messages, tools and extra context.
//
// Input: [flags byte][template bytes]\0[JSON]. The JSON is either a messages array or an object with
// "messages", "tools" and "extra_context". Without a NUL the template gets a single user message.
// flags: bit0 add_generation_prompt, bit1 apply_polyfills, bits2-3 patcher (none / undefined-tests /
// multiline-strings).
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

#include "minja/chat-template.hpp"
#include "minja/minja.hpp"
#include "json_utils.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/json_container.hpp"
#include "utils.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) {
        return -1;
    }
    const uint8_t flags = data[0];
    std::string rest(reinterpret_cast<const char*>(data + 1), size - 1);
    std::string tpl = rest;
    std::string json_text;
    const size_t nul = rest.find('\0');
    if (nul != std::string::npos) {
        tpl = rest.substr(0, nul);
        json_text = rest.substr(nul + 1);
    }
    using ov::genai::JsonContainer;
    try {
        switch ((flags >> 2) & 3) {
            case 1: tpl = ov::genai::utils::patch_chat_template_undefined_tests(tpl); break;
            case 2: tpl = ov::genai::utils::patch_chat_template_multiline_strings(tpl); break;
            default: break;
        }

        JsonContainer messages = JsonContainer::array();
        JsonContainer tools = JsonContainer::array();
        JsonContainer extra = JsonContainer::object();
        if (json_text.empty()) {
            JsonContainer msg = JsonContainer::object();
            msg["role"] = "user";
            msg["content"] = "Hello";
            messages.push_back(msg);
        } else {
            JsonContainer json = JsonContainer::from_json_string(json_text);
            if (json.is_array()) {
                messages = json;
            } else if (json.is_object()) {
                if (json.contains("messages")) messages = json["messages"];
                if (json.contains("tools")) tools = json["tools"];
                if (json.contains("extra_context")) extra = json["extra_context"];
            }
        }
        if (!messages.is_array() || !tools.is_array() || !extra.is_object()) {
            return 0;
        }
        ov::genai::ChatHistory history(messages);

        // TokenizerImpl::get_cached_minja_chat_template asserts a non-empty template ("Chat template
        // wasn't found") before it constructs minja::chat_template; mirror that guard.
        if (tpl.empty()) {
            return 0;
        }
        minja::chat_template tmpl(tpl, "<s>", "</s>");
        minja::chat_template_inputs inputs;
        inputs.messages = history.get_messages();
        if (!tools.empty()) {
            inputs.tools = tools;
        }
        inputs.add_generation_prompt = flags & 1;
        inputs.extra_context = nlohmann::ordered_json::object();
        inputs.extra_context["bos_token"] = "<s>";
        inputs.extra_context["eos_token"] = "</s>";
        inputs.extra_context["pad_token"] = "<pad>";
        if (!extra.empty()) {
            inputs.extra_context.update(nlohmann::ordered_json(extra));
        }
        minja::chat_template_options options;
        options.apply_polyfills = (flags & 2) != 0;
        (void)tmpl.apply(inputs, options);
    } catch (const std::exception&) {
    }
    return 0;
}
