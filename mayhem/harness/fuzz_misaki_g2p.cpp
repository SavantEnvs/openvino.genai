// Kokoro TTS front end: misaki English grapheme-to-phoneme conversion (tokenizer, lexicon lookup,
// number/currency/time expansion, stress and phoneme assembly).
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>

#include "misaki/g2p.hpp"

namespace {
const misaki::G2P& engine(bool british) {
    static const std::unique_ptr<misaki::G2P> us = misaki::make_english_engine("en-us");
    static const std::unique_ptr<misaki::G2P> gb = misaki::make_english_engine("en-gb");
    return british ? *gb : *us;
}
}  // namespace

// The lexicon JSON files are not part of the repository; the image caches the real misaki/data lexicon
// (hexgrad/misaki, the data misaki_cpp/README.md points at) under /opt/toolchains/deps.
extern "C" int LLVMFuzzerInitialize(int*, char***) {
    misaki::set_english_lexicon_data_root(MAYHEM_MISAKI_DATA);
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) {
        return -1;
    }
    try {
        const misaki::G2P& g2p = engine(data[0] & 1);
        misaki::PhonemizeResult result = g2p.phonemize_with_tokens(std::string(reinterpret_cast<const char*>(data + 1), size - 1));
        (void)result.phonemes.size();
    } catch (const std::exception&) {
    }
    return 0;
}
