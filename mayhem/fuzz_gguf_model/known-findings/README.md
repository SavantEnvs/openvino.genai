# Known findings

- `gguf_set_data_offset-segv.bin`: SEGV at `gguflib.c:286` in `gguf_set_data_offset` (gguflib, bac796a). A tensor/KV count or length taken from the file is used to walk past the end of the mapping with no bounds check. Also hits `fuzz_gguf_lora`. Fix: bounds-check the walk against `ctx->size` in gguflib's parser.

- `gguf_get_key-segv.bin`: SEGV at `gguflib.c:244` in `gguf_get_key`; same class (a length read from the file is trusted).
- `q4_1-implicit-conversion.bin`: UBSan `building_blocks.cpp:670:34: -23 is outside the range of representable values of type 'unsigned char'` in the legacy GGUF Q4_1 dequant path. Fix: cast through `int8_t`/mask explicitly before narrowing.
- Reproduce with `/mayhem/fuzz_gguf_model-standalone <file>`.
