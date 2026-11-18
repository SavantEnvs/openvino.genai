# Known findings

- `gguf_set_data_offset-segv.bin`: SEGV at `gguflib.c:286` in `gguf_set_data_offset` (gguflib trusts file-supplied counts/lengths; no bounds check). Reproduce with `/mayhem/fuzz_gguf_lora-standalone <file>`.

