# Known findings

- `eat-heap-overflow.bin`: heap-buffer-overflow read at `safetensors.h:232` in `eat` (8-byte file, header length 0). The parser reads the header without checking the buffer size. Fix: check `size >= 8 + header_len` before parsing.

- `eat_string-heap-overflow.bin`: heap-buffer-overflow read at `safetensors.h:301` in `eat_string`; the string length is not checked against the remaining buffer. Reproduce with `/mayhem/fuzz_safetensors_lora-standalone <file>`.
