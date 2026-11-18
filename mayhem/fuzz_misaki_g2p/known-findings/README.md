# Known findings

- `int_below_1000-segv.bin`: SEGV at `en_g2p.cpp:1667` in `EnglishG2P::int_below_1000_to_words`: `under_20[h]` with `h = value / 100` indexes past the 20-entry table when the caller passes a value outside 0..999 (large or negative number text). Fix: split the number into groups of three digits before calling it. Reproduce with `/mayhem/fuzz_misaki_g2p-standalone <file>`.
