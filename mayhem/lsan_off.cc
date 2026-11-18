// Leaks are not the defect class fuzzed here; ASan's memory-corruption and UBSan checks stay on.
extern "C" int __lsan_is_turned_off() { return 1; }
