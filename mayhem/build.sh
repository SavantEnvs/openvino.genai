#!/usr/bin/env bash
#
# mayhem/build.sh — builds, inside the commit image (as `mayhem`, in /mayhem):
#   1. the OpenVINO runtime unpacked from the cached archive (offline),
#   2. openvino_genai_obj (the library's objects) with $SANITIZER_FLAGS, DWARF 3 and SanCov,
#   3. every fuzz harness + its -standalone reproducer,
#   4. the upstream gtest suite (tests_continuous_batching) with NORMAL flags, for mayhem/test.sh.
# Re-runnable and offline: every dependency comes from /opt/toolchains/deps.
set -euo pipefail

[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
: "${SANITIZER_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer}"
: "${DEBUG_FLAGS:=-g -gdwarf-3}"
: "${CC:=clang}" ; : "${CXX:=clang++}" ; : "${LIB_FUZZING_ENGINE:=-fsanitize=fuzzer}"
: "${MAYHEM_JOBS:=$(nproc)}"
: "${COVERAGE_FLAGS=}"
: "${SRC:=/mayhem}"
: "${STANDALONE_FUZZ_MAIN:=/opt/mayhem/StandaloneFuzzTargetMain.c}"
export SANITIZER_FLAGS DEBUG_FLAGS CC CXX LIB_FUZZING_ENGINE MAYHEM_JOBS COVERAGE_FLAGS

cd "$SRC"

DEPS=/opt/toolchains/deps
OV_VER=2027.0.0.dev20261117
OV_TGZ="$DEPS/openvino-$OV_VER/src/openvino.tgz"
OV="$SRC/build-ov"
OV_LIB="$OV/runtime/lib/intel64"
TBB_LIB="$OV/runtime/3rdparty/tbb/lib"
RPATH="-Wl,--disable-new-dtags -Wl,-rpath,$OV_LIB -Wl,-rpath,$TBB_LIB"

# 1) OpenVINO runtime. The dependency cache may not hold .so files (SPEC §6.2 item 19), so the archive
#    is unpacked into the (gitignored) build tree; the GPU/NPU plugins are not needed and are dropped.
if [ ! -f "$OV_LIB/libopenvino.so" ]; then
  rm -rf "$OV" && mkdir -p "$OV"
  tar -xzf "$OV_TGZ" -C "$OV" --strip-components=1 --wildcards '*/runtime'
  rm -f "$OV_LIB"/libopenvino_intel_gpu_plugin.so "$OV_LIB"/libopenvino_intel_npu_* "$OV_LIB"/cache.json
fi

# CMake gets every FetchContent source from the cache and may not touch the network.
cmake_common=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DOpenVINO_DIR="$OV/runtime/cmake"
  -DBUILD_TOKENIZERS=OFF -DENABLE_PYTHON=OFF -DENABLE_SAMPLES=OFF -DENABLE_TOOLS=OFF
  -DENABLE_XGRAMMAR=OFF
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$DEPS/nlohmann_json-3.11.3/src"
  -DFETCHCONTENT_SOURCE_DIR_MINJA="$DEPS/minja-3e4c61c616eda133cfb1e440fc7a14bf1729bbee/src"
  -DFETCHCONTENT_SOURCE_DIR_SAFETENSORS.H="$DEPS/safetensors.h-974a85d7dfd6e010558353226638bb26d6b9d756/src"
  -DFETCHCONTENT_SOURCE_DIR_GGUFLIB="$DEPS/gguflib-bac796ada809ac293e685db59b075971181cb008/src"
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$DEPS/googletest-03597a01ee50ed33e9dfd640b249b4be3799d395/src"
)

# 2) The library with the sanitizers + SanCov (fuzzer-no-link), so the fuzzed code is instrumented.
# gguflib and the GGUF reader load unaligned fields straight from the file mapping (misaligned loads are the
# format's normal layout and benign on x86); that one UBSan check is off for the library so valid GGUF files
# are not rejected. ASan and every other UBSan check stay on and halting.
FUZZ_FLAGS="$SANITIZER_FLAGS -fno-sanitize=alignment -fsanitize=fuzzer-no-link $DEBUG_FLAGS"
cmake -S . -B build-fuzz "${cmake_common[@]}" -DENABLE_TESTS=OFF \
  -DCMAKE_C_FLAGS="$FUZZ_FLAGS" -DCMAKE_CXX_FLAGS="$FUZZ_FLAGS" \
  -DCMAKE_C_FLAGS_RELEASE="-O1 -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O1 -DNDEBUG"
cmake --build build-fuzz -j"$MAYHEM_JOBS" --target openvino_genai_obj gguflib openvino_genai_misaki_cpp

# 3) Harnesses. The library objects go into one static archive, so each link only pulls what it needs.
GENAI_OBJ_DIR=build-fuzz/src/cpp/CMakeFiles/openvino_genai_obj.dir
rm -f build-fuzz/libgenai_fuzz.a
find "$GENAI_OBJ_DIR" -name '*.o' | sort | xargs llvm-ar rcs build-fuzz/libgenai_fuzz.a
GGUFLIB_A=$(find build-fuzz -name libgguflib.a | head -1)
MISAKI_A=$(find build-fuzz -name libopenvino_genai_misaki_cpp.a | head -1)

INCLUDES=(
  -I"$SRC/src/cpp/src" -I"$SRC/src/cpp/include" -I"$SRC/build-fuzz/src/cpp"
  -I"$OV/runtime/include"
  -I"$DEPS/nlohmann_json-3.11.3/src/include"
  -I"$DEPS/gguflib-bac796ada809ac293e685db59b075971181cb008/src"
  -I"$SRC/src/cpp/src/speech_generation/misaki_cpp/include"
)
LIBS=(
  build-fuzz/libgenai_fuzz.a "$GGUFLIB_A" "$MISAKI_A"
  -L"$OV_LIB" -L"$TBB_LIB" -lopenvino_gguf_frontend -lopenvino -ltbb -ldl -lpthread
)

HARNESSES=(fuzz_gguf_model fuzz_gguf_lora fuzz_safetensors_lora fuzz_json_container fuzz_parsers
           fuzz_generation_config fuzz_misaki_g2p fuzz_clip_preprocess)

BUILD_DIR="$SRC/build-fuzz"
$CXX $SANITIZER_FLAGS $DEBUG_FLAGS -c "$SRC/mayhem/lsan_off.cc" -o "$BUILD_DIR/lsan_off.o"
$CC $SANITIZER_FLAGS $DEBUG_FLAGS -c "$STANDALONE_FUZZ_MAIN" -o "$BUILD_DIR/standalone_main.o"

for h in "${HARNESSES[@]}"; do
  src="$SRC/mayhem/harness/$h.cpp"
  $CXX -std=c++17 -O1 -DNDEBUG $SANITIZER_FLAGS -fno-sanitize=alignment $DEBUG_FLAGS "${INCLUDES[@]}" \
       -DMAYHEM_MISAKI_DATA="\"$SRC/mayhem/misaki_data\"" -I"$SRC/mayhem/harness" \
       -c "$src" -o "$BUILD_DIR/$h.o"
  $CXX $SANITIZER_FLAGS $DEBUG_FLAGS $LIB_FUZZING_ENGINE "$BUILD_DIR/$h.o" "$BUILD_DIR/lsan_off.o" \
       "${LIBS[@]}" $RPATH -o "/mayhem/$h"
  $CXX $SANITIZER_FLAGS $DEBUG_FLAGS "$BUILD_DIR/$h.o" "$BUILD_DIR/lsan_off.o" "$BUILD_DIR/standalone_main.o" \
       "${LIBS[@]}" $RPATH -o "/mayhem/$h-standalone"
done

# 4) The upstream gtest suite with the project's NORMAL flags (no sanitizers), for mayhem/test.sh.
cmake -S . -B build-tests "${cmake_common[@]}" -DENABLE_TESTS=ON \
  -DCMAKE_C_FLAGS="$COVERAGE_FLAGS" -DCMAKE_CXX_FLAGS="$COVERAGE_FLAGS"
cmake --build build-tests -j"$MAYHEM_JOBS" --target tests_continuous_batching
