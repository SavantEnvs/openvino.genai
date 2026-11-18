# syntax=docker/dockerfile:1.7
#
# Commit image for openvino.genai (SMALL form, on the org C/C++ base). The upstream build fetches
# its dependencies at configure time and needs an OpenVINO runtime, so everything it would download
# is cached below under /opt/toolchains/deps and mayhem/build.sh runs offline.
FROM ghcr.io/savantenvs/base:20261023

ARG COMMIT
ARG REPO_URL
ARG SOURCE_DATE_EPOCH
ARG SANITIZER_FLAGS
ARG COVERAGE_FLAGS=
ARG MAYHEM_JOBS=

# OpenVINO runtime (headers, cmake config, libopenvino + GGUF frontend + TBB): pinned nightly archive.
# GenAI master needs OpenVINO 2027.0 with the GGUF frontend, which no release or distro package has yet.
# It is a prebuilt binary distribution, so it stays a .tgz here; build.sh unpacks it under /mayhem.
# The other dependencies are the sources upstream's CMake FetchContent would download (same URL + hash).
USER root
ARG OV_VER=2027.0.0.dev20261117
ARG OV_BUILD=2027.0.0-23916-36ba7864363
ARG OV_SHA256=a2ca19694109d0b005ae0242193efe9b7d2e0b6c0696a093fab2cf7873ca8e00
RUN d=/opt/toolchains/deps/openvino-${OV_VER} && mkdir -p "$d/src" \
 && curl -fsSL -o "$d/src/openvino.tgz" \
      "https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/${OV_BUILD}/openvino_toolkit_ubuntu24_${OV_VER}_x86_64.tgz" \
 && echo "${OV_SHA256}  $d/src/openvino.tgz" | sha256sum -c - && chmod -R a+rX "$d"

RUN fetch() { \
      name="$1"; ver="$2"; url="$3"; sha="$4"; kind="$5"; \
      d=/opt/toolchains/deps/${name}-${ver}; mkdir -p "$d/src"; \
      curl -fsSL -o /tmp/dep.archive "$url"; \
      echo "${sha}  /tmp/dep.archive" | sha256sum -c -; \
      if [ "$kind" = zip ]; then unzip -q /tmp/dep.archive -d /tmp/dep-x && mv /tmp/dep-x/*/* "$d/src/" && rm -rf /tmp/dep-x; \
      else tar -xzf /tmp/dep.archive --strip-components=1 -C "$d/src"; fi; \
      rm -f /tmp/dep.archive; chmod -R a+rX "$d"; \
    }; \
    fetch nlohmann_json 3.11.3 https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz \
          0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406 tgz; \
    fetch minja 3e4c61c616eda133cfb1e440fc7a14bf1729bbee \
          https://github.com/google/minja/archive/3e4c61c616eda133cfb1e440fc7a14bf1729bbee.tar.gz \
          221f8f2279dd2552b20d5f749aab2cf3650dd144bbe013c73752d0cb507e6405 tgz; \
    fetch safetensors.h 974a85d7dfd6e010558353226638bb26d6b9d756 \
          https://github.com/hsnyder/safetensors.h/archive/974a85d7dfd6e010558353226638bb26d6b9d756.tar.gz \
          9aaf5961609601cf9aaa96582a207bce7c6e5fbf57ed2cc669bb7bde6a937d4b tgz; \
    fetch gguflib bac796ada809ac293e685db59b075971181cb008 \
          https://github.com/Lourdle/gguf-tools/archive/bac796ada809ac293e685db59b075971181cb008.zip \
          4d6eab5055468d222833f3f83fe2f7909ccd06114278c2c0b468570ef002c22d zip; \
    fetch googletest 03597a01ee50ed33e9dfd640b249b4be3799d395 \
          https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip \
          edd885a1ab32b6999515a880f669efadb80b3f880215f315985fa3f6eca7c4d3 zip

USER mayhem
WORKDIR /mayhem
COPY --chown=2000:2000 . /mayhem

# Pre-fill rlenv's compile cache when the base ships it (SPEC §6.2 item 18), else a plain build.
RUN if [ -x /opt/mayhem/ccache/prefill ]; then /opt/mayhem/ccache/prefill bash mayhem/build.sh; \
    else bash mayhem/build.sh; fi

# Run the project's tests (pre-built by build.sh) so the image fails to build if they don't pass.
RUN bash mayhem/test.sh

LABEL org.opencontainers.image.source="${REPO_URL}" \
      org.opencontainers.image.revision="${COMMIT}" \
      rlenv.tier="commit" \
      rlenv.source_date_epoch="${SOURCE_DATE_EPOCH}" \
      rlenv.user="mayhem"

USER mayhem
WORKDIR /mayhem
