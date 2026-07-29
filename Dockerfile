# adtc-profiler runtime image.
#
# Multi-stage: stage 1 builds llama.cpp from a pinned release for
# reproducibility, stage 2 builds Python wheels (llama-cpp-python compiles
# from source and needs a C/C++ toolchain), stage 3 ships only the runtime.
#
# Build (from the repo root):
#   docker build -t adtc-profiler:latest .

# -----------------------------------------------------------------------------
# Stage 1: build llama.cpp (CPU-only, for parity with Standard Laptop profile)
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS llama-build

ARG LLAMACPP_REF=b10175
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMACPP_REF}" \
      https://github.com/ggerganov/llama.cpp.git /src/llama.cpp \
    && cd /src/llama.cpp \
    && cmake -B build \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX=OFF \
        -DGGML_AVX2=OFF \
        -DGGML_AVX512=OFF \
        -DGGML_FMA=OFF \
        -DGGML_F16C=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_METAL=OFF \
    && cmake --build build --config Release --target llama-bench llama-cli llama-server -j2

# -----------------------------------------------------------------------------
# Stage 2: build Python wheels (llama-cpp-python compiles from source — the
# slim runtime image has no C/C++ toolchain, so wheels must be built here)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS py-build

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Portable CPU build — no -march=native, the wheel must run on any audit VM.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"

WORKDIR /opt/adtc-profiler
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# -----------------------------------------------------------------------------
# Stage 3: profiler runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates lm-sensors libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# llama-bench + llama-cli + llama-server on PATH
COPY --from=llama-build /src/llama.cpp/build/bin/llama-bench  /usr/local/bin/
COPY --from=llama-build /src/llama.cpp/build/bin/llama-cli    /usr/local/bin/
COPY --from=llama-build /src/llama.cpp/build/bin/llama-server /usr/local/bin/

# Install the profiler + all deps from the prebuilt wheels (no toolchain here)
COPY --from=py-build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels adtc-profiler \
    && rm -rf /wheels

WORKDIR /work
ENTRYPOINT ["adtc-profiler"]
