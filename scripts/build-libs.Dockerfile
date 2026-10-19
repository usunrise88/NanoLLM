# The build environment of NanoLLM's own llama.cpp libraries (scripts/build-libs.sh).
#
# Ubuntu 22.04, the same base as upstream's x64 release job: a binary built
# against its glibc (2.35) runs on anything newer, which is what the shipped
# archives need. The Vulkan SDK comes from LunarG's jammy repository, as in
# upstream's .github/workflows/release.yml.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates curl wget gnupg \
        libssl-dev pkg-config ccache \
    && rm -rf /var/lib/apt/lists/*
ARG VULKAN=1
RUN if [ "$VULKAN" = 1 ]; then \
        wget -qO /etc/apt/trusted.gpg.d/lunarg.asc https://packages.lunarg.com/lunarg-signing-key-pub.asc \
        && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list \
        && apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers vulkan-sdk \
        && rm -rf /var/lib/apt/lists/*; \
    fi
