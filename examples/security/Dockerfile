# Pinned multi-stage SageMath 10.6 + uv container for lattice-estimator execution (Phase 15).
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711 AS uv_source
FROM --platform=linux/amd64 sagemath/sagemath:10.6@sha256:19995db6194f4a4bab18ce9a88556fd15b9ed5e916b4504fefe618a7796ddbdb

COPY --from=uv_source /uv /usr/local/bin/uv

USER root
RUN chmod +x /usr/local/bin/uv && \
    mkdir -p /home/sage/.cache/uv && \
    chown -R sage:sage /home/sage/.cache
USER sage

ENV UV_CACHE_DIR=/tmp/uv-cache
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/uv"]
