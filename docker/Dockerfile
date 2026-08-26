# ============================================================================
# open-pacmuci -- Multi-stage Docker build
# Stage 1: Install conda env + pip package
# Stage 2: Copy resolved env into slim runtime image
# ============================================================================

# Pin base image for reproducibility
FROM mambaorg/micromamba:1.5.10-bookworm-slim AS builder

# Install conda environment first (layer cached unless environment.yml changes)
COPY --chown=$MAMBA_USER:$MAMBA_USER conda/environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes && \
    rm -f /tmp/environment.yml

# Install Clair3 into the same environment
RUN micromamba install -y -n open-pacmuci-tools -c bioconda -c conda-forge clair3 && \
    micromamba clean --all --yes

# Install open-pacmuci Python package
# Copy pyproject.toml first for layer caching (source changes don't bust dep cache)
# Data is bundled inside src/open_pacmuci/data/ -- no separate data/ dir needed
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md /app/
COPY --chown=$MAMBA_USER:$MAMBA_USER src/ /app/src/

WORKDIR /app
# Install from the build context (local fork / working tree), not PyPI.
# Do not rm /app here: WORKDIR is root-owned and $MAMBA_USER cannot delete it.
# The runtime stage only copies the conda env, so /app is discarded anyway.
RUN micromamba run -n open-pacmuci-tools pip install --no-cache-dir .

# ============================================================================
# Runtime stage -- slim image with only the resolved environment
# ============================================================================
FROM mambaorg/micromamba:1.5.10-bookworm-slim

# Copy the fully resolved conda environment from builder
COPY --from=builder /opt/conda/envs/open-pacmuci-tools /opt/conda/envs/open-pacmuci-tools

# Activate environment via PATH (faster than micromamba run wrapper)
ENV PATH="/opt/conda/envs/open-pacmuci-tools/bin:${PATH}"

# Run as non-root user
USER $MAMBA_USER
WORKDIR /data

# Metadata
LABEL org.opencontainers.image.source="https://github.com/berntpopp/open-pacmuci" \
      org.opencontainers.image.description="open-pacmuci: MUC1 VNTR analysis pipeline for PacBio HiFi amplicon data" \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["open-pacmuci"]
CMD ["--help"]
