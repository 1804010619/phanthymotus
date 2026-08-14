FROM python:3.12-slim

# Install git, docker CLI, and buildx
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        openssh-client \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-buildx-plugin \
    && rm -rf /var/lib/apt/lists/*

# Setup QEMU for ARM64 cross-compilation (will be run at container startup if needed)
# The actual QEMU registration happens via docker run --privileged at deploy time

WORKDIR /app

# Install Python dependencies
COPY agents/pr_review/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent source code
COPY agents/ /app/agents/

# Create data directory
RUN mkdir -p /data/repos

EXPOSE 15690

CMD ["python", "-m", "agents.pr_review.server"]
