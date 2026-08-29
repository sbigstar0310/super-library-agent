# sla-base: WebGen-Bench agent + eval image.
#
# Used for WebGen coding (the agent generates React+Vite apps) and for the
# in-container WebGen eval (appearance screenshot + WebVoyager UI-test).
# Bundles python3.12, Node 20/npm/vite, chromium + chromedriver, PM2, selenium,
# and Playwright. Agents can `pip install` anything else at runtime.
#
# The factory in mswe_agents/_factory.py defaults PATH to
# `/opt/conda/envs/agent/bin:/opt/conda/bin:...`. We don't actually use conda
# here — just create symlinks so that PATH default keeps working without a
# per-image extra_env override.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        jq \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Node 20 + npm for WebGen-Bench agents (`npm install && npx vite build`).
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g npm@11   # npm@latest now pulls npm@12 which needs node>=22; npm@11 is the newest node-20-compatible line

# Stub the conda-env layout the factory's default PATH expects.
RUN mkdir -p /opt/conda/envs/agent/bin /opt/conda/bin \
    && ln -sf /usr/local/bin/python3 /opt/conda/envs/agent/bin/python \
    && ln -sf /usr/local/bin/python3 /opt/conda/envs/agent/bin/python3 \
    && ln -sf /usr/local/bin/pip     /opt/conda/envs/agent/bin/pip     \
    && ln -sf /usr/local/bin/python3 /opt/conda/bin/python             \
    && ln -sf /usr/local/bin/python3 /opt/conda/bin/python3

# Tier-1 generic Python baseline. Agents can pip-install the rest at runtime.
RUN pip install --no-cache-dir \
        numpy \
        pandas \
        scipy \
        pyyaml \
        psutil \
        pillow \
        requests \
        pytest \
        tqdm

# WebGen-Bench eval (in-container). chromium + chromedriver for selenium-driven
# screenshot capture (`webgen_appearance`) and WebVoyager interaction
# (`webgen_ui_test`).
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        chromium-driver \
        fonts-liberation \
        libnss3 \
        libxss1 \
        libasound2 \
        libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# PM2: webgen eval starts vite preview as a daemon and tails pm2 logs for the
# bound port. Globally installed so `pm2` is on PATH inside the container.
RUN npm install -g pm2

# Eval-side python deps. selenium drives chromium; openai/anthropic power the
# VLM appearance grader and the WebVoyager subprocess.
RUN pip install --no-cache-dir \
        'selenium>=4.40,<5' \
        openai \
        anthropic

ENV CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER=/usr/bin/chromedriver \
    PYTHONUNBUFFERED=1

# Force chromium flags required for headless Chrome in a docker container
# (root user + small /dev/shm). Debian chromium's launcher sources every
# .conf under /etc/chromium.d/ and prepends `$CHROMIUM_FLAGS` to argv, so any
# caller (selenium, WebVoyager, our entrypoint) inherits these without code
# changes upstream.
RUN mkdir -p /etc/chromium.d \
    && echo 'CHROMIUM_FLAGS="$CHROMIUM_FLAGS --no-sandbox --disable-dev-shm-usage"' \
        > /etc/chromium.d/docker-flags.conf

# Playwright for the webgen-screenshot skill. Bundled chromium guarantees
# CDP-protocol compatibility (Debian's system chromium tracks a different
# version and can break on protocol mismatches). `install --with-deps`
# pulls the matching shared libs that the bundled binary expects.
RUN pip install --no-cache-dir 'playwright>=1.40' \
    && python -m playwright install --with-deps chromium

COPY webgen_eval_entrypoint.sh /usr/local/bin/webgen_eval_entrypoint.sh
COPY webgen_screenshot_entrypoint.sh /usr/local/bin/webgen_screenshot_entrypoint.sh
RUN chmod +x /usr/local/bin/webgen_eval_entrypoint.sh \
              /usr/local/bin/webgen_screenshot_entrypoint.sh

WORKDIR /home
CMD ["bash"]
