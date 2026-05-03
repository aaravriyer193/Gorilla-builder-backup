FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl jq unzip xz-utils ca-certificates findutils procps \
      build-essential python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /home/user/app

# Ensure package.json exists in ./boilerplate or npm install will fail
COPY ./boilerplate/ /home/user/app/

RUN if [ -f package.json ]; then \
    npm install --legacy-peer-deps --no-fund --no-audit --prefer-offline \
    && npm cache clean --force; \
    fi

RUN mkdir -p /home/user/app/public/generated \
             /home/user/app/src \
             /home/user/app/.gorilla \
  && chmod -R 755 /home/user/app
