#!/bin/sh
# Starts the stock ollama server, then pulls $KB_EMBED_MODEL on first boot
# if it is not already present in the kb-ollama-data volume. Idempotent: a
# restart with the model already pulled skips straight to serving. GPU is
# used automatically when the container has access to one (see the
# commented deploy.resources block in docker-compose.yml); with no GPU,
# ollama falls back to CPU on its own -- nothing here has to branch on that.
#
# KB_EMBED_MODEL (set by docker-compose.yml, default "bge-m3") names the
# base model to pull/serve -- a single source of truth so a deployment that
# overrides it (e.g. a pinned quantisation served under a different name)
# does not leave this script and the healthcheck still matching the
# literal "bge-m3".
#
# BGE_M3_DIGEST (set by docker-compose.yml from manifest.json, see
# manifest_probe.fetch_model_digest) is meant to pin the pull to the exact
# model build the dump was vectorised with, rather than trusting whatever
# "$KB_EMBED_MODEL:latest" currently resolves to in the registry -- the tag
# is mutable, the digest is not. `ollama pull name@sha256:digest` is the
# documented syntax for that, so it is tried first -- but measured directly
# (kb-smoke run, 2026-07-25, ollama 0.32.3): the CLI rejects it outright
# with "400 Bad Request: invalid model name" before any network access, so
# in practice this ALWAYS falls through to the plain tag pull on this
# ollama build. The attempt is kept anyway (harmless, ~0.2ms round trip
# to the local server) in case a future ollama release adds the syntax --
# but the real guarantee is smoke_checks.check_embedding_model_digest,
# which re-queries /api/tags after boot via HTTP and hard-fails the deploy
# if the SERVED digest disagrees with the manifest. That check, not this
# pull, is what actually catches a silently different model build.
set -e

model="${KB_EMBED_MODEL:-bge-m3}"

ollama serve &
server_pid=$!

until ollama list >/dev/null 2>&1; do
    sleep 1
done

if ! ollama list | grep -q "^${model}"; then
    if [ -n "$BGE_M3_DIGEST" ] && ollama pull "${model}@sha256:$BGE_M3_DIGEST"; then
        :
    else
        ollama pull "$model"
    fi
fi

wait "$server_pid"
