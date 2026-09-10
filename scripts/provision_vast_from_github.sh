#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 HOST PORT GIT_COMMIT" >&2
    echo "Example: $0 203.0.113.10 12345 \"\$(git rev-parse HEAD)\"" >&2
}

if [[ $# -ne 3 ]]; then
    usage
    exit 2
fi

HOST="$1"
PORT="$2"
GIT_COMMIT="$3"

if [[ ! "$HOST" =~ ^[A-Za-z0-9.-]+$ ]]; then
    echo "Invalid host: $HOST" >&2
    exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "Invalid port: $PORT" >&2
    exit 2
fi
if [[ ! "$GIT_COMMIT" =~ ^[0-9a-f]{7,40}$ ]]; then
    echo "Invalid Git commit: $GIT_COMMIT" >&2
    exit 2
fi

echo "Provisioning root@${HOST}:${PORT} from GitHub commit ${GIT_COMMIT}"
echo "This installs dependencies but does not download model/SAE/J-Lens weights."

REMOTE_MAIN="$(git ls-remote origin refs/heads/main | cut -f1)"
LOCAL_COMMIT="$(git rev-parse "$GIT_COMMIT")"
if [[ "$REMOTE_MAIN" != "$LOCAL_COMMIT" ]]; then
    echo "Refusing provision: commit $LOCAL_COMMIT is not pushed as origin/main" >&2
    exit 3
fi

BUNDLE="$(mktemp /tmp/mech-interp-vast.XXXXXX.bundle)"
trap 'rm -f "$BUNDLE"' EXIT
git bundle create "$BUNDLE" main
scp -P "$PORT" -i /home/beluga/.ssh/id_ed25519 \
    "$BUNDLE" "root@${HOST}:/workspace/mech-interp.bundle"

ssh -p "$PORT" -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
    -i /home/beluga/.ssh/id_ed25519 "root@${HOST}" \
    "GIT_COMMIT='$GIT_COMMIT' bash -s" <<'REMOTE'
set -euo pipefail

REPO=/workspace/mech-interp
if [[ ! -d "$REPO/.git" ]]; then
    git clone /workspace/mech-interp.bundle "$REPO"
else
    git -C "$REPO" fetch /workspace/mech-interp.bundle main
fi

cd "$REPO"
git checkout --detach "$GIT_COMMIT"
test "$(git rev-parse HEAD)" = "$(git rev-parse "$GIT_COMMIT")"

export HF_HOME=/workspace/.hf_home
export TRANSFORMERS_CACHE="$HF_HOME/hub"
./scripts/bootstrap_vast.sh

echo "Provisioned commit: $(git rev-parse HEAD)"
REMOTE
