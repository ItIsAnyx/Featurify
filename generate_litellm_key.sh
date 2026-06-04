#!/usr/bin/env bash
#
# Creates a LiteLLM "virtual key" with a spending budget, using your master key.
# Run from your project folder (with docker-compose.yml and .env), stack running:
#
#   bash generate_litellm_key.sh
#
# Optional overrides:
#   BUDGET=0.5 DURATION=30d RPM=20 TPM=40000 bash generate_litellm_key.sh
#
# If .env parsing gives trouble, pass the master key directly:
#   MASTER_KEY=sk-xxxxx bash generate_litellm_key.sh
#
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env}"
LITELLM_URL="${LITELLM_URL:-http://localhost:4000}"

BUDGET="${BUDGET:-5}"
DURATION="${DURATION:-30d}"
RPM="${RPM:-20}"
TPM="${TPM:-40000}"

# 1) Master key: use $MASTER_KEY if provided, otherwise read it from .env and
#    scrub carriage returns, surrounding whitespace, and surrounding quotes.
if [ -z "${MASTER_KEY:-}" ]; then
  if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: cannot find $ENV_FILE. Run from your project folder, or pass MASTER_KEY=... " >&2
    exit 1
  fi
  MASTER_KEY="$(grep -E '^[[:space:]]*LITELLM_API_KEY[[:space:]]*=' "$ENV_FILE" | head -n1 \
    | sed -E 's/^[^=]*=//' \
    | tr -d '\r' \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/^"//; s/"$//; s/^'\''//; s/'\''$//')"
fi

if [ -z "${MASTER_KEY:-}" ]; then
  echo "ERROR: could not determine the master key (LITELLM_API_KEY)." >&2
  exit 1
fi

# 2) Show a masked preview so you can sanity-check it.
LEN=${#MASTER_KEY}
PREVIEW="${MASTER_KEY:0:6}"
echo "Master key read: ${PREVIEW}... (${LEN} chars)"
case "$MASTER_KEY" in
  sk-*) ;;
  *) echo "WARNING: your master key does not start with 'sk-'. LiteLLM expects master keys to start with sk-." ;;
esac
echo

echo "Asking LiteLLM at $LITELLM_URL for a virtual key:"
echo "  budget = \$$BUDGET per $DURATION,  rpm = $RPM,  tpm = $TPM"
echo

RESP="$(curl -sS -X POST "$LITELLM_URL/key/generate" \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"models\":[\"deepseek-chat\"],\"max_budget\":$BUDGET,\"budget_duration\":\"$DURATION\",\"rpm_limit\":$RPM,\"tpm_limit\":$TPM}")"

echo "Raw response from LiteLLM:"
echo "$RESP"
echo

KEY="$(echo "$RESP" | python3 -c 'import sys,json
try:
    print(json.load(sys.stdin).get("key",""))
except Exception:
    print("")' 2>/dev/null || true)"

if [ -n "$KEY" ]; then
  echo "=============================================================="
  echo "SUCCESS. Your budgeted virtual key:"
  echo "  $KEY"
  echo
  echo "Add this exact line to your .env:"
  echo "  LITELLM_VIRTUAL_KEY=$KEY"
  echo "=============================================================="
else
  echo "No key returned. If the error says role=unknown, the master key sent was"
  echo "wrong - compare the masked preview above with LITELLM_API_KEY in your .env,"
  echo "or rerun as:  MASTER_KEY=sk-xxxxx bash generate_litellm_key.sh"
fi