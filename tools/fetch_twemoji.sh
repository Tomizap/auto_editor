#!/usr/bin/env bash
set -e

# ============================
# Config
# ============================

TWEMOJI_VERSION="14.0.2"
OUT_DIR="assets/twemoji/72x72"
TMP_DIR=".twemoji_tmp"

# Emojis réellement utilisés par ton pipeline
# (tu peux en rajouter sans problème)
EMOJIS=(
  "🤯"  # mind blown
  "💰"  # money
  "⚠️"  # warning
  "⚡"  # lightning
  "🧠"  # brain
  "✅"  # check
)

# ============================
# Helpers
# ============================

emoji_to_codepoint() {
  python3 - << 'EOF'
import sys
s = sys.stdin.read().strip()
codes = []
for ch in s:
    cp = ord(ch)
    if cp == 0xFE0F:  # strip variation selector
        continue
    codes.append(f"{cp:x}")
print("-".join(codes))
EOF
}

# ============================
# Main
# ============================

echo "▶ Fetching Twemoji PNGs (v${TWEMOJI_VERSION})"

mkdir -p "$OUT_DIR"
mkdir -p "$TMP_DIR"

BASE_URL="https://raw.githubusercontent.com/twitter/twemoji/v${TWEMOJI_VERSION}/assets/72x72"

for emoji in "${EMOJIS[@]}"; do
  codepoint=$(echo -n "$emoji" | emoji_to_codepoint)
  src="${BASE_URL}/${codepoint}.png"
  dst="${OUT_DIR}/${codepoint}.png"

  if [[ -f "$dst" ]]; then
    echo "  ✔ $emoji ($codepoint) already exists"
    continue
  fi

  echo "  ⬇ $emoji → $codepoint.png"
  curl -sSfL "$src" -o "$dst"
done

rm -rf "$TMP_DIR"

echo "✅ Twemoji setup complete"
echo "➡️  Emojis available in: $OUT_DIR"
