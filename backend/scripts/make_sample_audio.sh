#!/usr/bin/env bash
# Generate a test recording with the macOS `say` command -- no microphone needed.
#
#   ./scripts/make_sample_audio.sh                       -> sample.wav
#   ./scripts/make_sample_audio.sh "turn the light off" out.wav
set -euo pipefail

TEXT="${1:-could you get me some water}"
OUT="${2:-sample.wav}"

if ! command -v say >/dev/null 2>&1; then
  echo "error: 'say' is macOS-only. Record something with any tool and pass it to send_audio.py." >&2
  exit 1
fi

# 16 kHz mono 16-bit LE WAV -- exactly what Whisper wants.
say -o "$OUT" --data-format=LEI16@16000 --channels=1 "$TEXT"
echo "wrote $OUT  (\"$TEXT\")"
