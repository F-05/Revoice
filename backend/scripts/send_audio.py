#!/usr/bin/env python3
"""Send an audio file to /process-speech and pretty-print the response.

    python scripts/send_audio.py sample.wav
    python scripts/send_audio.py sample.m4a --url http://192.168.1.42:8000

Uses only the standard library, so it runs with or without the venv active.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def build_multipart(file_path: Path, field: str = "audio") -> tuple[bytes, str]:
    boundary = f"----speech{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field}"; filename="{file_path.name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="Path to an audio file.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Backend base URL.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Seconds to wait.")
    args = parser.parse_args()

    if not args.audio.is_file():
        print(f"error: no such file: {args.audio}", file=sys.stderr)
        return 2

    body, content_type = build_multipart(args.audio)
    endpoint = args.url.rstrip("/") + "/process-speech"
    request = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": content_type}, method="POST"
    )

    print(f"POST {endpoint}  ({args.audio.name}, {len(body) / 1024:.0f} KB)")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read())
            status = response.status
    except urllib.error.HTTPError as exc:
        payload, status = json.loads(exc.read() or b"{}"), exc.code
    except urllib.error.URLError as exc:
        print(f"error: could not reach {endpoint} ({exc.reason})", file=sys.stderr)
        print("hint: is the server running?  uvicorn app.main:app --host 0.0.0.0 --port 8000", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    print(f"HTTP {status} in {elapsed:.2f}s")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
