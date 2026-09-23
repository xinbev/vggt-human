#!/usr/bin/env python3
"""Generate an image through the a6api.com gpt-image relay."""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, TypeVar

try:
    from PIL import Image, ImageOps
except ImportError:  # Optional: generation still works without reference optimization.
    Image = None
    ImageOps = None


DEFAULT_BASE_URL = "https://a6api.com"
DEFAULT_MODEL = "gpt-image-2.5-sunburst"
DEFAULT_OUT = "output/imagegen/nbai-gpt-image.png"
DEFAULT_TIMEOUT = 90
DEFAULT_DOWNLOAD_TIMEOUT = 60
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_REFERENCE_MAX_SIDE = 1536
DEFAULT_REFERENCE_QUALITY = 88
DEFAULT_REFERENCE_OPTIMIZE_THRESHOLD = 512 * 1024
LOCAL_CONFIG = Path(r"C:/Users/ROG/.codex/skills/nbai-gpt-image/config.local.json")
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
)
TRANSIENT_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504, 524, 526}
MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
T = TypeVar("T")


class RelayError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def die(message: str, code: int = 1) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(code)


def get_api_key() -> str:
    key = (
        os.environ.get("A6_API_KEY")
        or os.environ.get("NBAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not key and LOCAL_CONFIG.exists():
        try:
            config = json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
            key = config.get("api_key")
        except Exception as exc:  # noqa: BLE001 - concise local config failure.
            die(f"Failed to read local config: {exc}")
    if not key:
        die("A6_API_KEY/NBAI_API_KEY is not set and config.local.json has no api_key.")
    return key


def normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def normalize_path_prefix(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().strip("/")
    return f"/{value}" if value else ""


def concise_body(body: str, limit: int = 300) -> str:
    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "empty response body"
    return text[:limit] + ("..." if len(text) > limit else "")


def request_json_once(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RelayError(
            f"HTTP {exc.code}: {concise_body(body)}",
            retryable=exc.code in TRANSIENT_HTTP_STATUS,
        ) from exc
    except urllib.error.URLError as exc:
        raise RelayError(f"Network error: {exc.reason}", retryable=True) from exc
    except TimeoutError as exc:
        raise RelayError(f"Timeout after {timeout}s", retryable=True) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RelayError(f"Invalid JSON response: {exc}", retryable=True) from exc


def run_with_retries(
    operation: Callable[[], T],
    *,
    label: str,
    max_attempts: int,
    retry_delay: float,
) -> T:
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        print(f"{label}: attempt {attempt}/{max_attempts}", file=sys.stderr, flush=True)
        try:
            result = operation()
            elapsed = time.monotonic() - started
            print(f"{label}: success in {elapsed:.1f}s", file=sys.stderr, flush=True)
            return result
        except RelayError as exc:
            elapsed = time.monotonic() - started
            print(
                f"{label}: failed in {elapsed:.1f}s: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if not exc.retryable or attempt >= max_attempts:
                raise RelayError(
                    f"{label} stopped after {attempt} attempt(s): {exc}",
                    retryable=exc.retryable,
                ) from exc
            print(
                f"{label}: retrying in {retry_delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(retry_delay)
    raise AssertionError("retry loop exited unexpectedly")


def extract_image_item(response: dict) -> dict:
    data = response.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise RelayError(
            f"No image data in response: {concise_body(json.dumps(response))}",
            retryable=True,
        )
    first = data[0]
    if first.get("b64_json"):
        try:
            base64.b64decode(first["b64_json"], validate=True)
        except Exception as exc:  # noqa: BLE001 - malformed provider response.
            raise RelayError(f"Invalid base64 image payload: {exc}", retryable=True) from exc
        return first
    if isinstance(first.get("url"), str) and first["url"]:
        return first
    raise RelayError(
        f"Unsupported image response item: {concise_body(json.dumps(first))}",
        retryable=True,
    )


def encode_multipart(fields: dict, files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    boundary = f"----codex-nbai-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        if value is None:
            continue
        chunks.extend(
            [
                f"--{boundary}".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"),
                b"",
                str(value).encode("utf-8"),
            ]
        )

    for field_name, path in files:
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{path.name}"'
                ).encode("utf-8"),
                f"Content-Type: {mime_type}".encode("utf-8"),
                b"",
                path.read_bytes(),
            ]
        )

    chunks.append(f"--{boundary}--".encode("utf-8"))
    chunks.append(b"")
    return b"\r\n".join(chunks), f"multipart/form-data; boundary={boundary}"


def request_multipart_once(
    url: str,
    fields: dict,
    files: list[tuple[str, Path]],
    api_key: str,
    timeout: int,
) -> dict:
    data, content_type = encode_multipart(fields, files)
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RelayError(
            f"HTTP {exc.code}: {concise_body(body)}",
            retryable=exc.code in TRANSIENT_HTTP_STATUS,
        ) from exc
    except urllib.error.URLError as exc:
        raise RelayError(f"Network error: {exc.reason}", retryable=True) from exc
    except TimeoutError as exc:
        raise RelayError(f"Timeout after {timeout}s", retryable=True) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RelayError(f"Invalid JSON response: {exc}", retryable=True) from exc


def download_url_once(url: str, timeout: int) -> tuple[bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            content = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.URLError as exc:
        raise RelayError(f"Failed to download image URL: {exc.reason}", retryable=True) from exc
    except TimeoutError as exc:
        raise RelayError(f"Image download timeout after {timeout}s", retryable=True) from exc

    return content, content_type


def detect_image_mime(content: bytes, reported: str = "") -> str:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return reported.split(";", 1)[0].strip().lower() or "application/octet-stream"


def output_path_for_mime(out_path: Path, mime_type: str) -> Path:
    expected = MIME_SUFFIXES.get(mime_type)
    if not expected:
        return out_path
    current = out_path.suffix.lower()
    compatible = current == expected or (mime_type == "image/jpeg" and current == ".jpeg")
    return out_path if compatible else out_path.with_suffix(expected)


def save_content(content: bytes, reported_mime: str, out_path: Path) -> tuple[Path, str, int]:
    mime_type = detect_image_mime(content, reported_mime)
    actual_path = output_path_for_mime(out_path, mime_type)
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    actual_path.write_bytes(content)
    return actual_path, mime_type, len(content)

def decode_b64(b64_json: str) -> bytes:
    try:
        return base64.b64decode(b64_json, validate=True)
    except Exception as exc:  # noqa: BLE001 - report concise decoding failure.
        die(f"Invalid base64 image payload: {exc}")


def image_to_base64(
    path: Path,
    *,
    optimize: bool,
    optimize_threshold: int,
    max_side: int,
    quality: int,
) -> tuple[str, dict]:
    original = path.read_bytes()
    encoded = original
    encoding = detect_image_mime(original)
    optimized = False

    if optimize and len(original) >= optimize_threshold and Image is not None and ImageOps is not None:
        try:
            with Image.open(io.BytesIO(original)) as source:
                image = ImageOps.exif_transpose(source).copy()
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            if has_alpha:
                image.save(buffer, format="PNG", optimize=True)
                candidate_encoding = "image/png"
            else:
                image.convert("RGB").save(
                    buffer,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                )
                candidate_encoding = "image/jpeg"
            candidate = buffer.getvalue()
            if len(candidate) < len(original):
                encoded = candidate
                encoding = candidate_encoding
                optimized = True
        except Exception as exc:  # noqa: BLE001 - fall back to original bytes.
            print(
                f"reference optimization skipped: {exc}",
                file=sys.stderr,
                flush=True,
            )

    info = {
        "original_bytes": len(original),
        "encoded_bytes": len(encoded),
        "optimized": optimized,
        "encoding": encoding,
    }
    return base64.b64encode(encoded).decode("ascii"), info


def build_payload(
    args: argparse.Namespace, image_path: Path | None = None
) -> tuple[dict, dict | None]:
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "quality": args.quality,
        "n": args.n,
    }
    reference_info = None
    if image_path and args.image_mode == "generations":
        b64_image, reference_info = image_to_base64(
            image_path,
            optimize=args.optimize_reference,
            optimize_threshold=args.reference_optimize_threshold,
            max_side=args.reference_max_side,
            quality=args.reference_quality,
        )
        if args.image_format == "array":
            payload["image"] = [b64_image]
        elif args.image_format == "string":
            payload["image"] = b64_image
        elif args.image_format == "object-data":
            payload["image"] = {"data": b64_image}
        else:
            die(f"Unsupported image format: {args.image_format}")
    return {key: value for key, value in payload.items() if value is not None}, reference_info


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate images through a6api.com gpt-image.")
    parser.add_argument("--prompt", required=True, help="Image prompt.")
    parser.add_argument(
        "--image",
        help="Reference image path.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("generations", "edits"),
        help=(
            "How to send --image. 'generations' embeds base64 JSON in /images/generations; "
            "'edits' uploads multipart to /images/edits."
        ),
    )
    parser.add_argument(
        "--image-format",
        choices=("array", "string", "object-data"),
        help="JSON shape for --image when --image-mode=generations.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output image path.")
    parser.add_argument("--size", default="1024x1024", help="Image size, for example 1024x1024.")
    parser.add_argument(
        "--quality",
        default="low",
        choices=("low", "medium", "high", "auto"),
        help="Image model quality setting.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--n", type=int, default=1, help="Number of images to request.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("A6_BASE_URL") or os.environ.get("NBAI_BASE_URL"),
        help="Relay base URL.",
    )
    parser.add_argument(
        "--images-path-prefix",
        default=os.environ.get("A6_IMAGES_PATH_PREFIX"),
        help="Optional path prefix before /images, for example /v1 for legacy relays.",
    )
    parser.add_argument("--timeout", type=int, help="Generation request timeout in seconds.")
    parser.add_argument("--download-timeout", type=int, help="Returned image URL timeout in seconds.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="Total attempts for transient failures; default 2 (initial request plus one retry).",
    )
    parser.add_argument("--retry-delay", type=float, help="Seconds before the one retry.")
    parser.add_argument(
        "--optimize-reference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compress large reference images before Base64 upload (enabled by default).",
    )
    parser.add_argument("--reference-max-side", type=int, help="Maximum reference image side.")
    parser.add_argument("--reference-quality", type=int, help="JPEG quality for reference optimization.")
    parser.add_argument(
        "--reference-optimize-threshold",
        type=int,
        help="Only optimize references at or above this byte size.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print request shape without network.")
    args = parser.parse_args()

    local_config = {}
    if LOCAL_CONFIG.exists():
        try:
            local_config = json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - concise local config failure.
            die(f"Failed to read local config: {exc}")

    base_url = normalize_base_url(args.base_url or local_config.get("base_url") or DEFAULT_BASE_URL)
    images_path_prefix = normalize_path_prefix(
        args.images_path_prefix or local_config.get("images_path_prefix")
    )
    args.image_mode = args.image_mode or local_config.get("image_mode") or "generations"
    args.image_format = args.image_format or local_config.get("image_format") or "array"
    args.timeout = args.timeout or local_config.get("timeout") or DEFAULT_TIMEOUT
    args.download_timeout = (
        args.download_timeout
        or local_config.get("download_timeout")
        or DEFAULT_DOWNLOAD_TIMEOUT
    )
    args.max_attempts = (
        args.max_attempts or local_config.get("max_attempts") or DEFAULT_MAX_ATTEMPTS
    )
    args.retry_delay = (
        args.retry_delay
        if args.retry_delay is not None
        else local_config.get("retry_delay", DEFAULT_RETRY_DELAY)
    )
    args.optimize_reference = (
        args.optimize_reference
        if args.optimize_reference is not None
        else local_config.get("optimize_reference", True)
    )
    args.reference_max_side = (
        args.reference_max_side
        or local_config.get("reference_max_side")
        or DEFAULT_REFERENCE_MAX_SIDE
    )
    args.reference_quality = (
        args.reference_quality
        or local_config.get("reference_quality")
        or DEFAULT_REFERENCE_QUALITY
    )
    args.reference_optimize_threshold = (
        args.reference_optimize_threshold
        or local_config.get("reference_optimize_threshold")
        or DEFAULT_REFERENCE_OPTIMIZE_THRESHOLD
    )
    if args.timeout <= 0 or args.download_timeout <= 0:
        die("Timeout values must be positive.")
    if not 1 <= args.max_attempts <= 4:
        die("--max-attempts must be between 1 and 4 for this user-authorized run.")
    if args.retry_delay < 0:
        die("--retry-delay cannot be negative.")
    if args.reference_max_side <= 0 or not 1 <= args.reference_quality <= 100:
        die("Reference size must be positive and quality must be between 1 and 100.")
    image_path = Path(args.image) if args.image else None
    if image_path and not image_path.exists():
        die(f"Reference image does not exist: {image_path}")
    if not image_path and args.image_mode == "edits":
        die("--image-mode=edits requires --image.")

    endpoint_path = "edits" if image_path and args.image_mode == "edits" else "generations"
    endpoint = f"{base_url}{images_path_prefix}/images/{endpoint_path}"
    payload, reference_info = build_payload(args, image_path)

    if args.dry_run:
        dry_payload = dict(payload)
        if image_path and args.image_mode == "generations":
            image_value = dry_payload.get("image")
            if isinstance(image_value, list):
                dry_payload["image"] = [f"<base64:{len(image_value[0])} chars>"]
            elif isinstance(image_value, str):
                dry_payload["image"] = f"<base64:{len(image_value)} chars>"
            elif isinstance(image_value, dict) and "data" in image_value:
                dry_payload["image"] = {"data": f"<base64:{len(image_value['data'])} chars>"}
        dry_run = {"endpoint": endpoint, "payload": dry_payload}
        if image_path:
            dry_run["image"] = str(image_path)
            dry_run["image_mode"] = args.image_mode
            dry_run["reference"] = reference_info
            if args.image_mode == "generations":
                dry_run["image_format"] = args.image_format
        dry_run["runtime"] = {
            "timeout_seconds": args.timeout,
            "download_timeout_seconds": args.download_timeout,
            "max_attempts": args.max_attempts,
            "retry_delay_seconds": args.retry_delay,
        }
        print(json.dumps(dry_run, indent=2, ensure_ascii=False))
        return 0

    api_key = get_api_key()
    try:
        if image_path and args.image_mode == "edits":
            first = run_with_retries(
                lambda: extract_image_item(
                    request_multipart_once(
                        endpoint, payload, [("image", image_path)], api_key, args.timeout
                    )
                ),
                label="image request",
                max_attempts=args.max_attempts,
                retry_delay=args.retry_delay,
            )
        else:
            first = run_with_retries(
                lambda: extract_image_item(
                    request_json_once(endpoint, payload, api_key, args.timeout)
                ),
                label="image request",
                max_attempts=args.max_attempts,
                retry_delay=args.retry_delay,
            )
    except RelayError as exc:
        die(str(exc))
    out_path = Path(args.out)
    if first.get("b64_json"):
        content = decode_b64(first["b64_json"])
        actual_path, mime_type, byte_count = save_content(content, "", out_path)
        source = "b64_json"
    elif first.get("url"):
        try:
            content, reported_mime = run_with_retries(
                lambda: download_url_once(first["url"], args.download_timeout),
                label="image download",
                max_attempts=args.max_attempts,
                retry_delay=args.retry_delay,
            )
        except RelayError as exc:
            die(str(exc))
        actual_path, mime_type, byte_count = save_content(content, reported_mime, out_path)
        source = "url"

    summary = {
        "ok": True,
        "model": args.model,
        "endpoint": endpoint,
        "source": source,
        "mime_type": mime_type,
        "bytes": byte_count,
        "output": str(actual_path.resolve()),
        "attempt_limit": args.max_attempts,
    }
    if reference_info:
        summary["reference"] = reference_info
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
