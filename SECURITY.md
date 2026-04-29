# EdukaAI Studio — Security Audit Report

> **Date:** 2026-04-29  
> **Scope:** Full codebase (backend + frontend)  
> **Risk Model:** CRITICAL / HIGH / MEDIUM / LOW / INFO

---

## Executive Summary

EdukaAI Studio is a **local-only** desktop application (backend serves frontend on `localhost:8000`, `allow_remote=false` by default). This significantly reduces the attack surface — most endpoints are unreachable from the network unless the user explicitly enables remote access. However, several code-quality issues and one **hardcoded secret** create real risks, especially if the app is ever exposed or if a malicious dataset/model is loaded.

| Severity | Count | Categories |
|----------|-------|------------|
| **CRITICAL** | 1 | Arbitrary code execution from models (C2) |
| **HIGH** | 3 | Exposed HF token (H3), path-traversal bypass (H1), unvalidated DB paths (H2) |
| **MEDIUM** | 6 | Weak filename sanitizer, content-type bypass, CSRF gap, `trust_proxy` undefined, duplicate constants, large in-memory uploads |
| **LOW / INFO** | 5 | Default secret key, debug traceback leakage, no rate limiting, missing `__all__`, formula test block UX |

---

## HIGH

### H3. Hardcoded HuggingFace API Token in Local `.env` File

**File:** `.env` (working directory only — **not tracked by git**)  
**Line:** 1

```
EDUKAAI_HF_TOKEN=############
```

**Verification:** `.env` is properly ignored by `.gitignore:56` and has **never been committed** (`git log --all --full-history -- .env` returns zero commits). It exists only in the local working directory.

**Impact:** The token is exposed to:
1. **Local filesystem** — any process or user with read access to the project directory
2. **Conversation logs** — the token was read during this audit session and may persist in system logs

Anyone with the token can access gated models on HuggingFace, consume rate-limited bandwidth, and potentially infer the account identity.

**Remediation (immediate):**
1. **Revoke the token** at https://huggingface.co/settings/tokens *now*.
2. Delete the local `.env` file: `rm .env`
3. Create `.env` from `.env.example` for any future local development.
4. Document in README that users must supply their own `EDUKAAI_HF_TOKEN`.

---

### C2. `trust_remote_code=True` Enables Arbitrary Code Execution from Downloaded Models

**Files:**
- `backend/app/ml/trainer.py:645`
- `backend/app/ml/trainer.py:1122`
- `backend/app/ml/trainer.py:1203`

**Code:**
```python
self.model, self.tokenizer = await asyncio.to_thread(
    load,
    self.model_path,
    tokenizer_config={"trust_remote_code": True}   # ← C2
)
```

**Impact:** Any HuggingFace model with a `tokenizer_config.json` containing `auto_map` or custom tokenizer class will cause Python to download and execute arbitrary code from the model repository. A malicious model (or a compromised legitimate one) can run shell commands, exfiltrate data, or install persistent malware.

**Why it exists:** The flag was likely added because some MLX-community models use custom tokenizers that crash without it.

**Remediation:**
- **Option A (best):** Remove `trust_remote_code=True`. Maintain a curated allow-list of model repositories (e.g., `mlx-community/*`) whose tokenizers are manually reviewed. Reject all others.
- **Option B (intermediate):** Gate the flag behind an explicit user opt-in with a scary warning: *"This model requires custom code. Enabling this allows the model author to execute arbitrary code on your machine."*
- **Option C (minimal):** At minimum, warn loudly in logs and UI every time it is triggered, and restrict it to known-safe namespaces.

---

## HIGH

### H1. Path-Traversal Bypass in Export Download Endpoint

**File:** `backend/app/routers/training.py`  
**Lines:** 2037–2041

**Code:**
```python
    export_path = os.path.normpath(f"{run.storage_path}/exports/{format}")
    if not export_path.startswith(os.path.normpath(run.storage_path)):
        raise ValidationError("Invalid export path")

    export_path = f"{run.storage_path}/exports/{format}"   # ← H1: overwrites the checked path!
```

**Impact:** The `format` variable is validated against `ALLOWED_EXPORT_FORMATS = {"adapter", "fused", "gguf"}` at line 2030, so an attacker *today* cannot inject `../etc/passwd`. However, this is a **logic bug** that defeats the defence-in-depth check. If the whitelist is ever relaxed, removed, or bypassed (e.g. through a future code path), the traversal guard is already nullified.

**Remediation:** Use the validated `export_path` variable. Do not reassign it after the check:
```python
    # After the check, export_path is already safe. Use it.
    if not os.path.exists(export_path):
        raise NotFoundError(f"Export not found for format: {format}")
```

---

### H2. Database-Stored File Paths Used Without Re-Validation

**Files:**
- `backend/app/routers/datasets.py:281` — `os.remove(existing_validation.file_path)`
- `backend/app/routers/datasets.py:504` — `shutil.rmtree(run.storage_path)` (via `run.storage_path` from DB)
- `backend/app/routers/datasets.py:514` — `os.remove(dataset.file_path)`
- `backend/app/routers/training.py:902` — `open(dataset.file_path, 'r')`
- `backend/app/routers/training.py:940` — `open(val_dataset.file_path, 'r')`

**Impact:** If an attacker ever gains SQL injection capability or manages to corrupt the SQLite database (e.g. via a malicious backup restore), poisoned `file_path` / `storage_path` values could cause arbitrary file deletion or reading. The dataset_id UUID generation makes *direct* injection hard, but DB compromise is a realistic secondary attack vector.

**Remediation:** Before every file operation on a DB-stored path, assert the resolved path is inside an allowed root:
```python
from pathlib import Path

def assert_safe_path(path: str, allowed_root: str) -> str:
    resolved = Path(path).resolve()
    root = Path(allowed_root).resolve()
    if not str(resolved).startswith(str(root)):
        raise ValidationError("Path escapes allowed directory")
    return str(resolved)
```

Apply this wrapper to **all** `open()`, `os.remove()`, `shutil.rmtree()`, and `shutil.copy()` calls that use DB-derived paths.

---

## MEDIUM

### M1. Weak `sanitize_filename()` — Does Not Block `..` Sequences

**File:** `backend/app/core/__init__.py`  
**Lines:** 12–19

**Code:**
```python
def sanitize_filename(filename: str) -> str:
    import re
    filename = re.sub(r'[\\/]', '', filename)
    filename = filename.replace('\x00', '')
    return filename[:255]
```

**Impact:** While `/` and `\` are stripped, `..` (parent-directory reference) is **not** removed. On its own this is harmless because there are no path separators left, but combined with code that later prepends a directory (e.g., `f"./storage/datasets/{filename}"`), a filename like `..` could in theory resolve outside the intended directory depending on how the path is constructed. Today the actual usage is `f"./storage/datasets/{dataset_id}.jsonl"` where `dataset_id` is a UUID, so the risk is low — but the utility itself is misleadingly named.

**Remediation:**
```python
def sanitize_filename(filename: str) -> str:
    import re
    filename = re.sub(r'[\\/]', '', filename)
    filename = filename.replace('\x00', '')
    filename = re.sub(r'\.+', '', filename)   # remove any dot sequences
    filename = filename.strip('.')              # no leading/trailing dots
    return filename[:255] or "unnamed"
```

---

### M2. Content-Type Validation Intentionally Bypassed

**File:** `backend/app/routers/datasets.py`  
**Lines:** 72–82

**Code:**
```python
allowed_content_types = [
    'application/json', 'application/jsonl', 'text/plain', 'text/json',
    'application/octet-stream', None
]
if file.content_type and not any(...):
    logger.warning(f"Unexpected content type: {file.content_type}")
    # Don't reject - try to parse anyway
```

**Impact:** The comment says it all — *any* file type is accepted. While the backend only parses JSONL/JSON, a malicious client can upload a binary payload disguised as a dataset. This opens the door to:
- **Polyglot attacks** (a file that is valid JSONL *and* something else)
- **Billion-laughs / zip-bomb style JSON** (deeply nested JSON that exhausts memory during `json.loads()`)
- **Parser differential attacks** if the JSON parser has CVEs

**Remediation:** Reject non-text uploads. Add a content-type check that actually blocks unknown types, and add a JSON depth limit (e.g., `json.loads(..., parse_constant=lambda x: None)` with a custom decoder that limits nesting depth).

---

### M3. No CSRF Protection

**File:** `backend/app/main.py`  
**Lines:** 170–176

**Code:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3030", "http://localhost:5173", ...],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Impact:** The app has **no CSRF tokens**, **no SameSite cookies**, and **no session system** at all. Because it is local-only by default, the practical risk is low. But if `EDUKAAI_ALLOW_REMOTE=true` is ever set (e.g. by a user who wants to access from another device on the LAN), any malicious website visited by the user can make cross-origin POST requests to `http://<lan-ip>:8000/api/...` and trigger training starts, dataset deletions, or model exports.

**Remediation:**
- Add a simple CSRF token mechanism (e.g. a random token returned on `GET /api/health` that must be included in the `X-CSRF-Token` header for state-changing requests).
- Alternatively, add a per-installation API key that the frontend must send in every request, rotated on first launch and stored in `localStorage`.

---

### M4. `trust_proxy` Used But Never Defined in Settings

**File:** `backend/app/main.py`  
**Line:** 198

**Code:**
```python
if getattr(settings, 'trust_proxy', False):
    forwarded_for = request.headers.get("x-forwarded-for")
```

**Impact:** `trust_proxy` is not declared in `backend/app/config.py` `Settings` class. It will always default to `False`, which is the safe default, but this is a **latent bug**. If someone adds it to the Settings class later and accidentally defaults it to `True`, the localhost-only middleware will trust `X-Forwarded-For` from any client, allowing trivial IP spoofing to bypass the localhost restriction.

**Remediation:** Either remove the `trust_proxy` branch entirely (the app should never be behind a proxy in its intended architecture), or explicitly add it to `Settings` with a default of `False` and a big warning comment.

---

### M5. Duplicate Constant Definition

**File:** `backend/app/routers/training.py`  
**Lines:** 1839 and 2018

**Code:**
```python
ALLOWED_EXPORT_FORMATS = {"adapter", "fused", "gguf"}      # line 1839
...
ALLOWED_EXPORT_FORMATS = {"adapter", "fused", "gguf"}      # line 2018
```

**Impact:** Low direct security impact, but code duplication is a maintenance hazard. If one is updated and the other forgotten, the `download_export` endpoint could accept a format that `export_model_endpoint` does not create (or vice versa), leading to a confused-deputy style bug.

**Remediation:** Move the constant to `backend/app/config.py` or `backend/app/core/constants.py` and import it.

---

### M6. Large File Uploads Read Entirely Into Memory

**File:** `backend/app/routers/datasets.py`  
**Lines:** 97–98

**Code:**
```python
content = await file.read()
if len(content) > 500 * 1024 * 1024:
    raise ValidationError("File too large for processing. Maximum 500MB allowed.")
```

**Impact:** A 500 MB file is read into RAM in one go. While there is a size check, it happens *after* the read. A malicious client can stream a multi-GB file and exhaust server memory before the check is reached (or simply cause a denial of service). FastAPI's `UploadFile` supports streaming — the code should use `file.file` as an iterator or write directly to disk in chunks.

**Remediation:** Stream the upload to a temporary file, then validate and process:
```python
import tempfile, shutil

with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tmp:
    chunk_size = 64 * 1024
    total = 0
    while chunk := await file.read(chunk_size):
        total += len(chunk)
        if total > 500 * 1024 * 1024:
            raise ValidationError("File too large")
        tmp.write(chunk)
# Now process tmp.name
```

---

## LOW / INFORMATIONAL

### L1. Default Secret Key in Production Warning Only

**File:** `backend/app/config.py`  
**Line:** 45

**Code:**
```python
secret_key: str = "change-me-in-production"
```

**Impact:** The secret key is only used for … nothing, currently. There are no sessions, no JWTs, no signed cookies. The warning at startup is good, but if the key is ever used for crypto in the future, this default is dangerous.

**Remediation:** Generate a random key at first launch and persist it in `~/.edukaai/.secret_key` instead of hardcoding.

---

### L2. Debug Mode Leaks Full Tracebacks and Internal Paths

**File:** `backend/app/main.py`  
**Lines:** 315–324

**Code:**
```python
if get_settings().debug:
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_code": "internal_error",
            "error_id": error_id,
            "traceback": tb_str          # ← L2
        }
    )
```

**Impact:** If `EDUKAAI_DEBUG=true` is set, internal file paths, function names, and exception details are returned to the client. This aids reconnaissance for attackers.

**Remediation:** Acceptable for a local-only dev tool, but document prominently that `debug` must never be enabled in any remotely accessible deployment.

---

### L3. No Rate Limiting

**Impact:** There is no rate limiting on any endpoint. A malicious local process (or remote client if `allow_remote` is on) can:
- Flood dataset uploads to fill disk
- Spam model validation requests to hit HuggingFace API rate limits
- Rapidly start/stop training to corrupt SQLite WAL

**Remediation:** Add `slowapi` or a simple in-memory rate limiter for state-changing endpoints.

---

### L4. `huggingface_id` Passed to `model_info()` with Expansive Metadata Flags

**File:** `backend/app/routers/training.py`  
**Lines:** 358–361

**Code:**
```python
info = hf_model_info(
    huggingface_id,
    expand=["config", "safetensors", "cardData"],
    token=settings.hf_token
)
```

**Impact:** The `cardData` expansion can return arbitrary markdown and URLs from the model card. If this data is ever rendered in the frontend without sanitization, it becomes an XSS vector. Currently the frontend does not render card data, but this is a latent risk.

**Remediation:** Sanitize any model-card fields before displaying them in the UI (the existing `sanitizeHtml()` in `messageSecurity.ts` can be reused).

---

### L5. Homebrew Formula Test Block Starts Full Server

**File:** `homebrew-formula.rb`  
**Lines:** 62–73

**Code:**
```ruby
test do
    pid = fork { exec bin/"edukaai-studio" }
    sleep 10
    begin
      output = shell_output("curl -sf http://127.0.0.1:8000/api/health || echo 'not healthy'")
      assert_match "healthy", output
    ensure
      Process.kill("TERM", pid) if pid
      Process.wait(pid) if pid
    end
end
```

**Impact:** Not a security vulnerability, but a **CI/UX risk**. Starting the full server (which loads MLX, checks models, etc.) in a Homebrew test block is heavy and may time out on slow CI runners. If MLX model loading hangs, the test will fail and block formula publication.

**Remediation:** Replace with a lighter test, e.g.:
```ruby
test do
  system "#{bin}/edukaai-studio", "--version"
end
```
(Add a `--version` flag to the wrapper script, or simply test that the binary exists and is executable.)

---

## Frontend-Specific Observations

### Axios Usage — No SSRF Risk in Current Architecture

All frontend axios instances point to `http://localhost:8000` (or relative `/api`). The recent Axios CVEs (e.g., SSRF via `proxy`, request body transformation) are **not exploitable** here because:
- The frontend runs in the user's browser and cannot reach internal network services beyond what the browser Same-Origin Policy allows.
- There are no proxy configurations in the axios instances.
- No user-supplied URLs are passed to axios.

**However**, if `EDUKAAI_ALLOW_REMOTE=true` is ever enabled, the frontend's CORS-reliant design becomes a target for CSRF (see M3).

### XSS Defence in Frontend

**File:** `frontend/src/utils/messageSecurity.ts`

The frontend has a well-thought-out defence hierarchy:
1. **User messages:** HTML-escaped via `escapeHtml()`.
2. **Assistant messages:** Parsed as Markdown, then passed through `DOMPurify.sanitize()` with a strict allow-list.
3. **DOMPurify config:** No `data-*` attributes, no inline event handlers, `SANITIZE_DOM: true`.

**Gap:** The `ALLOWED_ATTR` list includes `href` on `<a>` tags but does not validate the URL scheme. A model response containing `<a href="javascript:alert(1)">` would survive DOMPurify and create a click-XSS vector. Add an `href` validator:
```javascript
// In DOMPurify config, or post-process:
const allowedSchemes = ['http:', 'https:', 'mailto:'];
```

---

## Recommendations Priority Matrix

| Priority | Item | Effort |
|----------|------|--------|
| **P0 — Today** | Revoke exposed HF token (H3) | 10 min |
| **P0 — Today** | Remove `trust_remote_code=True` or gate behind opt-in (C2) | 2 hrs |
| **P1 — This week** | Fix path-traversal bypass in export download (H1) | 30 min |
| **P1 — This week** | Validate all DB-derived paths before file ops (H2) | 3 hrs |
| **P2 — Next sprint** | Harden `sanitize_filename()` (M1) | 15 min |
| **P2 — Next sprint** | Stream large uploads instead of `await file.read()` (M6) | 2 hrs |
| **P2 — Next sprint** | Add CSRF token or per-install API key (M3) | 4 hrs |
| **P3 — Backlog** | Add rate limiting (L3) | 2 hrs |
| **P3 — Backlog** | Replace formula test block with lightweight check (L5) | 30 min |

---

*Report generated by automated static analysis + manual review. No dynamic testing (fuzzing, penetration testing) was performed.*
