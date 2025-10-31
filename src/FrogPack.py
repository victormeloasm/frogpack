#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrogPack Pro — M9 Ultimate (Patched v2) 🐸
- FIX: Progressbar determinística (pack e unpack)
- pyzstd/LZ4 backends intactos
- Manifest-free TAR + extractor seguro
"""

from __future__ import annotations

import os, io, sys, tarfile, struct, time, json, hashlib, binascii, tempfile, shutil, threading, collections, re, math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Tuple, Optional, Callable

# ---------------------------------------------------------------
# Optional deps (auto-detected)
# ---------------------------------------------------------------

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    np = None
    NUMPY_AVAILABLE = False

try:
    import zstandard as zstd
except Exception:
    zstd = None

try:
    import brotli
except Exception:
    brotli = None

try:
    import lzma
except Exception:
    lzma = None

try:
    import pyzstd  # pyzstd.compress(..., level_or_option=int, threads=-1) supported
except Exception:
    pyzstd = None

try:
    import lz4.frame as lz4f
except Exception:
    lz4f = None

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except Exception:
    PILLOW_AVAILABLE = False

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
    logger.remove()
    logger.add("frogpack.log", rotation="10 MB", compression="zip")
except Exception:
    LOGURU_AVAILABLE = False

try:
    from rich.console import Console
    RICH_AVAILABLE = True
    _console = Console(stderr=True)
except Exception:
    RICH_AVAILABLE = False
    _console = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import magic as _magic  # python-magic
    _HAVE_PY_MAGIC = True
except Exception:
    _HAVE_PY_MAGIC = False

try:
    import filetype as _filetype  # python-filetype
    _HAVE_FILETYPE = True
except Exception:
    _HAVE_FILETYPE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

try:
    from multiprocess import cpu_count  # type: ignore
except Exception:
    try:
        from multiprocessing import cpu_count  # fallback
    except Exception:
        def cpu_count():
            return 1

APP_NAME = "FrogPack Pro — M9 Ultimate (Patched v2) 🐸"
APP_EXT  = ".frog"
MAGIC    = b'FROGPK\x00\x00'  # 8 bytes
VERSION_V8  = 8  # compatibility reader
VERSION_V9  = 9  # current

METHOD_LZMA2 = 1
METHOD_ZSTD  = 2
METHOD_STORE = 3
METHOD_BROTLI= 4
METHOD_ZSTD_THEN_BROTLI = 5
METHOD_LZ4   = 6  # NEW: real LZ4 frame stream

LOGO = "🐸"
THEMES = ("Frog", "Lagoon", "Light", "Dark")

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _log_file(msg: str) -> None:
    if LOGURU_AVAILABLE:
        logger.info(msg)

def _log_rich(msg: str) -> None:
    if RICH_AVAILABLE:
        try:
            _console.log(msg)
        except Exception:
            pass

def secure_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sanitize_path(path: str) -> str:
    if not path:
        return path
    return path.rstrip(" .\\/")

def get_safe_extraction_dir(preferred_dir: str) -> str:
    try:
        safe_dir = sanitize_path(preferred_dir)
        p = Path(safe_dir)
        p.mkdir(parents=True, exist_ok=True)
        t = p / "frogpack_write_test.tmp"
        t.write_bytes(b"ok")
        t.unlink(missing_ok=True)
        return safe_dir
    except Exception:
        fallback = tempfile.mkdtemp(prefix="frogpack_")
        return fallback

def calculate_entropy(sample: bytes) -> float:
    if not sample:
        return 0.0
    if NUMPY_AVAILABLE:
        arr = np.frombuffer(sample, dtype=np.uint8)
        if arr.size == 0: return 0.0
        unique, counts = np.unique(arr, return_counts=True)
        probs = counts.astype(np.float64) / float(arr.size)
        return float(-np.sum(probs * np.log2(probs, where=(probs>0))))
    # exact log2 without numpy
    freq = [0]*256
    for b in sample: freq[b]+=1
    n = len(sample)
    ent = 0.0
    for c in freq:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent

TEXT_EXTS = {
    ".txt",".md",".rst",".csv",".tsv",".json",".xml",".yaml",".yml",
    ".html",".htm",".css",".js",".ts",".py",".java",".c",".cpp",".h",".hpp",".ini",".cfg",".toml",".log"
}

def detect_mime(path: Path) -> str:
    if _HAVE_PY_MAGIC:
        try:
            return _magic.from_file(str(path), mime=True) or "application/octet-stream"
        except Exception:
            pass
    if _HAVE_FILETYPE:
        try:
            g = _filetype.guess(str(path))
            return (g.mime if g else None) or "application/octet-stream"
        except Exception:
            pass
    ext = path.suffix.lower()
    if ext in (".png",".jpg",".jpeg",".gif",".webp",".bmp",".tiff"):
        return "image/" + ext.lstrip(".")
    if ext in (".exe",".dll",".so",".dylib"):
        return "application/x-binary"
    return "application/octet-stream"

def is_probably_text(path: Path, max_probe=8192) -> bool:
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return True
    try:
        mime = detect_mime(path)
        if mime.startswith("text/"):
            return True
        if mime.startswith("image/") or mime in ("application/pdf","application/zip","application/x-binary"):
            return False
        b = path.read_bytes()[:max_probe]
        if not b:
            return True
        if NUMPY_AVAILABLE:
            arr = np.frombuffer(b, dtype=np.uint8)
            printable_mask = (arr >= 32) & (arr <= 126)
            ws_mask = (arr == 9) | (arr == 10) | (arr == 13)
            printable = int(np.sum(printable_mask | ws_mask))
        else:
            printable = sum(1 for x in b if (32 <= x <= 126) or x in (9,10,13))
        return printable / max(1, len(b)) > 0.9
    except Exception:
        return False

def normalize_text_bytes(b: bytes, path: Path) -> bytes:
    """Whitespace/BOM normalization to boost text ratio (reversible in container)."""
    try:
        # Strip BOM
        for bom in (b'\xef\xbb\xbf', b'\xff\xfe', b'\xfe\xff'):
            if b.startswith(bom):
                b = b[len(bom):]
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            s = b.decode("latin-1", errors="ignore")
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        ext = path.suffix.lower()
        if ext == ".json":
            try:
                data = json.loads(s)
                s = json.dumps(data, separators=(",", ":"))
            except Exception:
                pass
        elif ext in (".html",".htm"):
            s = re.sub(r">\s+<", "><", s)
            s = re.sub(r"\s+", " ", s)
        elif ext == ".css":
            s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
            s = re.sub(r"\s+", " ", s)
            s = re.sub(r";\s*", ";", s)
            s = re.sub(r":\s*", ":", s)
            s = re.sub(r"\s*\{\s*", "{", s)
            s = re.sub(r"\s*\}\s*", "}", s)
        return s.encode("utf-8")
    except Exception:
        return b

def optimize_png_lossless(data: bytes) -> bytes:
    if not PILLOW_AVAILABLE:
        return data
    try:
        img = Image.open(io.BytesIO(data))
        if img.format != "PNG":
            return data
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True, compress_level=9)
        new = out.getvalue()
        return new if len(new) < len(data) else data
    except Exception:
        return data

# ---------------------------------------------------------------
# TAR build / read (manifest-free)
# ---------------------------------------------------------------

def build_solid_tar(file_paths: List[Path], base_dir: Path, log, lossless_bytes: bool, image_opt: bool,
                    progress_cb: Optional[Callable[[float], None]]=None,
                    p_start: float=0.05, p_end: float=0.60) -> bytes:
    """
    Constrói TAR sólido e reporta progresso linear de p_start→p_end por arquivo.
    """
    if not file_paths:
        if progress_cb: progress_cb(p_end)
        return b""
    buf = io.BytesIO()
    total = max(1, len(file_paths))
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i, p in enumerate(file_paths, 1):
            try:
                try:
                    rel = p.relative_to(base_dir)
                except Exception:
                    rel = p.name
                rel = str(rel).replace("\\", "/")
                if rel in (".","..","") or rel.startswith("/") or ":" in rel:
                    rel = p.name
                log(f"🔍 [tar] Adding: {p} as '{rel}'")
            except Exception as e:
                log(f"🟨 [tar] Error processing {p}: {e}")
                rel = p.name

            b = p.read_bytes()
            if not lossless_bytes and is_probably_text(p):
                b = normalize_text_bytes(b, p)
            if image_opt and p.suffix.lower() == ".png":
                b = optimize_png_lossless(b)

            info = tarfile.TarInfo(name=rel)
            info.size = len(b)
            try:
                info.mtime = int(os.path.getmtime(p))
            except Exception:
                info.mtime = int(time.time())
            tf.addfile(info, io.BytesIO(b))

            if progress_cb:
                frac = p_start + (p_end - p_start) * (i / total)
                progress_cb(frac)
    if progress_cb: progress_cb(p_end)
    return buf.getvalue()

# ---------------------------------------------------------------
# Compression backends
# ---------------------------------------------------------------

def compress_lzma2_xz(data: bytes) -> bytes:
    if not lzma:
        raise RuntimeError("lzma not available")
    return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME, format=lzma.FORMAT_XZ)

def decompress_lzma2_xz(data: bytes) -> bytes:
    if not lzma:
        raise RuntimeError("lzma not available")
    return lzma.decompress(data, format=lzma.FORMAT_XZ)

def compress_zstd_max(data: bytes, level=22, threads=0) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard not available")
    cctx = zstd.ZstdCompressor(level=level, write_content_size=True, write_checksum=True, threads=threads)
    return cctx.compress(data)

def decompress_zstd(data: bytes) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard not available")
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)

def compress_pyzstd_max(data: bytes, level=22, threads=0) -> bytes:
    """pyzstd fast path: use level int; threads: -1 auto when 0 requested."""
    if pyzstd is None:
        raise RuntimeError("pyzstd not available")
    t = -1 if threads == 0 else int(threads)
    try:
        return pyzstd.compress(data, level_or_option=int(level), threads=t)
    except TypeError:
        # Older pyzstd: no threads kw
        return pyzstd.compress(data, level_or_option=int(level))

def compress_brotli_max(data: bytes) -> bytes:
    if brotli is None:
        raise RuntimeError("brotli not available")
    return brotli.compress(data, quality=11, lgwin=24)

def decompress_brotli(data: bytes) -> bytes:
    if brotli is None:
        raise RuntimeError("brotli not available")
    return brotli.decompress(data)

def compress_lz4_high(data: bytes) -> bytes:
    if lz4f is None:
        raise RuntimeError("lz4 not available")
    return lz4f.compress(data, compression_level=12, content_checksum=True, block_checksum=True)

def decompress_lz4(data: bytes) -> bytes:
    if lz4f is None:
        raise RuntimeError("lz4 not available")
    return lz4f.decompress(data)

def compress_two_stage_zstd_brotli(data: bytes, level=22, threads=0) -> bytes:
    if zstd is None or brotli is None:
        raise RuntimeError("two-stage requires zstd + brotli")
    stage1 = compress_zstd_max(data, level=level, threads=threads)
    if len(stage1) <= 10000:
        return stage1
    stage2 = brotli.compress(stage1, quality=10, lgwin=24)
    return stage2 if len(stage2) <= len(stage1) * 0.98 else stage1

# ---------------------------------------------------------------
# Container v8/v9
# ---------------------------------------------------------------

HEADER_V8 = struct.Struct("<8sBBIQQI16s")
HEADER_V9 = struct.Struct("<8sBBIQQQI16s")

def write_container(out_path: Path, method: int, tar_bytes: bytes, comp_bytes: bytes, dict_bytes: bytes | None = None, flags: int = 0, version: int = VERSION_V9):
    dict_len = len(dict_bytes) if dict_bytes else 0
    crc = binascii.crc32(tar_bytes) & 0xffffffff
    if version == VERSION_V9:
        hdr = HEADER_V9.pack(
            MAGIC, version, method, flags,
            len(tar_bytes), len(comp_bytes), dict_len, crc,
            b"\x00" * 16
        )
    else:
        hdr = HEADER_V8.pack(
            MAGIC, VERSION_V8, method, flags,
            len(tar_bytes), len(comp_bytes), crc,
            b"\x00" * 16
        )
    with open(out_path, "wb") as w:
        w.write(hdr)
        if version == VERSION_V9 and dict_len > 0:
            w.write(dict_bytes)
        w.write(comp_bytes)

def read_container(src_path: Path):
    with open(src_path, "rb") as f:
        head = f.read(HEADER_V8.size)
        if len(head) != HEADER_V8.size:
            raise ValueError("File too small")
        magic, ver, method, flags, tar_len, comp_len, crc, reserved = HEADER_V8.unpack(head)
        if magic != MAGIC:
            raise ValueError("Not a FROG container")
        if ver == VERSION_V9:
            rest = f.read(HEADER_V9.size - HEADER_V8.size)
            if len(rest) != (HEADER_V9.size - HEADER_V8.size):
                raise ValueError("Truncated v9 header")
            magic2, ver2, method2, flags2, tar_len2, comp_len2, dict_len, crc2, reserved2 = HEADER_V9.unpack(head + rest)
            if magic2 != MAGIC or ver2 != VERSION_V9:
                raise ValueError("Malformed v9 header")
            dict_bytes = f.read(dict_len) if dict_len else b""
            comp = f.read(comp_len)
            if len(comp) != comp_len:
                raise ValueError("Truncated container payload")
            return ver2, method2, flags2, dict_bytes, tar_len2, comp, crc2
        elif ver == VERSION_V8:
            comp = f.read(comp_len)
            if len(comp) != comp_len:
                raise ValueError("Truncated container payload")
            return ver, method, flags, b"", tar_len, comp, crc
        else:
            raise ValueError(f"Unsupported version: {ver}")

def _decompress_by_method(method: int, payload: bytes) -> bytes:
    if method == METHOD_ZSTD:
        return decompress_zstd(payload)
    if method == METHOD_LZMA2:
        return decompress_lzma2_xz(payload)
    if method == METHOD_BROTLI:
        return decompress_brotli(payload)
    if method == METHOD_STORE:
        return payload
    if method == METHOD_ZSTD_THEN_BROTLI:
        return decompress_zstd(decompress_brotli(payload))
    if method == METHOD_LZ4:
        return decompress_lz4(payload)
    raise ValueError(f"Unknown method: {method}")

def verify_container_in_memory(path: Path, log, progress_cb: Optional[Callable[[float], None]]=None, p_from: float=0.90, p_to: float=0.98) -> bool:
    try:
        if progress_cb: progress_cb(p_from)
        ver, method, flags, dict_b, tar_len, comp, crc = read_container(path)
        tar_b = _decompress_by_method(method, comp)
        if len(tar_b) != tar_len:
            log(f"🟨 [verify] Size mismatch: header {tar_len} vs actual {len(tar_b)}")
            if progress_cb: progress_cb(p_to)
            return False
        c = binascii.crc32(tar_b) & 0xffffffff
        ok = (c == crc)
        # count files
        cnt = 0
        with tarfile.open(fileobj=io.BytesIO(tar_b), mode="r:") as tf:
            for _ in tf: cnt += 1
        log(f"✅ [verify] OK — files={cnt}, method={method}, ver={ver}, crc={hex(c)}")
        if progress_cb: progress_cb(p_to)
        return ok
    except Exception as e:
        log(f"🟥 [verify] Failed: {e}")
        if progress_cb: progress_cb(p_to)
        return False

# ---------------------------------------------------------------
# Pack/unpack
# ---------------------------------------------------------------

class PackOptions:
    def __init__(self):
        self.zstd_level = 22
        self.threads = 0
        self.never_inflate_policy = "input"  # "off" | "input" | "tar"
        self.enable_brotli = True
        self.enable_lzma2 = True
        self.enable_lz4 = True
        self.enable_pyzstd = True
        self.enable_two_stage = True
        self.lossless_bytes = False
        self.image_optimize = True
        self.parallel_candidates = True

def _candidate_run(kind: str, tar_bytes: bytes, opts: PackOptions) -> Tuple[str,int,bytes]:
    if kind == "brotli" and brotli is not None:
        return ("FPKB(Brotli-11)", METHOD_BROTLI, compress_brotli_max(tar_bytes))
    if kind == "two" and (zstd is not None and brotli is not None):
        c = compress_two_stage_zstd_brotli(tar_bytes, level=opts.zstd_level, threads=opts.threads)
        meth = METHOD_ZSTD_THEN_BROTLI if len(c) < len(tar_bytes) else METHOD_ZSTD
        return ("FPKZB(Zstd→Brotli10)", meth, c)
    if kind == "pyzstd" and pyzstd is not None:
        return ("FPK2a(pyzstd)", METHOD_ZSTD, compress_pyzstd_max(tar_bytes, level=opts.zstd_level, threads=opts.threads))
    if kind == "zstd" and zstd is not None:
        return ("FPK2(Zstd)", METHOD_ZSTD, compress_zstd_max(tar_bytes, level=opts.zstd_level, threads=opts.threads))
    if kind == "lzma2" and lzma is not None:
        return ("FPK1(LZMA2)", METHOD_LZMA2, compress_lzma2_xz(tar_bytes))
    if kind == "lz4" and lz4f is not None:
        return ("FPKL(LZ4-12)", METHOD_LZ4, compress_lz4_high(tar_bytes))
    return ("STORE", METHOD_STORE, tar_bytes)

def choose_and_pack(file_paths: List[Path], base_dir: Path, out_path: Path, opts: PackOptions, log,
                    progress_cb: Optional[Callable[[float], None]]=None):
    t0 = time.time()
    if progress_cb: progress_cb(0.01)
    log("🧠 [info] Smart Ultra chooser starting…")
    base_dir = base_dir or (file_paths[0].parent if file_paths else Path.cwd())
    log(f"🔍 [base_dir] {base_dir}")

    # 0.05 → 0.60: TAR build
    tar_bytes = build_solid_tar(file_paths, base_dir, log, lossless_bytes=opts.lossless_bytes,
                                image_opt=opts.image_optimize, progress_cb=progress_cb,
                                p_start=0.05, p_end=0.60)
    if not tar_bytes:
        raise RuntimeError("Nothing to pack")

    tar_len = len(tar_bytes)
    log("🧰 [info] Building solid TAR …")
    log(f"ℹ️ [info] TAR size: {tar_len/1024:.2f}KB")

    input_payload = int(sum((p.stat().st_size for p in file_paths)))
    if NUMPY_AVAILABLE:
        try:
            sizes = np.array([p.stat().st_size for p in file_paths], dtype=np.int64)
            input_payload = int(np.sum(sizes))
        except Exception:
            pass
    log(f"📊 [ratio] Input payload: {input_payload/1024:.2f}KB")

    if PANDAS_AVAILABLE:
        try:
            df = pd.DataFrame({"size":[p.stat().st_size for p in file_paths]})
            mean_sz = float(df["size"].mean())
            std_sz  = float(df["size"].std(ddof=0))
            log(f"📊 [stats] files={len(df)} avg={mean_sz:.1f}B std={std_sz:.1f}B")
        except Exception:
            pass
    if SCIPY_AVAILABLE and len(file_paths) > 1:
        try:
            sizes = [p.stat().st_size for p in file_paths]
            vc = collections.Counter(sizes)
            probs = [c/len(sizes) for c in vc.values()]
            ent = float(scipy_stats.entropy(probs, base=2))
            log(f"📊 [stats] size-distribution entropy ≈ {ent:.3f} bits")
        except Exception:
            pass

    # 0.60 → 0.85: candidatos
    candidate_kinds = []
    if opts.enable_brotli and brotli is not None: candidate_kinds.append("brotli")
    if opts.enable_two_stage and (zstd is not None and brotli is not None): candidate_kinds.append("two")
    if opts.enable_pyzstd and pyzstd is not None: candidate_kinds.append("pyzstd")
    if zstd is not None: candidate_kinds.append("zstd")
    if opts.enable_lzma2 and lzma is not None: candidate_kinds.append("lzma2")
    if opts.enable_lz4 and lz4f is not None: candidate_kinds.append("lz4")
    candidate_kinds.append("store")

    results: List[Tuple[str,int,bytes]] = []
    def _progress_candidates_step(i, n):
        if progress_cb:
            progress_cb(0.60 + 0.25 * (i / max(1,n)))

    if opts.parallel_candidates and JOBLIB_AVAILABLE and len(tar_bytes) >= (128<<10):
        n_jobs = min(max(1, cpu_count()-1), 6)
        try:
            # Paralelo: damos um bump grosso e fechamos em 0.85 ao final
            progress_cb and progress_cb(0.80)
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_candidate_run)(k, tar_bytes, opts) for k in candidate_kinds
            )
            progress_cb and progress_cb(0.85)
        except Exception:
            tmp = []
            for i, k in enumerate(candidate_kinds, 1):
                tmp.append(_candidate_run(k, tar_bytes, opts)); _progress_candidates_step(i, len(candidate_kinds))
            results = tmp
    else:
        tmp = []
        for i, k in enumerate(candidate_kinds, 1):
            tmp.append(_candidate_run(k, tar_bytes, opts)); _progress_candidates_step(i, len(candidate_kinds))
        results = tmp
        progress_cb and progress_cb(0.85)

    for name, meth, comp in results:
        log(f"🧪 [cand] {name}: {len(comp)/1024:.2f}KB")

    best = min(results, key=lambda x: len(x[2]))
    best_name, best_method, best_comp = best

    if opts.never_inflate_policy == "input":
        if len(best_comp) >= input_payload:
            best_name, best_method, best_comp = ("STORE", METHOD_STORE, tar_bytes)
            log("🛑 [ratio] Never-Inflate(input) → STORE")
    elif opts.never_inflate_policy == "tar":
        if len(best_comp) >= tar_len:
            best_name, best_method, best_comp = ("STORE", METHOD_STORE, tar_bytes)
            log("🛑 [ratio] Never-Inflate(tar) → STORE")
    # "off" => do nothing

    # 0.85 → 0.90: write
    write_container(out_path, best_method, tar_bytes, best_comp, dict_bytes=None, flags=0, version=VERSION_V9)
    progress_cb and progress_cb(0.90)

    output_size = os.path.getsize(out_path)
    saved = input_payload - output_size
    red = 100.0 * (1 - (output_size / max(1, input_payload)))
    factor = (input_payload / max(1, output_size)) if output_size else 0.0

    log(f"✅ [ratio] Output: {output_size/1024:.2f}KB  | Reduction: {red:.2f}%  | "
        f"Factor: {factor:.2f}×  | Saved: {saved/1024:.2f}KB  | Picked: {best_name}")

    ok = verify_container_in_memory(out_path, log, progress_cb=progress_cb, p_from=0.90, p_to=0.98)
    if not ok:
        raise RuntimeError("Verify failed")
    elapsed = time.time() - t0
    progress_cb and progress_cb(1.00)
    log(f"✅ [ok] Done! Time: {elapsed:.2f}s (parallel={opts.parallel_candidates and JOBLIB_AVAILABLE})")

def safe_extract_tar_from_memory(tar_data: bytes, dest_dir: str, log,
                                 progress_cb: Optional[Callable[[float], None]]=None,
                                 p_start: float=0.05, p_end: float=1.00):
    dest_dir = get_safe_extraction_dir(dest_dir)
    root = Path(dest_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    skipped_count = 0
    total_files = 0

    # Primeiro pass: contar arquivos para progresso
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:") as tf:
        for m in tf:
            if m.isfile():
                total_files += 1

    done = 0
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:") as tf:
        for m in tf:
            try:
                if not m.isfile():
                    skipped_count += 1
                    continue
                name = m.name
                if not name or name.strip() in (".","..",""):
                    log(f"🛡️ [safe] Skipping invalid name: {name}")
                    skipped_count += 1
                    continue
                clean_name = Path(name).name
                if not clean_name or clean_name in (".",".."):
                    log(f"🛡️ [safe] Skipping invalid clean name: {name} -> {clean_name}")
                    skipped_count += 1
                    continue

                target = root / clean_name
                src = tf.extractfile(m)
                if src is None:
                    skipped_count += 1
                    continue
                with open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1<<20)

                extracted_count += 1
                done += 1
                if progress_cb and total_files:
                    frac = p_start + (p_end - p_start) * (done / total_files)
                    progress_cb(min(0.99, frac))
                log(f"📄 [extract] {clean_name}")
            except Exception as e:
                log(f"🟨 [extract] Failed {m.name}: {e}")
                skipped_count += 1

    if progress_cb: progress_cb(p_end)
    log(f"📦 [extract] Summary: {extracted_count} files extracted, {skipped_count} skipped")

def unpack_container(src_path: Path, out_dir: Path, log, progress_cb: Optional[Callable[[float], None]]=None):
    log("📂 [info] Unpacking .frog …")
    progress_cb and progress_cb(0.02)
    ver, method, flags, dict_b, tar_len, comp, crc = read_container(src_path)
    log(f"🔍 [format] method={method} ver={ver} dict={len(dict_b)}B tar_len={tar_len}")
    progress_cb and progress_cb(0.08)
    tar_b = _decompress_by_method(method, comp)
    progress_cb and progress_cb(0.12)
    if len(tar_b) != tar_len:
        raise ValueError("Length mismatch")
    if (binascii.crc32(tar_b) & 0xffffffff) != crc:
        raise ValueError("CRC mismatch")
    safe_extract_tar_from_memory(tar_b, str(out_dir), log, progress_cb=progress_cb, p_start=0.15, p_end=1.00)
    log(f"✅ [ok] Extracted to: {out_dir}")

# ---------------------------------------------------------------
# GUI
# ---------------------------------------------------------------

class FrogGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        try:
            # slightly larger scaling for 1080p
            self.tk.call('tk', 'scaling', 1.2)
        except Exception:
            pass
        self.minsize(1150, 700)
        self.geometry("1280x800+60+60")
        self.style = ttk.Style(self)
        self._theme_name = "Frog"

        self.files: List[Path] = []
        self.base_dir: Optional[Path] = None

        self.opts = PackOptions()
        self._build_ui()
        self._apply_theme(self._theme_name)

        numpy_status = "✅" if NUMPY_AVAILABLE else "❌"
        brotli_status = "✅" if brotli else "❌"
        zstd_status = "✅" if zstd else "❌"
        pyzstd_status = "✅" if pyzstd else "❌"
        lz4_status = "✅" if lz4f else "❌"

        self._log(f"ℹ️ [info] M9 ready: NumPy={numpy_status} Brotli={brotli_status} Zstd={zstd_status} pyzstd={pyzstd_status} LZ4={lz4_status}.")

    # ---- Theme skins
    def _apply_theme(self, name: str):
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self._theme_name = name
        if name == "Frog":
            bg = "#e7f8ea"; fg="#101010"
        elif name == "Lagoon":
            bg = "#e7f4fb"; fg="#101010"
        elif name == "Dark":
            bg = "#222"; fg="#eaeaea"
        else:
            bg = "#f2f2f2"; fg="#101010"

        self.configure(bg=bg)
        # ttk defaults
        self.style.configure(".", font=("Segoe UI", 10), background=bg, foreground=fg)
        for cls in ("TLabel", "TFrame", "TLabelframe", "TLabelframe.Label", "TButton", "TCheckbutton", "TCombobox", "TProgressbar"):
            try:
                self.style.configure(cls, background=bg, foreground=fg)
            except Exception:
                pass
        # Non-ttk widgets updated directly (e.g., Text)
        try:
            self.log_txt.configure(bg=bg, fg=fg, insertbackground=fg)
        except Exception:
            pass

        self.update_idletasks()
        self._log(f"🎨 [theme] Applied: {name} {LOGO}")

    def _ui_apply_theme(self):
        name = self.theme_cmb.get()
        self._apply_theme(name)

    # ---- Logging to GUI + optional sinks
    def _log(self, msg: str):
        _log_file(msg)
        _log_rich(msg)
        ts = time.strftime("%H:%M:%S ")
        try:
            self.log_txt.insert("end", ts + msg + "\n")
            self.log_txt.see("end")
        except Exception:
            pass
        self.update_idletasks()

    # ---- Progress (thread-safe)
    def _progress(self, v: float):
        v = float(max(0.0, min(1.0, v)))
        try:
            self.progress['value'] = v * 100.0
        except Exception:
            # widget may not be ready
            pass

    def _progress_async(self, v: float):
        try:
            self.after(0, lambda: self._progress(v))
        except Exception:
            pass

    # ---- File mgmt
    def _add_files(self):
        paths = filedialog.askopenfilenames(title="Add files")
        if not paths: return
        for p in paths:
            p = Path(p)
            self.files.append(p)
            self.file_list.insert("end", str(p))
        self._guess_base()

    def _add_folder(self):
        d = filedialog.askdirectory(title="Add folder")
        if not d: return
        d = Path(d)
        for root, _, files in os.walk(d):
            for fn in files:
                p = Path(root) / fn
                self.files.append(p)
                self.file_list.insert("end", str(p))
        self._guess_base()

    def _clear(self):
        self.files.clear()
        self.file_list.delete(0, "end")
        self.base_dir = None
        self._progress(0.0)

    def _guess_base(self):
        if not self.files:
            self.base_dir = None
            return
        try:
            common = os.path.commonpath([str(p.resolve()) for p in self.files])
            self.base_dir = Path(common)
        except Exception:
            self.base_dir = self.files[0].parent

    def _apply_profile(self):
        prof = self.profile_cmb.get()
        if prof == "Ultra-Max (All)":
            self.var_brotli.set(True)
            self.var_lzma2.set(True)
            self.var_lz4.set(True)
            self.var_pyzstd.set(True)
            self.var_two_stage.set(True)
            self.var_parallel.set(True)
            self.var_lossless.set(False)
            self.var_png_opt.set(True)
            self.spin_level.set("22")
        elif prof == "Balanced":
            self.var_brotli.set(True)
            self.var_lzma2.set(True)
            self.var_lz4.set(False)
            self.var_pyzstd.set(True)
            self.var_two_stage.set(False)
            self.var_parallel.set(True)
            self.var_lossless.set(False)
            self.var_png_opt.set(True)
            self.spin_level.set("19")
        elif prof == "Store-only":
            self.var_brotli.set(False)
            self.var_lzma2.set(False)
            self.var_lz4.set(False)
            self.var_pyzstd.set(False)
            self.var_two_stage.set(False)
            self.var_parallel.set(False)
            self.var_lossless.set(True)
            self.var_png_opt.set(False)
        self._log(f"🧩 [profile] Applied: {prof}")

    def _clear_dict_cache(self):
        base = Path(tempfile.gettempdir()) / "frogpack_dict_cache"
        try:
            if base.is_dir():
                shutil.rmtree(base, ignore_errors=True)
            base.mkdir(parents=True, exist_ok=True)
            self._log(f"🧹 [cache] Cleared: {str(base)}")
        except Exception as e:
            self._log(f"🟨 [cache] Failed to clear: {e}")

    def _pack(self):
        if not self.files:
            messagebox.showwarning("FrogPack", "Add some files first.")
            return
        out = filedialog.asksaveasfilename(
            title="Save .frog",
            defaultextension=APP_EXT,
            filetypes=[("FrogPack Container", f"*{APP_EXT}"), ("All files", "*.*")])
        if not out:
            return
        out_path = Path(out)

        opts = PackOptions()
        opts.never_inflate_policy = self.nevinf_cmb.get()
        opts.enable_brotli = self.var_brotli.get()
        opts.enable_lzma2 = self.var_lzma2.get()
        opts.enable_lz4   = self.var_lz4.get()
        opts.enable_pyzstd= self.var_pyzstd.get()
        opts.enable_two_stage = self.var_two_stage.get()
        opts.parallel_candidates = self.var_parallel.get()
        opts.lossless_bytes = self.var_lossless.get()
        opts.image_optimize = self.var_png_opt.get()
        opts.zstd_level = int(self.spin_level.get())
        opts.threads = int(self.spin_threads.get())

        base_dir = self.base_dir or (self.files[0].parent if self.files else Path.cwd())

        # Reset progress to 0
        self._progress(0.0)

        def worker():
            try:
                choose_and_pack(self.files, base_dir, out_path, opts, self._log,
                                progress_cb=self._progress_async)
            except Exception as e:
                self._log(f"🟥 [error] {e}")
                messagebox.showerror("FrogPack", str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _unpack(self):
        src = filedialog.askopenfilename(title="Choose .frog",
                                         filetypes=[("FrogPack Container", f"*{APP_EXT}"), ("All files", "*.*")])
        if not src:
            return
        out_dir = filedialog.askdirectory(title="Choose output folder")
        if not out_dir:
            return
        safe_out = get_safe_extraction_dir(out_dir)
        if safe_out != out_dir:
            self._log(f"🛡️ [safe] Using fallback directory: {safe_out}")

        # Reset progress to 0
        self._progress(0.0)

        def worker():
            try:
                unpack_container(Path(src), Path(safe_out), self._log,
                                 progress_cb=self._progress_async)
            except Exception as e:
                self._log(f"🟥 [error] {e}")
                messagebox.showerror("FrogPack", str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _build_ui(self):
        topbar = ttk.Frame(self)
        topbar.pack(side="top", fill="x", padx=6, pady=6)

        ttk.Button(topbar, text="Add files…", command=self._add_files).pack(side="left", padx=3)
        ttk.Button(topbar, text="Add folder…", command=self._add_folder).pack(side="left", padx=3)
        ttk.Button(topbar, text="Clear", command=self._clear).pack(side="left", padx=3)

        ttk.Label(topbar, text="Profile:").pack(side="left", padx=(12, 2))
        self.profile_cmb = ttk.Combobox(topbar, state="readonly",
                                        values=["Ultra-Max (All)", "Balanced", "Store-only"])
        self.profile_cmb.current(0)
        self.profile_cmb.pack(side="left", padx=3)
        ttk.Button(topbar, text="Apply", command=self._apply_profile).pack(side="left", padx=3)

        ttk.Label(topbar, text="Theme:").pack(side="left", padx=(18, 2))
        self.theme_cmb = ttk.Combobox(topbar, state="readonly", values=list(THEMES))
        self.theme_cmb.current(0)
        self.theme_cmb.pack(side="left", padx=3)
        ttk.Button(topbar, text="Apply theme", command=self._ui_apply_theme).pack(side="left", padx=3)

        ttk.Button(topbar, text="Clear dict cache", command=self._clear_dict_cache).pack(side="left", padx=12)

        ttk.Button(topbar, text="Unpack (.frog)", command=self._unpack).pack(side="right", padx=3)
        ttk.Button(topbar, text="Pack (.frog)", command=self._pack).pack(side="right", padx=3)

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        lf = ttk.LabelFrame(left, text="Files")
        lf.pack(fill="both", expand=True, padx=6, pady=6)

        self.file_list = tk.Listbox(lf, selectmode="extended")
        yscroll = ttk.Scrollbar(lf, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=yscroll.set)
        self.file_list.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        opt = ttk.LabelFrame(right, text="Compression")
        opt.pack(fill="x", padx=6, pady=6)

        self.var_brotli = tk.BooleanVar(value=True)
        self.var_lzma2 = tk.BooleanVar(value=True)
        self.var_lz4   = tk.BooleanVar(value=True)
        self.var_pyzstd= tk.BooleanVar(value=True)
        self.var_two_stage = tk.BooleanVar(value=True)
        self.var_parallel = tk.BooleanVar(value=True)
        self.var_lossless = tk.BooleanVar(value=False)
        self.var_png_opt = tk.BooleanVar(value=True)

        row = 0
        ttk.Checkbutton(opt, text="Enable Brotli-11 candidate", variable=self.var_brotli).grid(row=row, column=0, sticky="w", padx=6, pady=3, columnspan=2); row += 1
        ttk.Checkbutton(opt, text="Enable LZMA2 candidate", variable=self.var_lzma2).grid(row=row, column=0, sticky="w", padx=6, pady=3, columnspan=2); row += 1
        ttk.Checkbutton(opt, text="Enable pyzstd (if available)", variable=self.var_pyzstd).grid(row=row, column=0, sticky="w", padx=6, pady=3, columnspan=2); row += 1
        ttk.Checkbutton(opt, text="Enable LZ4 high-compression", variable=self.var_lz4).grid(row=row, column=0, sticky="w", padx=6, pady=3, columnspan=2); row += 1
        ttk.Checkbutton(opt, text="Enable two-stage Zstd→Brotli", variable=self.var_two_stage).grid(row=row, column=0, sticky="w", padx=6, pady=3, columnspan=2); row += 1
        ttk.Checkbutton(opt, text="Parallel candidate run (joblib)", variable=self.var_parallel).grid(row=row, column=0, sticky="w", padx=6, pady=3, columnspan=2); row += 1
        ttk.Checkbutton(opt, text="Lossless bytes (disable text normalization)", variable=self.var_lossless).grid(row=row, column=0, sticky="w", padx=6, pady=3, columnspan=2); row += 1
        ttk.Checkbutton(opt, text="PNG lossless optimize (Pillow)", variable=self.var_png_opt).grid(row=row, column=0, sticky="w", padx=6, pady=3, columnspan=2); row += 1

        ttk.Label(opt, text="Never-Inflate policy:").grid(row=row, column=0, sticky="e", padx=6)
        self.nevinf_cmb = ttk.Combobox(opt, state="readonly", values=["input","tar","off"], width=7)
        self.nevinf_cmb.set("input")
        self.nevinf_cmb.grid(row=row, column=1, sticky="w", padx=3); row += 1

        ttk.Label(opt, text="Zstd level:").grid(row=row, column=0, sticky="e", padx=6)
        self.spin_level = ttk.Spinbox(opt, from_=1, to=22, width=6); self.spin_level.set("22"); self.spin_level.grid(row=row, column=1, sticky="w", padx=3); row += 1

        ttk.Label(opt, text="Threads (0=auto):").grid(row=row, column=0, sticky="e", padx=6)
        self.spin_threads = ttk.Spinbox(opt, from_=0, to=max(64, cpu_count()), width=6); self.spin_threads.set("0"); self.spin_threads.grid(row=row, column=1, sticky="w", padx=3); row += 1

        pf = ttk.LabelFrame(right, text="Progress")
        pf.pack(fill="x", padx=6, pady=6)
        self.progress = ttk.Progressbar(pf, mode="determinate", maximum=100.0, value=0.0)
        self.progress.pack(fill="x", padx=8, pady=8)

        logf = ttk.LabelFrame(right, text="Log")
        logf.pack(fill="both", expand=True, padx=6, pady=6)
        self.log_txt = tk.Text(logf, height=12, wrap="word")
        self.log_txt.pack(fill="both", expand=True)

# Entry
if __name__ == "__main__":
    app = FrogGUI()
    app.mainloop()
