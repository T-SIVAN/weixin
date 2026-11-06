"""Local, atomic project snapshots. Credentials are deliberately not serialized."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

from .models import DownloadedPaper, PaperAnalysis, PaperInput, QuickReadArticle
from .pdf_reader import PdfContent

STATE_KEYS = ("papers", "pdfs", "images", "downloads", "articles", "paper_analyses",
              "analysis_cache", "vision_cache", "article_versions", "article_cache", "editor_drafts")


def project_root() -> Path:
    return Path(os.environ.get("WEIXIN_PROJECT_DIR", ".weixin/projects"))


def project_path(project_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", project_id):
        raise ValueError("项目编号无效。")
    return project_root() / (project_id + ".zip")


def snapshot_bytes(state) -> tuple[bytes, str]:
    blobs: dict[str, bytes] = {}
    def blob(value: bytes) -> str:
        name = "blobs/" + hashlib.sha256(value).hexdigest()
        blobs[name] = value
        return name
    manifest = {
        "schema": 1,
        "name": state.get("project_name", "未命名项目"),
        "papers": [item.to_dict() for item in state.get("papers", [])],
        "pdfs": {key: value.to_dict() for key, value in state.get("pdfs", {}).items()},
        "images": {key: blob(value) for key, value in state.get("images", {}).items()},
        "downloads": [{**item.to_dict(), "blob": blob(item.content_bytes)} for item in state.get("downloads", [])],
        "articles": [item.to_dict() for item in state.get("articles", [])],
        "paper_analyses": {key: value.to_dict() for key, value in state.get("paper_analyses", {}).items()},
        **{key: state.get(key, {}) for key in ("analysis_cache", "vision_cache", "article_cache", "editor_drafts")},
        "article_versions": state.get("article_versions", []),
    }
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("workspace.json", encoded)
        for name, value in blobs.items():
            archive.writestr(name, value)
    return output.getvalue(), digest


def restore_bytes(data: bytes) -> dict:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 512 * 1024 * 1024:
                raise ValueError("项目解压后超过 512 MB，无法载入。")
            manifest = json.loads(archive.read("workspace.json"))
            if manifest.get("schema") != 1:
                raise ValueError("不支持此项目版本。")
            images = {key: archive.read(name) for key, name in manifest["images"].items()}
            pdfs = {key: PdfContent.from_dict(value) for key, value in manifest["pdfs"].items()}
            for pdf in pdfs.values():
                names = {item.image_name for item in pdf.legends + pdf.all_figures}
                if pdf.lead_image:
                    names.add(pdf.lead_image.image_name)
                pdf.rendered_images = {name: images[name] for name in names if name in images}
            downloads = [DownloadedPaper(**{key: value for key, value in item.items() if key != "blob"},
                                         content_bytes=archive.read(item["blob"])) for item in manifest["downloads"]]
        return {
            "project_name": manifest.get("name", "未命名项目"),
            "papers": [PaperInput.from_dict(item) for item in manifest["papers"]],
            "pdfs": pdfs, "images": images, "downloads": downloads,
            "articles": [QuickReadArticle.from_dict(item) for item in manifest["articles"]],
            "paper_analyses": {key: PaperAnalysis.from_dict(value) for key, value in manifest["paper_analyses"].items()},
            **{key: manifest.get(key, {}) for key in ("analysis_cache", "vision_cache", "article_cache", "editor_drafts")},
            "article_versions": manifest.get("article_versions", []),
        }
    except (KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError("无法读取项目，请使用工作台导出的项目备份。") from exc


def save_workspace(state) -> None:
    if not any(state.get(key) for key in ("papers", "pdfs", "articles", "article_versions")) and not state.get("_saved_digest"):
        return
    data, digest = snapshot_bytes(state)
    if state.get("_saved_digest") == digest:
        return
    target = project_path(state["project_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    state["_saved_digest"] = digest


def load_workspace(project_id: str) -> dict:
    return restore_bytes(project_path(project_id).read_bytes())


def list_workspaces() -> list[tuple[str, str]]:
    result = []
    for path in sorted(project_root().glob("*.zip"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("workspace.json"))
            result.append((path.stem, manifest.get("name", "未命名项目")))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    return result


def preserve_article(state, article: QuickReadArticle) -> None:
    previous = [item for item in state.get("articles", []) if item.paper.paper_key == article.paper.paper_key]
    for item in previous:
        if item.to_dict() != article.to_dict():
            state.setdefault("article_versions", []).append(item.to_dict())
    state["articles"] = [article] + [item for item in state.get("articles", [])
                                   if item.paper.paper_key != article.paper.paper_key]
