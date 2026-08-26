from __future__ import annotations

import json
import html as html_lib
import mimetypes
import re
import uuid
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .exporter import article_document_html
from .models import QuickReadArticle


WECHAT_API = "https://api.weixin.qq.com"
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", flags=re.I | re.S)
_SRC_ATTR_RE = re.compile(r"(?<![\w:-])src(\s*=\s*)([\"'])(.*?)\2", flags=re.I | re.S)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class WechatPublishError(RuntimeError):
    pass


@dataclass
class WechatDraftConfig:
    app_id: str = ""
    app_secret: str = ""
    author: str = ""
    cover_image_name: str = ""
    show_cover_pic: bool = False
    content_source_url: str = ""
    need_open_comment: bool = False
    only_fans_can_comment: bool = False


def _http_json(url: str, data: bytes | None = None, headers: dict[str, str] | None = None, timeout: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode(response.headers.get_content_charset() or "utf-8"))
    if isinstance(payload, dict) and payload.get("errcode") not in (None, 0):
        raise WechatPublishError(f"WeChat API error {payload.get('errcode')}: {payload.get('errmsg')}")
    return payload


def get_access_token(app_id: str, app_secret: str) -> str:
    if not app_id or not app_secret:
        raise WechatPublishError("请填写公众号 APP_ID 和 APP_SECRET。")
    query = urllib.parse.urlencode({"grant_type": "client_credential", "appid": app_id, "secret": app_secret})
    payload = _http_json(f"{WECHAT_API}/cgi-bin/token?{query}")
    token = str(payload.get("access_token") or "")
    if not token:
        raise WechatPublishError(f"WeChat token response missing access_token: {payload}")
    return token


def _multipart_body(field_name: str, file_name: str, data: bytes, content_type: str = "") -> tuple[bytes, str]:
    boundary = f"----weixin-lite-{uuid.uuid4().hex}"
    mime = content_type or mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_name}"\r\n'.encode("utf-8"),
            f"Content-Type: {mime}\r\n\r\n".encode("utf-8"),
            data,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


def upload_cover_material(access_token: str, file_name: str, data: bytes) -> str:
    body, content_type = _multipart_body("media", file_name, data)
    query = urllib.parse.urlencode({"access_token": access_token, "type": "image"})
    payload = _http_json(
        f"{WECHAT_API}/cgi-bin/material/add_material?{query}",
        data=body,
        headers={"Content-Type": content_type},
        timeout=120,
    )
    media_id = str(payload.get("media_id") or "")
    if not media_id:
        raise WechatPublishError(f"WeChat material response missing media_id: {payload}")
    return media_id


def upload_content_image(access_token: str, file_name: str, data: bytes) -> str:
    body, content_type = _multipart_body("media", file_name, data)
    query = urllib.parse.urlencode({"access_token": access_token})
    payload = _http_json(
        f"{WECHAT_API}/cgi-bin/media/uploadimg?{query}",
        data=body,
        headers={"Content-Type": content_type},
        timeout=120,
    )
    url = str(payload.get("url") or "")
    if not url:
        raise WechatPublishError(f"WeChat uploadimg response missing url: {payload}")
    return url


def _image_sources(content_html: str) -> list[str]:
    sources: list[str] = []
    for tag_match in _IMG_TAG_RE.finditer(content_html or ""):
        attr = _SRC_ATTR_RE.search(tag_match.group(0))
        if attr:
            sources.append(html_lib.unescape(attr.group(3)).strip())
    return sources


def _classify_image_source(src: str) -> tuple[str, str]:
    lowered = src.lower()
    if lowered.startswith("file:") or _WINDOWS_ABSOLUTE_RE.match(src) or src.startswith("\\\\"):
        raise WechatPublishError(f"正文图片禁止使用本机绝对路径：{src}")
    if lowered.startswith(("http://", "https://")) or src.startswith("//"):
        return "external", src
    if lowered.startswith("data:image/"):
        return "embedded", src

    normalized = src.replace("\\", "/")
    prefix = next((item for item in ("./images/", "/images/", "images/") if normalized.startswith(item)), "")
    if not prefix:
        raise WechatPublishError(f"正文图片地址无法识别，请使用 images/ 相对路径或 HTTPS 外链：{src}")
    path = normalized[len(prefix) :].split("?", 1)[0].split("#", 1)[0]
    image_name = urllib.parse.unquote(path)
    parts = image_name.split("/")
    if not image_name or any(part in {"", ".", ".."} for part in parts):
        raise WechatPublishError(f"正文图片路径不安全：{src}")
    return "local", image_name


def referenced_content_images(content_html: str) -> list[str]:
    names: list[str] = []
    for src in _image_sources(content_html):
        kind, value = _classify_image_source(src)
        if kind == "local" and value not in names:
            names.append(value)
    return names


def external_content_images(content_html: str) -> list[str]:
    external: list[str] = []
    for src in _image_sources(content_html):
        kind, value = _classify_image_source(src)
        if kind == "external" and value not in external:
            external.append(value)
    return external


def missing_content_images(content_html: str, image_assets: dict[str, bytes]) -> list[str]:
    return [name for name in referenced_content_images(content_html) if not image_assets.get(name)]


def replace_content_image_sources(html: str, access_token: str, image_assets: dict[str, bytes]) -> str:
    missing = missing_content_images(html, image_assets)
    if missing:
        raise WechatPublishError("正文引用的图片资源缺失：" + "、".join(missing))

    uploaded_urls: dict[str, str] = {}

    def replace_tag(tag_match: re.Match[str]) -> str:
        tag = tag_match.group(0)
        attr = _SRC_ATTR_RE.search(tag)
        if not attr:
            return tag
        src = html_lib.unescape(attr.group(3)).strip()
        kind, image_name = _classify_image_source(src)
        if kind != "local":
            return tag
        if image_name not in uploaded_urls:
            uploaded_urls[image_name] = upload_content_image(access_token, image_name, image_assets[image_name])
        return tag[: attr.start(3)] + html_lib.escape(uploaded_urls[image_name], quote=True) + tag[attr.end(3) :]

    return _IMG_TAG_RE.sub(replace_tag, html)


def duyi_wechat_html(article: QuickReadArticle, content_html: str | None = None) -> str:
    return article_document_html(article, content_html=content_html)


def build_draft_payload(
    article: QuickReadArticle,
    config: WechatDraftConfig,
    thumb_media_id: str = "",
    content_html: str | None = None,
) -> dict[str, Any]:
    return {
        "articles": [
            {
                "title": article.title[:64],
                "author": config.author,
                "digest": article.digest[:120],
                "content": duyi_wechat_html(article, content_html=content_html),
                "content_source_url": config.content_source_url or article.paper.url,
                "thumb_media_id": thumb_media_id,
                "show_cover_pic": 1 if config.show_cover_pic else 0,
                "need_open_comment": 1 if config.need_open_comment else 0,
                "only_fans_can_comment": 1 if config.only_fans_can_comment else 0,
            }
        ]
    }


def create_draft(access_token: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    query = urllib.parse.urlencode({"access_token": access_token})
    return _http_json(
        f"{WECHAT_API}/cgi-bin/draft/add?{query}",
        data=data,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )


def publish_draft(
    article: QuickReadArticle,
    config: WechatDraftConfig,
    image_assets: dict[str, bytes] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    assets = image_assets or {}
    cover_name = config.cover_image_name or str(getattr(article, "cover_image_name", "") or "")
    cover_bytes = assets.get(cover_name, b"") if cover_name else b""
    payload = build_draft_payload(article, config, thumb_media_id="<thumb_media_id_after_upload>")
    content_html = payload["articles"][0]["content"]
    missing_images = missing_content_images(content_html, assets)
    external_images = external_content_images(content_html)
    if dry_run:
        diagnostics = [f"正文引用的图片资源缺失：{name}" for name in missing_images]
        if external_images:
            diagnostics.append("正文包含外链图片，发布时将保留原地址：" + "、".join(external_images))
        if cover_name and not cover_bytes:
            diagnostics.append(f"平台封面资源缺失：{cover_name}")
        elif not cover_name:
            diagnostics.append("尚未选择平台封面；真实创建草稿前必须提供封面图。")
        return {
            "dry_run": True,
            "cover_image_name": cover_name,
            "referenced_content_images": referenced_content_images(content_html),
            "external_images": external_images,
            "diagnostics": diagnostics,
            "payload": payload,
        }
    if not cover_name or not cover_bytes:
        detail = f"（{cover_name}）" if cover_name else ""
        raise WechatPublishError(f"真实发布草稿前必须上传或选择一张可用封面图{detail}。")
    if missing_images:
        raise WechatPublishError("正文引用的图片资源缺失：" + "、".join(missing_images))
    access_token = get_access_token(config.app_id, config.app_secret)
    thumb_media_id = upload_cover_material(access_token, cover_name, cover_bytes)
    payload["articles"][0]["thumb_media_id"] = thumb_media_id
    payload["articles"][0]["content"] = replace_content_image_sources(content_html, access_token, assets)
    result = create_draft(access_token, payload)
    return {"dry_run": False, "payload": payload, "result": result}


def export_wechat_payload(article: QuickReadArticle, config: WechatDraftConfig) -> bytes:
    payload = build_draft_payload(article, config, thumb_media_id="<thumb_media_id_after_upload>")
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
