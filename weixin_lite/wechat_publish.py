from __future__ import annotations

import json
import mimetypes
import uuid
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .exporter import article_html
from .models import QuickReadArticle


WECHAT_API = "https://api.weixin.qq.com"


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


def duyi_wechat_html(article: QuickReadArticle) -> str:
    body = article.body_html
    return f"""
<section style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;color:#22313f;line-height:1.85;font-size:16px;">
  <section style="border-left:4px solid #0f766e;padding:2px 0 2px 12px;margin:0 0 18px;">
    <h1 style="font-size:22px;line-height:1.35;margin:0;color:#0f172a;">{article.title}</h1>
    <p style="margin:8px 0 0;color:#64748b;font-size:14px;">{article.digest}</p>
  </section>
  <section style="height:1px;background:#dbe7e4;margin:18px 0;"></section>
  {body}
</section>
""".strip()


def build_draft_payload(article: QuickReadArticle, config: WechatDraftConfig, thumb_media_id: str = "") -> dict[str, Any]:
    return {
        "articles": [
            {
                "title": article.title[:64],
                "author": config.author,
                "digest": article.digest[:120],
                "content": duyi_wechat_html(article),
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
    cover_name = config.cover_image_name or article.cover_image_name
    cover_bytes = assets.get(cover_name, b"") if cover_name else b""
    if dry_run:
        return {
            "dry_run": True,
            "cover_image_name": cover_name,
            "payload": build_draft_payload(article, config, thumb_media_id="<thumb_media_id_after_upload>"),
        }
    if not cover_name or not cover_bytes:
        raise WechatPublishError("真实发布草稿前必须上传或选择一张封面图。")
    access_token = get_access_token(config.app_id, config.app_secret)
    thumb_media_id = upload_cover_material(access_token, cover_name, cover_bytes)
    payload = build_draft_payload(article, config, thumb_media_id=thumb_media_id)
    result = create_draft(access_token, payload)
    return {"dry_run": False, "payload": payload, "result": result}


def export_wechat_payload(article: QuickReadArticle, config: WechatDraftConfig) -> bytes:
    payload = build_draft_payload(article, config, thumb_media_id="<thumb_media_id_after_upload>")
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
