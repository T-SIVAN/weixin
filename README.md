# 微信公众号文献雷达与中文快读生成器

这是一个独立的 Streamlit 应用，用于跟踪 TdT、PUP、酶促 DNA/RNA 合成和相关酶工程方向的最新文献，并批量生成微信公众号中文单篇快读。

## 功能

- 只填关键词即可检索，无需手写 Boolean 检索式。
- 每天 08:00 北京时间通过 GitHub Actions 自动检索最新文章。
- 检索结果页展示英文标题/摘要与中文标题/摘要对照。
- 合法开放全文自动下载到当前会话并解析；付费文章只输出 DOI、题名、期刊和链接。
- 支持多 PDF 上传进入同一批次分析。
- 公众号正文只写中文，结构参考指定公众号文章：核心要点、关键数据/图例分析、创新意义、原文信息。
- 导出 Markdown、微信公众号 HTML、证据 JSON、图片、付费 DOI 列表和下载状态。
- 页面是一个整合流程，不拆成多个功能选项：关键词检索、开放全文/上传文件、批量分析、导出发布包连续完成。

## Streamlit Cloud 部署

入口文件必须填写：

```text
app.py
```

不要选择 `weixin_lite/exporter.py` 或其他内部模块，否则会出现相对导入错误。

## 本地运行

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 每日检索

默认关键词在 `config/topics.json`。GitHub Actions 文件为 `.github/workflows/daily-literature-radar.yml`，支持定时和手动触发。

可选仓库 Secrets：

- `OPENAI_API_KEY`：用于把检索结果标题和摘要翻译成中文。
- `OPENAI_BASE_URL`：OpenAI-compatible base URL。
- `OPENAI_MODEL`：翻译模型。
- `OPENALEX_MAILTO`：OpenAlex polite pool 邮箱。

如果没有配置模型 Key，系统会保留英文原文，并在中文字段标注“待翻译”，不会阻塞每日检索。

## 命令行

```powershell
python -m weixin_lite.daily_search --config config/topics.json --output data/latest_papers.json
python -m weixin_lite.translate_results --input data/latest_papers.json
python -m weixin_lite.batch_analyze --input data/latest_papers.json --limit 20
```

## 证据边界

只有合法开放全文或用户上传 PDF 可用时，系统才会分析关键图例和关键数据。没有全文时，公众号稿会明确提示“图例分析需开放全文或上传 PDF 后补充”，不会编造图表内容或性能数据。
