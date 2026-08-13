# 微信文献快读工具

这是一个 Streamlit 应用，用于按期刊清单追踪 Nature、Cell、Science 等顶刊的最新文章，也支持粘贴任意文章、论文、新闻稿、综述或技术正文，并生成可导出的中文微信公众号解读稿。PDF 全文和截图是增强材料，不是生成前提。

## 核心流程

1. **检索与翻译**：按 `config/journals.json` 中启用的期刊，从 PubMed、Europe PMC、OpenAlex、Crossref 抓取最近若干天的最新文章。检索默认不调用翻译模型，结果会先立即展示。
2. **标题翻译**：在网页中点击“翻译全部未翻译标题”或“重试失败翻译”后，才会翻译英文标题。工具会优先使用本地缓存，并显示进度、缓存命中、失败和待翻译统计。
3. **内容与生成**：自动下载合法开放全文、上传 PDF，或粘贴文章内容/摘要/正文。所有有题录、摘要、DOI、PDF 或手动文本的候选都可以生成公众号稿。
4. **导出与发布**：导出项目包、单篇 Markdown/HTML、待全文增强 DOI CSV，也可以 dry-run 预览或真实创建微信公众号草稿。

检索结果显示正式发表日期；Crossref 只读取 `published`、`published-online`、`published-print`、`issued`、`posted` 等发表字段，不把 `created`、`indexed`、`deposited` 这类入库日期当作发表日期。

## 翻译与缓存

支持 OpenAI-compatible 接口，并内置供应商预设：

- `openai`
- `deepseek`
- `siliconflow`
- `custom`

可用环境变量：

```powershell
$env:LLM_PROVIDER="deepseek"
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_BASE_URL="https://api.deepseek.com/v1"
$env:OPENAI_MODEL="deepseek-chat"
```

也兼容：

```powershell
$env:LLM_API_KEY="your-key"
$env:LLM_BASE_URL="https://api.example.com/v1"
$env:LLM_MODEL="your-model"
```

标题翻译缓存默认位于 `data/translation_cache.json`。缓存 key 优先使用 DOI，没有 DOI 时使用标准化英文标题。已有中文标题或缓存命中的记录不会再次调用模型。

翻译只处理标题，不翻译摘要，也不会写入 `abstract_zh`。没有 API Key 时，工具会保留英文标题并标记为“待翻译”，不会写入乱码占位，也不会阻塞后续下载、导出和生成。

遇到 `429 Too Many Requests`、5xx 或 timeout 时，工具会按指数退避自动重试，并尽量读取供应商返回的 `Retry-After`。如果整批失败，会自动拆成单篇重试；仍失败的条目会标记为“失败”，可在网页中点击“重试失败翻译”继续处理。

## 每日顶刊检索

页面默认显示“每日顶刊最新文章”入口。可以调整抓取天数、结果数量、数据源，并启用或停用期刊清单中的条目。检索不依赖固定关键词，也不使用合成生物 strict 相关性过滤。

期刊清单位于 `config/journals.json`，每项包含 `name`、`aliases`、`issn/eissn`、`publisher_family`、`priority` 和 `enabled`。结果按期刊优先级和发表日期排序。

OpenAlex 使用 API Key。未配置时会跳过 OpenAlex，PubMed、Europe PMC 和 Crossref 仍会继续检索。

```powershell
$env:OPENALEX_API_KEY="your-openalex-api-key"
```

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 命令行

默认只检索并写出 JSON：

```powershell
python -m weixin_lite.daily_search --journals config/journals.json --output data/latest_papers.json --since-days 7 --openalex-api-key $env:OPENALEX_API_KEY
```

需要同时翻译标题时显式加入 `--translate`：

```powershell
python -m weixin_lite.daily_search --translate --translation-cache data/translation_cache.json --journals config/journals.json --output data/latest_papers.json --provider deepseek --since-days 7 --openalex-api-key $env:OPENALEX_API_KEY
```

旧关键词模式仍可兼容调用：

```powershell
python -m weixin_lite.daily_search --mode keyword --config config/topics.json --output data/latest_papers.json --provider deepseek --since-years 1 --search-mode strict --openalex-api-key $env:OPENALEX_API_KEY
```

也可以对已有结果单独翻译：

```powershell
python -m weixin_lite.translate --input data/latest_papers.json --provider deepseek --batch-size 8 --delay-seconds 1 --translation-cache data/translation_cache.json
python -m weixin_lite.batch_analyze --input data/latest_papers.json --limit 20
```

`batch_analyze` 会复用网页的开放生成规则：没有解析 PDF 全文的记录也会基于题录、摘要和链接生成摘要级解读；PDF、图注和截图只作为增强材料。待补全文的记录仍会进入 DOI CSV 和下载状态文件，方便后续补证据。

## 微信公众号草稿

“导出与发布”页支持微信公众号草稿箱：

- 填写 `APP_ID`、`APP_SECRET`、作者、封面图、原文链接等信息。
- 默认勾选“只预览 payload”，不会写入公众号后台。
- 取消 dry-run 并勾选确认后，工具会获取 access token、上传封面素材、创建草稿。
- 失败时会显示微信接口返回的 `errcode/errmsg`。

## 测试

```powershell
python -m pytest -q
```
