# 微信文献快读工具

这是一个 Streamlit 应用，用于检索合成生物、代谢工程、生物制造、生物合成、工程菌和发酵生产方向文献，解析开放全文或用户上传 PDF，并生成可导出的中文微信公众号快读稿。

## 核心流程

1. **检索与翻译**：输入可编辑中文或英文关键词和年份范围，先解析成实际英文检索词，再检索 PubMed、Europe PMC、OpenAlex、Crossref，并只把英文标题翻译为中文。检索会自动叠加合成生物行业语境过滤，减少不相关结果。
2. **全文与生成**：自动下载合法开放全文或上传 PDF。只有已解析到 PDF 全文的论文才会生成公众号正文。
3. **导出与发布**：导出项目包、单篇 Markdown/HTML、未生成 DOI CSV，也可以 dry-run 预览或真实创建微信公众号草稿。

公众号正文采用“原文截图在上、短说明在下”的图文模板：系统会从 PDF 渲染关键图所在页面截图，网页预览中可直接查看并手动替换截图；导出的 Markdown、HTML 和草稿 payload 保持同一版式。

检索结果显示正式发表日期；Crossref 只读取 `published`、`published-online`、`published-print`、`issued`、`posted` 等发表字段，不把 `created`、`indexed`、`deposited` 这类入库日期当作发表日期。未下载到全文、下载失败、只拿到题录或只拿到 HTML 的论文不会生成公众号内容，会进入 `unavailable_dois.csv`，字段包括 DOI、题名、期刊、发表日期、年份、链接、全文状态和错误原因。导出包里也保留兼容文件名 `paywalled_dois.csv`。

## 翻译/生成模型

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

页面侧边栏提供“测试翻译模型”按钮，并可调节翻译批量和批间间隔。遇到 `429 Too Many Requests` 时，把批量调到 1、间隔调到 5-10 秒，或切换 DeepSeek/SiliconFlow/custom 等额度更高的 OpenAI-compatible 服务。模型失败时会显示具体错误，并保留英文标题和“待翻译”标记，不阻塞后续下载和导出。

翻译只处理标题。摘要详情保留英文原文，不会发送给模型，也不会写入 `abstract_zh`。

## 中文检索

页面会把中文关键词解析为可编辑的“实际检索词”。内置合成生物行业词典覆盖底盘细胞、细胞工厂、精准发酵、酶工程、天然产物、聚羟基脂肪酸酯等方向；未收录中文词会尝试用当前模型扩展为 1-3 个英文学术同义词。模型不可用时保留原词并显示提示，不阻塞检索。

OpenAlex 已改用 API Key。未配置时会跳过 OpenAlex，PubMed、Europe PMC 和 Crossref 仍会继续检索。

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

```powershell
python -m weixin_lite.daily_search --config config/topics.json --output data/latest_papers.json --provider deepseek --since-years 1 --search-mode strict --openalex-api-key $env:OPENALEX_API_KEY
python -m weixin_lite.translate_results --input data/latest_papers.json --provider deepseek --batch-size 1 --delay-seconds 5
python -m weixin_lite.batch_analyze --input data/latest_papers.json --limit 20
```

`batch_analyze` 会复用网页的生成准入规则：没有解析 PDF 全文的记录会被跳过，只进入 DOI CSV 和下载状态文件。

## 公众号草稿

“导出与发布”页支持微信公众号草稿箱：

- 填写 `APP_ID`、`APP_SECRET`、作者、封面图、原文链接等信息。
- 默认勾选“只预览 payload”，不会写入公众号后台。
- 取消 dry-run 并勾选确认后，工具会获取 access token、上传封面素材、创建草稿。
- 失败时会显示微信接口返回的 `errcode/errmsg`。

公众号 HTML 使用简洁的微信内联样式，面向 `duyi-wechat-skill-suite` 类工作流产出可复用的素材和草稿 payload，但不复制外部 skill 源码。

## 测试

```powershell
python -m pytest -q
```
