# 微信文献快读工具

这是一个 Streamlit 应用，用于检索 TdT、PUP、酶促 DNA/RNA 合成和酶工程方向文献，解析开放全文或用户上传 PDF，并生成可导出的中文微信公众号快读稿。

## 核心流程

1. **检索与翻译**：输入关键词和年份范围，检索 PubMed、Europe PMC、OpenAlex、Crossref，并把英文题名/摘要翻译为中文。
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

页面侧边栏提供“测试翻译模型”按钮，并可调节翻译批量和批间间隔。遇到 `429 Too Many Requests` 时，把批量调到 1、间隔调到 5-10 秒，或切换 DeepSeek/SiliconFlow/custom 等额度更高的 OpenAI-compatible 服务。模型失败时会显示具体错误，并保留英文原文和“待翻译”标记，不阻塞后续下载和导出。

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 命令行

```powershell
python -m weixin_lite.daily_search --config config/topics.json --output data/latest_papers.json --provider deepseek --since-years 1
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
