# 微信文献快读工具

这是一个 Streamlit 应用，用于按期刊清单追踪 Nature、Cell、Science 等生命科学顶刊的最新文章，解析开放全文或用户上传 PDF，并生成可导出的中文微信公众号快读稿。

## 核心流程

1. **文献检索**：选择明确的自然月和期刊，从 PubMed、Europe PMC、OpenAlex、Crossref 检索；先按推荐或自定义关键词筛选，再主动翻译筛选结果的标题。不会在检索后自动调用翻译模型。
2. **论文分析**：下载合法开放全文或上传 PDF，分段分析完整正文，确认关键图表，再进行 Gemini 视觉复核和公众号成稿。未完成复核的图表不能进入最终稿。
3. **成稿导出**：编辑、保存、预览，再准备下载可编辑 Word；便携 HTML（内嵌图片）、Markdown、项目 ZIP 和公众号草稿保留在次级入口。

公众号正文使用论文首页截图开篇、关键图在上和中文分析在下的图文顺序。用户可以确认、裁剪或替换推荐图表。分析和图解不设产品层字数上限；模型仍有技术上下文与输出限制。预览与便携 HTML 使用同一渲染器，微信发布时再将正文图片替换为微信素材地址。

检索时间使用可自由选择的开始日期和结束日期，精确到年月日，并按所选边界查询和复核结果。默认范围是本月第一天至今天。检索结果显示正式发表日期；Crossref 只读取 `published`、`published-online`、`published-print`、`issued`、`posted` 等发表字段，不把 `created`、`indexed`、`deposited` 这类入库日期当作发表日期。未下载到全文、下载失败、只拿到题录或只拿到 HTML 的论文不会生成公众号内容，会进入 `unavailable_dois.csv`，字段包括 DOI、题名、期刊、发表日期、年份、链接、全文状态和错误原因。导出包里也保留兼容文件名 `paywalled_dois.csv`。

## 翻译/生成模型

支持 OpenAI-compatible 接口，并内置供应商预设：

- `openai`
- `gemini`
- `deepseek`
- `siliconflow`
- `custom`

可用环境变量：

```powershell
$env:LLM_PROVIDER="deepseek"
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_BASE_URL="https://api.deepseek.com/v1"
$env:OPENAI_MODEL="deepseek-flash"
```

也兼容：

```powershell
$env:LLM_API_KEY="your-key"
$env:LLM_BASE_URL="https://api.example.com/v1"
$env:LLM_MODEL="your-model"
```

侧边栏的高级设置提供连接测试、翻译批量和批间间隔。正文模型不是 Gemini 时，可单独配置 Gemini 图表复核密钥和模型。

遇到 `429 Too Many Requests` 时，程序区分短期限流与日配额/余额耗尽。短期限流按服务端等待提示重试，最多三次请求；要求等待超过 60 秒则交还页面控制权。日配额、余额或零额度不会无效重试，也不会继续拆成逐标题请求。降低请求频率只能缓解短期限流，不能恢复已经用完的配额；请在供应商控制台查看具体限制。

全文分析按块保存成功结果，同一会话内可继续失败阶段；图表复核按图片、裁剪和模型缓存。刷新页面或重启服务可能清空这些会话缓存，并非持久化任务队列。模型失败时保留原文、已完成分析和图片，不把失败包装成完成稿。

翻译只处理标题。摘要详情保留英文原文，不会发送给模型，也不会写入 `abstract_zh`。

DeepSeek 默认使用官方当前模型名 `deepseek-flash`（DeepSeek V4.1 Flash）。模型字段仍可手动改为 `deepseek-v4-pro`；旧的 `deepseek-chat` 和 `deepseek-reasoner` 不再作为默认值。

## 每日顶刊检索

页面默认进入“文献检索”。可以自由选择开始日期、结束日期、结果数量和数据源，并选择本次要检索的期刊。期刊表按已有 2024 JIF 从高到低排列，首次进入页面默认全部未选；日期无效、未选择期刊或未选择数据源时不能启动检索。检索不依赖固定关键词，筛选只作用于已经返回的候选结果。

期刊清单位于 `config/journals.json`，每项包含 `name`、`aliases`、`issn/eissn`、`publisher_family`、`impact_factor`、`impact_factor_year`、`priority` 和 `enabled`。结果按影响因子和发表日期排序。

OpenAlex 已改用 API Key。未配置时会跳过 OpenAlex，PubMed、Europe PMC 和 Crossref 仍会继续检索。

```powershell
$env:OPENALEX_API_KEY="your-openalex-api-key"
```

## 本地运行

需要 Streamlit 1.58 或更高的 1.x 版本（内嵌文章预览使用 `st.iframe`）。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 命令行

```powershell
python -m weixin_lite.daily_search --journals config/journals.json --output data/latest_papers.json --provider deepseek --since-days 7 --openalex-api-key $env:OPENALEX_API_KEY
python -m weixin_lite.daily_search --mode keyword --config config/topics.json --output data/latest_papers.json --provider deepseek --since-years 1 --search-mode strict --openalex-api-key $env:OPENALEX_API_KEY
python -m weixin_lite.translate_results --input data/latest_papers.json --provider deepseek --batch-size 1 --delay-seconds 5
python -m weixin_lite.batch_analyze --input data/latest_papers.json --limit 20
```

`batch_analyze` 会复用网页的生成准入规则：没有解析 PDF 全文的记录会被跳过，只进入 DOI CSV 和下载状态文件。

## 公众号草稿

“成稿导出”页的“微信公众号草稿”折叠区支持草稿箱：

- 填写 `APP_ID`、`APP_SECRET`、作者、封面图、原文链接等信息。
- 默认勾选“仅预览草稿，不发布”，不会写入公众号后台。
- 取消预览选项、补齐凭据及封面后，点击“创建公众号草稿”才会上传素材和创建草稿。
- 失败时会显示微信接口返回的 `errcode/errmsg`。

公众号 HTML 使用简洁的微信内联样式，面向 `duyi-wechat-skill-suite` 类工作流产出可复用的素材和草稿 payload，但不复制外部 skill 源码。

## 测试

```powershell
python -m pytest -q
```
