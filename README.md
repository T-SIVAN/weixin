# 微信公众号文献快读生成器

这是一个独立的 Streamlit 应用，用于批量生成微信公众号单篇文献快读。当前版本面向 TdT、PUP、酶促 DNA/RNA 合成、相关酶机制与酶工程方向。

核心目标：

- 一次导入或检索 10-20 篇论文。
- 每篇生成 500-1500 字中文快读，默认约 1000-1300 字。
- 结构贴近参考公众号模式：标题、期刊标题图、文章核心要点简述、关键数据/关键图例分析、文章的创新意义、原文信息。
- 上传 PDF 后才进行关键图例和关键数据分析；没有全文时只生成摘要级快读，并明确提示证据边界。
- 导出可发布的 Markdown、HTML、证据 JSON 和 `.weixin-project.zip` 项目包。
- 可选对接自建固定 IP 桥接服务创建微信公众号草稿；不自动发布。

## 本地运行

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 部署

将本仓库部署到 Streamlit Community Cloud，入口文件选择 `app.py`。

LLM API Key 在页面侧边栏输入，只保存在当前浏览器会话，不写入项目包。不要把 Key 写进仓库。

## 使用流程

1. 在“批量导入论文”里输入主题检索式，或粘贴 DOI/PMID/标题。
2. 上传论文 PDF。只有 PDF 或合法开放全文可用时，系统才会抽取图注、关键数字和证据位置。
3. 在“生成公众号稿”里选择 10-20 篇文章批量生成。
4. 为每篇文章上传期刊网页标题图或截图。
5. 在“导出/公众号”里下载 `.weixin-project.zip`，其中包含 HTML、Markdown、图片和证据文件。

## 微信公众号对接说明

微信草稿接口需要稳定后端保存 `AppSecret` 并满足 IP 白名单。Streamlit Cloud 通常没有固定出口 IP，因此本应用默认只提供两种方式：

- 下载 HTML/Markdown 后手动粘贴到公众号后台。
- 配置你自己的固定 IP 桥接服务，调用 `/v1/drafts` 创建草稿。

草稿桥接服务建议只做“上传图片、创建草稿”，发布仍由人工在公众号后台确认。

## 证据边界

应用会把关键数字、Figure/Table 编号、页码和图注一起导出到 `evidence/*.json`。如果 PDF 文本层无法读取图注或坐标轴，文章会出现“需人工核对”的警告。不要发布无法追溯的数据。
