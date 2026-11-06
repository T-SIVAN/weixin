"""Create an explicitly synthetic local project for offline UI/export acceptance.

Run from the repository root: python -m scripts.verify_local_workbench
No credentials or real model calls are used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import fitz

from weixin_lite.article_analysis import ANALYSIS_PROMPT_VERSION
from weixin_lite.figure_analysis import analyze_selected_figures
from weixin_lite.generator import generate_article
from weixin_lite.models import AnalysisClaim, DownloadedPaper, PaperAnalysis, PaperInput
from weixin_lite.pdf_reader import parse_pdf
from weixin_lite.workspace_store import save_workspace
from weixin_lite.exporter import export_article_docx_bytes, export_article_html


def build_demo():
    project_id = "c81214917fc243f1a3a478459680cb9a"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((45, 55), "SYNTHETIC WORKBENCH TEST - NOT RESEARCH", fontsize=17)
        page.insert_text((45, 105), "Overview of a controlled comparison\nAbstract\nA synthetic example for local software acceptance.\n"
                         "Methods\nCompare two predefined groups under identical conditions.", fontsize=12)
        for i in range(1, 4):
            page = document.new_page()
            page.insert_text((45, 50), f"Synthetic chart {i} - conversion (%)", fontsize=15)
            page.draw_line((90, 290), (440, 290), width=2)
            page.draw_line((90, 290), (90, 80), width=2)
            for height, x, color, label in [(120, 140, (0.2, 0.4, 0.7), "Control: 60%"),
                                           (180, 290, (0.1, 0.6, 0.5), "Treatment: 90%")]:
                page.draw_rect(fitz.Rect(x, 290 - height, x + 75, 290), color=color, fill=color)
                page.insert_text((x - 10, 310), label, fontsize=11)
            page.insert_text((45, 355), f"Fig. {i}. Treatment reached 90% conversion compared with 60% for control.", fontsize=11)
            page.insert_text((45, 410), f"Results\nFig. {i} shows a 30 percentage point difference between groups.\n"
                             "Discussion\nThe evidence supports only the synthetic test conditions.", fontsize=11)
        raw = document.tobytes()
    pdf = parse_pdf(raw, mode="pypdf")
    paper = PaperInput(title_en="Synthetic workbench acceptance", title_zh="本地验收样例（非真实研究）",
                       pdf_name="synthetic-workbench.pdf", access_status="open")
    for index, figure in enumerate(pdf.legends):
        figure.selected = index < 2
        figure.order = index + 1
    analysis = PaperAnalysis(status="complete", source_hash=pdf.hash, version=ANALYSIS_PROMPT_VERSION,
                            **{name: [AnalysisClaim("这是用于验证软件流程的模拟比较，仅用于本地测试。", page="1",
                                                   evidence_text="A synthetic example for local software acceptance.")]
                               for name in ("research_question", "background", "methods", "innovation", "limitations", "conclusion")})
    config = dict(api_key="mock-only", model="mock-model", base_url="https://example.test/v1",
                  image_assets=pdf.rendered_images, cache={})
    response = {"heading": "处理组与对照组的转化率比较",
                "paragraphs": ["这张模拟图用于检查软件能否将原图与对应说明正确排列。蓝色与绿色分别表示两个预设组别。",
                               "处理组的转化率为 90%，对照组为 60%。两者相差 30 个百分点。这里的数字仅用于软件验收，不代表真实科研结果。",
                               "这项比较只能说明样例数据的关系，不能用于推断真实实验效果。"],
                "source_excerpt": "Treatment reached 90% conversion compared with 60% for control."}
    with patch("weixin_lite.figure_analysis.call_openai_compatible", return_value=json.dumps(response)):
        figures = analyze_selected_figures(paper, analysis, pdf.legends, config, pdf=pdf)
    assert len(figures) == 2, [f.vision_error for f in pdf.legends]
    payload = {"title": "本地验收：图文解读样例", "digest": "模拟数据，仅用于软件验收。",
               "intro": "这是本地验收样例，用于检查选图、图解、项目恢复以及 Word 导出。所有图表均为模拟数据。",
               "core_points": ["图解与所选图片一一对应。", "草稿修改可以独立保存并恢复。"],
               "research_question": "工作台需要将论文证据组织成可修改、可核对的中文稿件。该样例用于验证整个操作流程。",
               "method_overview": ["选择两张模拟图，按图号查找正文，并使用模拟模型响应生成图解。"],
               "innovation": ["将选图、分析和编辑集中到同一个本地项目。"],
               "limitations": ["本样例使用模拟数据和模拟模型响应，不评价真实论文的分析质量。"],
               "take_home": "图片和说明保持一致，修改后的稿件可导出和恢复。"}
    with patch("weixin_lite.generator.call_openai_compatible", return_value=json.dumps(payload)):
        article = generate_article(paper, pdf, api_key="mock-only", analysis=analysis,
                                   confirmed_figures=figures, allow_text_evidence_figures=True)
    key = hashlib.sha256(paper.paper_key.strip().lower().encode()).hexdigest()[:24]
    state = dict(project_id=project_id, project_name="本地验收样例（模拟数据）", papers=[paper], pdfs={paper.pdf_name: pdf},
                 images=pdf.rendered_images, articles=[article], paper_analyses={key: analysis},
                 downloads=[DownloadedPaper(paper_key=paper.paper_key, file_name=paper.pdf_name,
                                            status="open", content_type="application/pdf", content_bytes=raw)],
                 vision_cache=config["cache"])
    save_workspace(state)
    output = Path(".weixin/acceptance")
    output.mkdir(parents=True, exist_ok=True)
    (output / "sample.pdf").write_bytes(raw)
    (output / "article.docx").write_bytes(export_article_docx_bytes(article, pdf.rendered_images))
    (output / "article.html").write_bytes(export_article_html(article, pdf.rendered_images))
    print(json.dumps({"project": project_id, "figures": len(figures), "pages": pdf.page_count,
                      "url": f"http://127.0.0.1:8517/?project={project_id}"}))


if __name__ == "__main__":
    build_demo()
