import copy
import io
import json
import zipfile
from uuid import uuid4

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from app import paper_key as record_key

from weixin_lite.article_analysis import ANALYSIS_PROMPT_VERSION
from weixin_lite.evidence_context import figure_context
from weixin_lite.figure_analysis import analyze_selected_figures
from weixin_lite.generator import generate_article
from weixin_lite.editor import article_blocks, replace_block, replace_figure_note, rewrite_block
from weixin_lite.exporter import export_article_docx_bytes, export_article_html
from weixin_lite.llm import LLMError
from weixin_lite.models import AnalysisClaim, FigureAnalysis, PaperAnalysis, PaperInput, QuickReadArticle
from weixin_lite.pdf_reader import PdfContent
from weixin_lite.workspace_store import (snapshot_bytes, restore_bytes, save_workspace, load_workspace,
                                         preserve_article, project_path)


@pytest.fixture
def workspace():
    image = io.BytesIO()
    Image.new("RGB", (400, 220), "#226677").save(image, "PNG")
    figures = [FigureAnalysis(f"Fig. {i}", f"Fig. {i} compares treatment with control.", page=str(i),
                              image_name=f"fig{i}.png", selected=i < 3, order=i) for i in range(1, 4)]
    pdf = PdfContent(text="[Page 1]\nBackground and the study question.\n\n[Page 2]\nThe method is shown here.\n\n"
                         "[Page 3]\nFig. 1 shows treatment achieved 90% conversion compared with 60% for control.\n\n"
                         "[Page 4]\nFig. 2 shows treatment reduced costs by 20%.\n\n"
                         "[Page 5]\nFig. 3 UNSELECTED_PAGE_CONTEXT.",
                     page_count=5, hash="fixture", legends=figures)
    analysis = PaperAnalysis(status="complete", source_hash=pdf.hash, version=ANALYSIS_PROMPT_VERSION,
                            **{key: [AnalysisClaim("研究主线", page="1", evidence_text="Background and the study question.")]
                               for key in ("research_question", "background", "methods", "innovation", "limitations", "conclusion")})
    paper = PaperInput(title_en="Fixture paper", pdf_name="study.pdf", access_status="open")
    return dict(project_id=uuid4().hex, project_name="测试项目", papers=[paper], pdfs={"study.pdf": pdf},
                images={f.image_name: image.getvalue() for f in figures}, downloads=[], articles=[],
                paper_analyses={record_key(paper): analysis}, analysis_cache={}, vision_cache={}, article_cache={},
                article_versions=[], editor_drafts={})


def response_for(prompt):
    first = "只分析 Fig. 1。" in prompt
    return json.dumps({"heading": "转化效率" if first else "成本改善",
                       "paragraphs": ["这张图比较处理与对照。", "转化率达到 90%。" if first else "成本降低 20%。", "结论限定在论文条件下。"],
                       "source_excerpt": "treatment achieved 90% conversion compared with 60% for control." if first else "treatment reduced costs by 20%.",
                       "source_page": "3" if first else "4"})


def config_for(state):
    return dict(api_key="fake", model="text-model", base_url="https://example.test/v1",
                image_assets=state["images"], cache=state["vision_cache"], checkpoint=lambda: save_workspace(state))


def run_figures(state, config=None, **kwargs):
    paper = state["papers"][0]
    return analyze_selected_figures(paper, state["paper_analyses"][record_key(paper)], state["pdfs"]["study.pdf"].legends,
                                    config or config_for(state), pdf=state["pdfs"]["study.pdf"], **kwargs)


def test_cross_page_evidence_and_resuming_after_restart(workspace, monkeypatch):
    calls = []
    def call(**kwargs):
        calls.append(kwargs["user_prompt"])
        if len(calls) == 2:
            raise LLMError("quota", status_code=429, quota_exhausted=True)
        return response_for(kwargs["user_prompt"])
    monkeypatch.setattr("weixin_lite.figure_analysis.call_openai_compatible", call)
    monkeypatch.setattr("weixin_lite.figure_analysis.call_openai_compatible_with_images", lambda **_: pytest.fail("text mode must not use vision"))
    assert len(run_figures(workspace)) == 1
    assert "[Page 3]" in calls[0] and "90% conversion" in calls[0]
    assert "UNSELECTED_PAGE_CONTEXT" not in "".join(calls)
    restored = load_workspace(workspace["project_id"])
    restored["project_id"] = workspace["project_id"]
    assert len(run_figures(restored)) == 2
    assert len(calls) == 3
    assert "只分析 Fig. 2。" in calls[-1]


@pytest.mark.parametrize("change", ["image", "crop", "context", "model"])
def test_figure_cache_changes_when_its_inputs_change(workspace, monkeypatch, change):
    calls = []
    monkeypatch.setattr("weixin_lite.figure_analysis.call_openai_compatible", lambda **kw: calls.append(kw) or response_for(kw["user_prompt"]))
    config = config_for(workspace)
    assert len(run_figures(workspace, config)) == 2
    assert len(run_figures(workspace, config)) == 2
    assert len(calls) == 2
    if change == "image":
        workspace["images"]["fig1.png"] += b"changed"
    elif change == "crop":
        workspace["pdfs"]["study.pdf"].legends[0].crop_bbox = (0.1, 0.1, 0.9, 0.9)
    elif change == "context":
        workspace["pdfs"]["study.pdf"].text += "\n\nFig. 1 additional evidence."
    else:
        config["model"] = "different-model"
    run_figures(workspace, config)
    assert len(calls) > 2


@pytest.mark.parametrize("bad", ["excerpt", "page", "number"])
def test_invalid_model_evidence_preserves_previous_note(workspace, monkeypatch, bad):
    def call(**kwargs):
        data = json.loads(response_for(kwargs["user_prompt"]))
        if bad == "excerpt":
            data["source_excerpt"] = "An invented statement not present in this paper"
        elif bad == "page":
            data["source_page"] = "5"
        else:
            data["paragraphs"][1] = "转化率达到 999%。"
        return json.dumps(data)
    figure = workspace["pdfs"]["study.pdf"].legends[0]
    figure.interpretation = "旧稿需保留"
    monkeypatch.setattr("weixin_lite.figure_analysis.call_openai_compatible", call)
    assert not run_figures(workspace)
    assert figure.interpretation == "旧稿需保留"
    assert figure.vision_error


def test_snapshot_keeps_assets_versions_and_no_credentials(workspace):
    workspace["text_config"] = {"api_key": "SECRET_API_KEY"}
    workspace["editor_drafts"] = {"section-a": "正在编辑的草稿"}
    data, _ = snapshot_bytes(workspace)
    restored = restore_bytes(data)
    assert restored["pdfs"]["study.pdf"].text == workspace["pdfs"]["study.pdf"].text
    assert restored["images"] == workspace["images"]
    assert restored["pdfs"]["study.pdf"].legends[1].selected
    assert restored["editor_drafts"]["section-a"] == "正在编辑的草稿"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert b"SECRET_API_KEY" not in archive.read("workspace.json")
    with pytest.raises(ValueError):
        project_path("../../outside")
    with pytest.raises(ValueError):
        restore_bytes(b"not a zip")


def make_article(state, monkeypatch):
    monkeypatch.setattr("weixin_lite.figure_analysis.call_openai_compatible", lambda **kw: response_for(kw["user_prompt"]))
    figures = run_figures(state)
    payload = {key: "有依据的中文段落。" for key in ("intro", "research_question", "innovation", "limitations", "take_home")}
    payload.update(title="测试研究解读", digest="摘要", core_points=["论文核心问题"], method_overview=["简要方法"])
    monkeypatch.setattr("weixin_lite.generator.call_openai_compatible", lambda **kw: json.dumps(payload))
    paper = state["papers"][0]
    article = generate_article(paper, state["pdfs"]["study.pdf"], api_key="fake", analysis=state["paper_analyses"][record_key(paper)],
                               confirmed_figures=figures, allow_text_evidence_figures=True)
    preserve_article(state, article)
    return article


def test_selected_figures_prose_edit_history_and_exports(workspace, monkeypatch):
    article = make_article(workspace, monkeypatch)
    assert "图展示什么：" not in article.body_markdown
    assert "Fig. 3" not in article.body_markdown
    assert "未进行图像复核" in article.body_markdown
    edited = replace_block(article, 1, "## 文章核心要点简述\n\n修改后的要点。")
    preserve_article(workspace, edited)
    assert workspace["article_versions"][0]["body_markdown"] == article.body_markdown
    assert "转化率达到 90%" in edited.body_markdown
    fig = copy.deepcopy(edited.figures[0])
    fig.interpretation = "人工修改后的图解。"
    updated = replace_figure_note(edited, 0, fig)
    assert "人工修改后的图解。" in updated.body_markdown
    assert "成本降低 20%" in updated.body_markdown
    html = export_article_html(updated, workspace["images"]).decode()
    assert "data:image/png;base64" in html
    assert html.index("data:image") < html.index("人工修改后的图解")
    docx = export_article_docx_bytes(updated, workspace["images"])
    with zipfile.ZipFile(io.BytesIO(docx)) as archive:
        assert any(name.startswith("word/media/") for name in archive.namelist())
        assert "人工修改后的图解" in archive.read("word/document.xml").decode()


def test_app_restores_project_after_new_session(workspace):
    save_workspace(workspace)
    at = AppTest.from_file("app.py", default_timeout=20)
    at.query_params["project"] = workspace["project_id"]
    at.session_state["workspace-page"] = "论文分析"
    at.run()
    assert not at.exception
    assert len(at.session_state["papers"]) == 1
    assert len([checkbox for checkbox in at.checkbox if checkbox.value]) == 2
    assert next(button for button in at.button if button.label == "生成稿件").disabled
    assert not any(button.label in {"复核选中图表", "跳过复核，直接生成"} for button in at.button)


def test_section_rewrite_preserves_other_sections_and_images(workspace, monkeypatch):
    article = make_article(workspace, monkeypatch)
    part = article_blocks(article.body_markdown)[1][1]
    monkeypatch.setattr("weixin_lite.editor.call_openai_compatible", lambda **kw: json.dumps({"text": part.replace("论文核心问题", "重新组织的研究主线")}))
    updated = rewrite_block(article, 1, "精简重复", next(iter(workspace["paper_analyses"].values())), workspace["pdfs"]["study.pdf"], config_for(workspace))
    assert "重新组织的研究主线" in updated.body_markdown
    assert "论文核心问题" in article.body_markdown
    assert article_blocks(updated.body_markdown)[2:] == article_blocks(article.body_markdown)[2:]
    monkeypatch.setattr("weixin_lite.editor.call_openai_compatible", lambda **kw: json.dumps({"text": "## Changed heading\nUnexpected rewrite"}))
    with pytest.raises(ValueError, match="结构"):
        rewrite_block(article, 1, "重写", next(iter(workspace["paper_analyses"].values())), workspace["pdfs"]["study.pdf"], config_for(workspace))


def test_editing_repeated_figure_note_only_changes_the_chosen_figure(workspace, monkeypatch):
    article = make_article(workspace, monkeypatch)
    note = article.figures[0].interpretation
    second = copy.deepcopy(article.figures[1])
    second.interpretation = note
    duplicate = replace_figure_note(article, 1, second)
    assert duplicate.body_markdown.count(note) == 2
    first = copy.deepcopy(duplicate.figures[0])
    first.interpretation = "只修改第一张图。"
    changed = replace_figure_note(duplicate, 0, first)
    assert changed.body_markdown.count(note) == 1
    assert "只修改第一张图。" in changed.body_markdown


def test_switching_local_projects_does_not_carry_over_state(workspace):
    save_workspace(workspace)
    at = AppTest.from_file("app.py", default_timeout=20)
    at.query_params["project"] = workspace["project_id"]
    at.run()
    next(button for button in at.button if button.label == "新建项目").click().run()
    assert not at.exception
    assert not at.session_state["papers"]
    assert at.session_state["project_id"] != workspace["project_id"]
    next(button for button in at.button if button.label == "打开项目").click().run()
    assert not at.exception
    assert len(at.session_state["papers"]) == 1


def test_app_generate_and_edit_with_mock_models(workspace, monkeypatch):
    make_article(workspace, monkeypatch)
    save_workspace(workspace)
    at = AppTest.from_file("app.py", default_timeout=20)
    at.query_params["project"] = workspace["project_id"]
    at.session_state["workspace-page"] = "成稿导出"
    at.run()
    assert not at.exception
    assert any(item.label == "编辑位置" for item in at.selectbox)
    editor = next(item for item in at.text_area if item.label == "正文")
    editor.set_value(editor.value + "\n\n新的中文段落。").run()
    next(button for button in at.button if button.label == "保存本节").click().run()
    assert not at.exception
    assert "新的中文段落" in at.session_state["articles"][0].body_markdown
    assert at.session_state["article_versions"]
    assert "新的中文段落" in load_workspace(workspace["project_id"])["articles"][0].body_markdown


def test_unified_generation_button_calls_text_model_and_preserves_previous_version(workspace, monkeypatch):
    original = make_article(workspace, monkeypatch)
    workspace["pdfs"]["study.pdf"].legends[0].caption += " Updated caption."
    save_workspace(workspace)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    at = AppTest.from_file("app.py", default_timeout=20)
    at.query_params["project"] = workspace["project_id"]
    at.session_state["workspace-page"] = "论文分析"
    at.run()
    assert not at.exception
    button = next(button for button in at.button if button.label == "生成稿件")
    assert not button.disabled
    button.click().run()
    assert not at.exception
    assert any("稿件已生成" in notice.value for notice in at.success)
    assert at.session_state["article_versions"]
    assert at.session_state["articles"][0].figures[0].review_mode == "text"


def test_failed_vision_can_switch_to_text_without_losing_previous_article(workspace, monkeypatch):
    original = make_article(workspace, monkeypatch)
    save_workspace(workspace)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    def fail(**kwargs):
        raise LLMError("quota", status_code=429, quota_exhausted=True)
    monkeypatch.setattr("weixin_lite.figure_analysis.call_openai_compatible_with_images", fail)
    at = AppTest.from_file("app.py", default_timeout=20)
    at.query_params["project"] = workspace["project_id"]
    at.session_state["workspace-page"] = "论文分析"
    at.run()
    next(radio for radio in at.radio if radio.label == "图解方式").set_value("文字 + 视觉").run()
    next(button for button in at.button if button.label == "生成稿件").click().run()
    assert not at.exception
    assert at.error
    assert at.session_state["articles"][0].body_markdown == original.body_markdown
    next(radio for radio in at.radio if radio.label == "图解方式").set_value("文字证据").run()
    next(button for button in at.button if button.label == "生成稿件").click().run()
    assert not at.exception
    assert any("稿件已生成" in item.value for item in at.success)


def test_word_table_follows_interpretation_without_internal_marker(workspace, monkeypatch):
    from docx import Document
    from weixin_lite.generator import render_article_with_confirmed_figures, markdown_to_wechat_html
    from weixin_lite.models import EditableTable
    article = make_article(workspace, monkeypatch)
    article.figures[0].editable_table = EditableTable(headers=["Group", "Rate"], rows=[["Treatment", "90%"]], confidence=0.9)
    article.body_markdown = render_article_with_confirmed_figures(article.paper, article.sections, article.figures)
    article.body_html = markdown_to_wechat_html(article.body_markdown, article.figures)
    document = Document(io.BytesIO(export_article_docx_bytes(article, workspace["images"])))
    assert len(document.tables) == 1
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "editable-table" not in text
    children = list(document.element.body)
    table_index = next(i for i, child in enumerate(children) if child.tag.endswith("}tbl"))
    note_index = next(i for i, child in enumerate(children) if "转化率达到" in "".join(child.itertext()))
    assert table_index > note_index


def test_context_does_not_confuse_main_and_supplementary_figures():
    pdf = PdfContent(text="[Page 1]\nFig. 1 main result.\n\n[Page 2]\nSupplementary Fig. 1 unrelated supplement.\n\n[Page 3]\nFig. 10 unrelated figure.")
    source = figure_context(pdf, FigureAnalysis("Fig. 1", "", page="1"))
    assert "main result" in source
    assert "unrelated supplement" not in source
    assert "unrelated figure" not in source
