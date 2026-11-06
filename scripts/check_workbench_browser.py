"""Optional Playwright acceptance: run after verify_local_workbench and Streamlit."""
from pathlib import Path
import json

from playwright.sync_api import sync_playwright


def check_browser():
    output = Path(".weixin/acceptance")
    output.mkdir(parents=True, exist_ok=True)
    url = "http://127.0.0.1:8517/?project=c81214917fc243f1a3a478459680cb9a"
    results = []
    with sync_playwright() as runner:
        browser = runner.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.get_by_role("heading", name="微信文献快读工具").wait_for()
        page.get_by_text("论文分析", exact=True).first.click()
        page.get_by_text("3. 生成稿件", exact=True).wait_for()
        page.screenshot(path=str(output / "desktop-analysis.png"), full_page=True)
        page.get_by_role("button", name="生成稿件", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(output / "desktop-generation.png"))
        assert page.get_by_role("button", name="生成稿件", exact=True).count() == 1
        assert page.get_by_role("button", name="复核选中图表", exact=True).count() == 0
        assert page.locator('[data-testid="stImage"] img').count() >= 3
        page.get_by_text("成稿导出", exact=True).first.click()
        page.get_by_text("当前修改预览", exact=True).wait_for()
        assert "![论文首页]" not in page.get_by_role("textbox", name="正文", exact=True).input_value()
        page.screenshot(path=str(output / "desktop-editor.png"), full_page=True)
        page.get_by_role("button", name="准备下载文件", exact=True).click()
        page.get_by_role("button", name="下载 Word", exact=True).wait_for()
        page.screenshot(path=str(output / "desktop-export.png"), full_page=True)
        results.append({"viewport": "desktop", "overflow": page.evaluate("document.documentElement.scrollWidth > innerWidth"),
                        "word_download": True, "errors": list(errors)})
        sidebar_close = page.locator('[data-testid="stSidebarCollapseButton"] button')
        page.locator('[data-testid="stSidebar"]').hover()
        sidebar_close.click()
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_function("""() => {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            return !sidebar || sidebar.getBoundingClientRect().right <= 1 || getComputedStyle(sidebar).display === 'none';
        }""")
        page.get_by_text("编辑位置", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(output / "mobile-controls.png"), animations="disabled")
        page.get_by_text("当前修改预览", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(output / "mobile-editor.png"), full_page=True, animations="disabled")
        for frame in page.frames[1:]:
            assert not frame.evaluate("document.documentElement.scrollWidth > innerWidth"), "Preview overflows on mobile"
            for image in frame.locator("img").all():
                assert image.evaluate("img => img.complete && img.naturalWidth > 0"), "Missing preview image"
        results.append({"viewport": "mobile", "overflow": page.evaluate("document.documentElement.scrollWidth > innerWidth"),
                        "errors": list(errors)})
        page.reload()
        page.get_by_role("heading", name="微信文献快读工具").wait_for()
        assert "候选文献 1" in page.locator("body").inner_text()
        assert "稿件 1" in page.locator("body").inner_text()
        assert not page.locator('[data-testid="stException"]').count()
        browser.close()
    (output / "browser-report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))
    assert not any(item["overflow"] or item["errors"] for item in results)


if __name__ == "__main__":
    check_browser()
