from __future__ import annotations

import argparse
import json
from pathlib import Path

from .exporter import project_zip
from .generator import generate_article
from .llm import default_api_key, default_base_url, default_model
from .models import BatchProject, DownloadedPaper, PaperInput, SearchRun


def load_papers(path: Path) -> tuple[str, list[PaperInput]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        run = SearchRun.from_dict(data)
        return ", ".join(run.keywords), run.records
    if isinstance(data, list):
        return "batch", [PaperInput.from_dict(item) for item in data]
    return "batch", []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chinese WeChat quick-read drafts from search metadata.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/batch_analysis.weixin-project.zip")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument("--model", default=default_model())
    args = parser.parse_args()

    topic, papers = load_papers(Path(args.input))
    selected = papers[: max(1, min(args.limit, 20))]
    articles = [
        generate_article(paper, api_key=default_api_key(), base_url=args.base_url, model=args.model)
        for paper in selected
    ]
    project = BatchProject(topic=topic, papers=selected, articles=articles, downloads=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(project_zip(project, downloads=[]))


if __name__ == "__main__":
    main()
