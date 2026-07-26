from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm import default_api_key, default_base_url, default_model
from .models import SearchRun
from .search import DEFAULT_KEYWORDS, run_keyword_search
from .translate import translate_records


def load_keywords(config_path: Path) -> list[str]:
    if not config_path.exists():
        return DEFAULT_KEYWORDS
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [str(item) for item in data.get("keywords") or DEFAULT_KEYWORDS]
    if isinstance(data, list):
        return [str(item) for item in data]
    return DEFAULT_KEYWORDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily keyword literature radar search.")
    parser.add_argument("--config", default="config/topics.json")
    parser.add_argument("--output", default="data/latest_papers.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--since-days", type=int, default=7)
    parser.add_argument("--email", default="")
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument("--model", default=default_model())
    args = parser.parse_args()

    keywords = load_keywords(Path(args.config))
    run = run_keyword_search(keywords, limit=args.limit, email=args.email, since_days=args.since_days)
    translate_records(run.records, api_key=default_api_key(), base_url=args.base_url, model=args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    history_dir = output.parent / "search_runs"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / f"{run.run_id}.json").write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
