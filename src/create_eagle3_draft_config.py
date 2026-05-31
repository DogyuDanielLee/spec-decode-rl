#!/usr/bin/env python3
"""Create an EAGLE3 draft config outside the vendored SpecForge tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specforge-dir", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-download-dir", default=None)
    args = parser.parse_args()

    specforge_dir = Path(args.specforge_dir).resolve()
    sys.path.insert(0, str(specforge_dir))

    from specforge.utils import generate_draft_model_config, save_draft_model_config

    template = specforge_dir / "configs" / "llama3-8B-eagle3.json"
    config = generate_draft_model_config(
        target_model_path=args.target_model,
        template_config_path=str(template),
        cache_dir=args.model_download_dir,
    )
    save_draft_model_config(config, args.output)


if __name__ == "__main__":
    main()
