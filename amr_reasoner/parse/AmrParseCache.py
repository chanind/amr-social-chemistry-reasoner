from __future__ import annotations

import json
from typing import Optional
from os.path import exists


class AmrParseCache:
    """
    Helper file to keept track of already parsed AMRs and avoid needing to reparse
    """

    file_path: Optional[str]
    contents: dict[str, str]

    def __init__(self, file_path: Optional[str] = None) -> None:
        self.file_path = file_path
        self.contents = {}
        if file_path is not None and exists(file_path):
            with open(file_path) as f:
                self.contents = json.load(f)

    def save(self) -> None:
        if self.file_path is None:
            raise ValueError("No file path specified")
        with open(self.file_path, "w") as f:
            json.dump(self.contents, f, indent=2, ensure_ascii=False)

    def get(self, sentence: str) -> Optional[str]:
        return self.contents.get(sentence)

    def get_or_throw(self, sentence: str) -> str:
        return self.contents[sentence]

    def set(self, sentence: str, amr: str) -> None:
        self.contents[sentence] = amr
