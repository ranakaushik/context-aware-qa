"""Loads Markdown knowledge-base files and splits them into section-level chunks."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str  # "<file stem>#<section slug>", e.g. "leave_policy#sick-leave"
    source: str  # file name, e.g. "leave_policy.md"
    title: str  # document title (the "# " heading)
    section: str  # section heading (the "## " heading), "Overview" for the intro
    text: str


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def chunk_markdown(path: Path) -> list[Chunk]:
    """One chunk per "## " section; text before the first section becomes "Overview"."""
    title = path.stem.replace("_", " ").title()
    sections: list[tuple[str, list[str]]] = [("Overview", [])]

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)

    chunks = []
    for section, lines in sections:
        text = "\n".join(lines).strip()
        if text:
            chunks.append(
                Chunk(
                    id=f"{path.stem}#{slugify(section)}",
                    source=path.name,
                    title=title,
                    section=section,
                    text=text,
                )
            )
    return chunks


def load_knowledge_base(kb_dir: Path) -> list[Chunk]:
    files = sorted(list(kb_dir.glob("*.md")) + list(kb_dir.glob("*.txt")))
    if not files:
        raise FileNotFoundError(f"No .md or .txt files found in knowledge base folder: {kb_dir}")
    return [chunk for path in files for chunk in chunk_markdown(path)]
