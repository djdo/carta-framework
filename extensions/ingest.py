"""
Document-kind routing for ingestion (thesis 4.4): parse a file into structural blocks and
chunk it into token-bounded pieces with page metadata.

    chunks = chunk_file("Lab10.pdf", kind="worksheet")
    # [{"text": "...", "metadata": {"page_number": 3, "total_pages": 10, "topic": "...", "chunk_index": 0}}, ...]
"""
from typing import Collection, Dict, List, Optional

from .pdf_parser import ParsedPage, parse_document, parse_slides
from .structured_chunker import Block, StructuredChunker, TokenCounter, parse_markdown

# Document kinds and the parser each one uses
KINDS = ("slides", "worksheet", "manual", "code", "text")
CODE_EXTENSIONS = (".s", ".asm", ".a51", ".c", ".h", ".cpp", ".hpp", ".py", ".ino")


def detect_kind(filename: str, hint: Optional[str] = None) -> str:
    """Pick a parser from an explicit hint or from the file path (folder names count)."""
    if hint in KINDS:
        return hint
    lower = filename.lower()
    if lower.endswith(CODE_EXTENSIONS):
        return "code"
    if not lower.endswith(".pdf"):
        return "text"
    base = lower.replace("\\", "/").rsplit("/", 1)[-1]
    # The file's own name decides before its folder: a datasheet inside slides/ is a manual
    if any(k in base for k in ("datasheet", "manual", "spec", "reference")):
        return "manual"
    if "slide" in lower or "lecture" in lower or "aula" in lower:
        return "slides"
    return "worksheet"


def chunk_file(path: str, kind: Optional[str] = None, window: int = 450, overlap: int = 50,
               token_counter: Optional[TokenCounter] = None,
               pages: Optional[Collection[int]] = None) -> List[Dict]:
    """
    Parse and chunk one file. Every chunk text is at most `window` tokens; chunks never
    cross page boundaries, so `page_number` in the metadata is exact. `pages` restricts PDF
    parsing to those 1-based page numbers (running headers are still detected on all pages).
    """
    kind = detect_kind(path, kind)
    chunker = StructuredChunker(window=window, overlap=overlap, token_counter=token_counter)
    out: List[Dict] = []

    def emit(texts: List[str], meta: Dict):
        for text in texts:
            out.append({"text": text, "metadata": {**meta, "chunk_index": len(out)}})

    if kind in ("slides", "worksheet", "manual"):
        parsed: List[ParsedPage] = parse_slides(path, pages) if kind == "slides" else parse_document(path, pages)
        for page in parsed:
            if not page.blocks:
                continue
            meta = {"page_number": page.page_number, "total_pages": page.total_pages}
            if page.topic:
                meta["topic"] = page.topic
            starts_with_heading = page.blocks[0].kind == "heading"
            # Slides: the topic label heads every chunk of the slide. Documents: the running
            # section heading gives context to pages that continue a section.
            prefix = f"# {page.topic}" if page.topic and (kind == "slides" or not starts_with_heading) else ""
            emit(chunker.chunk(page.blocks, prefix=prefix), meta)
        return out

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return out
    blocks = [Block("code", text)] if kind == "code" else parse_markdown(text)
    emit(chunker.chunk(blocks), {})
    return out
