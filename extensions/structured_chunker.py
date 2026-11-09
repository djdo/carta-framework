"""
Structure-aware, token-bounded chunking (thesis 4.4.2).

A document is first parsed into blocks (headings, paragraphs, list items, tables, code).
The chunker then packs blocks into windows of at most W tokens with an overlap of O tokens,
following three structural rules:

1. Heading boundaries: a major heading (level <= 2) closes the current chunk early.
2. Table integrity: a table is never cut between rows; a table longer than W is split by
   rows with the header row repeated in every piece.
3. Code block cohesion: a code block stays whole if it fits in W; longer listings are split
   only at label or function boundaries.

Every emitted chunk is guaranteed to be at most W tokens (by the configured token counter).
"""
import math
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Default token estimate (about four characters per token), as used by ChatEngine."""
    return math.ceil(len(text) / 4) if text else 0


@dataclass
class Block:
    """One structural unit of a parsed document."""
    kind: str                     # "heading" | "paragraph" | "list_item" | "table" | "code"
    text: str = ""                # heading/paragraph/list text, or code body
    level: int = 0                # heading level (1-6) or list indent level
    rows: List[List[str]] = field(default_factory=list)   # table rows; rows[0] is the header

    def render(self) -> str:
        if self.kind == "heading":
            return f"{'#' * max(1, self.level)} {self.text}"
        if self.kind == "list_item":
            return f"{'  ' * self.level}- {self.text}"
        if self.kind == "table":
            return render_table(self.rows)
        if self.kind == "code":
            return f"```\n{self.text.rstrip()}\n```"
        return self.text


def _cell(value) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).replace("|", "\\|").strip()


def render_table(rows: List[List[str]]) -> str:
    """Render rows as a Markdown table; rows[0] is the header."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[_cell(c) for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(norm[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join(lines)


# Assembly labels ("timer_init:", "ISR_T0:") and C function definitions start a code segment.
_LABEL_RE = re.compile(r"^[A-Za-z_.$][\w.$]*:")
_C_FUNC_RE = re.compile(r"^[A-Za-z_][\w\s\*]*\b[A-Za-z_]\w*\s*\([^;]*\)\s*\{?\s*$")


class StructuredChunker:
    """Packs parsed blocks into chunks of at most `window` tokens (thesis 4.4.2)."""

    def __init__(self, window: int = 450, overlap: int = 50,
                 token_counter: Optional[TokenCounter] = None):
        if overlap >= window:
            raise ValueError("overlap must be smaller than window")
        self.window = window
        self.overlap = overlap
        self.count = token_counter or estimate_tokens

    # ── public API ──────────────────────────────────────────────────

    def chunk(self, blocks: List[Block], prefix: str = "") -> List[str]:
        """
        Chunk a sequence of blocks. `prefix` (e.g. the slide topic) is prepended to every
        chunk and counted against the window.
        """
        budget = self.window - (self.count(prefix + "\n\n") if prefix else 0)
        if budget <= self.overlap:
            raise ValueError("prefix leaves no room in the window")

        chunks: List[str] = []
        current: List[str] = []

        def tokens(parts: List[str]) -> int:
            return self.count("\n\n".join(parts)) if parts else 0

        def flush(carry_overlap: bool):
            nonlocal current
            if not current:
                return
            chunks.append("\n\n".join(current))
            current = [self._tail(current[-1])] if carry_overlap and self.overlap else []
            if current and not current[0]:
                current = []

        for block in blocks:
            if block.kind == "heading" and block.level <= 2 and tokens(current) >= budget // 4:
                flush(carry_overlap=False)                       # rule 1

            for piece in self._pieces(block, budget):
                atomic = block.kind in ("table", "code")
                if tokens(current + [piece]) <= budget:
                    current.append(piece)
                    continue
                flush(carry_overlap=not atomic)
                if tokens(current + [piece]) > budget:
                    current = []                                 # overlap does not fit beside it
                current.append(piece)
        flush(carry_overlap=False)

        result = [f"{prefix}\n\n{c}" if prefix else c for c in chunks if c.strip()]
        for c in result:
            assert self.count(c) <= self.window, "chunk exceeds window"
        return result

    # ── splitting a single block into window-sized pieces ───────────

    def _pieces(self, block: Block, budget: int) -> List[str]:
        rendered = block.render()
        if self.count(rendered) <= budget:
            return [rendered]
        if block.kind == "table":
            return self._split_table(block.rows, budget)         # rule 2
        if block.kind == "code":
            return self._split_code(block.text, budget)          # rule 3
        prefix = "  " * block.level + "- " if block.kind == "list_item" else ""
        return [prefix + p for p in self._split_prose(block.text, budget - self.count(prefix))]

    def _split_table(self, rows: List[List[str]], budget: int) -> List[str]:
        header, body = rows[0], rows[1:]
        pieces, current = [], []
        for row in body:
            if current and self.count(render_table([header] + current + [row])) > budget:
                pieces.append(render_table([header] + current))
                current = []
            if self.count(render_table([header, row])) > budget:
                # A single row longer than the window: truncate its cells evenly.
                row = self._shrink_row(header, row, budget)
            current.append(row)
        if current:
            pieces.append(render_table([header] + current))
        return pieces

    def _shrink_row(self, header, row, budget):
        cells = [_cell(c) for c in row]
        while self.count(render_table([header, cells])) > budget and any(len(c) > 8 for c in cells):
            longest = max(range(len(cells)), key=lambda i: len(cells[i]))
            cells[longest] = cells[longest][: max(8, int(len(cells[longest]) * 0.8))] + "…"
        return cells

    def _split_code(self, code: str, budget: int) -> List[str]:
        lines = code.rstrip().split("\n")
        # Segments start at labels / function definitions (with any comment lines just above).
        starts = [0]
        for i, line in enumerate(lines):
            if i and (_LABEL_RE.match(line) or _C_FUNC_RE.match(line)):
                j = i
                while j > starts[-1] + 1 and lines[j - 1].lstrip().startswith((";", "//", "/*", "*", "#")):
                    j -= 1
                if j > starts[-1]:
                    starts.append(j)
        segments = ["\n".join(lines[a:b]) for a, b in zip(starts, starts[1:] + [len(lines)])]

        fence = lambda body: f"```\n{body.rstrip()}\n```"
        pieces, current = [], ""
        for seg in segments:
            candidate = f"{current}\n{seg}" if current else seg
            if self.count(fence(candidate)) <= budget:
                current = candidate
                continue
            if current:
                pieces.append(fence(current))
            if self.count(fence(seg)) <= budget:
                current = seg
            else:
                # A single routine longer than the window: split it by lines (last resort).
                current = ""
                for line in (p for ln in seg.split("\n") for p in self._hard_split(ln, budget - self.count(fence("")))):
                    candidate = f"{current}\n{line}" if current else line
                    if current and self.count(fence(candidate)) > budget:
                        pieces.append(fence(current))
                        candidate = line
                    current = candidate
        if current:
            pieces.append(fence(current))
        return pieces

    def _split_prose(self, text: str, budget: int) -> List[str]:
        """Token-sliding window over sentences (falling back to words), with overlap."""
        units = [u for u in re.split(r"(?<=[.!?:;])\s+|\n+", text) if u.strip()]
        if any(self.count(u) > budget for u in units):
            units = [p for w in text.split() for p in self._hard_split(w, budget)]
        pieces, current = [], []
        for unit in units:
            if current and self.count(" ".join(current + [unit])) > budget:
                pieces.append(" ".join(current))
                tail = self._tail(" ".join(current))
                current = [tail] if tail and self.count(f"{tail} {unit}") <= budget else []
            current.append(unit)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _hard_split(self, text: str, budget: int) -> List[str]:
        """Slice a single over-long word or line by characters (last resort)."""
        if self.count(text) <= budget:
            return [text]
        size = max(1, len(text) * budget // max(1, self.count(text)))
        while size > 1 and self.count(text[:size]) > budget:
            size -= 1
        return [text[i:i + size] for i in range(0, len(text), size)]

    def _tail(self, text: str) -> str:
        """The last ~overlap tokens of a prose piece (whole words); empty for tables/code."""
        if text.startswith(("|", "```")):
            return ""
        words, tail = text.split(), []
        for word in reversed(words):
            if self.count(" ".join([word] + tail)) > self.overlap:
                break
            tail.insert(0, word)
        return " ".join(tail) if len(tail) < len(words) else ""


# ── Markdown / plain text parsing ──────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^(\s*)(?:[-*+•▪◦]|\d+[.)])\s+(.*)$")


def parse_markdown(text: str) -> List[Block]:
    """Parse Markdown-ish text into blocks (headings, lists, tables, fenced code, paragraphs)."""
    blocks: List[Block] = []
    lines = text.replace("\r\n", "\n").split("\n")
    i, para = 0, []

    def end_para():
        if para:
            blocks.append(Block("paragraph", " ".join(s.strip() for s in para)))
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            end_para()
            body = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            blocks.append(Block("code", "\n".join(body)))
        elif _HEADING_RE.match(line):
            end_para()
            m = _HEADING_RE.match(line)
            blocks.append(Block("heading", m.group(2).strip(), level=len(m.group(1))))
        elif line.strip().startswith("|"):
            end_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            blocks.append(Block("table", rows=rows))
            continue
        elif _LIST_RE.match(line):
            end_para()
            m = _LIST_RE.match(line)
            blocks.append(Block("list_item", m.group(2).strip(), level=len(m.group(1).expandtabs(2)) // 2))
        elif not line.strip():
            end_para()
        else:
            para.append(line)
        i += 1
    end_para()
    return blocks
