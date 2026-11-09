"""
Layout-aware PDF parsing into structural blocks (thesis 4.4.1).

- Slides: one page per slide; the largest-font line is the topic label; two-column layouts
  are read column by column; indented lines become nested Markdown list items.
- Documents (laboratory worksheets, manuals, datasheets): tables are reconstructed into
  Markdown rows with pdfplumber's table finder; larger-font lines become headings;
  monospace lines become code blocks.
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Collection, List, Optional

from .structured_chunker import Block

_MONO_HINTS = ("mono", "courier", "consol", "menlo", "inconsolata")
# \uf0b7 / \uf0a7: bullet glyphs of the Symbol / Wingdings fonts (private-use code points)
_BULLETS = ("\u2022", "\u25aa", "\u25e6", "\u2023", "\uf0b7", "\uf0a7", "\u2013", "-", "*", "o ")


# Symbol / Wingdings glyphs that PDFs expose as private-use code points (0xF000 + font code)
_PUA_GLYPHS = {
    0xF0AC: "\u2190", 0xF0AD: "\u2191", 0xF0AE: "\u2192", 0xF0AF: "\u2193", 0xF0AB: "\u2194",   # arrows
    0xF0DC: "\u21d0", 0xF0DE: "\u21d2", 0xF0DB: "\u21d4",                                    # double arrows
    0xF0A3: "\u2264", 0xF0B3: "\u2265", 0xF0B9: "\u2260", 0xF0BB: "\u2248", 0xF0B1: "\u00b1", # relations
    0xF0B4: "\u00d7", 0xF0B8: "\u00f7", 0xF0A5: "\u221e",
    0xF0B7: "\u2022", 0xF0A7: "\u25aa", 0xF0D8: "\u27a2", 0xF0FC: "\u2713", 0xF0E0: "\u2192",   # bullets, check
}


def _normalize(text: str) -> str:
    """Map Symbol/Wingdings private-use glyphs to Unicode; drop other private-use characters."""
    if not any(0xE000 <= ord(c) <= 0xF8FF for c in text):
        return text
    return "".join(_PUA_GLYPHS.get(ord(c), "") if 0xE000 <= ord(c) <= 0xF8FF else c for c in text)


def _split_at_gaps(line):
    """
    pdfplumber joins words at the same height into one line even across columns. Split a line
    where the gap between characters is far wider than a word space (1.5 em; 3 em for monospace,
    so a code line stays together with its aligned comment).
    """
    chars = [c for c in line.get("chars", []) if c.get("text")]
    if len(chars) < 2:
        return [line]
    size = median(c["size"] for c in chars) or 10
    limit = size * (3 if _is_mono(line) else 1.5)
    segments, current = [], [chars[0]]
    for prev, char in zip(chars, chars[1:]):
        if char["x0"] - prev["x1"] > limit and char["text"].strip():
            segments.append(current)
            current = []
        current.append(char)
    segments.append(current)
    if len(segments) == 1:
        return [line]
    out = []
    for seg in segments:
        text, last = "", None
        for c in seg:
            if last is not None and c["x0"] - last["x1"] > 0.2 * size and not text.endswith(" ") and c["text"] != " ":
                text += " "
            text += c["text"]
            last = c
        if text.strip():
            out.append({**line, "text": text.strip(), "chars": seg, "x0": min(c["x0"] for c in seg),
                        "x1": max(c["x1"] for c in seg)})
    return out


def _text_lines(page):
    lines = [part for line in page.extract_text_lines() for part in _split_at_gaps(line)]
    for line in lines:
        line["text"] = _normalize(line["text"])
    return lines


@dataclass
class ParsedPage:
    page_number: int
    total_pages: int
    blocks: List[Block] = field(default_factory=list)
    topic: Optional[str] = None


def _repeated_margin_lines(pdf, margin: float = 0.1, share: float = 0.3) -> set:
    """
    Running headers/footers: lines in the top or bottom margin whose text (digits removed)
    recurs on at least `share` of the pages.
    """
    if len(pdf.pages) < 4:
        return set()
    seen = Counter()
    for page in pdf.pages:
        keys = set()
        for line in _text_lines(page):
            if line["top"] < page.height * margin or line["bottom"] > page.height * (1 - margin):
                keys.add(_margin_key(line["text"]))
        seen.update(keys)
    return {k for k, n in seen.items() if k and n >= share * len(pdf.pages)}


def _is_noise(text: str, noise: set) -> bool:
    """A running header/footer, also when it is merged with other margin text on one line."""
    key = _margin_key(text)
    return key in noise or any(len(n) >= 12 and n in key for n in noise)


def _margin_key(text: str) -> str:
    """Comparable form of a margin line: digits as '#', spaces and punctuation ignored."""
    return re.sub(r"[\W_]+", "", re.sub(r"\d+", "#", text.lower())).replace("#", "#")


def _line_size(line) -> float:
    return median(c["size"] for c in line["chars"]) if line.get("chars") else 0.0


def _is_mono(line) -> bool:
    fonts = Counter(c.get("fontname", "").lower() for c in line.get("chars", []))
    return bool(fonts) and any(h in fonts.most_common(1)[0][0] for h in _MONO_HINTS)


_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+[A-Za-z]")


def _is_bold(line) -> bool:
    chars = line.get("chars", [])
    return bool(chars) and sum("bold" in c.get("fontname", "").lower() for c in chars) > len(chars) / 2


def _heading_level(line, size: float, body: float) -> int:
    """0 if not a heading; otherwise 1-3, from font size or from a bold section number."""
    text = line["text"].strip()
    # Headings have words and no final period (rules out bit-field rows and formula fragments)
    if len(text) >= 120 or len(re.findall(r"[A-Za-z]", text)) < 3 or text.endswith("."):
        return 0
    if size >= body * 1.15:
        return 1 if size >= body * 1.6 else 2 if size >= body * 1.3 else 3
    m = _NUMBERED_HEADING_RE.match(text)
    if m and _is_bold(line) and len(text) < 100:
        return min(3, m.group(1).count(".") + 1)
    return 0


def _inside(line, boxes) -> bool:
    cx, cy = (line["x0"] + line["x1"]) / 2, (line["top"] + line["bottom"]) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)


def _tables(page):
    """
    (bbox, rows) for each data table on the page, dropping empty rows/columns. Ruled layouts
    that pdfplumber also reports as tables (a framed slide, text boxes) are skipped: they hold
    paragraph-length cells, or cover most of the page with only a few rows.
    """
    found = []
    page_area = float(page.width * page.height) or 1.0
    for table in page.find_tables():
        x0, top, x1, bottom = table.bbox
        large = (x1 - x0) * (bottom - top) > 0.5 * page_area
        rows = [[_normalize((c or "").strip()) for c in r] for r in table.extract()]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            continue
        keep = [i for i in range(max(len(r) for r in rows)) if any(i < len(r) and r[i] for r in rows)]
        rows = [[r[i] if i < len(r) else "" for i in keep] for r in rows]
        cells = [c for r in rows for c in r if c]
        mean_cell = sum(map(len, cells)) / max(1, len(cells))
        if len(keep) >= 2 and mean_cell <= 120 and not (large and (len(rows) < 5 or mean_cell > 60)):
            found.append((table.bbox, rows))
    return found


def _strip_bullet(text: str) -> str:
    for b in _BULLETS:
        if text.startswith(b) and len(text) > len(b):
            return text[len(b):].lstrip()
    return text


def _columns(lines, page_width):
    """
    Split body lines into reading order: lines spanning most of the page come before or after
    the column block (by position); the remaining lines are clustered into side-by-side
    columns by their left edge, read left to right.
    """
    if len(lines) < 4:
        return [lines]
    wide = [l for l in lines if l["x1"] - l["x0"] > page_width * 0.55]
    rest = sorted((l for l in lines if l not in wide), key=lambda l: l["x0"])
    clusters = []
    for line in rest:
        if clusters and line["x0"] - max(l["x0"] for l in clusters[-1]) <= page_width * 0.12:
            clusters[-1].append(line)
        else:
            clusters.append([line])
    # A line reaching into the next cluster spans columns (e.g. a heading over them): read it
    # before or after the column block instead of inside the first column.
    for a, b in zip(clusters, clusters[1:]):
        start = min(l["x0"] for l in b)
        spanning = [l for l in a if l["x1"] > start + 4]
        wide += spanning
        a[:] = [l for l in a if l not in spanning]
    clusters = [c for c in clusters if c]
    columns = [c for c in clusters if len(c) >= 2]
    stray = [l for c in clusters if len(c) < 2 for l in c]
    # Side-by-side only if neighbouring columns share vertical space and do not overlap horizontally
    side_by_side = len(columns) >= 2 and all(
        min(max(l["bottom"] for l in a), max(l["bottom"] for l in b)) > max(min(l["top"] for l in a), min(l["top"] for l in b))
        and max(l["x1"] for l in a) <= min(l["x0"] for l in b) + 4
        for a, b in zip(columns, columns[1:]))
    if not side_by_side:
        return [lines]
    for line in stray:                                    # attach single lines to the nearest column
        min(columns, key=lambda c: abs(min(l["x0"] for l in c) - line["x0"])).append(line)
    block_top = min(l["top"] for c in columns for l in c)
    above = [l for l in wide if l["top"] < block_top]
    below = [l for l in wide if l["top"] >= block_top]
    return ([above] if above else []) + columns + ([below] if below else [])


def _starts_bullet(text: str) -> bool:
    return text.lstrip().startswith(_BULLETS[:6])


def _group_items(lines):
    """
    Group wrapped lines into items. A line continues the previous item when it follows
    closely and does not start with a bullet: under a bullet item any (hanging) indent
    counts, otherwise the indent must match.
    """
    items = []
    for line in sorted(lines, key=lambda l: (round(l["top"]), l["x0"])):
        size = _line_size(line) or 10
        text = line["text"].strip()
        if items and not _starts_bullet(text):
            prev = items[-1]
            close = line["top"] - prev["bottom"] < 0.6 * size
            aligned = prev["bullet"] and line["x0"] >= prev["x0"] - 2 or abs(line["x0"] - prev["x0"]) < 4
            if close and aligned:
                prev["text"] += " " + text
                prev["bottom"] = line["bottom"]
                continue
        items.append({"text": text, "x0": line["x0"], "top": line["top"], "bottom": line["bottom"],
                      "size": size, "bullet": _starts_bullet(text)})
    return items


def _indent_levels(items, tolerance: float = 6.0):
    """Nesting level of each item: rank of its x position among the distinct indents."""
    stops = []
    for x in sorted(i["x0"] for i in items):
        if not stops or x - stops[-1] > tolerance:
            stops.append(x)
    return [max(k for k, s in enumerate(stops) if i["x0"] >= s - tolerance) for i in items]


def _slide_column_blocks(column) -> List[Block]:
    """
    Blocks of one slide column in reading order: runs of two or more monospace lines become
    code blocks (assembly listings keep their line structure); the rest becomes list items.
    """
    ordered = sorted(column, key=lambda l: (round(l["top"]), l["x0"]))
    runs, current = [], []
    for line in ordered:
        mono = _is_mono(line)
        if current and current[0] != mono:
            runs.append(current)
            current = []
        if not current:
            current = [mono]
        current.append(line)
    if current:
        runs.append(current)
    # a single monospace line is inline code inside prose, not a listing
    merged = []
    for run in runs:
        is_code = run[0] and len(run) - 1 >= 2
        if merged and not is_code and not merged[-1][0]:
            merged[-1].extend(run[1:])
        else:
            merged.append([is_code] + run[1:])

    blocks: List[Block] = []
    items_of = {k: _group_items(run[1:]) for k, run in enumerate(merged) if not run[0]}
    text_items = [i for k in sorted(items_of) for i in items_of[k]]
    levels = dict(zip(map(id, text_items), _indent_levels(text_items))) if text_items else {}
    for k, run in enumerate(merged):
        if run[0]:
            base = min(l["x0"] for l in run[1:])
            width = max(1.0, median(c["width"] for l in run[1:] for c in l["chars"] if c.get("width")))
            blocks.append(Block("code", "\n".join(" " * int(round((l["x0"] - base) / width)) + l["text"].rstrip()
                                                  for l in run[1:])))
            continue
        for item in items_of[k]:
            text = _strip_bullet(item["text"])
            level = min(levels.get(id(item), 0), 3)
            blocks.append(Block("list_item", text, level=level) if len(text_items) > 1 else Block("paragraph", text))
    return blocks


def parse_slides(path: str, pages_filter: Optional[Collection[int]] = None) -> List[ParsedPage]:
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        noise = _repeated_margin_lines(pdf)
        for number, page in enumerate(pdf.pages, 1):
            if pages_filter is not None and number not in pages_filter:
                continue
            tables = _tables(page)
            boxes = [bbox for bbox, _ in tables]
            raw = [l for l in _text_lines(page) if l["text"].strip() and not _inside(l, boxes)]
            typical = median(_line_size(l) for l in raw) if raw else 0
            # A repeated large-font line at the top is the deck's section title, not a footer: keep it
            lines = [l for l in raw if not _is_noise(l["text"], noise)
                     or (_line_size(l) > typical * 1.25 and l["top"] < page.height * 0.25)]
            parsed = ParsedPage(number, total)
            if not lines and not tables:
                pages.append(parsed)
                continue

            sizes = [_line_size(l) for l in lines]
            body = median(sizes) if sizes else 0
            biggest = max(sizes) if sizes else 0
            title_lines = [l for l, s in zip(lines, sizes) if biggest >= body * 1.25 and s >= biggest - 0.5]
            sub_lines = [l for l, s in zip(lines, sizes) if l not in title_lines and s >= body * 1.25]
            body_lines = [l for l in lines if l not in title_lines and l not in sub_lines]
            if title_lines:
                parsed.topic = " ".join(l["text"].strip() for l in sorted(title_lines, key=lambda l: l["top"]))
            for l in sorted(sub_lines, key=lambda l: l["top"]):
                parsed.blocks.append(Block("heading", l["text"].strip(), level=3))

            for column in _columns(body_lines, page.width):
                parsed.blocks.extend(_slide_column_blocks(column))
            for _, rows in tables:
                parsed.blocks.append(Block("table", rows=rows))
            pages.append(parsed)
    return pages


def parse_document(path: str, pages_filter: Optional[Collection[int]] = None) -> List[ParsedPage]:
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        all_sizes = [round(c["size"], 1) for p in pdf.pages[: min(total, 30)] for c in p.chars[:2000]]
        body = Counter(all_sizes).most_common(1)[0][0] if all_sizes else 10.0
        section = None
        noise = _repeated_margin_lines(pdf)
        for number, page in enumerate(pdf.pages, 1):
            if pages_filter is not None and number not in pages_filter:
                continue
            parsed = ParsedPage(number, total, topic=section)
            tables = _tables(page)
            boxes = [bbox for bbox, _ in tables]
            lines = [l for l in _text_lines(page) if l["text"].strip() and not _inside(l, boxes)
                     and not _is_noise(l["text"], noise)]
            # Merge tables and text in vertical reading order
            events = [(l["top"], "line", l) for l in lines] + [(bbox[1], "table", rows) for bbox, rows in tables]
            para, code, prev = [], [], None

            def end_para():
                if para:
                    text = " ".join(para)
                    stripped = _strip_bullet(text)
                    parsed.blocks.append(Block("list_item", stripped) if stripped != text or text[:3].rstrip(".)").isdigit()
                                         else Block("paragraph", text))
                    para.clear()

            def end_code():
                if code:
                    parsed.blocks.append(Block("code", "\n".join(code)))
                    code.clear()

            for _, kind, item in sorted(events, key=lambda e: e[0]):
                if kind == "table":
                    end_para(); end_code()
                    parsed.blocks.append(Block("table", rows=item))
                    prev = None
                    continue
                line, size, text = item, _line_size(item), item["text"].strip()
                if _is_mono(line):
                    end_para()
                    code.append(line["text"].rstrip())
                    prev = line
                    continue
                end_code()
                level = _heading_level(line, size, body)
                if level:
                    end_para()
                    parsed.blocks.append(Block("heading", text, level=level))
                    if level <= 2:
                        section = text
                    prev = None
                    continue
                new_para = prev is None or line["top"] - prev["bottom"] > 0.6 * (size or 10) or \
                    text.startswith(_BULLETS[:6]) or (text[:3].rstrip(".)").isdigit() and text[1:3].strip()[:1] in ".)")
                if new_para:
                    end_para()
                para.append(text)
                prev = line
            end_para(); end_code()
            pages.append(parsed)
    return pages
