import unittest

from carta_framework.extensions.structured_chunker import (
    Block, StructuredChunker, estimate_tokens, parse_markdown, render_table,
)


def words(n, word="lorem"):
    return " ".join(f"{word}{i}." for i in range(n))


class TestStructuredChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = StructuredChunker(window=120, overlap=20)

    def assert_within_window(self, chunks, window=120):
        for c in chunks:
            self.assertLessEqual(estimate_tokens(c), window, c[:80])

    def test_small_document_is_one_chunk(self):
        blocks = [Block("heading", "Timers", 2), Block("paragraph", "Timer 0 counts machine cycles.")]
        self.assertEqual(self.chunker.chunk(blocks), ["## Timers\n\nTimer 0 counts machine cycles."])

    def test_long_prose_slides_with_overlap(self):
        chunks = self.chunker.chunk([Block("paragraph", words(200))])
        self.assertGreater(len(chunks), 2)
        self.assert_within_window(chunks)
        # Consecutive windows share their boundary words (overlap O)
        first_tail = chunks[0].split()[-1]
        self.assertIn(first_tail, chunks[1].split()[:15])

    def test_major_heading_closes_chunk_early(self):
        blocks = [Block("heading", "Timers", 2), Block("paragraph", words(30)),
                  Block("heading", "Serial port", 2), Block("paragraph", "UART text.")]
        chunks = self.chunker.chunk(blocks)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].startswith("## Serial port"))
        self.assertNotIn("Serial port", chunks[0])

    def test_minor_heading_does_not_force_a_break(self):
        blocks = [Block("heading", "Timers", 2), Block("paragraph", words(30)),
                  Block("heading", "Mode 1", 3), Block("paragraph", "16-bit mode.")]
        self.assertEqual(len(self.chunker.chunk(blocks)), 1)

    def test_table_fitting_the_window_stays_whole(self):
        rows = [["Bit", "Field", "Description"]] + [[str(b), f"F{b}", "enable"] for b in range(5)]
        blocks = [Block("paragraph", words(60)), Block("table", rows=rows)]
        chunks = self.chunker.chunk(blocks)
        table_chunks = [c for c in chunks if "| Bit |" in c]
        self.assertEqual(len(table_chunks), 1)
        self.assertIn(render_table(rows), table_chunks[0])

    def test_long_table_splits_by_rows_and_repeats_header(self):
        rows = [["Bit", "Field", "Description"]] + [[str(b), f"FIELD{b}", "Machine timer interrupt enable"] for b in range(40)]
        chunks = self.chunker.chunk([Block("table", rows=rows)])
        self.assertGreater(len(chunks), 2)
        self.assert_within_window(chunks)
        seen = []
        for c in chunks:
            lines = c.split("\n")
            self.assertEqual(lines[0], "| Bit | Field | Description |")
            self.assertEqual(lines[1], "| --- | --- | --- |")
            seen += [l.split("|")[1].strip() for l in lines[2:]]
        self.assertEqual(seen, [str(b) for b in range(40)])  # every row exactly once, in order

    def test_code_block_fitting_the_window_stays_whole(self):
        code = "timer_init:\n    mov TMOD, #01h\n    setb ET0\n    setb EA\n    setb TR0\n    ret"
        chunks = self.chunker.chunk([Block("paragraph", words(50)), Block("code", code)])
        self.assertTrue(any(f"```\n{code}\n```" in c for c in chunks))

    def test_long_code_splits_only_at_labels(self):
        routines = [f"; routine {r}\nroutine_{r}:\n" + "\n".join(f"    mov R{i % 8}, #{i}h" for i in range(12)) + "\n    ret"
                    for r in range(8)]
        chunks = self.chunker.chunk([Block("code", "\n".join(routines))])
        self.assertGreater(len(chunks), 1)
        self.assert_within_window(chunks)
        for c in chunks:
            body = c.strip("`\n").split("\n")
            self.assertTrue(body[0].startswith("; routine"), body[0])   # starts at a label (with its comment)
            self.assertEqual(c.count("ret"), c.count("routine_"))      # no routine cut in half

    def test_prefix_is_counted_and_repeated(self):
        chunks = self.chunker.chunk([Block("paragraph", words(150))], prefix="# Lecture 7: Timers")
        self.assertTrue(all(c.startswith("# Lecture 7: Timers\n\n") for c in chunks))
        self.assert_within_window(chunks)

    def test_pathological_input_still_respects_window(self):
        blob = "A" * 3000
        chunks = self.chunker.chunk([Block("paragraph", blob), Block("code", "x" * 2000),
                                     Block("table", rows=[["h"], ["y" * 2000]])])
        self.assert_within_window(chunks)

    def test_parse_markdown(self):
        md = "# Lab 3\n\nIntro text\nwrapped.\n\n## Steps\n1. Wire the LED\n   - use P1.0\n\n| Bit | Name |\n|---|---|\n| 7 | EA |\n\n```asm\nsetb EA\n```\n"
        kinds = [(b.kind, b.level) for b in parse_markdown(md)]
        self.assertEqual(kinds, [("heading", 1), ("paragraph", 0), ("heading", 2), ("list_item", 0),
                                 ("list_item", 1), ("table", 0), ("code", 0)])
        blocks = parse_markdown(md)
        self.assertEqual(blocks[1].text, "Intro text wrapped.")
        self.assertEqual(blocks[5].rows, [["Bit", "Name"], ["7", "EA"]])
        self.assertEqual(blocks[6].text, "setb EA")


if __name__ == "__main__":
    unittest.main()
