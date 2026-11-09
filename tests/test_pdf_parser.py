import unittest

from carta_framework.extensions import pdf_parser as pp


def line(text, x0=50.0, top=100.0, size=20.0, font="ArialMT", width=None):
    chars = [{"size": size, "fontname": font, "text": ch} for ch in text]
    return {"text": text, "x0": x0, "x1": x0 + (width or len(text) * size * 0.5), "top": top,
            "bottom": top + size, "chars": chars}


class TestPdfLayoutRules(unittest.TestCase):
    def test_bullet_glyphs_are_not_empty(self):
        # Regression: a private-use glyph lost in an edit became "", and every line was a bullet
        self.assertTrue(all(pp._BULLETS))
        self.assertIn("\uf0b7", pp._BULLETS)

    def test_wrapped_lines_join_their_bullet_with_hanging_indent(self):
        lines = [line("• Delay por software:", x0=56, top=145, size=24),
                 line("• O tempo de espera depende", x0=92, top=174, size=20),
                 line("dos ciclos máquina;", x0=110, top=193.2, size=20),
                 line("• Difíceis de controlar:", x0=56, top=260, size=24)]
        items = pp._group_items(lines)
        self.assertEqual([i["text"] for i in items],
                         ["• Delay por software:", "• O tempo de espera depende dos ciclos máquina;",
                          "• Difíceis de controlar:"])
        self.assertEqual(pp._indent_levels(items), [0, 1, 0])

    def test_plain_lines_join_only_when_aligned(self):
        lines = [line("Se o sinal de clock for", x0=7, top=100, size=18),
                 line("derivado do relógio", x0=7, top=118.5, size=18),
                 line("Outro bloco", x0=40, top=137, size=18)]
        self.assertEqual([i["text"] for i in pp._group_items(lines)],
                         ["Se o sinal de clock for derivado do relógio", "Outro bloco"])

    def test_two_columns_are_separated(self):
        left = [line(f"left {i}", x0=7, top=100 + 19 * i) for i in range(4)]
        right = [line(f"right {i}", x0=236, top=100 + 19 * i) for i in range(3)]
        cols = pp._columns(left + right, page_width=720)
        self.assertEqual(len(cols), 2)
        self.assertEqual([l["text"] for l in cols[1]], ["right 0", "right 1", "right 2"])

    def test_heading_levels(self):
        body = 9.0
        self.assertEqual(pp._heading_level(line("6.2.3 Interrupt Summary", font="Arial-BoldMT", size=9), 9, body), 3)
        self.assertEqual(pp._heading_level(line("6 Interrupts", font="Arial-BoldMT", size=9), 9, body), 1)
        self.assertEqual(pp._heading_level(line("6.2.3 Interrupt Summary", size=9), 9, body), 0)          # not bold
        self.assertEqual(pp._heading_level(line("2 RFLSH 0 RW RX FIFO Flush.", font="Arial-BoldMT", size=9), 9, body), 0)
        self.assertEqual(pp._heading_level(line("( n )", size=14), 14, body), 0)
        self.assertEqual(pp._heading_level(line("Timers and Counters", size=15), 15, body), 1)

    def test_running_footer_detected_also_when_merged(self):
        noise = {pp._margin_key("Microcontroladores -2024/2025 –Jorge Cabral")}
        self.assertTrue(pp._is_noise("Microcontroladores -2024/2025 –Jorge Cabral A07b-2", noise))
        self.assertFalse(pp._is_noise("Timer 0 overflow sets TF0", noise))


    def test_symbol_font_glyphs_are_mapped(self):
        # Symbol-font arrow (U+F0AC) and approx (U+F0BB) arrive as private-use code points
        self.assertEqual(pp._normalize("t0 \uf0ac t1+t2"), "t0 \u2190 t1+t2")
        self.assertEqual(pp._normalize("x \uf0bb 3"), "x \u2248 3")
        self.assertEqual(pp._normalize("odd\ue0ab glyph"), "odd glyph")

    def test_line_is_split_at_column_gaps(self):
        chars = []
        for text, x in (("Realizam", 31), ("Transferem", 256)):
            for i, ch in enumerate(text):
                chars.append({"text": ch, "x0": x + i * 7, "x1": x + i * 7 + 6, "size": 14, "fontname": "Calibri"})
        whole = {"text": "Realizam Transferem", "x0": 31, "x1": 326, "top": 86, "bottom": 100, "chars": chars}
        parts = pp._split_at_gaps(whole)
        self.assertEqual([p["text"] for p in parts], ["Realizam", "Transferem"])
        self.assertEqual(parts[1]["x0"], 256)

    def test_footer_variants_share_a_key(self):
        self.assertEqual(pp._margin_key("Microcontroladores-26/27-Jorge Cabral"),
                         pp._margin_key("Microcontroladores-26/27 -Jorge Cabral"))

    def test_heading_spanning_columns_is_read_before_them(self):
        lines = [line("Instruction Set Architecture: three groups", x0=13, top=13, size=14, width=313)]
        for col, x in enumerate((31, 256, 473)):
            lines += [line(f"col{col} row{r}", x0=x, top=53 + 17 * r, size=14, width=150) for r in range(3)]
        cols = pp._columns(lines, page_width=720)
        self.assertEqual([l["text"] for l in cols[0]], ["Instruction Set Architecture: three groups"])
        self.assertEqual([sorted({l["text"][:4] for l in c}) for c in cols[1:]], [["col0"], ["col1"], ["col2"]])

if __name__ == "__main__":
    unittest.main()
