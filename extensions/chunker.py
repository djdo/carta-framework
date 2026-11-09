import re
from typing import List, Dict, Generator, Optional

class PDFPageChunker:
    """Splits PDF documents page-by-page, retaining page number metadata."""
    
    @staticmethod
    def split(file_path: str) -> Generator[Dict[str, any], None, None]:
        """
        Extract text page-by-page from a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Yields:
            Dict containing 'text' (str) and 'metadata' (dict) with keys:
                - page_number: 1-based page index
                - total_pages: Total number of pages in PDF
        """
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(file_path)
        num_pages = len(pdf)
        
        try:
            for idx, page in enumerate(pdf):
                text_page = page.get_textpage()
                text = text_page.get_text_range()
                text = text.strip() if text else ""
                
                # Yield even empty/sparse pages to maintain chronological indexing
                yield {
                    "text": text,
                    "metadata": {
                        "page_number": idx + 1,
                        "total_pages": num_pages
                    }
                }
        finally:
            pdf.close()


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text using a list of separators (e.g. paragraphs, newlines, spaces).
    Aims to keep logically related lines (like assembly subroutines or functions) in the same chunk.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Splits the text into chunks of chunk_size with chunk_overlap."""
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting implementation."""
        if len(text) <= self.chunk_size:
            return [text]

        # Get separator
        separator = separators[-1]
        for s in separators:
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                break

        # Split text by separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits into chunks
        chunks = []
        current_chunk = []
        current_len = 0

        for split in splits:
            # Re-add separator except at the end
            split_text = split + (separator if separator != "" else "")
            split_len = len(split_text)

            if split_len > self.chunk_size:
                # If a single split is larger than chunk_size, split it recursively
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Recurse on remaining separators
                sub_separators = [s for s in separators if s != separator]
                chunks.extend(self._split(split, sub_separators))
            else:
                if current_len + split_len > self.chunk_size:
                    chunks.append(separator.join(current_chunk) if separator != "" else "".join(current_chunk))
                    
                    # Handle overlap by backing up
                    overlap_chunk = []
                    overlap_len = 0
                    for prev_split in reversed(current_chunk):
                        prev_len = len(prev_split) + (len(separator) if separator != "" else 0)
                        if overlap_len + prev_len <= self.chunk_overlap:
                            overlap_chunk.insert(0, prev_split)
                            overlap_len += prev_len
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_len = overlap_len
                
                current_chunk.append(split)
                current_len += split_len

        if current_chunk:
            chunks.append(separator.join(current_chunk) if separator != "" else "".join(current_chunk))

        # Filter out empty chunks and strip whitespaces
        return [c.strip() for c in chunks if c.strip()]
