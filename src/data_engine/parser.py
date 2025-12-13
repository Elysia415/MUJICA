import os
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

class PDFParser:
    def __init__(self):
        pass

    def parse_pdf(self, file_path: str, max_pages: int | None = None) -> str:
        """
        Extracts text from a PDF file.
        """
        if not PyPDF2 and not pdfplumber:
            print("PyPDF2/pdfplumber not installed. Please install one of them to parse PDFs.")
            return ""
            
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return ""

        text = ""
        try:
            # 优先用 pdfplumber（通常比 PyPDF2 提取效果更好）
            if pdfplumber:
                with pdfplumber.open(file_path) as pdf:
                    pages = pdf.pages[:max_pages] if max_pages else pdf.pages
                    for page in pages:
                        extracted = page.extract_text() or ""
                        if extracted.strip():
                            text += extracted.strip() + "\n"
                return text.strip()

            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                pages = reader.pages[:max_pages] if max_pages else reader.pages
                for page in pages:
                    extracted = page.extract_text() or ""
                    if extracted.strip():
                        text += extracted.strip() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return ""
