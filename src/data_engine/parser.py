import os
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

class PDFParser:
    def __init__(self):
        pass

    def parse_pdf(self, file_path: str) -> str:
        """
        Extracts text from a PDF file.
        """
        if not PyPDF2:
            print("PyPDF2 not installed. Please install it to parse PDFs.")
            return ""
            
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return ""

        text = ""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return ""
