from PyPDF2 import PdfReader


class PDFReaderTool:
    """
    A custom tool for reading and extracting text from PDF files.
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.name = "PDF Reader Tool"
        self.description = "A tool to read and extract text from PDF files."

    def run(self, query: str = None) -> str:
        """
        Extracts text from the PDF file.
        """
        try:
            with open(self.pdf_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                text = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text()
                return text
        except Exception as e:
            return f"Error extracting text from PDF: {e}"