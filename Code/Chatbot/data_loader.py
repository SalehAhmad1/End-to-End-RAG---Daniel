import os
from docx import Document
import PyPDF2

class ChatbotDataLoader:
    def __init__(self):
        pass

    def read_docx(self, file_path):
        """
        Reads content from a .docx file.
        """
        doc = Document(file_path)
        content = "\n".join([para.text for para in doc.paragraphs])
        return content

    def read_pdf(self, file_path):
        """
        Reads content from a .pdf file.
        """
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            content = ""
            for page in range(len(reader.pages)):
                content += reader.pages[page].extract_text()
        return content

    def load_file(self, file_path):
        """
        Reads content from a .docx or .pdf file based on the file extension.
        """
        if file_path.endswith(".docx"):
            return self.read_docx(file_path)
        elif file_path.endswith(".pdf"):
            return self.read_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def load_directory(self, dir_path):
        """
        Iterates through the directory, loads all .docx and .pdf files, and returns their content.
        """
        file_contents = {}
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith((".docx", ".pdf")):
                    try:
                        content = self.load_file(file_path)
                        file_contents[file_path] = content
                    except Exception as e:
                        print(f"Failed to load {file_path}: {str(e)}")
        return file_contents
