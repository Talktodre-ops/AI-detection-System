import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.mark.frontend
def test_file_processing_functions():
    """Test the file processing functions in the frontend"""
    try:
        from frontend.app import read_pdf, read_docx, read_txt
        
        # Test PDF reading with mocked PyPDF2
        with patch('PyPDF2.PdfReader') as mock_pdf_reader:
            mock_instance = MagicMock()
            mock_pdf_reader.return_value = mock_instance
            mock_instance.pages = [MagicMock(), MagicMock()]
            mock_instance.pages[0].extract_text.return_value = "Page 1 content"
            mock_instance.pages[1].extract_text.return_value = "Page 2 content"
            
            result = read_pdf("dummy.pdf")
            assert "Page 1 content" in result
            assert "Page 2 content" in result
        
        # Test DOCX reading with mocked docx
        with patch('docx.Document') as mock_document:
            mock_instance = MagicMock()
            mock_document.return_value = mock_instance
            mock_instance.paragraphs = [MagicMock(), MagicMock()]
            mock_instance.paragraphs[0].text = "Paragraph 1"
            mock_instance.paragraphs[1].text = "Paragraph 2"
            
            result = read_docx("dummy.docx")
            assert "Paragraph 1" in result
            assert "Paragraph 2" in result
        
        # Test TXT reading with mocked open
        mock_open = MagicMock()
        mock_open.return_value.__enter__.return_value.read.return_value = "Text file content"
        with patch('builtins.open', mock_open):
            result = read_txt("dummy.txt")
            assert result == "Text file content"
            
    except ImportError:
        pytest.skip("Could not import file processing functions from frontend.app")