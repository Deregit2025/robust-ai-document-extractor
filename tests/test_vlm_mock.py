import pytest
from unittest.mock import MagicMock, patch
from src.strategies.vision import VisionExtractor
from src.models.extracted_document import ExtractedDocument

@patch("openai.OpenAI")
@patch("dotenv.load_dotenv")
@patch("pdfplumber.open")
def test_vision_extractor_mock(mock_pdf_open, mock_load_dotenv, mock_openai_class):
    # Mock environment
    with patch("os.getenv", return_value="fake_key"):
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"text": "Extracted Text", "tables": [], "figures": []}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock PDF
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.width = 100
        mock_page.height = 200
        mock_page.to_image.return_value.original = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        
        extractor = VisionExtractor()
        result = extractor.extract("dummy.pdf")
        
        assert isinstance(result, ExtractedDocument)
        assert result.doc_id == "dummy.pdf"
        assert len(result.text_blocks) == 1
        assert result.text_blocks[0].content == "Extracted Text"
        assert result.total_pages == 1
        
        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "google/gemini-2.0-flash-001"
        assert kwargs["response_format"] == {"type": "json_object"}
