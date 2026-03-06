import pytest
from PIL import Image
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
        
        # Mock API response (moondream returns raw desc, not JSON format)
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='A highly detailed visual description of the extracted document containing more than 50 characters to pass the sanity check.'))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock PDF
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.width = 100
        mock_page.height = 200
        mock_image = Image.new('RGB', (100, 200), color = 'white')
        mock_page.to_image.return_value.original = mock_image
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        
        extractor = VisionExtractor()
        result = extractor.extract("dummy.pdf")
        
        assert isinstance(result, ExtractedDocument)
        assert result.doc_id == "dummy"
        assert len(result.text_blocks) == 1
        assert "visual description" in result.text_blocks[0].content
        assert result.total_pages == 1
        
        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "moondream"
