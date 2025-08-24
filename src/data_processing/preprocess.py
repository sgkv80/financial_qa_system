"""
preprocess.py

Class-based module for extracting and cleaning raw PDFs for RAG.
Uses configurations from YAML files and logs all processing steps.
"""

import os, io
import re
import json

from .enhanced_text_cleaner import TextCleaner

from pdf2image import convert_from_path
import pytesseract  # For OCR on images
import PyPDF2  # For extracting text from PDF files

from utils.logger import get_logger
from utils.config_loader import load_config, get_root_dir

from collections import Counter
from typing import List, Tuple, Dict  # For type annotations

class Preprocessor:
    """
    Handles extraction and preprocessing of raw PDF files.
    """

    def __init__(self, base_config_path: str = "configs/base_config.yaml"):
        """
        Initialize Preprocessor with configurations.
        """
        self.config = load_config(base_config_path)
        self.input_dir   = get_root_dir() / self.config["paths"]["raw_data"]
        self.output_dir  = get_root_dir() / self.config["paths"]["processed_data"]
        self.logger = get_logger(self.__class__.__name__)

        self.text_cleaner = TextCleaner()


    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from PDF using OCR and PyPDF2.
        Args: pdf_path (str): Path to PDF file.
        Returns: str: Extracted text.
        """
        self.logger.info(f"Extracted text from {pdf_path}")
        text = ""  # Initialize text variable
        try:
            with open(pdf_path, 'rb') as f:  # Open PDF file in binary mode
                reader = PyPDF2.PdfReader(f)  # Create PDF reader object
                num_pages = len(reader.pages)  # Get number of pages
                for i, page in enumerate(reader.pages):  # Iterate over each page
                    page_text = page.extract_text()  # Try to extract text from page
                    if page_text and page_text.strip():  # If text is found
                        text += page_text  # Add to text
                    else:
                        # OCR only if no text extracted
                        images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)  # Convert page to image
                        for img in images:  # Iterate over images (should be one per page)
                            ocr_text = pytesseract.image_to_string(img)  # Extract text using OCR
                            text += ocr_text  # Add OCR text to text
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")  # Log error if extraction fails
        self.logger.info(f"Extracted {len(text)} characters from {pdf_path}")  # Log number of characters extracted
        return text  # Return extracted text



    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing unwanted formatting.
        """
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"Page \d+", "", text)
        self.logger.debug("Text cleaned successfully.")
        return text.strip()





    
    def extract_segment_sections(self, text: str) -> Dict[str, str]:
        """
        Segments cleaned text into logical financial sections.
        Args: text (str): Cleaned document text.
        Returns: Dict[str, str]: Section name to text mapping.
        """
        self.logger.info(f'segment_sections called with text: {text[:100]}...')  # Log function call
        sections = {}  # Dictionary to store sections
        patterns = {
            'income_statement': r'(?i)(income statement|statement of income|consolidated statements of operations|statements of earnings)[\s\S]*?(?=balance sheet|statement of financial position|cash flow|statement of cash flows|shareholders|comprehensive|notes|management discussion|risk factors|auditor|summary|$)',
            'balance_sheet': r'(?i)(balance sheet|statement of financial position|consolidated balance sheets)[\s\S]*?(?=income statement|statement of income|cash flow|statement of cash flows|shareholders|comprehensive|notes|management discussion|risk factors|auditor|summary|$)',
            'cash_flow': r'(?i)(cash flow|statement of cash flows|consolidated statements of cash flows)[\s\S]*?(?=income statement|balance sheet|statement of income|statement of financial position|shareholders|comprehensive|notes|management discussion|risk factors|auditor|summary|$)',
            'shareholders_equity': r'(?i)(statement of shareholders\' equity|statement of stockholders\' equity|consolidated statements of shareholders\' equity)[\s\S]*?(?=income statement|balance sheet|cash flow|comprehensive|notes|management discussion|risk factors|auditor|summary|$)',
            'comprehensive_income': r'(?i)(statement of comprehensive income|consolidated statements of comprehensive income)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|notes|management discussion|risk factors|auditor|summary|$)',
            'notes': r'(?i)(notes to (the )?financial statements|notes)[\s\S]*?(?=management discussion|risk factors|auditor|summary|$)',
            'management_discussion': r'(?i)(management\'s discussion and analysis|md&a|management discussion)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|comprehensive|notes|risk factors|auditor|summary|$)',
            'risk_factors': r'(?i)(risk factors|risks related to)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|comprehensive|notes|management discussion|auditor|summary|$)',
            'auditor_report': r'(?i)(independent auditor\'s report|auditor\'s report|report of independent registered public accounting firm)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|comprehensive|notes|management discussion|risk factors|summary|$)',
            'summary_of_operations': r'(?i)(summary of operations|highlights|financial highlights)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|comprehensive|notes|management discussion|risk factors|auditor|$)',
            'liquidity_and_capital_resources': r'(?i)(liquidity and capital resources)[\s\S]*?(?=legal proceedings|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|human capital|forward-looking statements|$)',
            'legal_proceedings': r'(?i)(legal proceedings)[\s\S]*?(?=liquidity and capital resources|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|human capital|forward-looking statements|$)',
            'critical_accounting_estimates': r'(?i)(critical accounting estimates)[\s\S]*?(?=liquidity and capital resources|legal proceedings|overview|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|human capital|forward-looking statements|$)',
            'overview': r'(?i)(overview)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|human capital|forward-looking statements|$)',
            'technology_and_infrastructure': r'(?i)(technology and infrastructure)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|executive officers and directors|seasonality|business and industry risks|human capital|forward-looking statements|$)',
            'executive_officers_and_directors': r'(?i)(executive officers and directors)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|technology and infrastructure|seasonality|business and industry risks|human capital|forward-looking statements|$)',
            'seasonality': r'(?i)(seasonality)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|business and industry risks|human capital|forward-looking statements|$)',
            'business_and_industry_risks': r'(?i)(business and industry risks)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|seasonality|human capital|forward-looking statements|$)',
            'human_capital': r'(?i)(human capital)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|forward-looking statements|$)',
            'forward_looking_statements': r'(?i)(forward-looking statements)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|human capital|$)'
        }
        for name, pat in patterns.items():  # Iterate over each section pattern
            match = re.search(pat, text)
            if match:
                sections[name] = match.group(0)
            self.logger.info(f'Section found: {name}')
        # Log up to 10 segments with their first 100 characters
        self.logger.info('--- Segmentation Preview (up to 10 sections) ---')
        for i, (seg_name, seg_text) in enumerate(sections.items()):
            if i >= 10:
                break
            preview = seg_text[:100].replace('\n', ' ')
            self.logger.info(f'Segment {i+1}: {seg_name} | Preview: {preview}')
        # Detect and log all section headings in the cleaned text
        heading_pattern = r'(?m)^([A-Z][A-Za-z\'\-\s&]+):?$'
        headings = set(re.findall(heading_pattern, text))
        #logger.info(f"Detected section headings in document: {sorted(headings)}")
        self.logger.info(f"Segmented sections: {list(sections.keys())}")
        return sections


    def preprocess_pdfs(self):
        """
        Process all PDFs from raw_data directory, clean, and save as a single corpus.
        """
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        combined_text = ""

        self.logger.info(f"Preprocessing PDFs in directory: {self.input_dir}")
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.input_dir, filename)
                
                pdf_base_name = os.path.splitext(filename)[0]

                raw_text = self.extract_text_from_pdf(pdf_path)
                cleaned_text = self.text_cleaner.process(raw_text) #self.clean_text(raw_text)
                
                segment_sections  = self.extract_segment_sections(cleaned_text)

                with open(os.path.join(self.output_dir, f'{pdf_base_name}.clean_text'), "w", encoding="utf-8") as f:
                    f.write(cleaned_text)

                with open(os.path.join(self.output_dir, f'{pdf_base_name}.segment_sections'), "w", encoding="utf-8") as json_file:
                    json.dump(segment_sections, json_file, indent=4)

                self.logger.info(f"Processed {filename}")


        self.logger.info(f"Preprocessed text and sections saved to ")
