import re
import unicodedata
from typing import Dict, Any, Optional

from utils.logger import get_logger

class TextCleaner():
    """Enhanced text cleaner for financial documents with Unicode handling"""
    
    def __init__(self):

        self.logger = get_logger(self.__class__.__name__)
        
        # Unicode character mappings for common PDF artifacts
        self.unicode_replacements = {
            # Smart quotes
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark  
            '\u201a': "'",  # Single low-9 quotation mark
            '\u201b': "'",  # Single high-reversed-9 quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u201e': '"',  # Double low-9 quotation mark
            '\u201f': '"',  # Double high-reversed-9 quotation mark
            
            # Dashes and hyphens
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2015': '-',  # Horizontal bar
            '\u2212': '-',  # Minus sign
            
            # Spaces and separators
            '\u00a0': ' ',  # Non-breaking space
            '\u2000': ' ',  # En quad
            '\u2001': ' ',  # Em quad
            '\u2002': ' ',  # En space
            '\u2003': ' ',  # Em space
            '\u2004': ' ',  # Three-per-em space
            '\u2005': ' ',  # Four-per-em space
            '\u2006': ' ',  # Six-per-em space
            '\u2007': ' ',  # Figure space
            '\u2008': ' ',  # Punctuation space
            '\u2009': ' ',  # Thin space
            '\u200a': ' ',  # Hair space
            '\u200b': '',   # Zero width space
            '\u200c': '',   # Zero width non-joiner
            '\u200d': '',   # Zero width joiner
            
            # Bullets and symbols
            '\u2022': '•',  # Bullet
            '\u2023': '•',  # Triangular bullet
            '\u2024': '.',  # One dot leader
            '\u2025': '..',  # Two dot leader
            '\u2026': '...',  # Horizontal ellipsis
            
            # Financial symbols
            '\u00a2': 'cents',  # Cent sign
            '\u00a3': 'GBP',    # Pound sign
            '\u00a4': '$',      # Generic currency symbol
            '\u00a5': 'JPY',    # Yen sign
            '\u20ac': 'EUR',    # Euro sign
            
            # Mathematical symbols
            '\u00b1': '+/-',    # Plus-minus sign
            '\u00d7': 'x',      # Multiplication sign
            '\u00f7': '/',      # Division sign
            '\u2190': '<-',     # Leftwards arrow
            '\u2192': '->',     # Rightwards arrow
            '\u2194': '<->',    # Left right arrow
        }
        
        # Regex patterns for cleaning
        self.cleaning_patterns = [
            # Remove page headers/footers patterns
            (r'Page \d+ of \d+', ''),
            (r'^\d+\s*$', ''),  # Standalone page numbers
            
            # Remove excessive whitespace
            (r'\s+', ' '),      # Multiple spaces to single space
            (r'\n\s*\n\s*\n+', '\n\n'),  # Multiple newlines to double
            
            # Fix common OCR errors
            (r'\b(\d+)\s*,\s*(\d{3})\b', r'\1,\2'),  # Fix number formatting
            (r'\$\s+(\d)', r'$\1'),  # Fix currency spacing
            
            # Remove unwanted characters but keep structure
            (r'[^\w\s\$\.\,\%\(\)\-\+\=\:\;\!\?\"\'\n\t]', ''),
            
            # Clean up sentence boundaries
            (r'\.{2,}', '.'),   # Multiple periods
            (r'\?{2,}', '?'),   # Multiple question marks
            (r'\!{2,}', '!'),   # Multiple exclamation marks
        ]
    
    def process(self, text: str) -> str:
        """Main processing function"""
        try:
            cleaned_text = self.clean_unicode_characters(text)
            cleaned_text = self.apply_cleaning_patterns(cleaned_text)
            cleaned_text = self.normalize_financial_text(cleaned_text)
            cleaned_text = self.final_cleanup(cleaned_text)
            
            self.logger.info(f"Text cleaning completed. Original length: {len(text)}, "
                           f"Cleaned length: {len(cleaned_text)}")
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error in text cleaning: {e}")
            return text  # Return original text if cleaning fails
    
    def clean_unicode_characters(self, text: str) -> str:
        """Replace Unicode characters with ASCII equivalents"""
        
        # Method 1: Direct replacement using mapping
        for unicode_char, replacement in self.unicode_replacements.items():
            text = text.replace(unicode_char, replacement)
        
        # Method 2: Handle remaining Unicode escape sequences
        def replace_unicode_escapes(match):
            unicode_str = match.group(0)
            try:
                # Convert unicode escape to actual character, then to ASCII
                char = unicode_str.encode().decode('unicode_escape')
                # Try to find ASCII equivalent
                normalized = unicodedata.normalize('NFKD', char)
                ascii_char = normalized.encode('ascii', 'ignore').decode('ascii')
                return ascii_char if ascii_char else ''
            except:
                return ''
        
        # Replace remaining \uXXXX patterns
        text = re.sub(r'\\u[0-9a-fA-F]{4}', replace_unicode_escapes, text)
        
        return text
    
    def apply_cleaning_patterns(self, text: str) -> str:
        """Apply regex cleaning patterns"""
        for pattern, replacement in self.cleaning_patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def normalize_financial_text(self, text: str) -> str:
        """Normalize financial-specific text patterns"""
        
        # Normalize currency representations
        financial_patterns = [
            # Standardize currency formats
            (r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:million|Million|MILLION|M)\b', 
             r'$\1 million'),
            (r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:billion|Billion|BILLION|B)\b', 
             r'$\1 billion'),
            (r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:thousand|Thousand|THOUSAND|K)\b', 
             r'$\1 thousand'),
            
            # Normalize percentage
            (r'(\d+(?:\.\d+)?)\s*%', r'\1%'),
            
            # Fix decimal alignment
            (r'(\d+)\s*\.\s*(\d+)', r'\1.\2'),
            
            # Normalize date formats
            (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', r'\1/\2/\3'),
            
            # Clean up financial statement headers
            (r'(?i)(consolidated\s+statements?\s+of)', r'Consolidated Statements of'),
            (r'(?i)(balance\s+sheets?)', r'Balance Sheet'),
            (r'(?i)(income\s+statements?)', r'Income Statement'),
            (r'(?i)(cash\s+flows?)', r'Cash Flow'),
        ]
        
        for pattern, replacement in financial_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def final_cleanup(self, text: str) -> str:
        """Final cleanup and validation"""
        
        # Remove lines that are mostly noise
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip lines that are likely noise
            if (len(line) < 3 or 
                line.isdigit() or 
                re.match(r'^[\s\-\_\=\.\,]+$', line) or
                len(re.findall(r'[a-zA-Z]', line)) < 2):
                continue
                
            cleaned_lines.append(line)
        
        # Rejoin with proper spacing
        text = '\n'.join(cleaned_lines)
        
        # Final whitespace cleanup
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)     # Multiple spaces to single
        text = text.strip()
        
        return text
    
    def remove_headers_footers(self, text: str, 
                              header_indicators: Optional[list] = None,
                              footer_indicators: Optional[list] = None) -> str:
        """Remove common headers and footers from financial documents"""
        
        if header_indicators is None:
            header_indicators = [
                'Amazon.com, Inc.',
                'AMAZON.COM, INC.',
                'Annual Report',
                'Form 10-K',
                'SEC Filing',
                'Table of Contents'
            ]
        
        if footer_indicators is None:
            footer_indicators = [
                'Page',
                'Continued on next page',
                'See accompanying notes',
                'Notes to Financial Statements'
            ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Skip headers
            if any(indicator.lower() in line_lower for indicator in header_indicators):
                continue
                
            # Skip footers
            if any(indicator.lower() in line_lower for indicator in footer_indicators):
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """Get statistics about the cleaning process"""
        
        unicode_chars_removed = len([c for c in original_text if ord(c) > 127])
        
        return {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'reduction_percentage': ((len(original_text) - len(cleaned_text)) / len(original_text)) * 100,
            'unicode_chars_removed': unicode_chars_removed,
            'lines_original': len(original_text.split('\n')),
            'lines_cleaned': len(cleaned_text.split('\n'))
        }


# Utility function for quick cleaning
def quick_clean_text(text: str) -> str:
    """Quick text cleaning function for immediate use"""
    
    # Quick Unicode replacements
    unicode_map = {
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u00a0': ' ', '\u2026': '...'
    }
    
    for unicode_char, replacement in unicode_map.items():
        text = text.replace(unicode_char, replacement)
    
    # Remove escape sequences
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text
