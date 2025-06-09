#!/usr/bin/env python3
"""
File Extractors - Unified text extraction from PDF, EPUB, and TXT files
FIXED: Added proper imports
"""

import sys
import re
from pathlib import Path
from typing import Optional, List
from core.progress_display_manager import log_info, log_status

def extract_text(file_path: str, sections: Optional[List[int]] = None, config: dict = None) -> str:
    """Main entry point - extract text from any supported file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.txt':
        return extract_from_txt(file_path)
    elif suffix == '.epub':
        return extract_from_epub(file_path, sections)
    elif suffix == '.pdf':
        return extract_from_pdf(file_path, sections, config)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def extract_from_txt(file_path: Path) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode text file with any common encoding")

def extract_from_epub(file_path: Path, sections: Optional[List[int]] = None) -> str:
    """Extract text from EPUB file with optional section selection"""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("EPUB support requires: pip install ebooklib beautifulsoup4")
    
    try:
        book = epub.read_epub(str(file_path))
    except Exception as e:
        raise ValueError(f"Could not read EPUB file: {e}")
    
    # Get all content documents
    content_docs = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content_docs.append(item)
    
    if not content_docs:
        raise ValueError("No readable content found in EPUB")
    
    # Filter by sections if specified
    if sections:
        if max(sections) > len(content_docs):
            raise ValueError(f"Section {max(sections)} not found. Available: 1-{len(content_docs)}")
        content_docs = [content_docs[i-1] for i in sections if 1 <= i <= len(content_docs)]
    
    # Extract text from selected documents
    extracted_text = []
    for doc in content_docs:
        try:
            content = doc.get_content().decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if text.strip():
                extracted_text.append(text)
                
        except Exception as e:
            log_info(f"Warning: Could not extract text from section: {e}")
            continue
    
    if not extracted_text:
        raise ValueError("No text content extracted from EPUB")
    
    return ' '.join(extracted_text)

def extract_from_pdf(file_path: Path, sections: Optional[List[int]] = None, config: dict = None) -> str:
    """Extract text from PDF file - FIXED implementation"""
    try:
        import fitz  # type: ignore # PyMuPDF
    except ImportError:
        raise ImportError("PDF support requires: pip install pymupdf")
    
    try:
        doc = fitz.open(str(file_path))
        
        if doc.is_encrypted:
            doc.close()
            raise ValueError("PDF is encrypted or password-protected")
        
        total_pages = len(doc)
        if not total_pages:
            doc.close()
            raise ValueError("PDF has no pages")
        
        # Determine which pages to extract
        if sections:
            # Try to detect if we have chapters first
            chapters = _get_pdf_chapters(doc)
            
            if chapters and len(chapters) > 1:
                # Use chapters
                log_status(f"PDF has {len(chapters)} chapters, using chapter-based extraction")
                page_indices = _get_pages_from_chapters(chapters, sections)
            else:
                # Use page numbers directly
                log_status("Using page-based extraction")
                if max(sections) > total_pages:
                    doc.close()
                    raise ValueError(f"Page {max(sections)} not found. Available: 1-{total_pages}")
                page_indices = [i-1 for i in sections if 1 <= i <= total_pages]
        else:
            # Extract all pages
            page_indices = range(total_pages)
        
        # Extract text from selected pages
        extracted_text = []
        for page_idx in page_indices:
            try:
                page = doc[page_idx]
                text = page.get_text()
                
                if text and text.strip():
                    # Clean the text
                    text = _clean_pdf_text(text)
                    if text:
                        extracted_text.append(text)
                        
            except Exception as e:
                log_info(f"Warning: Could not extract text from page {page_idx + 1}")
                continue
        
        doc.close()
        
        if not extracted_text:
            raise ValueError("No text content extracted from PDF")
        
        result = ' '.join(extracted_text)
        log_status(f"Extracted {len(result.split()):,} words from PDF")
        return result
        
    except Exception as e:
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            raise ValueError("PDF is encrypted or password-protected")
        else:
            raise ValueError(f"Could not read PDF file: {e}")

def _get_pdf_chapters(doc):
    """Get chapters from PDF outline/bookmarks"""
    try:
        outline = doc.get_toc()
        if not outline:
            return []
        
        chapters = []
        for level, title, page_num in outline:
            chapters.append({
                'number': len(chapters) + 1,
                'title': title.strip(),
                'start_page': page_num,
                'level': level
            })
        
        # Calculate end pages
        for i, chapter in enumerate(chapters):
            if i + 1 < len(chapters):
                chapter['end_page'] = chapters[i + 1]['start_page'] - 1
            else:
                chapter['end_page'] = len(doc)
        
        return chapters
    except Exception:
        return []

def _get_pages_from_chapters(chapters, chapter_numbers):
    """Get page indices from chapter numbers"""
    page_indices = []
    
    for chapter_num in chapter_numbers:
        if 1 <= chapter_num <= len(chapters):
            chapter = chapters[chapter_num - 1]
            start_page = chapter['start_page'] - 1  # Convert to 0-indexed
            end_page = min(chapter['end_page'], len(chapters))
            
            page_indices.extend(range(start_page, end_page))
    
    return sorted(set(page_indices))

def _clean_pdf_text(text: str) -> str:
    """Clean extracted PDF text"""
    if not text or not text.strip():
        return ""
    
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    
    # Remove common PDF artifacts
    text = re.sub(r'\x00', '', text)  # Null characters
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f]', '', text)  # Control characters
    
    # Remove page numbers and headers/footers (simple heuristics)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip very short lines that might be page numbers
        if len(line) < 3:
            continue
        
        # Skip lines that are just numbers (likely page numbers)
        if line.isdigit():
            continue
        
        # Skip common header/footer patterns
        if re.match(r'^(page|p\.)\s*\d+', line, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)
    
    return ' '.join(cleaned_lines).strip()
