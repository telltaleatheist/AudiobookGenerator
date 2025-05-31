import os
import sys
import argparse
import re
import json
from pathlib import Path

def extract_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text.strip()
    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                return text.strip()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode text file with any common encoding")

def extract_from_epub(file_path, sections=None):
    """Extract text from EPUB file with optional section selection"""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("EPUB support requires: pip install ebooklib beautifulsoup4")
    
    try:
        book = epub.read_epub(file_path)
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
        # Convert to 0-indexed
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
            print(f"Warning: Could not extract text from section: {e}")
            continue
    
    if not extracted_text:
        raise ValueError("No text content extracted from EPUB")
    
    return ' '.join(extracted_text)

def extract_text_sections_for_interactive(file_path, selected_sections=None):
    """Extract text from file, keeping sections separate for interactive selection"""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.txt':
        # For TXT files, split on double newlines or common chapter markers
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to split intelligently
        sections = []
        if '\n\n' in content:
            sections = [s.strip() for s in content.split('\n\n') if s.strip()]
        else:
            # Fallback: split on chapter markers or just use the whole thing
            chapter_pattern = r'(?i)(?:^|\n)(?:chapter\s+\d+|ch\s+\d+|\d+\.)'
            if re.search(chapter_pattern, content):
                sections = re.split(chapter_pattern, content)
                sections = [s.strip() for s in sections if s.strip()]
            else:
                sections = [content]
        
        return sections
    
    elif file_path.suffix.lower() == '.epub':
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Interactive start requires: pip install ebooklib beautifulsoup4")
        
        try:
            book = epub.read_epub(file_path)
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
        if selected_sections:
            if max(selected_sections) > len(content_docs):
                raise ValueError(f"Section {max(selected_sections)} not found. Available: 1-{len(content_docs)}")
            # Convert to 0-indexed
            content_docs = [content_docs[i-1] for i in selected_sections if 1 <= i <= len(content_docs)]
        
        # Extract text from each document separately
        sections = []
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
                    sections.append(text)
                    
            except Exception as e:
                print(f"Warning: Could not extract text from section: {e}")
                continue
        
        return sections
    
    elif file_path.suffix.lower() == '.pdf':
        # Import PDF functionality from separate module
        try:
            from preprocessing_pdf import extract_pdf_sections_for_interactive
            return extract_pdf_sections_for_interactive(file_path, selected_sections)
        except ImportError:
            raise ImportError("PDF support requires: pip install pymupdf")
    
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

def split_section_into_subsections(section_text, chars_per_subsection=500):
    """Split a section into smaller subsections for granular start point selection"""
    # Split on sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', section_text)
    
    subsections = []
    current_subsection = ""
    
    for sentence in sentences:
        if len(current_subsection) + len(sentence) > chars_per_subsection and current_subsection:
            subsections.append(current_subsection.strip())
            current_subsection = sentence
        else:
            current_subsection += " " + sentence if current_subsection else sentence
    
    # Add the last subsection if it exists
    if current_subsection.strip():
        subsections.append(current_subsection.strip())
    
    return subsections

def show_word_selection(text, max_words=50):
    """Show words with numbers for precise selection"""
    words = text.split()
    
    # Show up to max_words for selection
    display_words = words[:max_words]
    
    print(f"\n📝 First {len(display_words)} words (showing positions):")
    print("-" * 60)
    
    # Display words in rows of 10 with numbers
    for i in range(0, len(display_words), 10):
        # Word numbers row
        numbers = []
        word_row = []
        
        for j in range(i, min(i + 10, len(display_words))):
            word_num = j + 1
            word = display_words[j]
            numbers.append(f"{word_num:3d}")
            word_row.append(f"{word:>3s}")
        
        print("  " + " ".join(numbers))
        print("  " + " ".join(word_row))
        print()
    
    if len(words) > max_words:
        remaining = len(words) - max_words
        print(f"... and {remaining} more words")
    
    return words

def interactive_word_selection(subsection_text):
    """Let user pick exact word to start from within a subsection"""
    print(f"\n🎯 Word-Level Start Selection")
    print("=" * 40)
    
    words = show_word_selection(subsection_text)
    
    if len(words) <= 1:
        print("✅ Only one word, using it")
        return 1
    
    max_word_num = min(50, len(words))  # Limit selection to first 50 words
    
    while True:
        try:
            choice = input(f"\n🎯 Start from which word? (1-{max_word_num}, 'show' to see text again, or 'q' to go back): ").strip()
            
            if choice.lower() == 'q':
                return None
            elif choice.lower() == 'show':
                show_word_selection(subsection_text)
                continue
            
            word_num = int(choice)
            if 1 <= word_num <= max_word_num:
                # Show preview of text starting from that word
                preview_words = words[word_num-1:word_num+10]  # Show 10 words from start point
                preview = " ".join(preview_words)
                if len(words) > word_num + 10:
                    preview += "..."
                
                print(f"\n📝 Text starting from word {word_num}: {preview}")
                
                confirm = input("❓ Start from this word? (y/N): ").strip().lower()
                if confirm == 'y':
                    return word_num
                else:
                    continue
            else:
                print(f"❌ Please enter a number between 1 and {max_word_num}")
        except ValueError:
            print("❌ Please enter a valid number, 'show', or 'q'")
    
    return None

def interactive_start_selection(file_path, selected_sections=None):
    """Interactive selection of start point within the text"""
    print("\n🎯 Interactive Start Point Selection")
    print("=" * 50)
    
    file_type = Path(file_path).suffix.lower()
    
    # For PDFs, use hierarchical selection
    if file_type == '.pdf':
        try:
            from preprocessing_pdf import interactive_hierarchical_start_selection
            return interactive_hierarchical_start_selection(file_path)
        except ImportError:
            raise ImportError("PDF support requires: pip install pymupdf")
    
    # For other file types, use the original section-based system
    # Extract sections
    try:
        sections = extract_text_sections_for_interactive(file_path, selected_sections)
    except Exception as e:
        print(f"❌ Could not extract sections: {e}")
        return None
    
    if not sections:
        print("❌ No sections found")
        return None
    
    section_label = "section"
    
    print(f"📖 Found {len(sections)} {section_label}(s)")
    
    # Let user choose which section to start from
    print(f"\n📋 Available {section_label}s:")
    for i, section in enumerate(sections, 1):
        # Show first 100 characters as preview
        preview = section[:100].replace('\n', ' ').strip()
        if len(section) > 100:
            preview += "..."
        print(f"  {i:2d}. {preview}")
    
    # Get section choice
    while True:
        try:
            section_choice = input(f"\n🎯 Which {section_label} to start from? (1-{len(sections)}, or 'q' to quit): ").strip()
            if section_choice.lower() == 'q':
                return None
            
            section_num = int(section_choice)
            if 1 <= section_num <= len(sections):
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(sections)}")
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
    
    # Selected section
    selected_section = sections[section_num - 1]
    
    # Split the selected section into subsections for granular control
    subsections = split_section_into_subsections(selected_section)
    
    if len(subsections) == 1:
        print(f"\n📝 {section_label.title()} {section_num} has only one subsection")
        
        # Go directly to word-level selection for the single subsection
        print(f"🎯 Selecting start word within {section_label} {section_num}")
        word_num = interactive_word_selection(subsections[0])
        
        if word_num is None:
            return None
        
        return {
            'section': section_num,
            'subsection': 1,
            'word': word_num,
            'start_from_section': section_num,
            'start_from_subsection': 1,
            'start_from_word': word_num
        }
    
    print(f"\n📝 {section_label.title()} {section_num} has {len(subsections)} subsections:")
    for i, subsection in enumerate(subsections, 1):
        # Show first 150 characters as preview
        preview = subsection[:150].replace('\n', ' ').strip()
        if len(subsection) > 150:
            preview += "..."
        print(f"  {i:2d}. {preview}")
    
    # Get subsection choice
    while True:
        try:
            subsection_choice = input(f"\n🎯 Which subsection to start from? (1-{len(subsections)}, or 'q' to quit): ").strip()
            if subsection_choice.lower() == 'q':
                return None
            
            subsection_num = int(subsection_choice)
            if 1 <= subsection_num <= len(subsections):
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(subsections)}")
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
    
    # Selected subsection
    selected_subsection = subsections[subsection_num - 1]
    
    print(f"\n✅ Selected: {section_label.title()} {section_num}, Subsection {subsection_num}")
    
    # Now get word-level selection
    print(f"🎯 Selecting start word within subsection {subsection_num}")
    word_num = interactive_word_selection(selected_subsection)
    
    if word_num is None:
        print("❌ Word selection cancelled")
        return None
    
    print(f"\n✅ Final selection: {section_label.title()} {section_num}, Subsection {subsection_num}, Word {word_num}")
    
    # Show final preview
    words = selected_subsection.split()
    start_words = words[word_num-1:word_num+15]  # Show 15 words from start point
    preview = " ".join(start_words)
    if len(words) > word_num + 15:
        preview += "..."
    
    print(f"📝 Will start from: '{preview}'")
    
    # Final confirmation
    confirm = input("\n❓ Confirm this exact start point? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Selection cancelled")
        return None
    
    return {
        'section': section_num,
        'subsection': subsection_num,
        'word': word_num,
        'start_from_section': section_num,
        'start_from_subsection': subsection_num,
        'start_from_word': word_num
    }

def apply_start_info(text_sections, start_info):
    """Apply start info to skip content before the selected start point"""
    if not start_info:
        return ' '.join(text_sections)
    
    # Handle hierarchical PDF selections
    if start_info.get('selection_type') in ['hierarchical_pdf', 'page_based_pdf']:
        # For PDF files, the extraction is handled in the PDF module
        # This function won't be called for PDF hierarchical selections
        return ' '.join(text_sections)
    
    section_num = start_info['start_from_section']
    subsection_num = start_info['start_from_subsection']
    word_num = start_info.get('start_from_word', 1)
    
    # Start from the selected section
    if section_num > len(text_sections):
        print(f"⚠️ Start section {section_num} not found, using full text")
        return ' '.join(text_sections)
    
    # Get sections from the start point onward
    remaining_sections = text_sections[section_num - 1:]
    
    # If we need to start from a specific subsection within the first section
    if subsection_num > 1 and remaining_sections:
        first_section = remaining_sections[0]
        subsections = split_section_into_subsections(first_section)
        
        if subsection_num <= len(subsections):
            # Replace first section with subsections from the start point
            remaining_subsections = subsections[subsection_num - 1:]
            
            # Apply word-level start point to the first subsection
            if word_num > 1 and remaining_subsections:
                first_subsection = remaining_subsections[0]
                words = first_subsection.split()
                
                if word_num <= len(words):
                    # Start from the specified word
                    remaining_words = words[word_num - 1:]
                    remaining_subsections[0] = ' '.join(remaining_words)
                    print(f"🎯 Started from section {section_num}, subsection {subsection_num}, word {word_num}")
                else:
                    print(f"⚠️ Start word {word_num} not found in subsection, using subsection from beginning")
            
            remaining_sections[0] = ' '.join(remaining_subsections)
        else:
            print(f"⚠️ Start subsection {subsection_num} not found, using section {section_num} from beginning")
    
    elif word_num > 1 and remaining_sections:
        # Apply word-level start point directly to the first section
        first_section = remaining_sections[0]
        words = first_section.split()
        
        if word_num <= len(words):
            remaining_words = words[word_num - 1:]
            remaining_sections[0] = ' '.join(remaining_words)
            print(f"🎯 Started from section {section_num}, word {word_num}")
        else:
            print(f"⚠️ Start word {word_num} not found, using section {section_num} from beginning")
    
    return ' '.join(remaining_sections)

def extract_section_candidates(source_file):
    """Extract section candidates from file"""
    sections = []
    path = Path(source_file)
    
    if path.suffix.lower() == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Split on double newlines
        sections = [s.strip() for s in text.split('\n\n') if s.strip()]
        
    elif path.suffix.lower() == '.epub':
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            
            book = epub.read_epub(str(path))
            items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            for item in items:
                soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                section_text = soup.get_text(separator=' ').strip()
                if section_text:
                    sections.append(section_text)
        except ImportError:
            raise ImportError("EPUB support requires: pip install ebooklib beautifulsoup4")
            
    elif path.suffix.lower() == '.pdf':
        try:
            from preprocessing_pdf import extract_pdf_sections_for_interactive
            sections = extract_pdf_sections_for_interactive(path)
        except ImportError:
            raise ImportError("PDF support requires: pip install pymupdf")
    else:
        raise ValueError("Unsupported file type")

    return sections

def list_epub_sections(file_path, output_json=False):
    """List available sections in EPUB file"""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("EPUB support requires: pip install ebooklib beautifulsoup4")
    
    try:
        book = epub.read_epub(file_path)
    except Exception as e:
        raise ValueError(f"Could not read EPUB file: {e}")
    
    content_docs = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content_docs.append(item)
    
    if not content_docs:
        raise ValueError("No readable sections found in EPUB")
    
    sections_info = []
    for i, doc in enumerate(content_docs, 1):
        try:
            content = doc.get_content().decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')
            
            title = "Untitled"
            for tag in ['h1', 'h2', 'h3', 'title']:
                heading = soup.find(tag)
                if heading and heading.get_text().strip():
                    title = heading.get_text().strip()[:50]
                    break
            
            # Get word count
            text = soup.get_text()
            word_count = len(text.split()) if text else 0
            
            section_info = {
                'section': i,
                'title': title,
                'filename': doc.get_name(),
                'word_count': word_count
            }
            sections_info.append(section_info)
            
        except Exception:
            sections_info.append({
                'section': i,
                'title': 'Error reading section',
                'filename': doc.get_name() if hasattr(doc, 'get_name') else 'unknown',
                'word_count': 0
            })
    
    if output_json:
        print(json.dumps(sections_info, indent=2))
    else:
        print(f"📖 Sections in {Path(file_path).name}:")
        for section in sections_info:
            print(f"  {section['section']:2d}. {section['title']} ({section['word_count']} words)")
        print(f"\nTotal: {len(sections_info)} sections")
    
    return sections_info

def list_pdf_sections(file_path, output_json=False):
    """List available chapters/pages in PDF file - DELEGATED TO PDF MODULE"""
    try:
        from preprocessing_pdf import list_pdf_sections as pdf_list_sections
        return pdf_list_sections(file_path, output_json)
    except ImportError:
        raise ImportError("PDF support requires: pip install pymupdf")

def get_epub_section_count(file_path):
    """Get number of sections in EPUB file"""
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        raise ImportError("EPUB support requires: pip install ebooklib beautifulsoup4")
    
    try:
        book = epub.read_epub(file_path)
        content_docs = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
        return len(content_docs)
    except Exception as e:
        raise ValueError(f"Could not read EPUB file: {e}")

def get_pdf_page_count(file_path):
    """Get number of pages in PDF file - DELEGATED TO PDF MODULE"""
    try:
        from preprocessing_pdf import get_pdf_page_count as pdf_page_count
        return pdf_page_count(file_path)
    except ImportError:
        raise ImportError("PDF support requires: pip install pymupdf")

def get_pdf_chapter_count(file_path):
    """Get number of chapters in PDF file - NEW FUNCTION"""
    try:
        from preprocessing_pdf import get_pdf_chapter_count as pdf_chapter_count
        return pdf_chapter_count(file_path)
    except ImportError:
        raise ImportError("PDF support requires: pip install pymupdf")

def parse_section_arguments(sections_args):
    """Parse section arguments like ['1', '3-5', '7'] into [1, 3, 4, 5, 7]"""
    if not sections_args:
        return None
    
    sections = []
    for arg in sections_args:
        if '-' in arg and not arg.startswith('-'):
            # Range like "3-7"
            try:
                start, end = map(int, arg.split('-'))
                sections.extend(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid section range: {arg}")
        else:
            # Single number
            try:
                sections.append(int(arg))
            except ValueError:
                raise ValueError(f"Invalid section number: {arg}")
    
    return sorted(list(set(sections)))

def extract_text(file_path, sections=None):
    """Main entry point - extract text from any supported file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.epub':
        return extract_from_epub(file_path, sections)
    elif suffix == '.txt':
        if sections:
            raise ValueError("Section selection not supported for TXT files")
        return extract_from_txt(file_path)
    elif suffix == '.pdf':
        # Delegate to PDF module
        try:
            from preprocessing_pdf import extract_from_pdf
            return extract_from_pdf(file_path, sections)
        except ImportError:
            raise ImportError("PDF support requires: pip install pymupdf")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def apply_phonetic_pronunciation_fixes(text):
    """Apply phonetic spelling fixes for non-SSML engines (Bark, F5, XTTS, EdgeTTS)"""
    print("STATUS: Applying phonetic pronunciation fixes...", file=sys.stderr)
    
    pronunciation_fixes = {
        # Religious/philosophical terms - NO HYPHENS to avoid pauses
        "atheist": "aytheeist",
        "atheists": "aytheeists", 
        "atheism": "aytheeism",
        "Jehovah's": "jehovas",
        
        # Common problem words - no hyphens/dashes
        "colonel": "kernel",
        "hierarchy": "hiyerarkey",
        "epitome": "ihpitomee",
        "hyperbole": "hyperbolee",
        "cache": "cash",
        "niche": "neesh",
        "facade": "fasahd",
        "gauge": "gayj",
        "receipt": "reeseet",
        "height": "hite",
        "leisure": "leezhur",
        
        # Religious/historical terms - no hyphens
        "bourgeois": "boorzhwah",
        "rendezvous": "rondayvoo",
        "regime": "rehzheem",
        "fascism": "fashism",
        "Nazi": "notsee",
        "Nazis": "notsees",
        "Aryan": "airy an",
        "pundits": "pundits",
        "ambiguous": "ambigyoous",
        "Christianity": "christianity",
        "religious": "rihliljus",
        
        # Geographic/names - no hyphens
        "Worcester": "wuster",
        "Leicester": "lester",
        "Arkansas": "arkansaw"
    }
    
    fixes_applied = 0
    for word, replacement in pronunciation_fixes.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        before_count = len(re.findall(pattern, text, flags=re.IGNORECASE))
        if before_count > 0:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            fixes_applied += before_count
    
    # Remove problematic names that cause repetition
    text = re.sub(r'\s+Owen Morgan\.?\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+[A-Z][a-z]+\s+[A-Z][a-z]+\.?\s*$', '', text)
    
    if fixes_applied > 0:
        print(f"STATUS: Applied {fixes_applied} phonetic pronunciation fixes", file=sys.stderr)
    
    return text

def clean_text_basic(text):
    """Apply universal text cleaning that works for all engines"""
    print("STATUS: Applying basic text cleaning...", file=sys.stderr)
    
    # Remove citations and references
    text = re.sub(r'\s*\[\d+\]', '', text)
    text = re.sub(r'\s*\(\d+\)', '', text)
    text = re.sub(r'(?<=[a-zA-Z,.])\s*\d{1,3}(?=[\s,.])', '', text)
    text = re.sub(r'\n\s*\d{1,3}\.\s+.*?(?=\n|$)', '', text)
    
    # Normalize whitespace and punctuation
    text = ' '.join(text.split())
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('—', ' -- ').replace('–', ' -- ').replace(' - ', ' -- ')
    
    # Common abbreviation fixes
    abbreviations = {
        "e.g.": "for example",
        "i.e.": "that is", 
        "etc.": "and so on",
        "vs.": "versus",
        "Dr.": "Doctor",
        "Mr.": "Mister", 
        "Mrs.": "Missus",
        "Ms.": "Miss",
    }
    
    for abbrev, expansion in abbreviations.items():
        pattern = r'\b' + re.escape(abbrev) + r'\b'
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    
    print("STATUS: Basic text cleaning completed", file=sys.stderr)
    return text.strip()

def save_clean_text(text, output_file):
    """Save cleaned text to output file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    word_count = len(text.split())
    print(f"STATUS: Saved clean text to {output_file} ({word_count:,} words)", file=sys.stderr)

def preprocess_file(input_file, output_file, config, sections=None):
    """Main preprocessing function - called by pipeline"""
    print(f"STATUS: Starting preprocessing...", file=sys.stderr)
    
    # Check if we have start_info from interactive selection
    start_info = config.get('preprocessing', {}).get('start_info')
    
    if start_info:
        file_type = Path(input_file).suffix.lower()
        
        # Handle PDF hierarchical selections
        if file_type == '.pdf' and start_info.get('selection_type') in ['hierarchical_pdf', 'page_based_pdf']:
            print(f"STATUS: Using PDF hierarchical selection", file=sys.stderr)
            try:
                from preprocessing_pdf import extract_pdf_from_hierarchical_selection
                text = extract_pdf_from_hierarchical_selection(input_file, start_info)
            except ImportError:
                raise ImportError("PDF support requires: pip install pymupdf")
        else:
            # Original section-based approach
            print(f"STATUS: Using interactive start point: Section {start_info['section']}, Subsection {start_info['subsection']}", file=sys.stderr)
            
            # Extract text sections separately to apply start_info
            text_sections = extract_text_sections_for_interactive(input_file, sections)
            text = apply_start_info(text_sections, start_info)
    else:
        # Normal extraction
        text = extract_text(input_file, sections)
    
    word_count = len(text.split())
    print(f"STATUS: Extracted {word_count:,} words from {Path(input_file).name}", file=sys.stderr)
    
    # Apply basic cleaning
    cleaned_text = clean_text_basic(text)
    
    # Apply phonetic pronunciation fixes for all engines
    cleaned_text = apply_phonetic_pronunciation_fixes(cleaned_text)
    
    # Save cleaned text
    save_clean_text(cleaned_text, output_file)
    
    print(f"STATUS: Preprocessing complete", file=sys.stderr)
    return True

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Universal text preprocessing")
    parser.add_argument("input_file", help="Input EPUB, PDF, or TXT file")
    parser.add_argument("output_file", nargs='?', help="Output clean text file")
    parser.add_argument("--sections", nargs="*", help="Section numbers for EPUB/PDF (e.g., 1 2 3 or 1-5)")
    parser.add_argument("--chapters", nargs="*", help="Chapter numbers for PDF (e.g., 1 2 3 or 1-5)")
    parser.add_argument("--list", action="store_true", help="List available sections/chapters/pages")
    parser.add_argument("--list-json", action="store_true", help="List sections/chapters/pages as JSON")
    parser.add_argument("--interactive-start", action="store_true", help="Interactive start point selection")
    
    args = parser.parse_args()
    
    # Handle section listing
    if args.list or args.list_json:
        file_path = Path(args.input_file)
        if file_path.suffix.lower() == '.epub':
            try:
                list_epub_sections(args.input_file, output_json=args.list_json)
                return 0
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1
        elif file_path.suffix.lower() == '.pdf':
            try:
                list_pdf_sections(args.input_file, output_json=args.list_json)
                return 0
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1
        else:
            print("ERROR: Section listing only available for EPUB and PDF files", file=sys.stderr)
            return 1
    
    # Handle interactive start
    if args.interactive_start:
        try:
            sections = None
            if args.sections:
                sections = parse_section_arguments(args.sections)
            elif args.chapters:
                sections = parse_section_arguments(args.chapters)
            
            start_info = interactive_start_selection(args.input_file, sections)
            if start_info:
                print(f"\n✅ Selected start point saved")
                print(f"📝 You can use this in your pipeline configuration")
                return 0
            else:
                print("❌ No start point selected")
                return 1
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    
    # Validate arguments for normal processing
    if not args.output_file:
        parser.error("output_file required for text extraction")
    
    if not Path(args.input_file).exists():
        print(f"ERROR: Input file not found: {args.input_file}", file=sys.stderr)
        return 1
    
    # Parse sections/chapters
    sections = None
    if args.sections:
        try:
            sections = parse_section_arguments(args.sections)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    elif args.chapters:
        try:
            sections = parse_section_arguments(args.chapters)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    
    # Run preprocessing
    try:
        success = preprocess_file(args.input_file, args.output_file, {}, sections)
        return 0 if success else 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())