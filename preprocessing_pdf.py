#!/usr/bin/env python3
"""
PDF Preprocessing - Dedicated PDF text extraction using PyMuPDF
Now with hierarchical Chapter → Page → Word selection
"""

import sys
import json
import re
from pathlib import Path

def get_pdf_chapters(file_path):
    """Get chapters using the best available method"""
    try:
        import fitz
    except ImportError:
        raise ImportError("PDF support requires: pip install pymupdf")
    
    # Try PDF bookmarks first (most reliable)
    try:
        doc = fitz.open(str(file_path))
        outline = doc.get_toc()
        doc.close()
        
        if outline:
            chapters = []
            for level, title, page_num in outline:
                chapters.append({
                    'number': len(chapters) + 1,
                    'title': title.strip(),
                    'start_page': page_num,
                    'level': level,
                    'type': 'bookmark'
                })
            
            # Calculate end pages
            for i, chapter in enumerate(chapters):
                if i + 1 < len(chapters):
                    chapter['end_page'] = chapters[i + 1]['start_page'] - 1
                else:
                    # Last chapter goes to end of document
                    doc = fitz.open(str(file_path))
                    chapter['end_page'] = len(doc)
                    doc.close()
            
            print(f"STATUS: Found {len(chapters)} chapters from PDF bookmarks", file=sys.stderr)
            return chapters
    except Exception as e:
        print(f"WARNING: Could not extract PDF bookmarks: {e}", file=sys.stderr)
    
    # Fallback: Text pattern detection
    try:
        chapters = detect_chapters_by_patterns(file_path)
        if chapters:
            print(f"STATUS: Found {len(chapters)} chapters from text patterns", file=sys.stderr)
            return chapters
    except Exception as e:
        print(f"WARNING: Text pattern detection failed: {e}", file=sys.stderr)
    
    # Final fallback: No chapters, use full document
    print(f"STATUS: No chapters detected, treating as single document", file=sys.stderr)
    try:
        doc = fitz.open(str(file_path))
        total_pages = len(doc)
        doc.close()
        
        return [{
            'number': 1,
            'title': 'Full Document',
            'start_page': 1,
            'end_page': total_pages,
            'level': 1,
            'type': 'fallback'
        }]
    except Exception:
        return []

def detect_chapters_by_patterns(file_path):
    """Detect chapters using text patterns"""
    try:
        import fitz
        doc = fitz.open(str(file_path))
        
        # Common chapter patterns
        patterns = [
            r'^\s*CHAPTER\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|\d+|[IVXLCDM]+)\s*$',
            r'^\s*Chapter\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+|[IVXLCDM]+)\s*$',
            r'^\s*CH\s*\.?\s*\d+\s*$',
            r'^\s*\d+\s*\.\s*[A-Z][A-Za-z\s]{3,50}\s*$',
            r'^\s*[IVXLCDM]+\s*\.\s*[A-Z][A-Za-z\s]{3,50}\s*$',
        ]
        
        chapters = []
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            
            if not text:
                continue
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                for pattern in patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        chapters.append({
                            'number': len(chapters) + 1,
                            'title': line,
                            'start_page': page_num + 1,
                            'level': 1,
                            'type': 'text_pattern'
                        })
                        break
        
        # Calculate end pages
        for i, chapter in enumerate(chapters):
            if i + 1 < len(chapters):
                chapter['end_page'] = chapters[i + 1]['start_page'] - 1
            else:
                chapter['end_page'] = total_pages
        
        doc.close()
        return chapters
        
    except Exception as e:
        raise ValueError(f"Could not detect chapters by patterns: {e}")

# BACKWARD COMPATIBILITY: Keep old function names but delegate to new system
def extract_from_pdf(file_path, sections=None):
    """Extract text from PDF - now supports both chapters and pages"""
    if sections:
        # Check if we're dealing with chapter numbers or page numbers
        chapters = get_pdf_chapters(file_path)
        if chapters and len(chapters) > 1:
            # Assume sections are chapter numbers if we have multiple chapters
            return extract_pdf_by_chapters(file_path, sections)
        else:
            # Fall back to page-based extraction
            return extract_pdf_by_pages(file_path, sections)
    else:
        # Extract everything
        return extract_pdf_by_pages(file_path, None)

def extract_pdf_by_pages(file_path, page_numbers=None):
    """Extract text from PDF by page numbers (original functionality)"""
    try:
        import fitz
        doc = fitz.open(str(file_path))
        
        if doc.is_encrypted:
            doc.close()
            raise ValueError("PDF is encrypted or password-protected")
        
        total_pages = len(doc)
        if not total_pages:
            doc.close()
            raise ValueError("PDF has no pages")
        
        # Filter by pages if specified
        if page_numbers:
            if max(page_numbers) > total_pages:
                doc.close()
                raise ValueError(f"Page {max(page_numbers)} not found. Available: 1-{total_pages}")
            page_indices = [i-1 for i in page_numbers if 1 <= i <= total_pages]
        else:
            page_indices = range(total_pages)
        
        # Extract text from selected pages
        extracted_text = []
        for page_idx in page_indices:
            try:
                page = doc[page_idx]
                text = page.get_text()
                
                if text and text.strip():
                    text = ' '.join(text.split())
                    extracted_text.append(text)
                    
            except Exception as e:
                print(f"Warning: Could not extract text from page {page_idx + 1}: {e}", file=sys.stderr)
                continue
        
        doc.close()
        
        if not extracted_text:
            raise ValueError("No text content extracted from PDF")
        
        return ' '.join(extracted_text)
        
    except Exception as e:
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            raise ValueError("PDF is encrypted or password-protected")
        else:
            raise ValueError(f"Could not read PDF file: {e}")

def extract_pdf_by_chapters(file_path, chapter_numbers=None):
    """Extract text from PDF by chapter numbers"""
    chapters = get_pdf_chapters(file_path)
    
    if not chapters:
        raise ValueError("No chapters detected in PDF")
    
    # Filter by requested chapters
    if chapter_numbers:
        if max(chapter_numbers) > len(chapters):
            raise ValueError(f"Chapter {max(chapter_numbers)} not found. Available: 1-{len(chapters)}")
        
        selected_chapters = [chapters[i-1] for i in chapter_numbers if 1 <= i <= len(chapters)]
    else:
        selected_chapters = chapters
    
    # Extract text from chapter ranges
    try:
        import fitz
        doc = fitz.open(str(file_path))
        
        extracted_text = []
        
        for chapter in selected_chapters:
            start_page = chapter['start_page'] - 1  # Convert to 0-indexed
            end_page = min(chapter['end_page'], len(doc))  # Ensure we don't exceed document
            
            # Extract text from chapter pages
            chapter_text = []
            for page_idx in range(start_page, end_page):
                try:
                    page = doc[page_idx]
                    text = page.get_text()
                    if text and text.strip():
                        chapter_text.append(' '.join(text.split()))
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_idx + 1}: {e}", file=sys.stderr)
                    continue
            
            if chapter_text:
                extracted_text.append(' '.join(chapter_text))
        
        doc.close()
        
        if not extracted_text:
            raise ValueError("No text content extracted from selected chapters")
        
        return ' '.join(extracted_text)
        
    except Exception as e:
        raise ValueError(f"Could not extract text by chapters: {e}")

def extract_pdf_sections_for_interactive(file_path, selected_sections=None):
    """Extract chapters as sections for interactive selection"""
    chapters = get_pdf_chapters(file_path)
    
    if not chapters:
        # Fall back to page-based sections
        return extract_pdf_pages_as_sections(file_path, selected_sections)
    
    # Filter by selected chapters if specified
    if selected_sections:
        if max(selected_sections) > len(chapters):
            raise ValueError(f"Chapter {max(selected_sections)} not found. Available: 1-{len(chapters)}")
        chapters = [chapters[i-1] for i in selected_sections if 1 <= i <= len(chapters)]
    
    # Extract text from each chapter separately
    try:
        import fitz
        doc = fitz.open(str(file_path))
        
        sections = []
        for chapter in chapters:
            start_page = chapter['start_page'] - 1
            end_page = min(chapter['end_page'], len(doc))
            
            chapter_text = []
            for page_idx in range(start_page, end_page):
                try:
                    page = doc[page_idx]
                    text = page.get_text()
                    if text and text.strip():
                        chapter_text.append(' '.join(text.split()))
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_idx + 1}: {e}", file=sys.stderr)
                    continue
            
            if chapter_text:
                sections.append(' '.join(chapter_text))
        
        doc.close()
        return sections
        
    except Exception as e:
        raise ValueError(f"Could not extract chapter sections: {e}")

def extract_pdf_pages_as_sections(file_path, selected_pages=None):
    """Fallback: Extract pages as sections when no chapters detected"""
    try:
        import fitz
        doc = fitz.open(str(file_path))
        
        if doc.is_encrypted:
            doc.close()
            raise ValueError("PDF is encrypted or password-protected")
        
        total_pages = len(doc)
        if not total_pages:
            doc.close()
            raise ValueError("PDF has no pages")
        
        # Filter by pages if specified
        if selected_pages:
            if max(selected_pages) > total_pages:
                doc.close()
                raise ValueError(f"Page {max(selected_pages)} not found. Available: 1-{total_pages}")
            page_indices = [i-1 for i in selected_pages if 1 <= i <= total_pages]
        else:
            page_indices = range(total_pages)
        
        # Extract text from each page separately
        sections = []
        for page_idx in page_indices:
            try:
                page = doc[page_idx]
                text = page.get_text()
                
                if text and text.strip():
                    text = ' '.join(text.split())
                    sections.append(text)
                    
            except Exception as e:
                print(f"Warning: Could not extract text from page {page_idx + 1}: {e}", file=sys.stderr)
                continue
        
        doc.close()
        return sections
        
    except Exception as e:
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            raise ValueError("PDF is encrypted or password-protected")
        else:
            raise ValueError(f"Could not read PDF file: {e}")

def list_pdf_sections(file_path, output_json=False):
    """List chapters (preferred) or pages as sections"""
    chapters = get_pdf_chapters(file_path)
    
    if chapters and len(chapters) > 1:
        # We have chapters - list them
        if output_json:
            print(json.dumps(chapters, indent=2))
        else:
            print(f"📖 Chapters in {Path(file_path).name}:")
            for chapter in chapters:
                chapter_num = chapter['number']
                title = chapter['title']
                start_page = chapter['start_page']
                end_page = chapter['end_page']
                page_count = end_page - start_page + 1
                chapter_type = chapter['type']
                
                # Truncate long titles
                if len(title) > 50:
                    title = title[:47] + "..."
                
                print(f"  {chapter_num:2d}. {title}")
                print(f"      Pages {start_page}-{end_page} ({page_count} pages) [{chapter_type}]")
            
            print(f"\nTotal: {len(chapters)} chapters")
            print("💡 Use --chapters to select specific chapters, or --interactive-start for precise selection")
        
        return chapters
    else:
        # No chapters detected - fall back to page listing
        return list_pdf_pages_fallback(file_path, output_json)

def list_pdf_pages_fallback(file_path, output_json=False):
    """Fallback page listing when no chapters detected"""
    try:
        import fitz
        doc = fitz.open(str(file_path))
        
        if doc.is_encrypted:
            doc.close()
            raise ValueError("PDF is encrypted or password-protected")
        
        total_pages = len(doc)
        if not total_pages:
            doc.close()
            raise ValueError("PDF has no pages")
        
        pages_info = []
        for page_num in range(total_pages):
            try:
                page = doc[page_num]
                text = page.get_text()
                word_count = len(text.split()) if text and text.strip() else 0
                
                # Get first line as preview
                preview = ""
                if text and text.strip():
                    first_line = text.strip().split('\n')[0]
                    preview = first_line[:50] + ("..." if len(first_line) > 50 else "")
                
                page_info = {
                    'page': page_num + 1,
                    'preview': preview or 'No text found',
                    'word_count': word_count
                }
                pages_info.append(page_info)
                
            except Exception as e:
                pages_info.append({
                    'page': page_num + 1,
                    'preview': f'Error reading page: {e}',
                    'word_count': 0
                })
        
        doc.close()
        
        if output_json:
            print(json.dumps(pages_info, indent=2))
        else:
            print(f"📄 Pages in {Path(file_path).name} (no chapters detected):")
            for page in pages_info:
                print(f"  {page['page']:2d}. {page['preview']} ({page['word_count']} words)")
            print(f"\nTotal: {len(pages_info)} pages")
            print("💡 Use --sections to select specific pages, or --interactive-start for precise selection")
        
        return pages_info
        
    except Exception as e:
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            raise ValueError("PDF is encrypted or password-protected")
        else:
            raise ValueError(f"Could not read PDF file: {e}")

def get_pdf_page_count(file_path):
    """Get number of pages in PDF file"""
    try:
        import fitz
        doc = fitz.open(str(file_path))
        
        if doc.is_encrypted:
            doc.close()
            raise ValueError("PDF is encrypted or password-protected")
        
        page_count = len(doc)
        doc.close()
        return page_count
        
    except Exception as e:
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            raise ValueError("PDF is encrypted or password-protected")
        else:
            raise ValueError(f"Could not read PDF file: {e}")

def get_pdf_chapter_count(file_path):
    """Get number of chapters in PDF file"""
    chapters = get_pdf_chapters(file_path)
    return len(chapters) if chapters else 0

def interactive_hierarchical_start_selection(file_path):
    """Interactive Chapter → Page → Word selection"""
    print("\n🎯 Hierarchical Start Point Selection")
    print("=" * 60)
    
    # Step 1: List and select chapter
    chapters = get_pdf_chapters(file_path)
    if not chapters:
        print("❌ No chapters found, falling back to page-based selection")
        return interactive_page_word_selection(file_path)
    
    if len(chapters) == 1:
        print(f"📖 Single chapter detected: '{chapters[0]['title']}'")
        chapter_num = 1
        selected_chapter = chapters[0]
    else:
        print("📖 Step 1: Choose Chapter")
        print(f"📖 Chapters in {Path(file_path).name}:")
        for chapter in chapters:
            chapter_num_display = chapter['number']
            title = chapter['title']
            start_page = chapter['start_page']
            end_page = chapter['end_page']
            page_count = end_page - start_page + 1
            
            if len(title) > 50:
                title = title[:47] + "..."
            
            print(f"  {chapter_num_display:2d}. {title} (Pages {start_page}-{end_page}, {page_count} pages)")
        
        while True:
            try:
                choice = input(f"\n🎯 Which chapter to start from? (1-{len(chapters)}, or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return None
                
                chapter_num = int(choice)
                if 1 <= chapter_num <= len(chapters):
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(chapters)}")
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
        
        selected_chapter = chapters[chapter_num - 1]
    
    print(f"\n✅ Selected Chapter {chapter_num}: '{selected_chapter['title']}'")
    
    # Step 2: List and select page within chapter
    start_page = selected_chapter['start_page']
    end_page = selected_chapter['end_page']
    page_count = end_page - start_page + 1
    
    if page_count == 1:
        print(f"📄 Chapter has only 1 page (page {start_page})")
        page_in_chapter = 1
        absolute_page = start_page
    else:
        print(f"\n📄 Step 2: Choose Page within Chapter {chapter_num}")
        
        # Show pages in chapter
        try:
            import fitz
            doc = fitz.open(str(file_path))
            
            print(f"📄 Pages in Chapter {chapter_num}:")
            for i, page_num in enumerate(range(start_page, end_page + 1), 1):
                try:
                    page = doc[page_num - 1]  # Convert to 0-indexed
                    text = page.get_text()
                    word_count = len(text.split()) if text and text.strip() else 0
                    
                    # Get first line as preview
                    preview = ""
                    if text and text.strip():
                        first_line = text.strip().split('\n')[0]
                        preview = first_line[:60] + ("..." if len(first_line) > 60 else "")
                    
                    print(f"  {i:2d}. (Page {page_num:3d}) {preview or 'No text found'} ({word_count} words)")
                    
                except Exception as e:
                    print(f"  {i:2d}. (Page {page_num:3d}) Error reading page: {e} (0 words)")
            
            doc.close()
            
        except Exception as e:
            print(f"❌ Could not read chapter pages: {e}")
            return None
        
        while True:
            try:
                choice = input(f"\n🎯 Which page in this chapter? (1-{page_count}, or 'q' to go back): ").strip()
                if choice.lower() == 'q':
                    return None
                
                page_in_chapter = int(choice)
                if 1 <= page_in_chapter <= page_count:
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {page_count}")
            except ValueError:
                print("❌ Please enter a valid number or 'q' to go back")
        
        absolute_page = start_page + page_in_chapter - 1
    
    print(f"\n✅ Selected Page {page_in_chapter} in Chapter {chapter_num} (Document Page {absolute_page})")
    
    # Step 3: Show words and select start word
    print(f"\n📝 Step 3: Choose Starting Word")
    try:
        import fitz
        doc = fitz.open(str(file_path))
        
        page = doc[absolute_page - 1]  # Convert to 0-indexed
        text = page.get_text()
        doc.close()
        
        if not text or not text.strip():
            print("❌ No text found on this page")
            return None
        
        # Clean up text and get words
        text = ' '.join(text.split())
        words = text.split()
        
        if not words:
            print("❌ No words found on this page")
            return None
        
        # Show word selection
        max_words = min(50, len(words))
        
        print(f"\n📝 Chapter {chapter_num}, Page {page_in_chapter} - First {max_words} words:")
        print(f"    (Document page {absolute_page})")
        print("-" * 80)
        
        # Display words in rows of 10 with numbers
        for i in range(0, max_words, 10):
            numbers = []
            word_row = []
            
            for j in range(i, min(i + 10, max_words)):
                word_num = j + 1
                word = words[j]
                numbers.append(f"{word_num:3d}")
                word_row.append(f"{word[:8]:>8s}")  # Truncate long words
            
            print("  " + " ".join(numbers))
            print("  " + " ".join(word_row))
            print()
        
        if len(words) > max_words:
            remaining = len(words) - max_words
            print(f"... and {remaining} more words on this page")
        
        while True:
            try:
                choice = input(f"\n🎯 Start from which word? (1-{max_words}, 'show' to see again, or 'q' to go back): ").strip()
                
                if choice.lower() == 'q':
                    return None
                elif choice.lower() == 'show':
                    # Redisplay words (already shown above)
                    continue
                
                word_num = int(choice)
                if 1 <= word_num <= max_words:
                    # Show preview
                    preview_words = words[word_num-1:word_num+10]
                    preview = " ".join(preview_words)
                    if len(words) > word_num + 10:
                        preview += "..."
                    
                    print(f"\n📝 Starting from word {word_num}: '{preview}'")
                    
                    confirm = input("❓ Confirm this start point? (y/N): ").strip().lower()
                    if confirm == 'y':
                        break
                    else:
                        continue
                else:
                    print(f"❌ Please enter a number between 1 and {max_words}")
            except ValueError:
                print("❌ Please enter a valid number, 'show', or 'q'")
        
        # Return the hierarchical selection
        result = {
            'chapter': chapter_num,
            'chapter_title': selected_chapter['title'],
            'page_in_chapter': page_in_chapter,
            'absolute_page': absolute_page,
            'word': word_num,
            'start_from_chapter': chapter_num,
            'start_from_page': absolute_page,
            'start_from_word': word_num,
            'selection_type': 'hierarchical_pdf'
        }
        
        print(f"\n✅ Final Selection:")
        print(f"   📖 Chapter {chapter_num}: '{selected_chapter['title']}'")
        print(f"   📄 Page {page_in_chapter} in chapter (Document page {absolute_page})")
        print(f"   📝 Starting from word {word_num}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error selecting words: {e}")
        return None

def interactive_page_word_selection(file_path):
    """Fallback: Page → Word selection when no chapters detected"""
    print("\n📄 Page-based selection (no chapters detected)")
    
    try:
        import fitz
        doc = fitz.open(str(file_path))
        total_pages = len(doc)
        
        # Show first few pages for selection
        print(f"📄 First 10 pages in {Path(file_path).name}:")
        for page_num in range(min(10, total_pages)):
            try:
                page = doc[page_num]
                text = page.get_text()
                word_count = len(text.split()) if text and text.strip() else 0
                
                preview = ""
                if text and text.strip():
                    first_line = text.strip().split('\n')[0]
                    preview = first_line[:60] + ("..." if len(first_line) > 60 else "")
                
                print(f"  {page_num + 1:2d}. {preview or 'No text found'} ({word_count} words)")
                
            except Exception:
                print(f"  {page_num + 1:2d}. Error reading page (0 words)")
        
        if total_pages > 10:
            print(f"... and {total_pages - 10} more pages")
        
        # Get page selection
        while True:
            try:
                choice = input(f"\n🎯 Which page to start from? (1-{total_pages}, or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return None
                
                page_num = int(choice)
                if 1 <= page_num <= total_pages:
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {total_pages}")
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
        
        # Get words from selected page and do word selection
        page = doc[page_num - 1]
        text = page.get_text()
        doc.close()
        
        if not text or not text.strip():
            print("❌ No text found on this page")
            return None
        
        text = ' '.join(text.split())
        words = text.split()
        
        if not words:
            print("❌ No words found on this page")
            return None
        
        # Word selection (simplified version)
        max_words = min(50, len(words))
        print(f"\n📝 Page {page_num} - First {max_words} words:")
        print("-" * 60)
        
        for i in range(0, max_words, 10):
            numbers = []
            word_row = []
            
            for j in range(i, min(i + 10, max_words)):
                word_num = j + 1
                word = words[j]
                numbers.append(f"{word_num:3d}")
                word_row.append(f"{word[:8]:>8s}")
            
            print("  " + " ".join(numbers))
            print("  " + " ".join(word_row))
            print()
        
        if len(words) > max_words:
            print(f"... and {len(words) - max_words} more words")
        
        while True:
            try:
                choice = input(f"\n🎯 Start from which word? (1-{max_words}, or 'q' to go back): ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                word_num = int(choice)
                if 1 <= word_num <= max_words:
                    preview_words = words[word_num-1:word_num+10]
                    preview = " ".join(preview_words)
                    if len(words) > word_num + 10:
                        preview += "..."
                    
                    print(f"\n📝 Starting from word {word_num}: '{preview}'")
                    
                    confirm = input("❓ Confirm this start point? (y/N): ").strip().lower()
                    if confirm == 'y':
                        break
                    else:
                        continue
                else:
                    print(f"❌ Please enter a number between 1 and {max_words}")
            except ValueError:
                print("❌ Please enter a valid number or 'q'")
        
        return {
            'section': page_num,
            'subsection': 1,
            'word': word_num,
            'start_from_section': page_num,
            'start_from_page': page_num,
            'start_from_word': word_num,
            'selection_type': 'page_based_pdf'
        }
        
    except Exception as e:
        print(f"❌ Error with page selection: {e}")
        return None

def extract_pdf_from_hierarchical_selection(file_path, selection_info):
    """Extract PDF text starting from hierarchical selection"""
    if not selection_info:
        # Extract everything
        return extract_pdf_by_pages(file_path, None)
    
    selection_type = selection_info.get('selection_type', 'unknown')
    
    if selection_type == 'hierarchical_pdf':
        # Extract from specific chapter/page/word
        start_page = selection_info['absolute_page'] - 1  # Convert to 0-indexed
        word_offset = selection_info.get('word', 1) - 1  # Convert to 0-indexed
        
        try:
            import fitz
            doc = fitz.open(str(file_path))
            
            extracted_text = []
            
            # Process pages from start point to end
            for page_num in range(start_page, len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if not text or not text.strip():
                    continue
                
                # Clean text
                text = ' '.join(text.split())
                
                # For the first page, apply word offset
                if page_num == start_page and word_offset > 0:
                    words = text.split()
                    if word_offset < len(words):
                        text = ' '.join(words[word_offset:])
                        print(f"STATUS: Started from word {word_offset + 1} on page {start_page + 1}", file=sys.stderr)
                    else:
                        print(f"WARNING: Word offset {word_offset + 1} beyond page length, using full page", file=sys.stderr)
                
                extracted_text.append(text)
            
            doc.close()
            
            if not extracted_text:
                raise ValueError("No text extracted from selection")
            
            result = ' '.join(extracted_text)
            word_count = len(result.split())
            print(f"STATUS: Extracted {word_count:,} words from hierarchical selection", file=sys.stderr)
            
            return result
            
        except Exception as e:
            raise ValueError(f"Could not extract from hierarchical selection: {e}")
    
    elif selection_type == 'page_based_pdf':
        # Extract from specific page/word (fallback mode)
        start_page = selection_info['start_from_page'] - 1
        word_offset = selection_info.get('start_from_word', 1) - 1
        
        try:
            import fitz
            doc = fitz.open(str(file_path))
            
            extracted_text = []
            for page_num in range(start_page, len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if not text or not text.strip():
                    continue
                
                text = ' '.join(text.split())
                
                if page_num == start_page and word_offset > 0:
                    words = text.split()
                    if word_offset < len(words):
                        text = ' '.join(words[word_offset:])
                        print(f"STATUS: Started from word {word_offset + 1} on page {start_page + 1}", file=sys.stderr)
                
                extracted_text.append(text)
            
            doc.close()
            
            if not extracted_text:
                raise ValueError("No text extracted from selection")
            
            result = ' '.join(extracted_text)
            word_count = len(result.split())
            print(f"STATUS: Extracted {word_count:,} words from page selection", file=sys.stderr)
            
            return result
            
        except Exception as e:
            raise ValueError(f"Could not extract from page selection: {e}")
    
    else:
        # Unknown selection type, extract everything
        print(f"WARNING: Unknown selection type '{selection_type}', extracting full document", file=sys.stderr)
        return extract_pdf_by_pages(file_path, None)

def test_pdf_file(file_path):
    """Test if PDF file can be opened and read"""
    try:
        import fitz
        print(f"DEBUG: Testing PDF file: {file_path}", file=sys.stderr)
        
        doc = fitz.open(str(file_path))
        print(f"DEBUG: PDF opened successfully", file=sys.stderr)
        
        if doc.is_encrypted:
            doc.close()
            print(f"DEBUG: PDF is encrypted", file=sys.stderr)
            return False, "PDF is encrypted"
        
        page_count = len(doc)
        print(f"DEBUG: PDF has {page_count} pages", file=sys.stderr)
        
        if page_count > 0:
            # Test first page
            page = doc[0]
            text = page.get_text()
            text_length = len(text) if text else 0
            print(f"DEBUG: First page has {text_length} characters", file=sys.stderr)
        
        # Test chapter detection
        chapters = get_pdf_chapters(file_path)
        chapter_count = len(chapters) if chapters else 0
        print(f"DEBUG: Detected {chapter_count} chapters", file=sys.stderr)
        
        doc.close()
        print(f"DEBUG: PDF test completed successfully", file=sys.stderr)
        return True, f"PDF is readable, {page_count} pages, {chapter_count} chapters"
        
    except ImportError:
        return False, "PyMuPDF not installed: pip install pymupdf"
    except Exception as e:
        print(f"DEBUG: PDF test failed: {e}", file=sys.stderr)
        return False, f"Error: {e}"

if __name__ == "__main__":
    """Command-line interface for PDF testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF text extraction and testing with hierarchical support")
    parser.add_argument("input_file", help="Input PDF file")
    parser.add_argument("--list", action="store_true", help="List chapters/pages")
    parser.add_argument("--list-chapters", action="store_true", help="List chapters only")
    parser.add_argument("--test", action="store_true", help="Test PDF file")
    parser.add_argument("--interactive", action="store_true", help="Interactive hierarchical selection")
    parser.add_argument("--extract", help="Extract text to file")
    parser.add_argument("--pages", nargs="*", type=int, help="Specific pages to process")
    parser.add_argument("--chapters", nargs="*", type=int, help="Specific chapters to process")
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"ERROR: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.test:
            success, message = test_pdf_file(args.input_file)
            print(f"{'✅' if success else '❌'} {message}")
            sys.exit(0 if success else 1)
        
        elif args.list or args.list_chapters:
            list_pdf_sections(args.input_file)
            sys.exit(0)
        
        elif args.interactive:
            selection = interactive_hierarchical_start_selection(args.input_file)
            if selection:
                print(f"\n📋 Selection Info:")
                print(json.dumps(selection, indent=2))
            else:
                print("❌ No selection made")
            sys.exit(0)
        
        elif args.extract:
            if args.interactive:
                selection = interactive_hierarchical_start_selection(args.input_file)
                if selection:
                    text = extract_pdf_from_hierarchical_selection(args.input_file, selection)
                else:
                    print("❌ No selection made")
                    sys.exit(1)
            elif args.chapters:
                text = extract_pdf_by_chapters(args.input_file, args.chapters)
            elif args.pages:
                text = extract_pdf_by_pages(args.input_file, args.pages)
            else:
                text = extract_pdf_by_pages(args.input_file, None)
            
            with open(args.extract, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"✅ Text extracted to {args.extract}")
            sys.exit(0)
        
        else:
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)