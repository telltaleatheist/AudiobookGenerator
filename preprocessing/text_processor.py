#!/usr/bin/env python3
"""
Text Processor - Universal text cleaning and engine-specific preprocessing
"""

import re
import sys
from pathlib import Path

def preprocess_file(input_file: str, output_file: str, config: dict, sections=None) -> bool:
    """Main preprocessing function - extract, clean, and apply engine-specific processing"""
    print(f"STATUS: Starting text preprocessing", file=sys.stderr)
    
    try:
        # Extract text from source file
        from preprocessing.file_extractors import extract_text
        text = extract_text(input_file, sections, config)
        
        word_count = len(text.split())
        print(f"STATUS: Extracted {word_count:,} words", file=sys.stderr)
        
        # Apply universal cleaning
        cleaned_text = apply_universal_cleaning(text)
        
        # Apply engine-specific preprocessing
        engine = config['metadata']['tts_engine']
        final_text = apply_engine_preprocessing(cleaned_text, engine, config)
        
        # Save processed text
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        
        final_word_count = len(final_text.split())
        print(f"STATUS: Saved {final_word_count:,} words to {output_file}", file=sys.stderr)
        
        return True
        
    except Exception as e:
        print(f"❌ Text preprocessing failed: {e}", file=sys.stderr)
        return False

def apply_universal_cleaning(text: str) -> str:
    """Apply universal text cleaning that works for all engines"""
    
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
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

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
    
    return text.strip()

def apply_engine_preprocessing(text: str, engine: str, config: dict) -> str:
    """Apply engine-specific preprocessing"""
    
    if engine.lower() == 'bark':
        return apply_bark_preprocessing(text, config)
    elif engine.lower() in ['edge', 'f5', 'xtts']:
        return apply_minimal_preprocessing(text, config)
    elif engine.lower() == 'openai':
        return text  # OpenAI needs minimal preprocessing
    else:
        return text

def apply_bark_preprocessing(text: str, config: dict) -> str:
    """Apply Bark-specific preprocessing including pronunciation fixes"""
    
    # Bark needs pronunciation help
    pronunciation_fixes = {
        # Religious/philosophical terms
        "atheist": "aytheeist",
        "atheists": "aytheeists", 
        "atheism": "aytheeism",
        "Jehovah's": "jehovas",
        
        # Common problem words
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
        
        # Religious/historical terms
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
        
        # Geographic/names
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
        print(f"STATUS: Applied {fixes_applied} Bark pronunciation fixes", file=sys.stderr)
    
    return text

def apply_minimal_preprocessing(text: str, config: dict) -> str:
    """Apply minimal preprocessing for engines that handle pronunciation well"""
    # These engines (EdgeTTS, XTTS, F5) handle pronunciation better
    # Just return the universally cleaned text
    return text