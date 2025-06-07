#!/usr/bin/env python3
"""
Section Manager - Smart text splitting for manageable audio sections
Splits text into equal-sized sections based on target duration (minutes)
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any

class SectionManager:
    """Handles intelligent text splitting for pipeline processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        try:
            self.target_minutes = config['pipeline']['target_section_length']
        except KeyError:
            raise ValueError("Missing required config: pipeline.target_section_length")
        
        # Rough estimation: 150 words per minute of speech
        self.words_per_minute = 150
        self.target_words = self.target_minutes * self.words_per_minute
        
        print(f"STATUS: Target section length: {self.target_minutes} minutes (~{self.target_words} words)", file=sys.stderr)
    
    def estimate_audio_duration_minutes(self, text: str) -> float:
        """Estimate audio duration in minutes based on word count"""
        word_count = len(text.split())
        return word_count / self.words_per_minute
    
    def split_text_into_sections(self, text: str) -> List[str]:
        """Split text into manageable sections based on target duration"""
        total_words = len(text.split())
        estimated_duration = self.estimate_audio_duration_minutes(text)
        
        print(f"STATUS: Text analysis - {total_words:,} words, ~{estimated_duration:.1f} minutes", file=sys.stderr)
        
        # If already under target, return as single section
        if estimated_duration <= self.target_minutes:
            print(f"STATUS: Text fits in single section", file=sys.stderr)
            return [text]
        
        # Calculate optimal number of sections
        num_sections = max(2, int(estimated_duration / self.target_minutes) + 1)
        target_words_per_section = total_words // num_sections
        
        print(f"STATUS: Splitting into {num_sections} sections (~{target_words_per_section:,} words each)", file=sys.stderr)
        
        # Try paragraph-based splitting first
        sections = self._split_by_paragraphs(text, target_words_per_section, num_sections)
        
        # Fallback to sentence-based if paragraphs don't work
        if not sections:
            print(f"STATUS: Paragraph splitting failed, using sentence boundaries", file=sys.stderr)
            sections = self._split_by_sentences(text, target_words_per_section, num_sections)
        
        # Final fallback to word-based
        if not sections:
            print(f"STATUS: Sentence splitting failed, using word boundaries", file=sys.stderr)
            sections = self._split_by_words(text, target_words_per_section)
        
        # Log final sections
        for i, section in enumerate(sections, 1):
            section_words = len(section.split())
            section_duration = self.estimate_audio_duration_minutes(section)
            print(f"STATUS: Section {i}: {section_words:,} words (~{section_duration:.1f} min)", file=sys.stderr)
        
        return sections
    
    def _split_by_paragraphs(self, text: str, target_words: int, num_sections: int) -> List[str]:
        """Split text at paragraph boundaries"""
        # Split on double newlines (paragraph markers)
        paragraphs = re.split(r'\n\s*\n', text)
        
        if len(paragraphs) < 2:
            return []  # No paragraph structure
        
        sections = []
        current_section = ""
        current_word_count = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            para_words = len(paragraph.split())
            
            # If adding this paragraph would exceed target and we have content
            if current_word_count > 0 and current_word_count + para_words > target_words * 1.3:
                sections.append(current_section.strip())
                current_section = paragraph
                current_word_count = para_words
            else:
                if current_section:
                    current_section += "\n\n" + paragraph
                else:
                    current_section = paragraph
                current_word_count += para_words
        
        # Add final section
        if current_section.strip():
            sections.append(current_section.strip())
        
        # Verify we got reasonable sections
        if len(sections) >= num_sections * 0.8:  # Allow some flexibility
            return sections
        
        return []  # Not enough sections, fallback needed
    
    def _split_by_sentences(self, text: str, target_words: int, num_sections: int) -> List[str]:
        """Split text at sentence boundaries"""
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) < num_sections:
            return []  # Not enough sentences
        
        sections = []
        current_section = ""
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed target and we have content
            if current_word_count > 0 and current_word_count + sentence_words > target_words * 1.2:
                sections.append(current_section.strip())
                current_section = sentence
                current_word_count = sentence_words
            else:
                if current_section:
                    current_section += " " + sentence
                else:
                    current_section = sentence
                current_word_count += sentence_words
        
        # Add final section
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections
    
    def _split_by_words(self, text: str, target_words: int) -> List[str]:
        """Split text by word count (fallback)"""
        words = text.split()
        sections = []
        
        for i in range(0, len(words), target_words):
            section_words = words[i:i + target_words]
            sections.append(" ".join(section_words))
        
        return sections
    
    def save_sections(self, sections: List[str], output_dir: Path) -> List[Path]:
        """Save sections as numbered text files"""
        output_dir = Path(output_dir)
        sections_dir = output_dir / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        
        section_files = []
        for i, section in enumerate(sections, 1):
            section_file = sections_dir / f"section_{i:03d}.txt"
            with open(section_file, 'w', encoding='utf-8') as f:
                f.write(section)
            section_files.append(section_file)
        
        print(f"STATUS: Saved {len(sections)} sections to {sections_dir}", file=sys.stderr)
        return section_files