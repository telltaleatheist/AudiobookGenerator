#!/usr/bin/env python3
"""
AudiobookGenerator - CLEANED: Simplified startup and removed duplicate completion message
"""

import sys
import argparse
from pathlib import Path
import re
from core.progress_display_manager import log_error, log_info, log_success, log_warning
from managers.project_manager import ProjectManager
from core.pipeline_manager import PipelineManager
from managers.config_manager import ConfigManager

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Project-based audiobook generation with hierarchical chapter support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create new project
  python AudiobookGenerator.py --init mybook
  
  # Process entire book
  python AudiobookGenerator.py --project mybook
  
  # Process with new input file
  python AudiobookGenerator.py --project mybook --input book.epub
  python AudiobookGenerator.py --project mybook --input document.pdf
  
  # Process specific chapters (PDF) or sections (EPUB)
  python AudiobookGenerator.py --project mybook --chapters 1 2 3
  python AudiobookGenerator.py --project mybook --sections 1 2 3
  
  # Interactive hierarchical start point selection (Chapter → Page → Word for PDFs)
  python AudiobookGenerator.py --project mybook --interactive-start
  
  # Use existing config
  python AudiobookGenerator.py --project mybook --config batch1.json
  
  # List chapters/sections/pages
  python AudiobookGenerator.py --project mybook --list
  
  # Use specific voices for any engine
  python AudiobookGenerator.py --project mybook --tts-engine edge --engine-voice en-US-JennyNeural
  python AudiobookGenerator.py --project mybook --tts-engine bark --engine-voice v2/en_speaker_6
  python AudiobookGenerator.py --project mybook --tts-engine openai --engine-voice nova
        """
    )
    
    # Project operations
    parser.add_argument("--init", help="Initialize new project")
    parser.add_argument("--project", help="Use existing project")
    parser.add_argument("--input", help="Add input file to project")
    parser.add_argument("--config", help="Use specific config file")
    
    # TTS engine
    parser.add_argument("--tts-engine", choices=['bark', 'edge', 'f5', 'xtts', 'openai'], default='bark',
                       help="TTS engine (default: bark)")
    
    # Processing options
    parser.add_argument("--chapters", nargs="*", type=str,
                       help="Chapters to process for PDF files (e.g., 1 2 3 or 1-5)")
    parser.add_argument("--sections", nargs="*", type=str,
                       help="Sections to process for EPUB files (e.g., 1 2 3 or 1-5)")
    parser.add_argument("--list", action="store_true", 
                       help="List available chapters/sections/pages")
    parser.add_argument("--skip-rvc", action="store_true", 
                       help="Skip RVC conversion")
    parser.add_argument("--interactive-start", action="store_true",
                       help="Interactive start point selection (hierarchical for PDFs)")
    
    # Config overrides
    parser.add_argument("--voice", help="Voice model (deprecated, use --engine-voice)")
    parser.add_argument("--engine-voice", help="Voice model for any TTS engine (e.g., nova, en-US-JennyNeural, v2/en_speaker_6)")
    parser.add_argument("--rvc-model", help="RVC model name")
    parser.add_argument("--list-rvc-voices", action="store_true", help="List available RVC voice profiles")
    parser.add_argument("--rvc-voice", help="RVC voice profile name (e.g., 'sigma_male_narrator', 'my_voice')")
    parser.add_argument("--speed", type=float, help="Speed factor")
    parser.add_argument("--bark-text-temp", type=float, help="Bark text temperature")
    parser.add_argument("--bark-waveform-temp", type=float, help="Bark waveform temperature")
    parser.add_argument("--edge-rate", help="EdgeTTS speech rate")
    parser.add_argument("--edge-pitch", help="EdgeTTS pitch")
    parser.add_argument("--edge-volume", help="EdgeTTS volume")
    parser.add_argument("--edge-delay", type=float, help="EdgeTTS delay between chunks")
    parser.add_argument("--silence-gap", type=float, help="Silence gap between chunks")
    parser.add_argument("--job", type=str, help="Custom name for this job (e.g., 'bark-test', 'chapter-5')")

    return parser

def parse_sections(section_args):
    """Parse section arguments into list of integers"""
    if not section_args:
        return None
    
    sections = []
    for arg in section_args:
        if '-' in arg and not arg.startswith('-'):
            # Range like "3-7"
            try:
                start, end = map(int, arg.split('-'))
                sections.extend(range(start, end + 1))
            except ValueError:
                log_error(f"Invalid section range: {arg}")
                return None
        else:
            # Single number
            try:
                sections.append(int(arg))
            except ValueError:
                log_error(f"Invalid section number: {arg}")
                return None
    
    return sorted(list(set(sections)))

def create_cli_overrides(args):
    """Convert CLI args to config overrides"""
    overrides = {}
    
    # Handle voice selection with new --engine-voice parameter
    voice_to_use = None
    if args.engine_voice:
        voice_to_use = args.engine_voice
        # CLEANED: Remove verbose voice message
    elif args.voice:
        voice_to_use = args.voice
        log_warning(f"--voice is deprecated, use --engine-voice instead")
    
    # Apply voice to the selected TTS engine
    if voice_to_use:
        engine = args.tts_engine
        
        # Apply to the specific engine (let the engine validate compatibility)
        if engine == 'bark':
            overrides['bark'] = overrides.get('bark', {})
            overrides['bark']['voice'] = voice_to_use
        elif engine == 'edge':
            overrides['edge'] = overrides.get('edge', {})
            overrides['edge']['voice'] = voice_to_use
        elif engine == 'openai':
            overrides['openai'] = overrides.get('openai', {})
            overrides['openai']['voice'] = voice_to_use
        elif engine == 'xtts':
            # XTTS uses 'speaker' for built-in voices
            overrides['xtts'] = overrides.get('xtts', {})
            overrides['xtts']['speaker'] = voice_to_use
        elif engine == 'f5':
            # CLEANED: Remove verbose F5 message
            pass
        else:
            log_warning(f"Unknown engine '{engine}', applying voice parameter anyway")
            overrides[engine] = overrides.get(engine, {})
            overrides[engine]['voice'] = voice_to_use
    
    # Bark-specific overrides
    bark = overrides.get('bark', {})
    if args.bark_text_temp is not None:
        bark['text_temp'] = args.bark_text_temp
    if args.bark_waveform_temp is not None:
        bark['waveform_temp'] = args.bark_waveform_temp
    if bark:
        overrides['bark'] = bark
    
    # Edge-specific overrides
    edge = overrides.get('edge', {})
    if args.edge_rate:
        edge['rate'] = args.edge_rate
    if args.edge_pitch:
        edge['pitch'] = args.edge_pitch
    if args.edge_volume:
        edge['volume'] = args.edge_volume
    if args.edge_delay is not None:
        edge['delay'] = args.edge_delay
    if edge:
        overrides['edge'] = edge
    
    # RVC voice selection (metadata override)
    if hasattr(args, 'rvc_voice') and args.rvc_voice:
        # Set the RVC voice in metadata
        if 'metadata' not in overrides:
            overrides['metadata'] = {}
        overrides['metadata']['rvc_voice'] = args.rvc_voice
    
    # Handle legacy --rvc-model parameter (map to rvc_voice)
    elif hasattr(args, 'rvc_model') and args.rvc_model:
        if 'metadata' not in overrides:
            overrides['metadata'] = {}
        overrides['metadata']['rvc_voice'] = args.rvc_model
        log_info("💡 Note: --rvc-model is deprecated, use --rvc-voice instead")
    
    # Global RVC settings (applies to all voices)
    rvc_global = {}
    if args.speed is not None:
        rvc_global['speed_factor'] = args.speed
    if rvc_global:
        overrides['rvc_global'] = rvc_global
    
    # Voice-specific RVC overrides (for advanced users)
    # This allows CLI overrides for specific voice settings
    # Example: --rvc-semitones -3 would override n_semitones for the selected voice
    if hasattr(args, 'rvc_voice') and args.rvc_voice:
        voice_specific_overrides = {}
        
        # Add any voice-specific CLI parameters here
        if hasattr(args, 'rvc_semitones') and args.rvc_semitones is not None:
            voice_specific_overrides['n_semitones'] = args.rvc_semitones
        if hasattr(args, 'rvc_index_rate') and args.rvc_index_rate is not None:
            voice_specific_overrides['index_rate'] = args.rvc_index_rate
        if hasattr(args, 'rvc_protect_rate') and args.rvc_protect_rate is not None:
            voice_specific_overrides['protect_rate'] = args.rvc_protect_rate
        
        # Apply voice-specific overrides to the selected voice profile
        if voice_specific_overrides and args.rvc_voice:
            rvc_voice_key = f'rvc_{args.rvc_voice}'
            overrides[rvc_voice_key] = voice_specific_overrides
    
    # Audio overrides
    audio = {}
    if args.silence_gap is not None:
        audio['silence_gap'] = args.silence_gap
    if audio:
        overrides['audio'] = audio
    
    return overrides if overrides else None

def validate_sections(source_file, sections, section_type="sections"):
    """Validate sections/chapters exist in source file"""
    if not sections:
        return True
    
    source_path = Path(source_file)
    suffix = source_path.suffix.lower()
    
    if suffix not in ['.epub', '.pdf']:
        log_error(f"{section_type.title()} selection only available for EPUB and PDF files")
        return False
    
    try:
        if suffix == '.epub':
            from preprocessing.text_processor import get_epub_section_count
            total_sections = get_epub_section_count(source_file)
            section_label = "section"
        elif suffix == '.pdf':
            if section_type == "chapters":
                from preprocessing import get_pdf_chapter_count
                total_sections = get_pdf_chapter_count(source_file)
                section_label = "chapter"
            else:
                from preprocessing import get_pdf_page_count
                total_sections = get_pdf_page_count(source_file)
                section_label = "page"
        
        for section in sections:
            if section < 1 or section > total_sections:
                log_error(f"{section_label.title()} {section} out of range (available: 1-{total_sections})")
                return False
        
        log_success(f"Processing {section_label}s: {sections}")
        return True
        
    except ImportError:
        log_info("⚠️ Could not validate sections (missing dependencies)")
        return True  # Allow processing to continue
    except Exception as e:
        log_warning(f"Could not validate sections: {e}")
        return True  # Allow processing to continue

def handle_list_sections(project_manager, project_name):
    """List available chapters/sections/pages in source file"""
    try:
        source_file = project_manager.find_source_file(project_name)
        
        suffix = source_file.suffix.lower()
        
        if suffix == '.epub':
            try:
                from preprocessing import list_epub_sections
                list_epub_sections(source_file, output_json=False)
                return True
            except ImportError:
                log_info("❌ Section listing requires: pip install ebooklib beautifulsoup4")
                return False
        elif suffix == '.pdf':
            try:
                from preprocessing import list_pdf_sections
                list_pdf_sections(source_file, output_json=False)
                return True
            except ImportError:
                log_info("❌ PDF chapter listing requires: pip install pymupdf")
                return False
        else:
            log_info("❌ Section listing only available for EPUB and PDF files")
            return False
        
    except Exception as e:
        log_error(f"{e}")
        return False

def handle_interactive_start(project_manager, project_name, sections=None):
    """Handle interactive start point selection"""
    try:
        source_file = project_manager.find_source_file(project_name)
        
        # Import the interactive start functionality
        try:
            from preprocessing import interactive_start_selection
            start_info = interactive_start_selection(source_file, sections)
            return start_info
        except ImportError:
            suffix = Path(source_file).suffix.lower()
            if suffix == '.epub':
                log_info("❌ Interactive start requires: pip install ebooklib beautifulsoup4")
            elif suffix == '.pdf':
                log_info("❌ Interactive start requires: pip install pymupdf")
            else:
                log_info("❌ Interactive start requires additional dependencies")
            return None
        
    except Exception as e:
        log_error(f"{e}")
        return None

def determine_section_type_and_values(args, source_file):
    """Determine whether to use chapters or sections based on file type and arguments"""
    suffix = Path(source_file).suffix.lower()
    
    # Parse arguments
    chapters = parse_sections(args.chapters) if args.chapters else None
    sections = parse_sections(args.sections) if args.sections else None
    
    # Validation logic
    if suffix == '.pdf':
        if chapters and sections:
            log_info("❌ Cannot specify both --chapters and --sections for PDF files")
            return None, None
        elif chapters:
            return chapters, "chapters"
        elif sections:
            log_info("💡 Using --sections as page numbers for PDF file")
            return sections, "pages"
        else:
            return None, None
    
    elif suffix == '.epub':
        if chapters and sections:
            log_info("❌ Cannot specify both --chapters and --sections for EPUB files")
            return None, None
        elif chapters:
            log_info("💡 EPUB files use sections, not chapters. Using --chapters as section numbers")
            return chapters, "sections"
        elif sections:
            return sections, "sections"
        else:
            return None, None
    
    else:  # .txt files
        if chapters or sections:
            log_info("⚠️ Section/chapter selection not supported for TXT files")
        return None, None

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    project_manager = ProjectManager()
    
    if args.list_rvc_voices:
        if not args.project:
            parser.error("--list-rvc-voices requires --project")
        
        try:
            config = project_manager.load_config(
                project_manager.validate_project(args.project) / "config" / "config.json"
            )
            voices = list_available_rvc_voices(config)
            
            log_info("🎭 Available RVC voices for project '{args.project}':")
            for voice_name, model_name in voices:
                log_info("  • {voice_name} → {model_name}")
            return 0
            
        except Exception as e:
            log_error(f"{e}")
            return 1

    # Handle project creation
    if args.init:
        try:
            project_manager.create_project(args.init)
            return 0
        except Exception as e:
            log_error(f"{e}")
            return 1
    
    # Require project for all other operations
    if not args.project:
        parser.error("Must specify --project or --init")
    
    try:
        # Validate project exists
        project_manager.validate_project(args.project)
        
        # Validate RVC voice if specified
        if hasattr(args, 'rvc_voice') and args.rvc_voice:
            if not validate_rvc_voice(project_manager, args.project, args.rvc_voice):
                return 1
        
        # Handle section listing
        if args.list:
            return 0 if handle_list_sections(project_manager, args.project) else 1
        
        # Find or add source file
        if args.input:
            source_file = project_manager.add_source_file(args.project, args.input)
        else:
            source_file = project_manager.find_source_file(args.project)
        
        # Determine section type and values
        sections, section_type = determine_section_type_and_values(args, source_file)
        if args.chapters or args.sections:
            if sections is None:
                return 1  # Error already printed
            
            # Validate sections/chapters
            if not validate_sections(source_file, sections, section_type):
                return 1
        
        # Handle interactive start selection
        start_info = None
        if args.interactive_start:
            start_info = handle_interactive_start(project_manager, args.project, sections)
            if start_info is None:
                log_info("❌ Interactive start selection failed or cancelled")
                return 1
        
        # Generate batch name and paths
        if args.job:
            batch_name = re.sub(r'[^\w\-_]', '_', args.job)
            # Handle collisions by adding counter if needed
            project_dir = project_manager.validate_project(args.project)
            counter = 2
            original_batch_name = batch_name
            while (project_dir / batch_name).exists():
                batch_name = f"{original_batch_name}_{counter}"
                counter += 1
        else:
            # Use automatic batch naming based on sections
            batch_name = project_manager.get_batch_name(args.project, sections)

        paths = project_manager.get_batch_paths(args.project, batch_name, args.tts_engine)
        
        # Create configuration
        cli_overrides = create_cli_overrides(args)
        config = project_manager.create_config(
            args.project, batch_name, args.tts_engine, sections,
            source_file, args.config, cli_overrides
        )
        
        # Add start_info to config if interactive start was used
        if start_info:
            config['preprocessing'] = config.get('preprocessing', {})
            config['preprocessing']['start_info'] = start_info
        
        # Save configuration to both locations
        project_manager.save_config(config, paths['config'])
        project_manager.save_config(config, paths['job_config'])  # Copy to job directory
        
        # CLEANED: Simplified project display - no verbose config summary
        # The pipeline manager will show the clean header
        
        if start_info:
            file_type = Path(source_file).suffix.lower()
            selection_type = start_info.get('selection_type', 'unknown')
            
            if selection_type == 'hierarchical_pdf':
                chapter = start_info.get('chapter', '?')
                chapter_title = start_info.get('chapter_title', 'Unknown')
                page_in_chapter = start_info.get('page_in_chapter', '?')
                absolute_page = start_info.get('absolute_page', '?')
                word = start_info.get('word', 1)
                
                # Truncate long chapter titles
                if len(chapter_title) > 40:
                    chapter_title = chapter_title[:37] + "..."
                
                log_info("🎯 Start: Chapter {chapter}, Page {page_in_chapter} (Doc Page {absolute_page}), Word {word}")
                log_info("📖 Chapter: '{chapter_title}'")
                
            elif selection_type == 'page_based_pdf':
                page = start_info.get('start_from_page', '?')
                word = start_info.get('start_from_word', 1)
                log_info("🎯 Start: Page {page}, Word {word}")
                
            else:
                # Original section-based system
                section_label = "page" if file_type == '.pdf' else "section"
                section = start_info.get('section', '?')
                subsection = start_info.get('subsection', '?')
                word = start_info.get('word', 1)
                word_display = f", Word {word}" if word > 1 else ""
                log_info("🎯 Start: {section_label.title()} {section}, Subsection {subsection}{word_display}")
        
        # Show section/chapter info if specified
        if sections:
            if section_type == "chapters":
                log_info("📖 Processing chapters: {sections}")
            elif section_type == "sections":
                log_info("📝 Processing sections: {sections}")
            elif section_type == "pages":
                log_info("📄 Processing pages: {sections}")
        
        # Execute pipeline
        pipeline_manager = PipelineManager()
        
        # Convert Path objects to strings for JSON serialization
        string_paths = {k: str(v) for k, v in paths.items()}
        
        success = pipeline_manager.run(
            source_file=str(source_file),
            paths=string_paths,
            config=config,
            sections=sections,
            skip_rvc=args.skip_rvc
        )
                
        # CLEANED: Remove duplicate completion message - pipeline_manager already shows completion
        if success:
            return 0
        else:
            log_info("❌ Pipeline failed")
            return 1
            
    except Exception as e:
        log_error(f"{e}")
        return 1
    
def list_available_rvc_voices(config):
    """List all available RVC voice profiles"""
    voices = []
    for key in config.keys():
        if key.startswith('rvc_') and key != 'rvc_global':
            voice_name = key.replace('rvc_', '')
            model_name = config[key].get('model', 'Unknown')
            voices.append((voice_name, model_name))
    return voices

def validate_rvc_voice(project_manager, project_name, rvc_voice):
    """Validate that the specified RVC voice exists"""
    try:
        config = project_manager.load_config(
            project_manager.validate_project(project_name) / "config" / "config.json"
        )
        
        available_voices = list_available_rvc_voices(config)
        voice_names = [v[0] for v in available_voices]
        
        if rvc_voice not in voice_names:
            log_error(f"RVC voice '{rvc_voice}' not found!")
            log_info("📋 Available voices:")
            for voice_name, model_name in available_voices:
                log_info("  • {voice_name} → {model_name}")
            return False
        
        return True
        
    except Exception as e:
        log_warning(f"Could not validate RVC voice: {e}")
        return True  # Allow processing to continue

if __name__ == "__main__":
    sys.exit(main())