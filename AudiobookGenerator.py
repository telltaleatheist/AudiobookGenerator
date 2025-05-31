#!/usr/bin/env python3
"""
AudiobookGenerator - Clean project-based audiobook generation
Simple, focused command-line interface for audiobook processing
"""

import sys
import argparse
from pathlib import Path
import re
from project_manager import ProjectManager
from pipeline_manager import PipelineManager

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Project-based audiobook generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create new project
  python AudiobookGenerator.py --init mybook
  
  # Process entire book
  python AudiobookGenerator.py --project mybook
  
  # Process with new input file
  python AudiobookGenerator.py --project mybook --input book.epub
  
  # Process specific sections
  python AudiobookGenerator.py --project mybook --sections 1 2 3
  
  # Interactive start point selection
  python AudiobookGenerator.py --project mybook --interactive-start
  
  # Use existing config
  python AudiobookGenerator.py --project mybook --config batch1.json
  
  # List sections
  python AudiobookGenerator.py --project mybook --list
        """
    )
    
    # Project operations
    parser.add_argument("--init", help="Initialize new project")
    parser.add_argument("--project", help="Use existing project")
    parser.add_argument("--input", help="Add input file to project")
    parser.add_argument("--config", help="Use specific config file")
    
    # TTS engine
    parser.add_argument("--tts-engine", choices=['bark', 'edge', 'f5', 'xtts'], default='bark',
                       help="TTS engine (default: bark)")
    
    # Processing options
    parser.add_argument("--sections", nargs="*", type=str,
                       help="Sections to process (e.g., 1 2 3 or 1-5)")
    parser.add_argument("--list", action="store_true", 
                       help="List available sections")
    parser.add_argument("--skip-rvc", action="store_true", 
                       help="Skip RVC conversion")
    parser.add_argument("--interactive-start", action="store_true",
                       help="Show text sections and choose where to start processing")
    
    # Config overrides
    parser.add_argument("--voice", help="Voice model")
    parser.add_argument("--rvc-model", help="RVC model name")
    parser.add_argument("--speed", type=float, help="Speed factor")
    parser.add_argument("--bark-text-temp", type=float, help="Bark text temperature")
    parser.add_argument("--bark-waveform-temp", type=float, help="Bark waveform temperature")
    parser.add_argument("--edge-rate", help="EdgeTTS speech rate")
    parser.add_argument("--edge-pitch", help="EdgeTTS pitch")
    parser.add_argument("--edge-volume", help="EdgeTTS volume")
    parser.add_argument("--edge-delay", type=float, help="EdgeTTS delay between chunks")
    parser.add_argument("--silence-gap", type=float, help="Silence gap between chunks")
    parser.add_argument("--batch-name", type=str, help="Custom name for this batch (e.g., 'bark-test', 'chapter-5')")

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
                print(f"❌ Invalid section range: {arg}")
                return None
        else:
            # Single number
            try:
                sections.append(int(arg))
            except ValueError:
                print(f"❌ Invalid section number: {arg}")
                return None
    
    return sorted(list(set(sections)))

def create_cli_overrides(args):
    """Convert CLI args to config overrides"""
    overrides = {}
    
    # Bark overrides
    bark = {}
    if args.bark_text_temp is not None:
        bark['text_temp'] = args.bark_text_temp
    if args.bark_waveform_temp is not None:
        bark['waveform_temp'] = args.bark_waveform_temp
    if args.voice and args.tts_engine == 'bark':
        bark['voice'] = args.voice
    if bark:
        overrides['bark'] = bark
    
    # Edge overrides
    edge = {}
    if args.voice and args.tts_engine == 'edge':
        edge['voice'] = args.voice
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
    
    # RVC overrides
    rvc = {}
    if args.rvc_model:
        rvc['model'] = args.rvc_model
    if args.speed is not None:
        rvc['speed_factor'] = args.speed
    if rvc:
        overrides['rvc'] = rvc
    
    # Audio overrides
    audio = {}
    if args.silence_gap is not None:
        audio['silence_gap'] = args.silence_gap
    if audio:
        overrides['audio'] = audio
    
    return overrides if overrides else None

def validate_sections(source_file, sections):
    """Validate sections exist in source file"""
    if not sections:
        return True
    
    source_path = Path(source_file)
    if source_path.suffix.lower() != '.epub':
        print("❌ Section selection only available for EPUB files")
        return False
    
    try:
        # Try to get section count (this would need the preprocessing module)
        # For now, just assume sections are valid
        print(f"✅ Processing sections: {sections}")
        return True
    except Exception as e:
        print(f"⚠️ Could not validate sections: {e}")
        return True  # Allow processing to continue

def handle_list_sections(project_manager, project_name):
    """List available sections in source file"""
    try:
        source_file = project_manager.find_source_file(project_name)
        
        if source_file.suffix.lower() != '.epub':
            print("❌ Section listing only available for EPUB files")
            return False
        
        # Use our new text_extractor
        try:
            from preprocessing import list_epub_sections
            list_epub_sections(source_file, output_json=False)
            return True
        except ImportError:
            print("❌ Section listing requires: pip install ebooklib beautifulsoup4")
            return False
        
    except Exception as e:
        print(f"❌ {e}")
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
            print("❌ Interactive start requires: pip install ebooklib beautifulsoup4")
            return None
        
    except Exception as e:
        print(f"❌ {e}")
        return None

def validate_sections(source_file, sections):
    """Validate sections exist in source file"""
    if not sections:
        return True
    
    source_path = Path(source_file)
    if source_path.suffix.lower() != '.epub':
        print("❌ Section selection only available for EPUB files")
        return False
    
    try:
        from text_extractor import get_epub_section_count
        total_sections = get_epub_section_count(source_file)
        
        for section in sections:
            if section < 1 or section > total_sections:
                print(f"❌ Section {section} out of range (available: 1-{total_sections})")
                return False
        
        print(f"✅ Processing sections: {sections}")
        return True
        
    except ImportError:
        print("⚠️ Could not validate sections (missing dependencies)")
        return True  # Allow processing to continue
    except Exception as e:
        print(f"⚠️ Could not validate sections: {e}")
        return True  # Allow processing to continue implemented in this refactor)")
        return True
        
    except Exception as e:
        print(f"❌ {e}")
        return False

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    project_manager = ProjectManager()
    
    # Handle project creation
    if args.init:
        try:
            project_manager.create_project(args.init)
            return 0
        except Exception as e:
            print(f"❌ {e}")
            return 1
    
    # Require project for all other operations
    if not args.project:
        parser.error("Must specify --project or --init")
    
    try:
        # Validate project exists
        project_manager.validate_project(args.project)
        
        # Handle section listing
        if args.list:
            return 0 if handle_list_sections(project_manager, args.project) else 1
        
        # Find or add source file
        if args.input:
            source_file = project_manager.add_source_file(args.project, args.input)
        else:
            source_file = project_manager.find_source_file(args.project)
        
        # Parse and validate sections
        sections = parse_sections(args.sections)
        if sections and not validate_sections(source_file, sections):
            return 1
        
        # Handle interactive start selection
        start_info = None
        if args.interactive_start:
            start_info = handle_interactive_start(project_manager, args.project, sections)
            if start_info is None:
                print("❌ Interactive start selection failed or cancelled")
                return 1
        
        # Generate batch name and paths
        if args.batch_name:
            # Use custom batch name, sanitize it
            batch_name = re.sub(r'[^\w\-_]', '_', args.batch_name)
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
        
        # Display project info
        print(f"🚀 Audiobook Pipeline")
        print(f"📁 Project: {args.project}")
        print(f"📦 Batch: {batch_name}")
        print(f"📖 Source: {source_file.name}")
        print(f"🎤 TTS Engine: {args.tts_engine.upper()}")
        
        if start_info:
            word_display = f", Word {start_info['word']}" if start_info.get('word', 1) > 1 else ""
            print(f"🎯 Start: Section {start_info['section']}, Subsection {start_info['subsection']}{word_display}")
        
        project_manager.display_config_summary(config)
        
        # Execute pipeline
        pipeline_manager = PipelineManager()
        success = pipeline_manager.run(
            source_file=str(source_file),
            paths=paths,
            config=config,
            sections=sections,
            skip_rvc=args.skip_rvc
        )
        
        if success:
            print(f"\n🎉 Processing Complete!")
            print(f"🎵 Final audio: {paths['final']}")
            return 0
        else:
            print("❌ Pipeline failed")
            return 1
            
    except Exception as e:
        print(f"❌ {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())