#!/usr/bin/env python3
"""
Pipeline Rewind - Interactive tool to view and control pipeline progress
Usage: python pipeline_rewind.py --project PROJECT_NAME --job JOB_NAME
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import os

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Interactive pipeline checkpoint viewer and control tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_rewind.py --project mybook --job complete
  python pipeline_rewind.py --project eisenhauer --job arc_genesis
        """
    )
    
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--job", required=True, help="Job/batch name")
    
    return parser

def find_progress_file(project_name, job_name):
    """Find the progress.log file for the given project and job"""
    # Look in output directory structure
    base_dir = Path("output")
    progress_file = base_dir / project_name / "jobs" / job_name / "progress.log"
    
    if progress_file.exists():
        return progress_file
    
    # Alternative paths to check
    alt_paths = [
        Path(project_name) / "jobs" / job_name / "progress.log",
        Path(f"{project_name}_{job_name}") / "progress.log",
        Path("jobs") / project_name / job_name / "progress.log"
    ]
    
    for alt_path in alt_paths:
        if alt_path.exists():
            return alt_path
    
    return None

def load_progress_data(progress_file):
    """Load and parse progress data"""
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading progress file: {e}")
        return None

def format_timestamp(timestamp_str):
    """Format timestamp into human-readable format"""
    try:
        # Handle different timestamp formats
        if 'T' in timestamp_str:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        
        return {
            'date': dt.strftime('%Y-%m-%d'),
            'time': dt.strftime('%H:%M:%S'),
            'readable': dt.strftime('%b %d, %Y at %I:%M:%S %p'),
            'datetime': dt
        }
    except Exception:
        return {
            'date': 'Unknown',
            'time': 'Unknown', 
            'readable': timestamp_str,
            'datetime': None
        }

def calculate_duration(checkpoints, index):
    """Calculate duration from previous checkpoint"""
    if index == 0:
        return "Start"
    
    try:
        current_cp = checkpoints[index]
        prev_cp = checkpoints[index - 1]
        
        current_time = format_timestamp(current_cp.get('timestamp', ''))
        prev_time = format_timestamp(prev_cp.get('timestamp', ''))
        
        if current_time['datetime'] and prev_time['datetime']:
            duration = current_time['datetime'] - prev_time['datetime']
            total_seconds = duration.total_seconds()
            
            if total_seconds < 60:
                return f"{total_seconds:.1f}s"
            elif total_seconds < 3600:
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                return f"{minutes}m {seconds}s"
            else:
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                return f"{hours}h {minutes}m"
        
        return "Unknown"
    except:
        return "Unknown"

def categorize_checkpoint(checkpoint_type):
    """Categorize checkpoint type for display"""
    if 'PIPELINE_START' in checkpoint_type:
        return {'category': 'PIPELINE', 'color': '\033[92m'}  # Green
    elif 'PREPROCESSING' in checkpoint_type:
        return {'category': 'PREPROCESS', 'color': '\033[94m'}  # Blue
    elif 'TTS' in checkpoint_type:
        return {'category': 'TTS', 'color': '\033[95m'}  # Magenta
    elif 'RVC' in checkpoint_type:
        return {'category': 'RVC', 'color': '\033[96m'}  # Cyan
    elif 'MASTER' in checkpoint_type:
        return {'category': 'MASTER', 'color': '\033[93m'}  # Yellow
    elif 'COMPLETE' in checkpoint_type:
        return {'category': 'COMPLETE', 'color': '\033[92m'}  # Green
    elif 'FAILED' in checkpoint_type or 'ERROR' in checkpoint_type:
        return {'category': 'ERROR', 'color': '\033[91m'}  # Red
    else:
        return {'category': 'OTHER', 'color': '\033[90m'}  # Gray

def display_progress_summary(progress_data):
    """Display high-level progress summary"""
    print("\n" + "="*80)
    print("📊 PROGRESS SUMMARY")
    print("="*80)
    
    # Project info
    metadata = progress_data.get('metadata', {})
    project_name = progress_data.get('project_name', metadata.get('project_name', 'Unknown'))
    batch_name = progress_data.get('batch_name', metadata.get('batch_name', 'Unknown'))
    tts_engine = progress_data.get('tts_engine', metadata.get('tts_engine', 'Unknown'))
    rvc_voice = progress_data.get('rvc_voice', metadata.get('rvc_voice', 'Unknown'))
    
    print(f"Project: {project_name}")
    print(f"Batch: {batch_name}")
    print(f"TTS Engine: {tts_engine}")
    print(f"RVC Voice: {rvc_voice}")
    
    # Section progress
    sections = progress_data.get('sections', {})
    total = sections.get('total', 0)
    completed = sections.get('completed', [])
    current = sections.get('current')
    remaining = sections.get('remaining', [])
    
    if total > 0:
        completion_pct = (len(completed) / total) * 100
        print(f"\nSection Progress: {len(completed)}/{total} ({completion_pct:.1f}%)")
        print(f"   Completed: {completed}")
        if current:
            print(f"   Current: {current}")
        print(f"   Remaining: {remaining}")
    
    # Timing info
    checkpoints = progress_data.get('checkpoints', [])
    if checkpoints:
        start_time = format_timestamp(checkpoints[0].get('timestamp', ''))
        latest_time = format_timestamp(checkpoints[-1].get('timestamp', ''))
        
        print(f"\nTimeline:")
        print(f"   Started: {start_time['readable']}")
        print(f"   Latest: {latest_time['readable']}")
        
        if start_time['datetime'] and latest_time['datetime']:
            total_duration = latest_time['datetime'] - start_time['datetime']
            total_seconds = total_duration.total_seconds()
            if total_seconds > 3600:
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                print(f"   Duration: {hours}h {minutes}m")
            elif total_seconds > 60:
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                print(f"   Duration: {minutes}m {seconds}s")
            else:
                print(f"   Duration: {total_seconds:.1f}s")

def display_checkpoints(checkpoints):
    """Display checkpoints in a formatted table with improved layout"""
    if not checkpoints:
        print("\nNo checkpoints found")
        return
    
    print(f"\nCHECKPOINT HISTORY ({len(checkpoints)} entries)")
    print("="*100)
    print(f"{'#':<3} {'Type':<30} {'Message':<35} {'Time':<10} {'Duration':<10}")
    print("-"*100)
    
    for i, checkpoint in enumerate(checkpoints):
        timestamp_info = format_timestamp(checkpoint.get('timestamp', ''))
        duration = calculate_duration(checkpoints, i)
        checkpoint_type = checkpoint.get('type', 'Unknown')
        message = checkpoint.get('message', 'No message')
        
        # Get category info for styling
        cat_info = categorize_checkpoint(checkpoint_type)
        
        # Truncate long messages
        if len(message) > 32:
            message = message[:29] + "..."
        
        # Truncate long types
        display_type = checkpoint_type
        if len(display_type) > 27:
            display_type = display_type[:24] + "..."
        
        # Color coding (if terminal supports it)
        color = cat_info['color']
        reset = '\033[0m'
        
        print(f"{i:<3} {color} {display_type:<28}{reset} {message:<35} {timestamp_info['time']:<10} {duration:<10}")

def get_checkpoint_details(checkpoint, index, checkpoints):
    """Get detailed information about a specific checkpoint"""
    timestamp_info = format_timestamp(checkpoint.get('timestamp', ''))
    duration = calculate_duration(checkpoints, index)
    cat_info = categorize_checkpoint(checkpoint.get('type', ''))
    
    print(f"\n{'='*60}")
    print(f"CHECKPOINT #{index} DETAILS")
    print(f"{'='*60}")
    print(f"Time: {timestamp_info['readable']}")
    print(f"Duration from previous: {duration}")
    print(f"Type: {checkpoint.get('type', 'Unknown')}")
    print(f"Message: {checkpoint.get('message', 'No message')}")
    print(f"Category: {cat_info['category']}")

def perform_rewind(progress_file, checkpoint_index, checkpoints):
    """Rewind to a specific checkpoint"""
    if checkpoint_index < 0 or checkpoint_index >= len(checkpoints):
        print(f"Invalid checkpoint index: {checkpoint_index}")
        return False
    
    target_checkpoint = checkpoints[checkpoint_index]
    removed_count = len(checkpoints) - (checkpoint_index + 1)
    
    print(f"\n{'='*60}")
    print("REWIND OPERATION")
    print(f"{'='*60}")
    
    get_checkpoint_details(target_checkpoint, checkpoint_index, checkpoints)
    
    print(f"\n  REWIND LOGIC:")
    print(f"    KEEP: Checkpoints 0 through {checkpoint_index} (this checkpoint and everything before)")
    print(f"    REMOVE: Checkpoints {checkpoint_index + 1} through {len(checkpoints) - 1} (everything after this checkpoint)")
    print(f"    RESTART: Pipeline will resume from the next action after this checkpoint")
    
    if removed_count > 0:
        print(f"\nThis will remove {removed_count} checkpoint(s):")
        for i in range(checkpoint_index + 1, len(checkpoints)):
            cp = checkpoints[i]
            cat_info = categorize_checkpoint(cp.get('type', ''))
            timestamp_info = format_timestamp(cp.get('timestamp', ''))
            print(f"   [{timestamp_info['time']}] {cp.get('type', 'Unknown')} WILL BE DELETED")
        
        # Show what the next action will be
        next_action = predict_next_action(target_checkpoint, checkpoints)
        if next_action:
            print(f"\nNEXT ACTION: Pipeline will resume with: {next_action}")
    else:
        print(f"\nNo checkpoints to remove (already at latest)")
        return False
    
    # Confirm
    while True:
        print(f"\n  SUMMARY:")
        print(f"   • Keep checkpoint #{checkpoint_index}: {target_checkpoint.get('type', 'Unknown')}")
        print(f"   • Remove {removed_count} checkpoints after this point")
        print(f"   • Resume from: {next_action}")
        
        response = input(f"\nProceed with rewind? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no']:
            print("Rewind cancelled")
            return False
        else:
            print("Please enter 'y' or 'n'")
    
    # Create backup
    backup_file = progress_file.with_suffix('.backup.json')
    try:
        import shutil
        shutil.copy2(progress_file, backup_file)
        print(f"Created backup: {backup_file}")
    except Exception as e:
        print(f"Could not create backup: {e}")
    
    # Perform rewind
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        # Truncate checkpoints
        progress_data['checkpoints'] = checkpoints[:checkpoint_index + 1]
        
        # Save modified progress
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"✅ Successfully rewound to checkpoint #{checkpoint_index}")
        print(f"🔄 Pipeline will resume from: {next_action}")
        return True
        
    except Exception as e:
        print(f"Error during rewind: {e}")
        return False

def predict_next_action(target_checkpoint, checkpoints):
    """Predict what the next action will be after rewinding to this checkpoint"""
    checkpoint_type = target_checkpoint.get('type', '')
    
    if 'SECTION_' in checkpoint_type:
        try:
            parts = checkpoint_type.split('_')
            section_num = int(parts[1])
            action = '_'.join(parts[2:])
            
            if action == 'START':
                return f"TTS generation for Section {section_num}"
            elif action == 'TTS_COMPLETE':
                return f"RVC processing for Section {section_num}"
            elif action == 'RVC_COMPLETE':
                return f"Master combination for Section {section_num}"
            elif action == 'MASTER_COMPLETE' or action == 'MASTER_FAILED':
                return f"Mark Section {section_num} complete"
            elif action == 'COMPLETE':
                return f"Start processing Section {section_num + 1}"
            else:
                return f"Continue Section {section_num} processing"
        except:
            pass
    
    if 'PREPROCESSING_COMPLETE' in checkpoint_type:
        return "Start section processing loop"
    elif 'PIPELINE_START' in checkpoint_type:
        return "Begin preprocessing phase"
    
    return "Resume pipeline processing"

def interactive_session(progress_file, progress_data):
    """Run interactive checkpoint management session"""
    checkpoints = progress_data.get('checkpoints', [])
    
    while True:
        print(f"\n{'='*80}")
        print("  PIPELINE REWIND - INTERACTIVE MODE")
        print(f"{'='*80}")
        print("Commands:")
        print("  s  - Show progress summary")
        print("  l  - List all checkpoints")
        print("  d  - Show details for specific checkpoint")
        print("  r  - Rewind to specific checkpoint")
        print("  q  - Quit")
        print("-"*80)
        
        try:
            command = input("Enter command: ").strip().lower()
            
            if command in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
                
            elif command in ['s', 'summary']:
                display_progress_summary(progress_data)
                
            elif command in ['l', 'list']:
                display_checkpoints(checkpoints)
                
            elif command in ['d', 'details']:
                if not checkpoints:
                    print(" No checkpoints available")
                    continue
                
                try:
                    index = int(input(f"Enter checkpoint index (0-{len(checkpoints)-1}): "))
                    if 0 <= index < len(checkpoints):
                        get_checkpoint_details(checkpoints[index], index, checkpoints)
                    else:
                        print(f" Invalid index. Must be 0-{len(checkpoints)-1}")
                except ValueError:
                    print(" Please enter a valid number")
                    
            elif command in ['r', 'rewind']:
                if not checkpoints:
                    print(" No checkpoints available")
                    continue
                
                if len(checkpoints) <= 1:
                    print(" Need at least 2 checkpoints to rewind")
                    continue
                
                display_checkpoints(checkpoints)
                
                try:
                    index = int(input(f"\nEnter checkpoint index to rewind to (0-{len(checkpoints)-2}): "))
                    if 0 <= index < len(checkpoints) - 1:
                        if perform_rewind(progress_file, index, checkpoints):
                            # Reload data after rewind
                            progress_data = load_progress_data(progress_file)
                            if progress_data:
                                checkpoints = progress_data.get('checkpoints', [])
                            else:
                                print(" Error reloading progress data")
                                break
                    else:
                        print(f" Invalid index. Must be 0-{len(checkpoints)-2}")
                except ValueError:
                    print(" Please enter a valid number")
                    
            else:
                print(f" Unknown command: {command}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Find progress file
    progress_file = find_progress_file(args.project, args.job)
    
    if not progress_file:
        print(f" Could not find progress.log for project '{args.project}' job '{args.job}'")
        print(f"\nLooked for:")
        print(f"  - output/{args.project}/jobs/{args.job}/progress.log")
        print(f"  - {args.project}/jobs/{args.job}/progress.log")
        print(f"  - {args.project}_{args.job}/progress.log")
        print(f"  - jobs/{args.project}/{args.job}/progress.log")
        sys.exit(1)
    
    print(f" Found progress file: {progress_file}")
    
    # Load progress data
    progress_data = load_progress_data(progress_file)
    if not progress_data:
        sys.exit(1)
    
    # Start interactive session
    interactive_session(progress_file, progress_data)

if __name__ == "__main__":
    main()
