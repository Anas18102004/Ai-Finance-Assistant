#!/usr/bin/env python3
"""
Script to move unnecessary files to Backend2 folder.
"""

import shutil
import os
from pathlib import Path

def move_files_to_backend2():
    """Move unnecessary files to Backend2 folder."""
    
    backend_dir = Path(".")
    backend2_dir = Path("../Backend2")
    
    # Create Backend2 directory if it doesn't exist
    backend2_dir.mkdir(exist_ok=True)
    
    # Files to move (unnecessary/legacy files)
    files_to_move = [
        "main.py",
        "test_system.py", 
        "test_agents.py",
        "test_fixed_agents.py",
        "test_user_switching.py",
        "load_transactions_to_chroma.py"
    ]
    
    # Testing files to move (keep only essential ones)
    testing_files_to_move = [
        "testing/compare_semantic_vs_keyword.py",
        "testing/test_chromadb_rag.py", 
        "testing/test_new_architecture.py",
        "testing/test_proper_rag.py",
        "testing/test_refined_architecture.py",
        "testing/test_sentence_transformers_rag.py"
        # Keep: testing/test_complete_workflow.py (main test)
    ]
    
    moved_files = []
    
    print("🗂️  Moving unnecessary files to Backend2...")
    print("="*50)
    
    # Move main directory files
    for file_name in files_to_move:
        source = backend_dir / file_name
        if source.exists():
            destination = backend2_dir / file_name
            try:
                shutil.move(str(source), str(destination))
                moved_files.append(file_name)
                print(f"✅ Moved: {file_name}")
            except Exception as e:
                print(f"❌ Error moving {file_name}: {e}")
        else:
            print(f"⚠️  File not found: {file_name}")
    
    # Create testing subdirectory in Backend2
    testing_backend2_dir = backend2_dir / "testing"
    testing_backend2_dir.mkdir(exist_ok=True)
    
    # Move testing files
    for file_path in testing_files_to_move:
        source = backend_dir / file_path
        if source.exists():
            destination = backend2_dir / file_path
            try:
                shutil.move(str(source), str(destination))
                moved_files.append(file_path)
                print(f"✅ Moved: {file_path}")
            except Exception as e:
                print(f"❌ Error moving {file_path}: {e}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"\n📊 Summary:")
    print(f"   • Files moved: {len(moved_files)}")
    print(f"   • Destination: ../Backend2/")
    
    print(f"\n🎯 Remaining Essential Files:")
    essential_files = [
        "setup_and_run.py",
        "config.py", 
        "generate_data.py",
        "agents/",
        "api/",
        "services/",
        "tools/",
        "nodes/",
        "index_build/",
        "testing/test_complete_workflow.py"
    ]
    
    for file in essential_files:
        if (backend_dir / file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
    
    print(f"\n🧹 Backend directory cleaned up!")
    print(f"   • Moved legacy/test files to Backend2")
    print(f"   • Kept essential multi-agent system files")
    print(f"   • Kept main workflow test for validation")

if __name__ == "__main__":
    move_files_to_backend2()
