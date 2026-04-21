import os
import glob
import re

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements:
    # 1. Match `# ═══ <TEXT> ═══` or similar and change to `# <TEXT>`
    content = re.sub(r'#\s*[═─]+\s*(.*?)\s*[═─]+', r'# \1', content)
    
    # 2. Sometimes it's just `# ════════════` or `# ─────────`. Remove those entirely.
    content = re.sub(r'#\s*[═─]+\n', '\n', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    py_files = glob.glob("*.py")
    for f in py_files:
        if f != "clean_comments.py":
            clean_file(f)
    print("Files cleaned successfully.")
