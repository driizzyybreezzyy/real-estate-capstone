import os
import zipfile
import fnmatch

def load_gitignore(root_dir):
    gitignore_path = os.path.join(root_dir, '.gitignore')
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    # Add default ignores
    patterns.extend(['.git', '__pycache__', '*.pyc', '*.zip'])
    return patterns

def is_ignored(path, root_dir, patterns):
    rel_path = os.path.relpath(path, root_dir)
    # Check if any part of the path matches a pattern (directory ignores)
    parts = rel_path.split(os.sep)
    for i in range(len(parts)):
        subpath = "/".join(parts[:i+1]) # gitignore uses forward slashes
        for pattern in patterns:
            # Handle directory matches (ending with /)
            if pattern.endswith('/'):
                p = pattern.rstrip('/')
                if fnmatch.fnmatch(subpath, p) or fnmatch.fnmatch(parts[i], p):
                    return True
            # Handle file matches
            elif fnmatch.fnmatch(subpath, pattern) or fnmatch.fnmatch(parts[i], pattern):
                return True
    return False

def backup_project():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backup_name = 'real_estate_capstone_backup.zip'
    backup_path = os.path.join(root_dir, backup_name)
    
    patterns = load_gitignore(root_dir)
    
    print(f"Starting backup of {root_dir}...")
    print(f"Ignoring patterns: {patterns}")
    
    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            # Filter directories in-place to avoid walking ignored trees
            dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), root_dir, patterns)]
            
            for file in files:
                file_path = os.path.join(root, file)
                if file == backup_name:
                    continue
                if not is_ignored(file_path, root_dir, patterns):
                    arcname = os.path.relpath(file_path, root_dir)
                    print(f"Adding: {arcname}")
                    zipf.write(file_path, arcname)
                    
    print(f"\n[+] Backup created successfully: {backup_path}")

if __name__ == "__main__":
    backup_project()
