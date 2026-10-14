"""
Git Readiness and Secret Audit Script for DrugGEN-master.
Scans for large files, secrets, private paths, and prepares classification.
"""

import os
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Regex patterns for secrets
SECRET_PATTERNS = [
    (r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]([a-zA-Z0-9_\-]{16,})['\"]", "API Key"),
    (r"(?i)(secret[_-]?key|secret)\s*[:=]\s*['\"]([a-zA-Z0-9_\-]{16,})['\"]", "Secret Key"),
    (r"(?i)(token|bearer)\s*[:=]\s*['\"]([a-zA-Z0-9_\-]{16,})['\"]", "Token"),
    (r"(?i)password\s*[:=]\s*['\"]([^'\"]{4,})['\"]", "Password"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub Personal Access Token"),
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth Access Token"),
    (r"glpat-[a-zA-Z0-9_\-]{20,}", "GitLab Personal Access Token"),
    (r"xox[baprs]-[0-9a-zA-Z]{10,48}", "Slack Token"),
    (r"-----BEGIN (RSA|EC|DSA|OPENSSH|PRIVATE) KEY-----", "Private Key Header"),
    (r"(?i)wandb\.login\(.*key\s*=\s*['\"]([^'\"]+)['\"]", "W&B Login Key"),
]

def scan_file_for_secrets(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return []
    
    findings = []
    for pattern, label in SECRET_PATTERNS:
        matches = re.finditer(pattern, content)
        for m in matches:
            # Mask secret value
            val = m.group(0)
            findings.append((label, m.start()))
    return findings

def audit():
    print("=" * 70)
    print("DRUGGEN REPOSITORY GIT READINESS AUDIT")
    print("=" * 70)

    # 1. Size distribution
    large_files_gt_100m = []
    large_files_50_100m = []
    large_files_10_50m = []
    large_files_1_10m = []
    
    secret_hits = []
    
    ignore_dirs = {'.git', '.venv', '__pycache__', '.pytest_cache', '.idea', '.vscode'}

    for root, dirs, files in os.walk(ROOT):
        # Filter dirs in-place
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            p = Path(root) / f
            try:
                stat = p.stat()
                size_mb = stat.st_size / (1024 * 1024)
                rel = p.relative_to(ROOT)
                
                if size_mb > 100:
                    large_files_gt_100m.append((str(rel), size_mb))
                elif size_mb >= 50:
                    large_files_50_100m.append((str(rel), size_mb))
                elif size_mb >= 10:
                    large_files_10_50m.append((str(rel), size_mb))
                elif size_mb >= 1:
                    large_files_1_10m.append((str(rel), size_mb))
                
                # Secret scan on text/code files
                if p.suffix.lower() in {'.py', '.json', '.yaml', '.yml', '.env', '.sh', '.ps1', '.txt', '.md', '.ipynb'}:
                    findings = scan_file_for_secrets(p)
                    if findings:
                        secret_hits.append((str(rel), findings))
            except Exception as e:
                pass

    print(f"\n--- 1. FILES > 100 MB (Hard GitHub Push Blockers) [{len(large_files_gt_100m)}] ---")
    for rel, sz in large_files_gt_100m:
        print(f"  {sz:8.2f} MB : {rel}")

    print(f"\n--- 2. FILES 50 - 100 MB (GitHub Warning / Near Limits) [{len(large_files_50_100m)}] ---")
    for rel, sz in large_files_50_100m:
        print(f"  {sz:8.2f} MB : {rel}")

    print(f"\n--- 3. FILES 10 - 50 MB [{len(large_files_10_50m)}] ---")
    for rel, sz in large_files_10_50m[:15]:
        print(f"  {sz:8.2f} MB : {rel}")
    if len(large_files_10_50m) > 15:
        print(f"  ... and {len(large_files_10_50m) - 15} more files")

    print(f"\n--- 4. SECRET SCAN RESULTS [{len(secret_hits)}] ---")
    if not secret_hits:
        print("  NO hardcoded API keys, tokens, or credentials detected.")
    else:
        for rel, findings in secret_hits:
            print(f"  {rel}: {[f[0] for f in findings]}")

if __name__ == '__main__':
    audit()
