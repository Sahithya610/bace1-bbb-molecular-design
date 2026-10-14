"""Download AutoDock-Vina v1.2.7 Windows executable."""
import urllib.request, os, zipfile, sys

os.makedirs("docking/bin", exist_ok=True)
VINA_EXE = "docking/bin/vina.exe"

if os.path.exists(VINA_EXE):
    print(f"Already exists: {VINA_EXE}")
    sys.exit(0)

# Official GitHub release asset
URL = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_win.exe"
print(f"Downloading Vina executable from:\n  {URL}")
try:
    urllib.request.urlretrieve(URL, VINA_EXE)
    print(f"Saved: {VINA_EXE}")
except Exception as e:
    print(f"Direct download failed: {e}")
    # Try zip
    ZIP_URL = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_win.zip"
    ZIP_FILE = "docking/bin/vina_win.zip"
    print(f"Trying zip: {ZIP_URL}")
    try:
        urllib.request.urlretrieve(ZIP_URL, ZIP_FILE)
        with zipfile.ZipFile(ZIP_FILE) as z:
            for name in z.namelist():
                if name.endswith(".exe"):
                    z.extract(name, "docking/bin/")
                    print(f"Extracted: {name}")
        os.remove(ZIP_FILE)
    except Exception as e2:
        print(f"Zip download also failed: {e2}")
        print("\nManual install: download from https://vina.scripps.edu/downloads/")
        print("and place vina.exe in docking/bin/")
        sys.exit(1)

# Verify
import subprocess
r = subprocess.run([VINA_EXE, "--version"], capture_output=True, text=True)
print(f"Vina test: {r.stdout.strip() or r.stderr.strip()}")
