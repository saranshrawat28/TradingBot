"""
Installs ApexTrade background runner into Windows user Startup folder.
"""

import os
from pathlib import Path

def install_startup():
    startup_dir = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    vbs_path = startup_dir / "ApexTrade_AutoStart.vbs"
    runner_path = Path(__file__).resolve().parent / "background_runner.py"

    vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw ""{runner_path}""", 0, False
Set WshShell = Nothing
'''

    with open(vbs_path, "w", encoding="utf-8") as f:
        f.write(vbs_content)

    print(f"[Installer] Created Windows Autostart shortcut: {vbs_path}")
    print("[Installer] ApexTrade will now automatically run silently whenever Windows starts!")

if __name__ == "__main__":
    install_startup()
