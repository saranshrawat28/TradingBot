' Invisible VBScript Launcher for ApexTrade Background Service
Set WshShell = CreateObject("WScript.Shell")
strPath = WshShell.CurrentDirectory
WshShell.Run "pythonw """ & strPath & "\scripts\background_runner.py""", 0, False
Set WshShell = Nothing
