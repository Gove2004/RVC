Set ws = CreateObject("WScript.Shell")
ws.CurrentDirectory = ws.CurrentDirectory
ws.Run """" & ws.CurrentDirectory & "\.venv\Scripts\pythonw.exe"" app.py", 0, False
