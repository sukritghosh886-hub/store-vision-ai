import os
import subprocess
import sys

port = os.environ.get("PORT", "8501")

cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "app.py",
    "--server.address=0.0.0.0",
    "--server.port=" + port,
    "--server.headless=true",
]

print(f"Starting Store Vision AI UI on port {port}", flush=True)

os.execvp(cmd[0], cmd)