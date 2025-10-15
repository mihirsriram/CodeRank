import sys, os, subprocess

# Ensure coderank_lc is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Launch Streamlit from within the package
subprocess.run([
    "streamlit", "run", "coderank_lc/ui/streamlit_app.py",
    "--server.port=7860", "--server.address=0.0.0.0"
])
