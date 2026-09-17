"""
Headless compilation and initialization test for SolarVision Streamlit dashboard.
"""

import ast
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def test_app_compilation():
    app_path = Path("app.py")
    assert app_path.exists(), "app.py does not exist"
    
    with open(app_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    # Parse AST to ensure zero syntax errors
    tree = ast.parse(source)
    print("app.py AST parse: SUCCESS")

    # Verify key section headers exist in the file
    required_sections = [
        "🏠 Overview",
        "🔬 Solar Image Analysis",
        "🎯 Detection Results",
        "🏷️ Region Classification",
        "🛰️ Multi-Day Tracking",
        "📊 Historical Activity",
        "📈 Scientific Evaluation",
        "📚 Methodology & Limitations",
    ]
    for sec in required_sections:
        assert sec in source, f"Missing section in app.py: {sec}"
        print(f"Verified section presence: {sec}")

    print("\nALL DASHBOARD SYNTAX & SECTION CHECKS PASSED!")

if __name__ == "__main__":
    test_app_compilation()
