"""
Automated Red Teaming Script
----------------------------
Executes a suite of adversarial attacks against the Phoenix Controller.
Generates a security compliance report.
"""
import sys
import os

# Auto-mount
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.redteam.scanner import RedTeamScanner
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)


def main():
    scanner = RedTeamScanner()
    report_df = scanner.scan_all()

    # Save Report
    output_path = os.path.join(current_dir, "results", "red_team_report.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # [FIX] Force UTF-8 encoding to support Emojis on Windows
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 🛡️ DynaAlign Red Team Safety Report\n\n")
        f.write("Automated adversarial testing results for Phoenix V9.3 Controller.\n\n")
        f.write(report_df.to_markdown(index=False))

    print("-" * 60)
    print(f"📄 Security Report generated: {output_path}")


if __name__ == "__main__":
    main()