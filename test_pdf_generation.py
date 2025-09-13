#!/usr/bin/env python3
"""
Test PDF generation from existing simulation results.
"""

from pathlib import Path
from analysis.pdf_generator import PDFReportGenerator

def test_pdf_generation():
    """Test PDF generation with existing simulation results."""
    
    # Find the most recent simulation run with a markdown report
    results_dir = Path("simulation_results")
    
    # Look for run with markdown report
    md_report_path = results_dir / "run_20250912_221951" / "reports" / "simulation_report.md"
    
    if not md_report_path.exists():
        print(f"❌ Markdown report not found at: {md_report_path}")
        return
    
    print(f"✓ Found markdown report: {md_report_path}")
    
    # Create PDF generator
    output_dir = md_report_path.parent
    pdf_generator = PDFReportGenerator(output_dir)
    
    # Look for charts directory
    charts_dir = md_report_path.parent.parent / "plots"
    if not charts_dir.exists():
        print("⚠ Charts directory not found, generating PDF without embedded charts")
        charts_dir = None
    else:
        print(f"✓ Found charts directory: {charts_dir}")
    
    # Generate PDF
    try:
        print("\nGenerating PDF report...")
        pdf_path = pdf_generator.generate_pdf(
            markdown_report_path=md_report_path,
            charts_dir=charts_dir,
            output_filename="quantum_risk_report_test.pdf"
        )
        print(f"✅ PDF successfully generated: {pdf_path}")
        print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")
        
        # Open the PDF (macOS specific)
        import subprocess
        try:
            subprocess.run(["open", str(pdf_path)], check=True)
            print("   PDF opened in default viewer")
        except:
            print(f"   Please open: {pdf_path}")
            
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pdf_generation()
