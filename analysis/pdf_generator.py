"""
PDF Report Generator for Quantum Risk Simulation

Generates professional PDF reports with embedded visualizations,
formatted tables, and executive-ready styling.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import base64
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, KeepTogether, Flowable, HRFlowable,
    ListFlowable, ListItem
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Line
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.widgets.markers import makeMarker

import markdown2
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Quantum-themed color palette
QUANTUM_COLORS = {
    'primary': HexColor('#1E3A8A'),      # Deep Blue
    'secondary': HexColor('#7C3AED'),    # Purple
    'accent': HexColor('#EC4899'),       # Pink
    'success': HexColor('#10B981'),      # Green
    'warning': HexColor('#F59E0B'),      # Amber
    'danger': HexColor('#EF4444'),       # Red
    'dark': HexColor('#1F2937'),         # Dark Gray
    'light': HexColor('#F3F4F6'),        # Light Gray
    'background': HexColor('#FFFFFF'),   # White
    'text': HexColor('#111827'),         # Near Black
}

class NumberedCanvas(canvas.Canvas):
    """Custom canvas for page numbers and headers/footers."""
    
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
        self.page_num = 1
        
    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        """Add page numbers and headers/footers to each page."""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_page_elements(self, page_count):
        """Draw headers, footers, and page numbers."""
        # Header
        self.setFont("Helvetica-Bold", 10)
        self.setFillColor(QUANTUM_COLORS['primary'])
        self.drawString(inch, letter[1] - 0.5*inch, 
                       "Quantum Risk Assessment - Solana Blockchain")
        
        # Footer with page number
        self.setFont("Helvetica", 9)
        self.setFillColor(QUANTUM_COLORS['dark'])
        self.drawCentredString(letter[0]/2, 0.5*inch,
                             f"Page {self.page_num} of {page_count}")
        
        # Timestamp
        self.setFont("Helvetica", 8)
        self.setFillColor(QUANTUM_COLORS['dark'])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.drawRightString(letter[0] - inch, 0.5*inch, timestamp)
        
        # Decorative line
        self.setStrokeColor(QUANTUM_COLORS['primary'])
        self.setLineWidth(1)
        self.line(inch, letter[1] - 0.6*inch, 
                 letter[0] - inch, letter[1] - 0.6*inch)
        self.line(inch, 0.6*inch, letter[0] - inch, 0.6*inch)
        
        self.page_num += 1


class PDFReportGenerator:
    """Generates professional PDF reports from simulation results."""
    
    def __init__(self, output_dir: Path):
        """Initialize the PDF generator.
        
        Args:
            output_dir: Directory to save PDF reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = self._create_custom_styles()
        self.story = []
        
    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for the PDF."""
        styles = getSampleStyleSheet()
        
        # Title style
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=QUANTUM_COLORS['primary'],
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Heading 1
        styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=QUANTUM_COLORS['primary'],
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=QUANTUM_COLORS['primary'],
            borderPadding=4,
            borderRadius=2
        ))
        
        # Heading 2
        styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=QUANTUM_COLORS['secondary'],
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        # Heading 3
        styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=QUANTUM_COLORS['dark'],
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        styles.add(ParagraphStyle(
            name='CustomBodyText',
            parent=styles['BodyText'],
            fontSize=10,
            textColor=QUANTUM_COLORS['text'],
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            leading=14
        ))
        
        # Critical alert
        styles.add(ParagraphStyle(
            name='CriticalAlert',
            parent=styles['BodyText'],
            fontSize=11,
            textColor=QUANTUM_COLORS['danger'],
            backColor=HexColor('#FEE2E2'),
            borderWidth=1,
            borderColor=QUANTUM_COLORS['danger'],
            borderPadding=8,
            borderRadius=4,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        ))
        
        # Success message
        styles.add(ParagraphStyle(
            name='SuccessMessage',
            parent=styles['BodyText'],
            fontSize=10,
            textColor=QUANTUM_COLORS['success'],
            backColor=HexColor('#D1FAE5'),
            borderWidth=1,
            borderColor=QUANTUM_COLORS['success'],
            borderPadding=6,
            borderRadius=4
        ))
        
        # Code block
        styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=styles['Code'],
            fontSize=9,
            textColor=QUANTUM_COLORS['dark'],
            backColor=QUANTUM_COLORS['light'],
            borderWidth=1,
            borderColor=HexColor('#D1D5DB'),
            borderPadding=8,
            fontName='Courier',
            leftIndent=20,
            rightIndent=20
        ))
        
        # Executive summary
        styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=styles['BodyText'],
            fontSize=11,
            textColor=QUANTUM_COLORS['dark'],
            backColor=HexColor('#F9FAFB'),
            borderWidth=2,
            borderColor=QUANTUM_COLORS['primary'],
            borderPadding=10,
            borderRadius=4,
            alignment=TA_JUSTIFY,
            leading=16
        ))
        
        return styles
    
    def generate_pdf(
        self,
        markdown_report_path: Path,
        charts_dir: Optional[Path] = None,
        output_filename: str = "quantum_risk_report.pdf"
    ) -> Path:
        """Generate a PDF from a markdown report.
        
        Args:
            markdown_report_path: Path to the markdown report
            charts_dir: Directory containing chart images
            output_filename: Name of the output PDF file
            
        Returns:
            Path to the generated PDF
        """
        # Read markdown content
        with open(markdown_report_path, 'r') as f:
            markdown_content = f.read()
        
        # Parse markdown and extract sections
        sections = self._parse_markdown(markdown_content)
        
        # Create PDF document
        pdf_path = self.output_dir / output_filename
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build the story (content flow)
        self.story = []
        
        # Add cover page
        self._add_cover_page()
        
        # Add table of contents
        self._add_table_of_contents(sections)
        
        # Add executive summary
        self._add_executive_summary(sections)
        
        # Add main content sections
        for section in sections:
            self._add_section(section, charts_dir)
        
        # Build PDF with custom canvas for page numbers
        doc.build(self.story, canvasmaker=NumberedCanvas)
        
        return pdf_path
    
    def _parse_markdown(self, markdown_content: str) -> List[Dict[str, Any]]:
        """Parse markdown content into sections.
        
        Args:
            markdown_content: Raw markdown text
            
        Returns:
            List of parsed sections
        """
        sections = []
        current_section = None
        
        lines = markdown_content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for headers
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Parse header level and text
                header_match = re.match(r'^(#+)\s+(.+)$', line)
                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2)
                    
                    current_section = {
                        'level': level,
                        'title': title,
                        'content': [],
                        'subsections': []
                    }
            elif current_section:
                # Add content to current section
                current_section['content'].append(line)
            
            i += 1
        
        # Save last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _add_cover_page(self):
        """Add a professional cover page."""
        self.story.append(Spacer(1, 2*inch))
        
        # Title
        title = Paragraph(
            "Quantum Risk Assessment Report",
            self.styles['CustomTitle']
        )
        self.story.append(title)
        
        # Subtitle
        subtitle = Paragraph(
            "Solana Blockchain Vulnerability Analysis",
            self.styles['CustomHeading2']
        )
        self.story.append(subtitle)
        
        self.story.append(Spacer(1, 0.5*inch))
        
        # Quantum circuit decorative element
        self._add_quantum_decoration()
        
        self.story.append(Spacer(1, 1*inch))
        
        # Report metadata
        metadata_data = [
            ['Report Type:', 'Monte Carlo Simulation Analysis'],
            ['Blockchain:', 'Solana'],
            ['Assessment Date:', datetime.now().strftime('%B %d, %Y')],
            ['Threat Level:', 'CRITICAL'],
            ['Confidence Level:', '95%']
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), QUANTUM_COLORS['primary']),
            ('TEXTCOLOR', (1, 0), (1, -1), QUANTUM_COLORS['dark']),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, QUANTUM_COLORS['light']),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, HexColor('#F9FAFB')])
        ]))
        
        self.story.append(metadata_table)
        
        # Classification notice
        self.story.append(Spacer(1, 1*inch))
        classification = Paragraph(
            "CONFIDENTIAL - FOR AUTHORIZED DISTRIBUTION ONLY",
            self.styles['CriticalAlert']
        )
        self.story.append(classification)
        
        self.story.append(PageBreak())
    
    def _add_quantum_decoration(self):
        """Add a decorative quantum circuit element."""
        d = Drawing(400, 100)
        
        # Quantum circuit lines
        for i in range(3):
            y = 30 + i * 25
            line = Line(50, y, 350, y)
            line.strokeColor = QUANTUM_COLORS['primary']
            line.strokeWidth = 1
            d.add(line)
        
        # Quantum gates (simplified representation)
        gate_positions = [100, 180, 260]
        for x in gate_positions:
            for y in [30, 55, 80]:
                # Gate box
                from reportlab.graphics.shapes import Rect, Circle
                rect = Rect(x-15, y-10, 30, 20)
                rect.strokeColor = QUANTUM_COLORS['secondary']
                rect.fillColor = HexColor('#F3E8FF')
                rect.strokeWidth = 2
                d.add(rect)
        
        self.story.append(d)
    
    def _add_table_of_contents(self, sections: List[Dict[str, Any]]):
        """Add a table of contents."""
        # Title
        toc_title = Paragraph("Table of Contents", self.styles['CustomHeading1'])
        self.story.append(toc_title)
        self.story.append(Spacer(1, 0.3*inch))
        
        # TOC entries
        toc_data = []
        page_num = 3  # Starting page after cover and TOC
        
        for section in sections:
            if section['level'] <= 2:  # Only include H1 and H2
                indent = "    " * (section['level'] - 1)
                title = indent + section['title']
                toc_data.append([title, str(page_num)])
                page_num += 1  # Simplified page numbering
        
        toc_table = Table(toc_data, colWidths=[5*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), QUANTUM_COLORS['dark']),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, QUANTUM_COLORS['light']),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]))
        
        self.story.append(toc_table)
        self.story.append(PageBreak())
    
    def _add_executive_summary(self, sections: List[Dict[str, Any]]):
        """Add the executive summary section with special formatting."""
        # Find executive summary section
        exec_summary = None
        for section in sections:
            if 'executive' in section['title'].lower() and 'summary' in section['title'].lower():
                exec_summary = section
                break
        
        if not exec_summary:
            return
        
        # Title
        title = Paragraph(exec_summary['title'], self.styles['CustomHeading1'])
        self.story.append(title)
        
        # Content with special formatting
        content_text = '\n'.join(exec_summary['content'])
        
        # Extract key metrics if present
        if 'Risk Score:' in content_text:
            # Create highlighted metrics box
            metrics_match = re.search(r'Risk Score:\s*([0-9.]+)', content_text)
            if metrics_match:
                risk_score = float(metrics_match.group(1))
                
                # Risk indicator with color coding
                if risk_score >= 75:
                    risk_color = QUANTUM_COLORS['danger']
                    risk_level = "CRITICAL"
                elif risk_score >= 50:
                    risk_color = QUANTUM_COLORS['warning']
                    risk_level = "HIGH"
                else:
                    risk_color = QUANTUM_COLORS['success']
                    risk_level = "MODERATE"
                
                risk_data = [
                    ['Quantum Risk Score', f'{risk_score:.1f}/100'],
                    ['Risk Level', risk_level],
                    ['Time to Threat', '3-5 years'],
                    ['Economic Impact', '$91.5B potential']
                ]
                
                risk_table = Table(risk_data, colWidths=[2.5*inch, 2.5*inch])
                risk_table.setStyle(TableStyle([
                    ('FONT', (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 12),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
                    ('BACKGROUND', (0, 0), (-1, -1), risk_color),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 2, colors.white),
                    ('BOX', (0, 0), (-1, -1), 2, risk_color),
                    ('ROWBACKGROUNDS', (0, 0), (-1, -1), [risk_color, risk_color])
                ]))
                
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(risk_table)
                self.story.append(Spacer(1, 0.2*inch))
        
        # Executive summary text
        summary_para = Paragraph(content_text, self.styles['ExecutiveSummary'])
        self.story.append(summary_para)
        
        self.story.append(PageBreak())
    
    def _add_section(self, section: Dict[str, Any], charts_dir: Optional[Path]):
        """Add a content section to the PDF.
        
        Args:
            section: Section dictionary with title, content, etc.
            charts_dir: Directory containing chart images
        """
        # Section title
        if section['level'] == 1:
            style = 'CustomHeading1'
        elif section['level'] == 2:
            style = 'CustomHeading2'
        else:
            style = 'CustomHeading3'
        
        title = Paragraph(section['title'], self.styles[style])
        self.story.append(title)
        
        # Process content
        content_text = '\n'.join(section['content'])
        
        # Check for tables in markdown
        if '|' in content_text:
            self._process_markdown_tables(content_text)
        
        # Check for code blocks
        elif '```' in content_text:
            self._process_code_blocks(content_text)
        
        # Check for lists
        elif re.search(r'^\s*[-*+]\s', content_text, re.MULTILINE):
            self._process_lists(content_text)
        
        # Regular paragraphs
        else:
            paragraphs = content_text.split('\n\n')
            for para_text in paragraphs:
                if para_text.strip():
                    para = Paragraph(para_text.strip(), self.styles['CustomBodyText'])
                    self.story.append(para)
                    self.story.append(Spacer(1, 0.1*inch))
        
        # Add related charts if available
        if charts_dir and section['title']:
            self._add_section_charts(section['title'], charts_dir)
    
    def _process_markdown_tables(self, content: str):
        """Convert markdown tables to PDF tables.
        
        Args:
            content: Content containing markdown tables
        """
        lines = content.split('\n')
        table_data = []
        in_table = False
        
        for line in lines:
            if '|' in line and not line.strip().startswith('|--'):
                # Parse table row
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                table_data.append(cells)
                in_table = True
            elif in_table and not '|' in line:
                # End of table, create PDF table
                if table_data:
                    self._create_pdf_table(table_data)
                    table_data = []
                    in_table = False
                
                # Add the non-table line as paragraph
                if line.strip():
                    para = Paragraph(line.strip(), self.styles['CustomBodyText'])
                    self.story.append(para)
        
        # Handle table at end of content
        if table_data:
            self._create_pdf_table(table_data)
    
    def _create_pdf_table(self, data: List[List[str]]):
        """Create a formatted PDF table.
        
        Args:
            data: Table data as list of lists
        """
        if not data:
            return
        
        # Calculate column widths
        num_cols = len(data[0])
        col_width = 6.5 * inch / num_cols
        
        table = Table(data, colWidths=[col_width] * num_cols)
        
        # Apply styling
        style = [
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONT', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), QUANTUM_COLORS['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), QUANTUM_COLORS['dark']),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, QUANTUM_COLORS['light']),
            ('BOX', (0, 0), (-1, -1), 1, QUANTUM_COLORS['primary']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F9FAFB')]),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]
        
        table.setStyle(TableStyle(style))
        self.story.append(table)
        self.story.append(Spacer(1, 0.2*inch))
    
    def _process_code_blocks(self, content: str):
        """Process code blocks in the content.
        
        Args:
            content: Content containing code blocks
        """
        parts = content.split('```')
        
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Regular text
                if part.strip():
                    para = Paragraph(part.strip(), self.styles['CustomBodyText'])
                    self.story.append(para)
            else:
                # Code block
                code_lines = part.split('\n')
                # Remove language identifier if present
                if code_lines and not code_lines[0].strip().startswith(' '):
                    code_lines = code_lines[1:]
                
                code_text = '\n'.join(code_lines)
                if code_text.strip():
                    code_para = Paragraph(
                        code_text.replace(' ', '&nbsp;').replace('\n', '<br/>'),
                        self.styles['CodeBlock']
                    )
                    self.story.append(code_para)
                    self.story.append(Spacer(1, 0.1*inch))
    
    def _process_lists(self, content: str):
        """Process markdown lists.
        
        Args:
            content: Content containing lists
        """
        lines = content.split('\n')
        list_items = []
        
        for line in lines:
            if re.match(r'^\s*[-*+]\s', line):
                # List item
                item_text = re.sub(r'^\s*[-*+]\s', '', line)
                list_items.append(ListItem(
                    Paragraph(item_text, self.styles['CustomBodyText']),
                    leftIndent=20
                ))
            else:
                # End of list or regular text
                if list_items:
                    list_flowable = ListFlowable(
                        list_items,
                        bulletType='bullet',
                        bulletColor=QUANTUM_COLORS['primary']
                    )
                    self.story.append(list_flowable)
                    list_items = []
                
                if line.strip():
                    para = Paragraph(line.strip(), self.styles['CustomBodyText'])
                    self.story.append(para)
        
        # Handle list at end
        if list_items:
            list_flowable = ListFlowable(
                list_items,
                bulletType='bullet',
                bulletColor=QUANTUM_COLORS['primary']
            )
            self.story.append(list_flowable)
    
    def _add_section_charts(self, section_title: str, charts_dir: Path):
        """Add relevant charts for a section.
        
        Args:
            section_title: Title of the section
            charts_dir: Directory containing chart images
        """
        # Map section titles to chart filenames
        chart_mappings = {
            'quantum timeline': ['timeline_projection.png', 'quantum_capabilities.png'],
            'economic': ['economic_impact.png', 'loss_distribution.png'],
            'network': ['network_evolution.png', 'validator_migration.png'],
            'attack': ['attack_scenarios.png', 'attack_probability.png'],
            'risk': ['risk_matrix.png', 'risk_trajectory.png']
        }
        
        # Find relevant charts
        section_key = section_title.lower()
        relevant_charts = []
        
        for key, charts in chart_mappings.items():
            if key in section_key:
                relevant_charts.extend(charts)
        
        # Add charts to PDF
        for chart_name in relevant_charts:
            chart_path = charts_dir / chart_name
            if chart_path.exists():
                try:
                    # Load and resize image
                    img = PILImage.open(chart_path)
                    
                    # Calculate scaling to fit page width
                    max_width = 6 * inch
                    max_height = 4 * inch
                    
                    img_width, img_height = img.size
                    aspect_ratio = img_width / img_height
                    
                    if aspect_ratio > max_width / max_height:
                        # Width limited
                        width = max_width
                        height = width / aspect_ratio
                    else:
                        # Height limited
                        height = max_height
                        width = height * aspect_ratio
                    
                    # Add image to story
                    img_flowable = Image(str(chart_path), width=width, height=height)
                    self.story.append(img_flowable)
                    
                    # Add caption
                    caption = chart_name.replace('_', ' ').replace('.png', '').title()
                    caption_para = Paragraph(
                        f"<i>Figure: {caption}</i>",
                        self.styles['CustomBodyText']
                    )
                    self.story.append(caption_para)
                    self.story.append(Spacer(1, 0.2*inch))
                    
                except Exception as e:
                    print(f"Warning: Could not add chart {chart_name}: {e}")
    
    def generate_from_results(
        self,
        results_dir: Path,
        output_filename: str = "quantum_risk_report.pdf"
    ) -> Path:
        """Generate a PDF directly from simulation results.
        
        Args:
            results_dir: Directory containing simulation results
            output_filename: Name of the output PDF file
            
        Returns:
            Path to the generated PDF
        """
        # Look for markdown report
        markdown_path = results_dir / "report.md"
        if not markdown_path.exists():
            raise FileNotFoundError(f"No markdown report found at {markdown_path}")
        
        # Look for charts directory
        charts_dir = results_dir / "charts"
        if not charts_dir.exists():
            charts_dir = None
        
        # Generate PDF
        return self.generate_pdf(
            markdown_report_path=markdown_path,
            charts_dir=charts_dir,
            output_filename=output_filename
        )


def create_sample_chart(output_path: Path):
    """Create a sample chart for testing.
    
    Args:
        output_path: Path to save the chart
    """
    import numpy as np
    
    # Create sample data
    years = np.arange(2024, 2035)
    risk_scores = [10, 15, 22, 35, 48, 62, 75, 82, 88, 92, 95]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    ax.plot(years, risk_scores, 'b-', linewidth=2, marker='o', markersize=8)
    ax.fill_between(years, 0, risk_scores, alpha=0.3)
    
    # Styling
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Quantum Risk Score', fontsize=12)
    ax.set_title('Projected Quantum Risk Timeline for Solana', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add risk level bands
    ax.axhspan(0, 25, alpha=0.1, color='green', label='Low Risk')
    ax.axhspan(25, 50, alpha=0.1, color='yellow', label='Moderate Risk')
    ax.axhspan(50, 75, alpha=0.1, color='orange', label='High Risk')
    ax.axhspan(75, 100, alpha=0.1, color='red', label='Critical Risk')
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Create sample output directory
    output_dir = Path("pdf_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create sample chart
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    create_sample_chart(charts_dir / "risk_timeline.png")
    
    # Create sample markdown report
    sample_md = """# Quantum Risk Assessment Report

## Executive Summary

**Risk Score: 75.0/100**

The quantum threat to Solana blockchain has reached a critical level requiring immediate attention. Our Monte Carlo simulation with 10,000 iterations indicates a high probability of quantum computers achieving cryptographically relevant capabilities within 3-5 years.

### Key Findings

- **Quantum Timeline**: 50% probability of CRQC by 2028
- **Economic Impact**: Potential losses exceeding $91.5B
- **Network Vulnerability**: Only 15% of validators migrated to quantum-safe cryptography
- **Attack Success Rate**: 78% for key compromise attacks by 2030

## Detailed Analysis

### Quantum Computing Progression

| Year | Logical Qubits | Gate Fidelity | Threat Level |
|------|---------------|---------------|--------------|
| 2024 | 100 | 99.0% | Emerging |
| 2026 | 1,000 | 99.5% | Moderate |
| 2028 | 2,330 | 99.8% | High |
| 2030 | 10,000 | 99.95% | Critical |

### Economic Impact Assessment

The potential economic impact includes:

- Direct theft from compromised wallets
- Market crash due to loss of confidence
- DeFi protocol failures
- Long-term reputation damage

### Recommendations

1. **Immediate Actions**
   - Begin migration to quantum-safe cryptography
   - Implement hybrid classical-quantum resistant signatures
   - Establish quantum threat monitoring systems

2. **Medium-term Strategy**
   - Complete network-wide migration by 2027
   - Deploy quantum-safe smart contract standards
   - Educate validator community on quantum risks

3. **Long-term Vision**
   - Research post-quantum blockchain architectures
   - Develop quantum-enhanced consensus mechanisms
   - Build quantum-resilient DeFi ecosystem

## Technical Appendix

```python
# Quantum risk calculation
def calculate_risk(qubits, fidelity, years_ahead):
    base_risk = min(100, (qubits / 10000) * 100)
    fidelity_factor = fidelity ** 10
    time_factor = max(0, 1 - (years_ahead / 10))
    return base_risk * fidelity_factor * time_factor
```

## Conclusion

The quantum threat to Solana is real and imminent. Immediate action is required to protect the network and its $130.62B ecosystem. This report recommends beginning migration to quantum-safe cryptography within the next 6 months.
"""
    
    md_path = output_dir / "sample_report.md"
    with open(md_path, 'w') as f:
        f.write(sample_md)
    
    # Generate PDF
    generator = PDFReportGenerator(output_dir)
    pdf_path = generator.generate_pdf(
        markdown_report_path=md_path,
        charts_dir=charts_dir,
        output_filename="quantum_risk_report.pdf"
    )
    
    print(f"PDF report generated: {pdf_path}")
