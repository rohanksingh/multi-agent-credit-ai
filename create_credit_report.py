
from pathlib import Path
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

output = Path("data/companies/ABC_CORP/credit/abc_credit_2025.pdf")
output.parent.mkdir(parents=True, exist_ok=True)

doc = SimpleDocTemplate(str(output))
styles = getSampleStyleSheet()
story = []

story.append(Paragraph(
    "ABC Corporation - Synthetic Corporate Credit Report",
    styles["Title"]
))
story.append(Paragraph(
    "FICTIONAL TEST DATA - NOT A REAL CREDIT REPORT",
    styles["Heading2"]
))
story.append(Spacer(1, 16))

data = [
    ["Metric", "Value"],
    ["Company ID", "ABC_CORP"],
    ["Reporting Year", "2025"],
    ["Total Credit Limit", "$10,000,000"],
    ["Outstanding Credit Balance", "$6,500,000"],
    ["Credit Utilization", "65%"],
    ["On-Time Payment Rate", "92%"],
    ["Payments 30+ Days Late", "3"],
    ["Payments 60+ Days Late", "1"],
    ["Defaults", "0"],
    ["Synthetic Internal Risk Score", "68 / 100"],
]

table = Table(data, colWidths=[240, 220])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("PADDING", (0, 0), (-1, -1), 8),
]))

story.append(table)
story.append(Spacer(1, 20))

story.append(Paragraph("Credit Risk Observations", styles["Heading2"]))
story.append(Paragraph(
    "ABC Corporation has moderate credit utilization. "
    "The fictional payment history includes late payments, "
    "which may indicate elevated repayment risk. "
    "The internal risk score is illustrative and is not "
    "a bureau score or validated credit rating.",
    styles["BodyText"]
))

doc.build(story)
print(f"Created: {output}")
