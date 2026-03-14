from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

styles = getSampleStyleSheet()

doc = SimpleDocTemplate("test_fir.pdf", pagesize=letter)
story = []
story.append(Paragraph("FIRST INFORMATION REPORT", styles["Title"]))
story.append(Paragraph("FIR No: 2024/001", styles["Normal"]))
story.append(Spacer(1, 12))
story.append(Paragraph("1. BACKGROUND", styles["Heading1"]))
story.append(Paragraph(
    "On the date of the Valencia Grand Prix the defendant driver "
    "Kevin Magnussen was operating vehicle No. 20. Station House "
    "Officer received complaint at 14:32 local time.", styles["Normal"]))
story.append(Spacer(1, 12))
story.append(Paragraph("2. STATEMENT OF FACTS", styles["Heading1"]))
story.append(Paragraph(
    "The gearbox of vehicle No. 20 showed signs of prior damage "
    "before the race commenced. Engineering telemetry confirmed "
    "anomalous gear shift patterns from lap 3 onwards. The driver "
    "was informed via radio communication at lap 5 that the gearbox "
    "was operating outside normal parameters.", styles["Normal"]))
story.append(Spacer(1, 12))
story.append(Paragraph("3. ALLEGATIONS", styles["Heading1"]))
story.append(Paragraph(
    "It is alleged that the defendant had prior knowledge of the "
    "mechanical fault and continued racing. The defendant denies "
    "all knowledge of the fault prior to lap 12 when the gearbox "
    "failed completely.", styles["Normal"]))
doc.build(story)
print("Created test_fir.pdf")

doc2 = SimpleDocTemplate("test_witness_statement.pdf", pagesize=letter)
story2 = []
story2.append(Paragraph("WITNESS STATEMENT", styles["Title"]))
story2.append(Paragraph("Witness: Guenther Steiner, Team Principal", styles["Normal"]))
story2.append(Spacer(1, 12))
story2.append(Paragraph("EXAMINATION", styles["Heading1"]))
story2.append(Paragraph(
    "I hereby state that on the morning of the race I personally "
    "reviewed the pre-race engineering report. The gearbox was "
    "declared fit for competition by our chief engineer. "
    "I was not aware of any anomaly at the time of race start.",
    styles["Normal"]))
story2.append(Spacer(1, 12))
story2.append(Paragraph("CROSS EXAMINATION", styles["Heading1"]))
story2.append(Paragraph(
    "When pressed on the telemetry data I confirmed that the "
    "engineering team did flag a minor irregularity on lap 3 "
    "but we assessed it as within acceptable tolerance. "
    "The driver was not informed because we did not consider "
    "it a safety risk at that point.", styles["Normal"]))
story2.append(Spacer(1, 12))
story2.append(Paragraph("RE-EXAMINATION", styles["Heading1"]))
story2.append(Paragraph(
    "I stand by my earlier statement. The driver had no knowledge "
    "of the gearbox irregularity until the failure on lap 12. "
    "The decision not to inform the driver was made collectively "
    "by the engineering team.", styles["Normal"]))
doc2.build(story2)
print("Created test_witness_statement.pdf")

doc3 = SimpleDocTemplate("test_court_order.pdf", pagesize=letter)
story3 = []
story3.append(Paragraph("FIA INTERNATIONAL TRIBUNAL", styles["Title"]))
story3.append(Paragraph("COURT ORDER — Case 2024/F1/001", styles["Normal"]))
story3.append(Spacer(1, 12))
story3.append(Paragraph("WHEREAS", styles["Heading1"]))
story3.append(Paragraph(
    "The Tribunal has reviewed all submitted evidence including "
    "telemetry data witness statements and video footage of "
    "the Valencia Grand Prix proceedings.", styles["Normal"]))
story3.append(Spacer(1, 12))
story3.append(Paragraph("THEREFORE", styles["Heading1"]))
story3.append(Paragraph(
    "It is hereby ordered that the defendant team shall produce "
    "all engineering logs and radio communications from lap 1 "
    "through lap 12 of the Valencia Grand Prix.", styles["Normal"]))
story3.append(Spacer(1, 12))
story3.append(Paragraph("ORDER", styles["Heading1"]))
story3.append(Paragraph(
    "The Tribunal finds sufficient evidence to proceed to a full "
    "hearing. The burden of proof rests with the defense to "
    "demonstrate the driver had no prior knowledge of the fault.",
    styles["Normal"]))
doc3.build(story3)
print("Created test_court_order.pdf")
print("\nAll test documents ready.")
