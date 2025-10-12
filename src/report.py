"""
Medical report generation.
"""


def generate_medical_report(prediction_result):
    """Generate detailed medical report from prediction."""
    severity = prediction_result['class']
    confidence = prediction_result['confidence']
    class_name = prediction_result['class_name']
    
    report = f"""
{'='*70}
           DIABETIC RETINOPATHY ANALYSIS REPORT              
{'='*70}

DIAGNOSIS: {class_name}
CONFIDENCE: {confidence*100:.2f}%
SEVERITY LEVEL: {severity}/4

DETAILED PROBABILITIES:
{'─'*70}
"""
    
    for cls_name, prob in prediction_result['all_probabilities'].items():
        bar = '█' * int(prob * 30)
        report += f"  {cls_name:20s}: {prob*100:5.2f}% {bar}\n"
    
    report += "\n" + "─" * 70 + "\n"
    report += "CLINICAL RECOMMENDATIONS:\n\n"
    
    if severity == 0:
        report += "  ✓ No signs of diabetic retinopathy detected.\n"
        report += "  ✓ Continue regular eye examinations annually.\n"
        report += "  ✓ Maintain good blood glucose control.\n"
    elif severity == 1:
        report += "  ⚠ Mild diabetic retinopathy detected.\n"
        report += "  • Schedule follow-up examination in 6-12 months.\n"
        report += "  • Optimize diabetes management.\n"
    elif severity == 2:
        report += "  ⚠ Moderate diabetic retinopathy detected.\n"
        report += "  • Schedule comprehensive eye examination soon.\n"
        report += "  • More frequent monitoring recommended (3-6 months).\n"
    elif severity == 3:
        report += "  ⚠⚠ Severe diabetic retinopathy detected.\n"
        report += "  • URGENT: Consult ophthalmologist immediately.\n"
        report += "  • May require laser treatment.\n"
    else:
        report += "  ⚠⚠⚠ PROLIFERATIVE diabetic retinopathy detected.\n"
        report += "  • CRITICAL: Immediate ophthalmologist referral required.\n"
        report += "  • High risk of vision loss.\n"
    
    report += "\n" + "═" * 70 + "\n"
    report += "IMPORTANT: This is an AI-assisted analysis. Always consult\n"
    report += "with a qualified ophthalmologist for definitive diagnosis.\n"
    report += "═" * 70
    
    return report