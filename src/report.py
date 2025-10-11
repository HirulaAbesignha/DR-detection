"""
Medical report generation.
"""


def generate_medical_report(prediction_result):
    """
    Generate detailed medical report from prediction.
    
    Args:
        prediction_result: Dictionary with prediction results
        
    Returns:
        Formatted medical report string
    """
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
        report += "  ✓ Monitor HbA1c levels regularly.\n"
        report += "  ✓ Control blood pressure and cholesterol.\n"
    
    elif severity == 1:
        report += "  ⚠ Mild diabetic retinopathy detected.\n"
        report += "  • Schedule follow-up examination in 6-12 months.\n"
        report += "  • Optimize diabetes management and glycemic control.\n"
        report += "  • Monitor blood pressure and lipid levels.\n"
        report += "  • Consider lifestyle modifications (diet, exercise).\n"
        report += "  • Regular monitoring by ophthalmologist recommended.\n"
    
    elif severity == 2:
        report += "  ⚠ Moderate diabetic retinopathy detected.\n"
        report += "  • Schedule comprehensive dilated eye examination soon.\n"
        report += "  • More frequent monitoring recommended (every 3-6 months).\n"
        report += "  • Strict diabetes control is essential (target HbA1c <7%).\n"
        report += "  • Blood pressure control critical (target <140/90 mmHg).\n"
        report += "  • May require referral to retinal specialist.\n"
        report += "  • Consider increased frequency of follow-ups.\n"
    
    elif severity == 3:
        report += "  ⚠⚠ Severe non-proliferative diabetic retinopathy detected.\n"
        report += "  • URGENT: Consult ophthalmologist or retinal specialist immediately.\n"
        report += "  • High risk of progression to proliferative stage.\n"
        report += "  • May require laser photocoagulation (panretinal photocoagulation).\n"
        report += "  • Anti-VEGF injections may be considered.\n"
        report += "  • Very strict blood sugar control mandatory.\n"
        report += "  • Close monitoring every 2-4 months essential.\n"
        report += "  • Control systemic risk factors (BP, lipids, kidney function).\n"
    
    else:  # severity == 4
        report += "  ⚠⚠⚠ PROLIFERATIVE diabetic retinopathy detected.\n"
        report += "  • CRITICAL: Immediate ophthalmologist/retinal specialist referral REQUIRED.\n"
        report += "  • HIGH RISK of severe vision loss or blindness.\n"
        report += "  • Neovascularization present - laser surgery likely necessary.\n"
        report += "  • Panretinal photocoagulation (PRP) may be indicated.\n"
        report += "  • Anti-VEGF therapy (bevacizumab, ranibizumab, aflibercept) recommended.\n"
        report += "  • Vitrectomy may be necessary if vitreous hemorrhage present.\n"
        report += "  • Emergency diabetes management protocol required.\n"
        report += "  • Monthly follow-up appointments mandatory.\n"
        report += "  • Address all cardiovascular risk factors urgently.\n"
    
    report += "\n" + "─" * 70 + "\n"
    report += "ADDITIONAL RECOMMENDATIONS:\n\n"
    report += "  • Maintain HbA1c below 7% (or as advised by physician)\n"
    report += "  • Keep blood pressure below 140/90 mmHg\n"
    report += "  • Manage cholesterol levels (LDL <100 mg/dL)\n"
    report += "  • Avoid smoking and limit alcohol consumption\n"
    report += "  • Regular physical activity (as approved by physician)\n"
    report += "  • Healthy diet with controlled carbohydrate intake\n"
    report += "  • Monitor kidney function regularly\n"
    report += "  • Take prescribed medications as directed\n"
    
    report += "\n" + "═" * 70 + "\n"
    report += "IMPORTANT DISCLAIMER:\n\n"
    report += "This is an AI-assisted analysis for screening and educational purposes.\n"
    report += "It should NOT be used as a substitute for professional medical diagnosis.\n"
    report += "Always consult with a qualified ophthalmologist or healthcare provider\n"
    report += "for definitive diagnosis, treatment planning, and medical decisions.\n\n"
    report += "The confidence score indicates the model's certainty but does not\n"
    report += "replace clinical judgment. False positives and false negatives may occur.\n"
    report += "═" * 70
    
    return report


if __name__ == "__main__":
    # Test report generation
    test_result = {
        'class': 2,
        'class_name': 'Moderate',
        'confidence': 0.87,
        'all_probabilities': {
            'No DR': 0.02,
            'Mild': 0.06,
            'Moderate': 0.87,
            'Severe': 0.03,
            'Proliferative DR': 0.02
        }
    }
    
    report = generate_medical_report(test_result)
    print(report)