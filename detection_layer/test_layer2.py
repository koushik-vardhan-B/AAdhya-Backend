"""
Test script for Layer 2 — Fraud Type Classification
Runs Layer 1 → Layer 2 pipeline on test messages.
Run from detection_layer/: python test_layer2.py
"""

from layer1 import run_layer1
from layer2 import run_layer2

# Test messages — all scams that Layer 1 should flag
test_messages = [
    # UPI Fraud
    ("Congratulations! You won ₹25 lakh KBC lottery. Send Aadhaar now.", "Lottery Scam"),
    ("Your UPI ID has been blocked. Verify now at http://sbi-fake.xyz", "UPI Fraud"),
    ("Dear customer, your SBI account will be suspended. Click here to update KYC: http://sbi-update.in", "UPI Fraud"),

    # Job Scam
    ("Earn ₹50,000/month work from home. Pay ₹2000 registration fee.", "Job Scam"),
    ("Part time job! Data entry work, earn ₹500 per task. No experience needed. WhatsApp 9876543210", "Job Scam"),

    # Lottery Scam
    ("You have received ₹15,000 cashback from PhonePe. Claim now: bit.ly/fakelink", "Lottery Scam"),
    ("Lucky draw winner! You won ₹10 crore jackpot prize. Call now to claim your reward.", "Lottery Scam"),

    # Phishing
    ("URGENT: Your PAN card is linked to illegal activity. Call 9876543210 immediately.", "Phishing"),
    ("Security alert: Suspicious login detected on your account. Click here to verify: http://fake-verify.com", "Phishing"),

    # Safe messages (should be filtered by Layer 1, not reach Layer 2)
    ("Hi, are you coming to the meeting tomorrow at 3pm?", None),
    ("Your OTP is 847291. Do NOT share with anyone. - HDFC Bank", None),
]

print("=" * 70)
print("        FULL PIPELINE TEST — LAYER 1 → LAYER 2")
print("=" * 70)

correct = 0
total_classified = 0

for msg, expected_type in test_messages:
    # Run Layer 1
    l1 = run_layer1(msg)

    print(f"\n{'─' * 70}")
    print(f"📩 Message: {msg[:65]}{'...' if len(msg) > 65 else ''}")
    print(f"   L1 Result: {l1['label'].upper()} | Risk: {l1['risk_score']}/100 | {l1['risk_level']}")

    # Only run Layer 2 if Layer 1 flags it
    if l1["proceed_to_layer2"]:
        l2 = run_layer2(msg, layer1_result=l1)
        total_classified += 1

        is_correct = l2["fraud_type"] == expected_type
        if is_correct:
            correct += 1
        check = "✓" if is_correct else f"✗ (expected {expected_type})"

        print(f"   L2 Fraud Type    : {l2['fraud_type']}  {check}")
        print(f"   Phishing Score   : {l2['phishing_confidence'] * 100:.1f}%")
        print(f"   Type Confidence  : {l2['type_confidence'] * 100:.1f}%")
        print(f"   Final Risk Score : {l2['risk_score']}/100 ({l2['risk_level']})")
        print(f"   Keywords         : {l2['matched_keywords'][:4]}")
        print(f"   💡 {l2['explanation'][:100]}...")
    else:
        if expected_type is None:
            correct += 1
            total_classified += 1
        print(f"   ✅ Filtered as SAFE — not sent to Layer 2")

print(f"\n{'=' * 70}")
print(f"  PIPELINE SUMMARY")
print(f"{'=' * 70}")
print(f"  Total messages    : {len(test_messages)}")
print(f"  Sent to Layer 2   : {sum(1 for _, t in test_messages if t is not None)}")
print(f"  Correctly typed   : {correct}/{total_classified}")
print(f"  📊 Accuracy       : {correct/total_classified*100:.0f}%" if total_classified > 0 else "  📊 Accuracy: N/A")
print(f"{'=' * 70}")
