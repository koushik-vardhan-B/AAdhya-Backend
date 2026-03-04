"""
Test script for Layer 1 — Spam Detection with Keyword Booster
Run from detection_layer/: python test_layer1.py
"""

from layer1 import run_layer1

# Mix of real Indian scam patterns + legitimate messages
test_messages = [
    # --- Scam / Spam messages (should be flagged) ---
    "Congratulations! You won ₹25 lakh KBC lottery. Send Aadhaar now.",
    "Your UPI ID has been blocked. Verify now at http://sbi-fake.xyz",
    "Earn ₹50,000/month work from home. Pay ₹2000 registration fee.",
    "Dear customer, your SBI account will be suspended. Click here to update KYC: http://sbi-update.in",
    "You have received ₹15,000 cashback from PhonePe. Claim now: bit.ly/fakelink",
    "URGENT: Your PAN card is linked to illegal activity. Call 9876543210 immediately.",

    # --- Legitimate / Safe messages (should NOT be flagged) ---
    "Hi, are you coming to the meeting tomorrow at 3pm?",
    "Your OTP is 847291. Do NOT share with anyone. - HDFC Bank",
    "Reminder: Your electricity bill of ₹1,250 is due on 10th March.",
    "Hey! Happy birthday 🎂 Have a great day!",
]

expected = [
    "SPAM", "SPAM", "SPAM", "SPAM", "SPAM", "SPAM",
    "HAM", "HAM", "HAM", "HAM",
]

print("=" * 65)
print("     LAYER 1 — SPAM DETECTION + KEYWORD BOOSTER TEST")
print("=" * 65)

stats = {"Safe": 0, "Suspicious": 0, "High Risk": 0}
correct = 0

for i, msg in enumerate(test_messages):
    result = run_layer1(msg)
    stats[result["risk_level"]] += 1

    actual = result["label"].upper()
    is_correct = actual == expected[i]
    if is_correct:
        correct += 1

    emoji = {"High Risk": "🚨", "Suspicious": "⚠️", "Safe": "✅"}[result["risk_level"]]
    check = "✓" if is_correct else "✗ WRONG"

    print(f"\n{emoji} Message  : {msg[:58]}{'...' if len(msg) > 58 else ''}")
    print(f"   Label      : {actual}  ({check}, expected {expected[i]})")
    print(f"   Risk Score : {result['risk_score']}/100")
    print(f"   Risk Level : {result['risk_level']}")
    if result["matched_patterns"]:
        print(f"   Patterns   : {', '.join(result['matched_patterns'])}")
    if result["matched_keywords"]:
        print(f"   Keywords   : {result['matched_keywords']}")
    print(f"   → Layer 2  : {'Yes' if result['proceed_to_layer2'] else 'No'}")

print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print(f"  Total messages : {len(test_messages)}")
print(f"  ✅ Safe         : {stats['Safe']}")
print(f"  ⚠️  Suspicious  : {stats['Suspicious']}")
print(f"  🚨 High Risk    : {stats['High Risk']}")
print(f"  📊 Accuracy     : {correct}/{len(test_messages)} ({correct/len(test_messages)*100:.0f}%)")
print("=" * 65)
