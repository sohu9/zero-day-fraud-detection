import json
import requests

BASE_URL = "http://127.0.0.1:5000"


def pretty_print(title, data):
    print(f"\n{title}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def run_tests():
    # 1. Health Check
    r1 = requests.get(f"{BASE_URL}/api/health", timeout=10)
    pretty_print("1. Checking /api/health (19-Feature Model & 3-Tier Engine)...", r1.json())

    # 2. Tier 1: Genuine Payment (Aun on Trusted iPhone 15 in Nashik)
    r2 = requests.post(f"{BASE_URL}/api/simulate/genuine_aun", timeout=10)
    pretty_print("2. [TIER 1] Testing GENUINE Payment (Aun's Trusted iPhone 15 in Nashik)...", r2.json())

    # 3. Tier 2: Aun Bought a New Phone in Mumbai -> Hold & Verify -> Biometric Pass
    r3 = requests.post(f"{BASE_URL}/api/simulate/new_phone_aun", timeout=10)
    tier2_txn = r3.json()
    pretty_print("3. [TIER 2] Testing AUN ON NEW MOBILE IN MUMBAI (Expected: Hold & Verify 🚨)...", tier2_txn)

    # 3B. Simulate Biometric / Face Scan Approval for Tier 2
    txn_id = tier2_txn.get("transaction_id")
    ver_id = tier2_txn.get("verification_id")
    r3_verify = requests.post(
        f"{BASE_URL}/api/verify",
        json={
            "transaction_id": txn_id,
            "verification_id": ver_id,
            "status": "APPROVED",
        },
        timeout=10,
    )
    pretty_print(
        f"3B. [BIOMETRIC FACE SCAN PASSED] Verifying Transaction #{txn_id} & Auto-Trusting New Phone...",
        r3_verify.json(),
    )

    # 4. Tier 3: Critical Hacker Account Drain Attack -> Instant Auto-Block + Full Kundali
    r4 = requests.post(f"{BASE_URL}/api/simulate/hacker_attack", timeout=10)
    pretty_print(
        "4. [TIER 3] Testing CRITICAL HACKER ATTACK (Expected: Auto-Blocked 🛑 + Full Hacker Kundali)...",
        r4.json(),
    )


if __name__ == "__main__":
    run_tests()