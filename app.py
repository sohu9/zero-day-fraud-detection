import os
import random
from datetime import datetime
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_socketio import SocketIO
from tensorflow.keras.models import load_model  # type: ignore
from werkzeug.security import check_password_hash

from data_pipeline import ID_COLUMNS, TARGET_COLUMN, add_time_features
from database import (
    clear_all_transactions,
    create_customer,
    get_auth_user,
    get_customer_devices,
    get_dashboard_stats,
    get_transaction,
    get_transactions,
    get_verification_events,
    init_database,
    is_device_trusted,
    register_device,
    save_transaction,
    save_verification_event,
    update_verification_status,
)

# ============================================================
# FLASK & SOCKET.IO CONFIGURATION
# ============================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY",
    "zero-day-fraudguard-dev-secret-key-2026"
)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

socketio = SocketIO(app, cors_allowed_origins="*")

init_database()

# ============================================================
# LOAD 19-FEATURE AUTOENCODER MODEL & ARTIFACTS
# ============================================================

MODEL_FILE = "zero_day_autoencoder.keras"
PREPROCESSOR_FILE = "preprocessor.joblib"
THRESHOLD_FILE = "threshold.joblib"
EVALUATION_FILE = "evaluation_results.joblib"

model = load_model(MODEL_FILE, compile=False)
artifact = joblib.load(PREPROCESSOR_FILE)
preprocessor = artifact["preprocessor"]
feature_names = list(artifact["feature_names"])
numerical_columns = list(artifact["numerical_columns"])
categorical_columns = list(artifact["categorical_columns"])
expected_columns = numerical_columns + categorical_columns

threshold_data = joblib.load(THRESHOLD_FILE)
threshold = float(
    threshold_data["threshold"]
    if isinstance(threshold_data, dict)
    else threshold_data
)

# Tier-3 Critical Instant Block MSE Multiplier / Threshold
CRITICAL_BLOCK_MSE_THRESHOLD = max(threshold * 4.5, 5.0)


# ============================================================
# AUTHENTICATION DECORATOR & ROUTES
# ============================================================

def login_required_api(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            # Allow local CLI scripts (like test_api.py) from localhost
            if request.remote_addr in ("127.0.0.1", "::1"):
                return view_func(*args, **kwargs)
            return jsonify({"error": "Authentication required"}), 401
        return view_func(*args, **kwargs)
    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if session.get("user_id"):
            return redirect(url_for("index"))
        return render_template("login.html")

    data = request.get_json(silent=True) or request.form
    account_id = str(data.get("account_id") or data.get("username") or "").strip()
    password = str(data.get("password") or "").strip()

    user = get_auth_user(account_id)
    if user and check_password_hash(user["password_hash"], password):
        session["user_id"] = user["account_id"]
        if request.is_json:
            return jsonify({"status": "ok", "redirect": url_for("index")})
        return redirect(url_for("index"))

    if request.is_json:
        return jsonify({"error": "Invalid Account ID or Password"}), 401
    return render_template("login.html", error="Invalid Account ID or Password"), 401


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return render_template("index.html", threshold=threshold)


# ============================================================
# EXPLAINABLE AI (PURE BEHAVIORAL RESIDUAL ANALYSIS)
# ============================================================

def _normalise_feature_name(raw_name):
    cleaned = str(raw_name)
    for prefix in ("num__", "cat__", "remainder__"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    for cat_col in categorical_columns:
        if cleaned.startswith(f"{cat_col}_"):
            return cat_col
    return cleaned


def _format_behavioral_reason(base_feature, data, residual):
    amt = float(data.get("transaction_amount") or 0)
    dev_amt = float(data.get("amount_deviation_from_user_mean") or 0)
    geo_km = float(data.get("geo_distance_from_last_txn") or 0)
    ip_risk = float(data.get("ip_risk_score") or 0)
    failed_24h = int(data.get("failed_txn_count_24h") or 0)
    txn_1h = int(data.get("txn_count_1h") or 0)
    txn_24h = int(data.get("txn_count_24h") or 0)
    acct_age = int(data.get("account_age_days") or 365)
    channel = str(data.get("payment_channel") or "card")
    dev_type = str(data.get("device_type") or "mobile")
    is_intl = int(data.get("is_international") or 0)

    descriptions = {
        "transaction_amount": f"Transaction amount (₹{amt:,.2f}) deviates significantly from user baseline (residual={residual:.3f})",
        "amount_deviation_from_user_mean": f"Amount spike (+₹{dev_amt:,.2f} from user mean) breaks historical spending pattern (residual={residual:.3f})",
        "geo_distance_from_last_txn": f"Geo-location jump ({geo_km:.1f} km from last transaction) indicates anomalous travel/remote access (residual={residual:.3f})",
        "ip_risk_score": f"Network IP risk score ({ip_risk:.2f}) signals anonymizer/VPN/high-risk subnet (residual={residual:.3f})",
        "failed_txn_count_24h": f"Multiple failed authentication attempts ({failed_24h} in 24h) suggest brute-force/credential stuffing (residual={residual:.3f})",
        "txn_count_1h": f"1-hour transaction velocity ({txn_1h} txns/hr) exceeds normal frequency (residual={residual:.3f})",
        "txn_count_24h": f"24-hour transaction frequency ({txn_24h} txns/24h) differs from learned pattern (residual={residual:.3f})",
        "account_age_days": f"Account tenure profile ({acct_age} days) vs transaction magnitude mismatch (residual={residual:.3f})",
        "payment_channel": f"Payment channel ('{channel}') diverges from customer's primary instrument (residual={residual:.3f})",
        "device_type": f"Endpoint device class ('{dev_type}') differs from established hardware profile (residual={residual:.3f})",
        "is_international": f"Cross-border/international routing flag (is_international={is_intl}) triggered (residual={residual:.3f})",
    }
    return descriptions.get(
        base_feature,
        f"Behavioral feature '{base_feature}' differs from learned normal pattern (residual={residual:.3f})"
    )


def generate_xai_explanation(data, features, reconstruction, mse_error):
    if mse_error <= threshold:
        return (
            "Reconstruction error is below the calibrated anomaly threshold (1.0347). "
            "Transaction matches the customer's learned normal behavioral pattern."
        )

    residuals = np.abs(np.asarray(features) - np.asarray(reconstruction))[0]
    sorted_indices = np.argsort(residuals)[::-1]

    explanations = []
    seen_base_features = set()

    for idx in sorted_indices:
        base_name = _normalise_feature_name(feature_names[idx])
        # Exclude synthetic composite score so XAI highlights pure behavioral features
        if base_name == "post_auth_risk_score" or base_name in seen_base_features:
            continue
        seen_base_features.add(base_name)
        explanations.append(
            _format_behavioral_reason(base_name, data, float(residuals[idx]))
        )
        if len(explanations) == 3:
            break

    return " | ".join(explanations)


# ============================================================
# HACKER FORENSIC TELEMETRY & 3-TIER ADAPTIVE RISK ENGINE
# ============================================================

def build_forensic_telemetry_and_tier(data, mse_error):
    customer_id = str(data.get("customer_id") or "11453")
    device_type = str(data.get("device_type") or "mobile").lower()
    ip_risk = float(data.get("ip_risk_score") or 0.0)
    geo_km = float(data.get("geo_distance_from_last_txn") or 0.0)
    failed_24h = int(data.get("failed_txn_count_24h") or 0)
    amt = float(data.get("transaction_amount") or 0.0)
    amt_dev = float(data.get("amount_deviation_from_user_mean") or 0.0)
    is_intl = int(data.get("is_international") or 0)

    # Determine if critical hacker attack (Tier 3) vs Step-Up Verification (Tier 2) vs Safe (Tier 1)
    critical_hacker_signals = (
        (ip_risk >= 0.80 and geo_km >= 400.0)
        or (failed_24h >= 4 and amt >= 20000.0)
        or (is_intl == 1 and ip_risk >= 0.75 and amt_dev >= 15000.0)
        or (mse_error >= CRITICAL_BLOCK_MSE_THRESHOLD and (ip_risk >= 0.70 or failed_24h >= 3 or geo_km >= 500.0))
        or bool(data.get("force_critical_block"))
    )

    is_anomaly = mse_error > threshold

    # 1. TIER 3: CRITICAL HACKER ATTACK -> INSTANT AUTO-BLOCK
    if is_anomaly and critical_hacker_signals:
        security_tier = "TIER_3_CRITICAL_BLOCK"
        transaction_status = "Auto-Blocked 🛑"
        risk_level = "High"

        ip_address = str(
            data.get("ip_address")
            or f"185.220.101.{random.randint(10, 250)} (Tor Exit Node / Anonymous VPN)"
        )
        geo_location = str(
            data.get("geo_location")
            or (
                f"Moscow, RU / Offshore Node ({geo_km:,.0f} km from Nashik)"
                if is_intl or geo_km >= 1000
                else f"Unverified Remote Zone ({geo_km:,.0f} km from last login)"
            )
        )
        device_name = str(
            data.get("device_name")
            or (
                "Unregistered Kali Linux Desktop • Headless Chrome"
                if device_type == "desktop"
                else "Unregistered Rooted Android Emulator (SM-G998B)"
            )
        )
        device_identifier = str(
            data.get("device_identifier") or f"DEV-HACKER-UNTRUSTED-{customer_id}"
        )
        device_trust_status = "Blacklisted / Untrusted Host 🛑"
        forensic_summary = (
            f"CRITICAL ACCOUNT TAKEOVER INTERCEPTED: Attacker attempted ₹{amt:,.2f} transfer "
            f"from {geo_location} via {device_name} [{ip_address}] after {failed_24h} failed auth attempts. "
            f"Autoencoder MSE ({mse_error:.4f}) exceeded critical safety ceiling."
        )

    # 2. TIER 2: SUSPICIOUS / NEW DEVICE OR LOCATION -> HOLD & VERIFY (2-STEP BIOMETRIC / FACE SCAN)
    elif is_anomaly:
        security_tier = "TIER_2_STEP_UP"
        transaction_status = "Hold & Verify 🚨"
        risk_level = "High"

        ip_address = str(
            data.get("ip_address")
            or f"49.36.{random.randint(40, 210)}.{random.randint(10, 240)} (Reliance Jio Cellular 5G)"
        )
        geo_location = str(
            data.get("geo_location")
            or (
                f"Mumbai / Pune Transit Corridor ({geo_km:.1f} km from Nashik)"
                if geo_km > 40
                else f"Nashik Metro Region ({geo_km:.1f} km from usual spot)"
            )
        )
        device_name = str(
            data.get("device_name")
            or (
                "New Unverified OnePlus 12R (Android 15)"
                if device_type == "mobile"
                else "New Unregistered Windows 11 Laptop (Edge 131)"
            )
        )
        device_identifier = str(
            data.get("device_identifier") or f"DEV-NEW-{device_type.upper()}-{customer_id}"
        )

        already_trusted = is_device_trusted(customer_id, device_identifier, device_name)
        device_trust_status = (
            "Trusted Device ✅"
            if already_trusted
            else "Unverified New Device ⚠️ (Awaiting Face/Biometric Scan)"
        )
        forensic_summary = (
            f"STEP-UP BIOMETRIC REQUIRED: Unusual pattern detected (MSE={mse_error:.4f} > {threshold:.4f}) "
            f"on {device_name} from {geo_location} for ₹{amt:,.2f}. "
            f"Held for 2-Step Biometric / Face ID confirmation."
        )

    # 3. TIER 1: GENUINE NORMAL TRANSACTION -> INSTANT APPROVAL
    else:
        security_tier = "TIER_1_SAFE"
        transaction_status = "Approved ✅"
        risk_level = "Low"

        ip_address = str(
            data.get("ip_address") or "103.48.19.112 (Trusted Home Broadband / JioFiber)"
        )
        geo_location = str(
            data.get("geo_location") or f"Nashik, Maharashtra, IN ({geo_km:.1f} km)"
        )
        device_name = str(
            data.get("device_name") or "Aun's Trusted iPhone 15 Pro (iOS 18.2)"
        )
        device_identifier = str(
            data.get("device_identifier") or "DEV-AUN-IPHONE15-PRO"
        )
        device_trust_status = "Trusted Device ✅"
        forensic_summary = (
            f"VERIFIED GENUINE PATTERN: Transaction of ₹{amt:,.2f} from {geo_location} "
            f"using {device_name} matches learned behavioral profile (MSE={mse_error:.4f})."
        )

    return {
        "security_tier": security_tier,
        "transaction_status": transaction_status,
        "risk_level": risk_level,
        "ip_address": ip_address,
        "geo_location": geo_location,
        "device_name": device_name,
        "device_identifier": device_identifier,
        "device_trust_status": device_trust_status,
        "forensic_summary": forensic_summary,
    }


# ============================================================
# CORE SCORING PIPELINE
# ============================================================

def score_single_transaction(raw_data):
    data = dict(raw_data)

    # Defaults for required fields
    data.setdefault("transaction_id", random.randint(300000, 999999))
    data.setdefault("customer_id", 11453)
    data.setdefault("merchant_id", 7397)
    data.setdefault("transaction_amount", 1250.0)
    data.setdefault("transaction_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    data.setdefault("account_age_days", 420)
    data.setdefault("payment_channel", "card")
    data.setdefault("device_type", "mobile")
    data.setdefault("is_international", 0)
    data.setdefault("ip_risk_score", 0.12)
    data.setdefault("txn_count_1h", 1)
    data.setdefault("txn_count_24h", 2)
    data.setdefault("failed_txn_count_24h", 0)
    data.setdefault("geo_distance_from_last_txn", 4.5)
    data.setdefault("amount_deviation_from_user_mean", 120.0)
    data.setdefault("post_auth_risk_score", 0.15)

    df_single = pd.DataFrame([data])
    df_features = df_single.drop(
        columns=[c for c in ID_COLUMNS + [TARGET_COLUMN] if c in df_features_cols(df_single)],
        errors="ignore"
    )
    df_features = add_time_features(df_features)

    for col in expected_columns:
        if col not in df_features.columns:
            df_features[col] = 0

    df_features = df_features[expected_columns]

    X_input = np.asarray(preprocessor.transform(df_features), dtype=np.float32)
    reconstruction = model.predict(X_input, verbose=0)
    mse_error = float(np.mean(np.square(X_input - reconstruction), axis=1)[0])

    xai_text = generate_xai_explanation(data, X_input, reconstruction, mse_error)
    forensic_info = build_forensic_telemetry_and_tier(data, mse_error)

    result = {
        "transaction_id": data["transaction_id"],
        "customer_id": data["customer_id"],
        "merchant_id": data["merchant_id"],
        "transaction_amount": float(data["transaction_amount"]),
        "transaction_time": str(data["transaction_time"]),
        "account_age_days": int(data["account_age_days"]),
        "payment_channel": str(data["payment_channel"]),
        "device_type": str(data["device_type"]),
        "is_international": int(data["is_international"]),
        "ip_risk_score": float(data["ip_risk_score"]),
        "txn_count_1h": int(data["txn_count_1h"]),
        "txn_count_24h": int(data["txn_count_24h"]),
        "failed_txn_count_24h": int(data["failed_txn_count_24h"]),
        "geo_distance_from_last_txn": float(data["geo_distance_from_last_txn"]),
        "amount_deviation_from_user_mean": float(data["amount_deviation_from_user_mean"]),
        "post_auth_risk_score": float(data["post_auth_risk_score"]),
        "reconstruction_error": mse_error,
        "threshold": threshold,
        "risk_level": forensic_info["risk_level"],
        "transaction_status": forensic_info["transaction_status"],
        "security_tier": forensic_info["security_tier"],
        "ip_address": forensic_info["ip_address"],
        "geo_location": forensic_info["geo_location"],
        "device_name": forensic_info["device_name"],
        "device_identifier": forensic_info["device_identifier"],
        "device_trust_status": forensic_info["device_trust_status"],
        "forensic_summary": forensic_info["forensic_summary"],
        "xai_explanation": xai_text,
    }

    create_customer(result["customer_id"])
    save_transaction(result)

    if result["security_tier"] == "TIER_2_STEP_UP":
        verification_id = save_verification_event(
            transaction_id=result["transaction_id"],
            customer_id=result["customer_id"],
            verification_method="BIOMETRIC_FACE_SCAN",
            verification_status="PENDING",
        )
        result["verification_id"] = verification_id
    elif result["security_tier"] == "TIER_3_CRITICAL_BLOCK":
        verification_id = save_verification_event(
            transaction_id=result["transaction_id"],
            customer_id=result["customer_id"],
            verification_method="AUTO_DEFENSE_FIREWALL",
            verification_status="AUTO_BLOCKED",
        )
        result["verification_id"] = verification_id

    socketio.emit("new_transaction", result)
    return result


def df_features_cols(df):
    return list(df.columns)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "service": "zero-day-fraud-detection",
        "database": "initialized",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_input_features": int(model.input_shape[-1]),
        "preprocessor_features": len(feature_names),
        "feature_dimensions_match": int(model.input_shape[-1]) == len(feature_names),
        "threshold": threshold,
        "critical_block_threshold": CRITICAL_BLOCK_MSE_THRESHOLD,
        "security_tiers": [
            "TIER_1_SAFE (Approved)",
            "TIER_2_STEP_UP (Hold & Biometric/Face Scan)",
            "TIER_3_CRITICAL_BLOCK (Instant Hacker Auto-Block)",
        ],
    })


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
@login_required_api
def predict_transaction():
    payload = request.get_json(silent=True) or {}
    result = score_single_transaction(payload)
    return jsonify(result)


@app.route("/api/simulate/<scenario>", methods=["POST", "GET"])
@login_required_api
def simulate_scenario(scenario):
    scenario_key = str(scenario).lower().strip()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if scenario_key in ("1", "normal", "genuine", "genuine_aun"):
        preset = {
            "transaction_id": random.randint(400000, 499999),
            "customer_id": 11453,
            "merchant_id": 7397,
            "transaction_amount": 1783.56,
            "transaction_time": "2023-10-01 00:02:41",
            "account_age_days": 540,
            "payment_channel": "card",
            "device_type": "mobile",
            "is_international": 0,
            "ip_risk_score": 0.14,
            "txn_count_1h": 1,
            "txn_count_24h": 2,
            "failed_txn_count_24h": 0,
            "geo_distance_from_last_txn": 4.2,
            "amount_deviation_from_user_mean": 180.0,
            "post_auth_risk_score": 0.18,
            "ip_address": "103.48.19.112 (JioFiber Home Broadband)",
            "geo_location": "Nashik, Maharashtra, IN (4.2 km)",
            "device_name": "Aun's Trusted iPhone 15 Pro (iOS 18.2)",
            "device_identifier": "DEV-AUN-IPHONE15-PRO",
        }
    elif scenario_key in ("2", "new_phone", "new_device", "new_phone_aun", "step_up"):
        preset = {
            "transaction_id": random.randint(500000, 599999),
            "customer_id": 11453,
            "merchant_id": 4062,
            "transaction_amount": 18450.00,
            "transaction_time": now_str,
            "account_age_days": 540,
            "payment_channel": "UPI",
            "device_type": "mobile",
            "is_international": 0,
            "ip_risk_score": 0.42,
            "txn_count_1h": 2,
            "txn_count_24h": 5,
            "failed_txn_count_24h": 1,
            "geo_distance_from_last_txn": 168.5,
            "amount_deviation_from_user_mean": 4200.0,
            "post_auth_risk_score": 0.88,
            "ip_address": "49.36.114.89 (Reliance Jio 5G Roaming)",
            "geo_location": "Mumbai, Maharashtra, IN (168.5 km from Nashik)",
            "device_name": "New Unverified OnePlus 12R (Android 15)",
            "device_identifier": "DEV-AUN-ONEPLUS-12R",
        }
    elif scenario_key in ("3", "hacker", "hacker_attack", "critical"):
        preset = {
            "transaction_id": random.randint(800000, 899999),
            "customer_id": 11453,
            "merchant_id": 9912,
            "transaction_amount": 92500.00,
            "transaction_time": now_str,
            "account_age_days": 540,
            "payment_channel": "netbanking",
            "device_type": "desktop",
            "is_international": 1,
            "ip_risk_score": 0.97,
            "txn_count_1h": 6,
            "txn_count_24h": 14,
            "failed_txn_count_24h": 7,
            "geo_distance_from_last_txn": 4820.0,
            "amount_deviation_from_user_mean": 89000.0,
            "post_auth_risk_score": 0.99,
            "ip_address": "185.220.101.47 (Tor Exit Node / DarkWeb Proxy)",
            "geo_location": "Moscow, RU / Offshore VPN (4,820 km from Nashik)",
            "device_name": "Unregistered Kali Linux x86_64 • Headless Chrome",
            "device_identifier": "DEV-HACKER-KALI-99",
            "force_critical_block": True,
        }
    else:
        return jsonify({"error": f"Unknown simulation scenario: {scenario}"}), 400

    if request.method == "GET":
        return jsonify(preset)

    result = score_single_transaction(preset)
    return jsonify(result)


@app.route("/api/transactions", methods=["GET"])
@login_required_api
def api_list_transactions():
    filter_type = request.args.get("filter", "all").lower()
    transactions = get_transactions(filter_type)
    stats = get_dashboard_stats()
    return jsonify({
        "transactions": transactions,
        "stats": stats,
    })


@app.route("/api/transactions/<transaction_id>", methods=["GET"])
@login_required_api
def api_get_transaction_detail(transaction_id):
    txn = get_transaction(transaction_id)
    if not txn:
        return jsonify({"error": "Transaction not found"}), 404
    verifications = get_verification_events(transaction_id)
    devices = get_customer_devices(txn.get("customer_id") or "11453")
    return jsonify({
        "transaction": txn,
        "verification_events": verifications,
        "registered_devices": devices,
    })


@app.route("/api/verify", methods=["POST"])
@app.route("/verify_transaction", methods=["POST"])
@login_required_api
def api_verify_transaction():
    data = request.get_json(silent=True) or {}
    transaction_id = str(data.get("transaction_id") or "").strip()
    verification_id = data.get("verification_id")
    raw_status = str(
        data.get("status") or data.get("decision") or data.get("action") or ""
    ).upper().strip()

    if raw_status in ("APPROVE", "APPROVED", "PASS", "VERIFIED"):
        new_status = "APPROVED"
    elif raw_status in ("REJECT", "REJECTED", "DENY", "BLOCK"):
        new_status = "REJECTED"
    else:
        return jsonify({"error": "Status must be APPROVED or REJECTED"}), 400

    if not transaction_id:
        return jsonify({"error": "transaction_id is required"}), 400

    updated = update_verification_status(verification_id, transaction_id, new_status)
    if not updated:
        return jsonify({
            "error": "Transaction is not in 'Hold & Verify' state or was already resolved."
        }), 409

    updated_txn = get_transaction(transaction_id)
    stats = get_dashboard_stats()
    devices = get_customer_devices(updated_txn.get("customer_id") or "11453")

    socketio.emit("transaction_updated", {
        "transaction": updated_txn,
        "stats": stats,
    })

    return jsonify({
        "status": "ok",
        "message": (
            "Biometric / Face ID Verified! Transaction Approved & New Device Bound as Trusted."
            if new_status == "APPROVED"
            else "Biometric Verification Denied! Transaction Blocked."
        ),
        "transaction": updated_txn,
        "registered_devices": devices,
        "stats": stats,
    })


@app.route("/api/customer/<customer_id>/devices", methods=["GET"])
@login_required_api
def api_customer_devices(customer_id):
    return jsonify({
        "customer_id": customer_id,
        "devices": get_customer_devices(customer_id),
    })


@app.route("/upload_csv", methods=["POST"])
@app.route("/api/upload_csv", methods=["POST"])
@login_required_api
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No CSV file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    df = pd.read_csv(file)
    max_rows = min(len(df), 250)
    df_batch = df.head(max_rows).copy()

    processed_count = 0
    anomalies_count = 0
    blocked_count = 0

    for _, row in df_batch.iterrows():
        res = score_single_transaction(row.to_dict())
        processed_count += 1
        if res["reconstruction_error"] > threshold:
            anomalies_count += 1
        if res["security_tier"] == "TIER_3_CRITICAL_BLOCK":
            blocked_count += 1

    return jsonify({
        "status": "ok",
        "processed": processed_count,
        "anomalies_detected": anomalies_count,
        "auto_blocked": blocked_count,
        "stats": get_dashboard_stats(),
    })


@app.route("/api/evaluation", methods=["GET"])
@app.route("/evaluation", methods=["GET"])
@login_required_api
def api_evaluation():
    if os.path.exists(EVALUATION_FILE):
        results = joblib.load(EVALUATION_FILE)
        return jsonify(results)
    return jsonify({"error": "evaluation_results.joblib not found. Run python evaluate.py first."}), 404


@app.route("/api/clear_transactions", methods=["POST"])
@app.route("/api/clear", methods=["POST"])
@login_required_api
def api_clear():
    clear_all_transactions()
    return jsonify({"status": "ok", "stats": get_dashboard_stats()})


if __name__ == "__main__":
    print("🚀 Starting FraudGuard AI 3-Tier SOC Backend on http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)