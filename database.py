import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash


DATABASE_FILE = "transactions.db"


def get_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# DATABASE INITIALIZATION & AUTO-MIGRATION
# ============================================================

def init_database():
    conn = get_connection()

    # 1. CUSTOMERS / ACCOUNTS
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            account_reference TEXT UNIQUE,
            full_name TEXT,
            email TEXT,
            phone TEXT,
            home_city TEXT DEFAULT 'Nashik, MH, IN',
            created_at TEXT
        )
    """)

    # 1A. USER AUTHENTICATION
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_auth (
            auth_id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # 2. REGISTERED DEVICES (Trusted Device Binding)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS registered_devices (
            device_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            device_identifier TEXT,
            device_model TEXT,
            device_type TEXT,
            is_active INTEGER DEFAULT 1,
            registered_at TEXT,
            FOREIGN KEY (customer_id)
                REFERENCES customers(customer_id)
        )
    """)

    # 3. TRANSACTIONS TABLE
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            customer_id TEXT,
            merchant_id TEXT,
            transaction_amount REAL,
            transaction_status TEXT,
            risk_level TEXT,
            reconstruction_error REAL,
            threshold REAL,
            xai_explanation TEXT,
            created_at TEXT
        )
    """)

    # 4. TRANSACTION CONTEXT & FORENSIC TELEMETRY COLUMNS (Auto-Migration)
    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(transactions)").fetchall()
    }

    new_columns = {
        "transaction_time": "TEXT",
        "payment_channel": "TEXT",
        "device_type": "TEXT",
        "is_international": "INTEGER",
        "ip_risk_score": "REAL",
        "txn_count_1h": "INTEGER",
        "txn_count_24h": "INTEGER",
        "failed_txn_count_24h": "INTEGER",
        "geo_distance_from_last_txn": "REAL",
        "amount_deviation_from_user_mean": "REAL",
        "post_auth_risk_score": "REAL",
        "account_age_days": "INTEGER",
        "security_tier": "TEXT",
        "ip_address": "TEXT",
        "geo_location": "TEXT",
        "device_name": "TEXT",
        "device_identifier": "TEXT",
        "device_trust_status": "TEXT",
        "forensic_summary": "TEXT",
    }

    for column_name, column_type in new_columns.items():
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE transactions ADD COLUMN {column_name} {column_type}"
            )

    # 5. VERIFICATION EVENTS (Step-Up Biometric / Face Scan Trail)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS verification_events (
            verification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            customer_id TEXT,
            verification_method TEXT,
            verification_status TEXT,
            verified_at TEXT,
            FOREIGN KEY (transaction_id)
                REFERENCES transactions(transaction_id)
        )
    """)

    # 6. DEFAULT ADMIN ACCOUNT SEEDING
    existing_admin = conn.execute(
        "SELECT 1 FROM user_auth LIMIT 1"
    ).fetchone()
    if existing_admin is None:
        conn.execute(
            """
            INSERT INTO user_auth (account_id, password_hash, created_at)
            VALUES (?, ?, ?)
            """,
            (
                "admin",
                generate_password_hash("admin123"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

    # 7. SEED DEFAULT CUSTOMER PROFILE (Aun's Primary Account & Trusted Device)
    conn.execute("""
        INSERT OR IGNORE INTO customers (
            customer_id, account_reference, full_name, email, phone, home_city, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        "11453",
        "ACC-AUN-11453",
        "Mohammad Aun Saeed Akhtar",
        "aun.akhtar@fraudguard.ai",
        "+91-9876543210",
        "Nashik, Maharashtra, IN",
        datetime.now().isoformat(timespec="seconds"),
    ))

    existing_device = conn.execute(
        "SELECT 1 FROM registered_devices WHERE customer_id = ? AND device_identifier = ?",
        ("11453", "DEV-AUN-IPHONE15-PRO")
    ).fetchone()
    if existing_device is None:
        conn.execute("""
            INSERT INTO registered_devices (
                customer_id, device_identifier, device_model, device_type, is_active, registered_at
            )
            VALUES (?, ?, ?, ?, 1, ?)
        """, (
            "11453",
            "DEV-AUN-IPHONE15-PRO",
            "Aun's Trusted iPhone 15 Pro (iOS 18.2)",
            "mobile",
            datetime.now().isoformat(timespec="seconds"),
        ))

    conn.commit()
    conn.close()

    print("✅ SQLite database initialized successfully.")
    print("   • Customers & Default Profile ready")
    print("   • Registered trusted devices table ready")
    print("   • Transactions & Hacker Forensic Telemetry fields ready")
    print("   • 3-Tier Adaptive Verification events table ready")


# ============================================================
# CUSTOMER & DEVICE TRUST MANAGEMENT
# ============================================================

def create_customer(
    customer_id,
    account_reference=None,
    full_name=None,
    email=None,
    phone=None
):
    conn = get_connection()
    conn.execute("""
        INSERT OR IGNORE INTO customers (
            customer_id,
            account_reference,
            full_name,
            email,
            phone,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        str(customer_id),
        account_reference or f"ACC-{customer_id}",
        full_name or f"Customer #{customer_id}",
        email,
        phone,
        datetime.now().isoformat(timespec="seconds")
    ))
    conn.commit()
    conn.close()


def get_customer(customer_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM customers WHERE customer_id = ?",
        (str(customer_id),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def is_device_trusted(customer_id, device_identifier=None, device_model=None):
    conn = get_connection()
    try:
        if device_identifier:
            row = conn.execute("""
                SELECT * FROM registered_devices
                WHERE customer_id = ? AND device_identifier = ? AND is_active = 1
            """, (str(customer_id), str(device_identifier))).fetchone()
            if row:
                return True

        if device_model:
            row = conn.execute("""
                SELECT * FROM registered_devices
                WHERE customer_id = ? AND device_model = ? AND is_active = 1
            """, (str(customer_id), str(device_model))).fetchone()
            if row:
                return True

        return False
    finally:
        conn.close()


def register_device(
    customer_id,
    device_identifier=None,
    device_model=None,
    device_type=None
):
    conn = get_connection()
    try:
        identifier = device_identifier or f"DEV-{customer_id}-{device_type or 'GENERIC'}"
        existing = conn.execute("""
            SELECT device_id FROM registered_devices
            WHERE customer_id = ? AND device_identifier = ? AND is_active = 1
        """, (str(customer_id), str(identifier))).fetchone()

        if existing:
            return int(existing["device_id"])

        cursor = conn.execute("""
            INSERT INTO registered_devices (
                customer_id,
                device_identifier,
                device_model,
                device_type,
                is_active,
                registered_at
            )
            VALUES (?, ?, ?, ?, 1, ?)
        """, (
            str(customer_id),
            str(identifier),
            device_model or "Biometric Verified Device",
            device_type or "mobile",
            datetime.now().isoformat(timespec="seconds")
        ))
        device_id = cursor.lastrowid
        conn.commit()
        return device_id
    finally:
        conn.close()


def get_customer_devices(customer_id):
    conn = get_connection()
    rows = conn.execute("""
        SELECT *
        FROM registered_devices
        WHERE customer_id = ?
        AND is_active = 1
        ORDER BY registered_at DESC
    """, (str(customer_id),)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ============================================================
# SAVE TRANSACTION (WITH FORENSIC TELEMETRY)
# ============================================================

def save_transaction(result):
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO transactions (
            transaction_id,
            customer_id,
            merchant_id,
            transaction_amount,
            transaction_status,
            risk_level,
            reconstruction_error,
            threshold,
            xai_explanation,
            created_at,
            transaction_time,
            payment_channel,
            device_type,
            is_international,
            ip_risk_score,
            txn_count_1h,
            txn_count_24h,
            failed_txn_count_24h,
            geo_distance_from_last_txn,
            amount_deviation_from_user_mean,
            post_auth_risk_score,
            account_age_days,
            security_tier,
            ip_address,
            geo_location,
            device_name,
            device_identifier,
            device_trust_status,
            forensic_summary
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        str(result.get("transaction_id")),
        str(result.get("customer_id")),
        str(result.get("merchant_id")),
        float(result.get("transaction_amount") or 0),
        result.get("transaction_status"),
        result.get("risk_level"),
        float(result.get("reconstruction_error") or 0),
        float(result.get("threshold") or 0),
        result.get("xai_explanation", ""),
        datetime.now().isoformat(timespec="seconds"),
        result.get("transaction_time"),
        result.get("payment_channel"),
        result.get("device_type"),
        int(result.get("is_international") or 0),
        float(result.get("ip_risk_score") or 0),
        int(result.get("txn_count_1h") or 0),
        int(result.get("txn_count_24h") or 0),
        int(result.get("failed_txn_count_24h") or 0),
        float(result.get("geo_distance_from_last_txn") or 0),
        float(result.get("amount_deviation_from_user_mean") or 0),
        float(result.get("post_auth_risk_score") or 0),
        int(result.get("account_age_days") or 365),
        result.get("security_tier", "TIER_1_SAFE"),
        result.get("ip_address", "103.48.19.112"),
        result.get("geo_location", "Nashik, MH, IN"),
        result.get("device_name", "Registered Mobile Device"),
        result.get("device_identifier", "DEV-DEFAULT"),
        result.get("device_trust_status", "Trusted Device ✅"),
        result.get("forensic_summary", "Standard transaction telemetry within normal bounds.")
    ))
    conn.commit()
    conn.close()


# ============================================================
# TRANSACTION LIST & DASHBOARD STATS
# ============================================================

def get_transactions(filter_type="all"):
    conn = get_connection()

    if filter_type == "approved":
        rows = conn.execute("""
            SELECT * FROM transactions
            WHERE transaction_status LIKE 'Approved%'
            ORDER BY created_at DESC
        """).fetchall()
    elif filter_type == "hold":
        rows = conn.execute("""
            SELECT * FROM transactions
            WHERE transaction_status LIKE 'Hold & Verify%'
            ORDER BY created_at DESC
        """).fetchall()
    elif filter_type == "blocked":
        rows = conn.execute("""
            SELECT * FROM transactions
            WHERE transaction_status LIKE 'Auto-Blocked%'
               OR transaction_status LIKE 'Rejected%'
            ORDER BY created_at DESC
        """).fetchall()
    elif filter_type == "high":
        rows = conn.execute("""
            SELECT * FROM transactions
            WHERE risk_level IN ('High', 'Critical')
              AND transaction_status NOT LIKE 'Approved%'
            ORDER BY created_at DESC
        """).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM transactions
            ORDER BY created_at DESC
        """).fetchall()

    conn.close()
    return [dict(row) for row in rows]


def get_dashboard_stats():
    conn = get_connection()
    row = conn.execute("""
        SELECT
            COUNT(*) AS total_count,
            SUM(CASE WHEN transaction_status LIKE 'Approved%' THEN 1 ELSE 0 END) AS approved_count,
            SUM(CASE WHEN transaction_status LIKE 'Hold & Verify%' THEN 1 ELSE 0 END) AS hold_count,
            SUM(CASE WHEN risk_level IN ('High', 'Critical') AND transaction_status NOT LIKE 'Approved%' THEN 1 ELSE 0 END) AS high_count,
            SUM(CASE WHEN transaction_status LIKE 'Rejected%' OR transaction_status LIKE 'Auto-Blocked%' THEN 1 ELSE 0 END) AS rejected_count
        FROM transactions
    """).fetchone()
    conn.close()
    return {
        "all": int(row["total_count"] or 0),
        "approved": int(row["approved_count"] or 0),
        "hold": int(row["hold_count"] or 0),
        "high": int(row["high_count"] or 0),
        "rejected": int(row["rejected_count"] or 0),
    }


def get_transaction(transaction_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM transactions WHERE transaction_id = ?",
        (str(transaction_id),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_customer_transactions(customer_id, limit=20):
    conn = get_connection()
    rows = conn.execute("""
        SELECT *
        FROM transactions
        WHERE customer_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (str(customer_id), int(limit))).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def clear_all_transactions():
    conn = get_connection()
    conn.execute("DELETE FROM verification_events")
    conn.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()


# ============================================================
# STEP-UP VERIFICATION & BIOMETRIC DEVICE BINDING WORKFLOW
# ============================================================

def save_verification_event(
    transaction_id,
    customer_id,
    verification_method="BIOMETRIC_FACE_SCAN",
    verification_status="PENDING"
):
    conn = get_connection()
    try:
        existing_pending = conn.execute("""
            SELECT verification_id
            FROM verification_events
            WHERE transaction_id = ? AND verification_status = 'PENDING'
        """, (str(transaction_id),)).fetchone()

        if existing_pending is not None:
            return int(existing_pending["verification_id"])

        cursor = conn.execute("""
            INSERT INTO verification_events (
                transaction_id,
                customer_id,
                verification_method,
                verification_status,
                verified_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(transaction_id),
            str(customer_id),
            verification_method,
            verification_status,
            datetime.now().isoformat(timespec="seconds")
        ))
        verification_id = cursor.lastrowid
        conn.commit()
        return verification_id
    finally:
        conn.close()


def get_verification_events(transaction_id):
    conn = get_connection()
    rows = conn.execute("""
        SELECT *
        FROM verification_events
        WHERE transaction_id = ?
        ORDER BY verified_at DESC
    """, (str(transaction_id),)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_verification_status(verification_id, transaction_id, new_status):
    if new_status not in {"APPROVED", "REJECTED"}:
        raise ValueError("Status must be APPROVED or REJECTED.")

    conn = get_connection()
    try:
        txn = conn.execute(
            "SELECT * FROM transactions WHERE transaction_id = ?",
            (str(transaction_id),)
        ).fetchone()
        if txn is None or not str(txn["transaction_status"] or "").startswith("Hold & Verify"):
            return False

        if verification_id:
            cursor = conn.execute("""
                UPDATE verification_events
                SET verification_status = ?, verified_at = ?
                WHERE verification_id = ?
                  AND transaction_id = ?
                  AND verification_status = 'PENDING'
            """, (
                new_status,
                datetime.now().isoformat(timespec="seconds"),
                int(verification_id),
                str(transaction_id)
            ))
        else:
            cursor = conn.execute("""
                UPDATE verification_events
                SET verification_status = ?, verified_at = ?
                WHERE transaction_id = ?
                  AND verification_status = 'PENDING'
            """, (
                new_status,
                datetime.now().isoformat(timespec="seconds"),
                str(transaction_id)
            ))

        if cursor.rowcount == 0:
            # Create and immediately resolve if no pending row existed
            conn.execute("""
                INSERT INTO verification_events (
                    transaction_id, customer_id, verification_method, verification_status, verified_at
                )
                VALUES (?, ?, 'BIOMETRIC_FACE_SCAN', ?, ?)
            """, (
                str(transaction_id),
                str(txn["customer_id"]),
                new_status,
                datetime.now().isoformat(timespec="seconds")
            ))

        if new_status == "APPROVED":
            txn_status = "Approved (Biometric Verified) ✅"
            new_risk = "Verified"
            new_trust = "Trusted (Biometric Bound) ✅"
            new_tier = "TIER_2_VERIFIED"

            # AUTO-TRUST NEW DEVICE ON BIOMETRIC APPROVAL
            customer_id = str(txn["customer_id"] or "11453")
            dev_id = str(txn["device_identifier"] or f"DEV-{customer_id}-NEW")
            dev_name = str(txn["device_name"] or "Biometric Verified Mobile")
            dev_type = str(txn["device_type"] or "mobile")

            conn.execute("""
                INSERT OR IGNORE INTO registered_devices (
                    customer_id, device_identifier, device_model, device_type, is_active, registered_at
                )
                VALUES (?, ?, ?, ?, 1, ?)
            """, (
                customer_id,
                dev_id,
                dev_name,
                dev_type,
                datetime.now().isoformat(timespec="seconds")
            ))
        else:
            txn_status = "Rejected (Biometric Failed) ❌"
            new_risk = "High"
            new_trust = "Untrusted / Blocked Device 🛑"
            new_tier = "TIER_3_CRITICAL_BLOCK"

        conn.execute("""
            UPDATE transactions
            SET transaction_status = ?,
                risk_level = ?,
                device_trust_status = ?,
                security_tier = ?
            WHERE transaction_id = ?
        """, (
            txn_status,
            new_risk,
            new_trust,
            new_tier,
            str(transaction_id)
        ))
        conn.commit()
        return True
    finally:
        conn.close()


# ============================================================
# USER AUTHENTICATION
# ============================================================

def create_auth_user(account_id, password_hash):
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO user_auth (account_id, password_hash, created_at)
            VALUES (?, ?, ?)
            """,
            (account_id, password_hash, datetime.now().isoformat(timespec="seconds"))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_auth_user(account_id):
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM user_auth WHERE account_id = ?",
            (account_id,)
        ).fetchone()
    finally:
        conn.close()