import sqlite3
from datetime import datetime


DATABASE_FILE = "transactions.db"


def get_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():
    conn = get_connection()

    # --------------------------------------------------------
    # 1. CUSTOMERS / ACCOUNTS
    # --------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            account_reference TEXT UNIQUE,
            full_name TEXT,
            email TEXT,
            phone TEXT,
            created_at TEXT
        )
    """)

    # --------------------------------------------------------
    # 2. REGISTERED DEVICES
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 3. EXISTING TRANSACTIONS TABLE
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 4. ADD NEW TRANSACTION CONTEXT COLUMNS
    # --------------------------------------------------------
    existing_columns = {
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(transactions)"
        ).fetchall()
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
    }

    for column_name, column_type in new_columns.items():
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE transactions ADD COLUMN "
                f"{column_name} {column_type}"
            )

    # --------------------------------------------------------
    # 5. VERIFICATION EVENTS
    # --------------------------------------------------------
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

    conn.commit()
    conn.close()

    print("✅ SQLite database initialized successfully.")
    print("   • Customers table ready")
    print("   • Registered devices table ready")
    print("   • Transactions table ready")
    print("   • Transaction context fields ready")
    print("   • Verification events table ready")


# ============================================================
# CUSTOMER / ACCOUNT
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
        account_reference,
        full_name,
        email,
        phone,
        datetime.now().isoformat(timespec="seconds")
    ))

    conn.commit()
    conn.close()


def get_customer(customer_id):
    conn = get_connection()

    row = conn.execute("""
        SELECT *
        FROM customers
        WHERE customer_id = ?
    """, (str(customer_id),)).fetchone()

    conn.close()

    return dict(row) if row else None


# ============================================================
# REGISTERED DEVICES
# ============================================================

def register_device(
    customer_id,
    device_identifier=None,
    device_model=None,
    device_type=None
):
    conn = get_connection()

    conn.execute("""
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
        device_identifier,
        device_model,
        device_type,
        datetime.now().isoformat(timespec="seconds")
    ))

    conn.commit()
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
# SAVE TRANSACTION
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
            post_auth_risk_score
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (

        # Existing ML / transaction fields
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

        # New transaction context
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

        float(result.get("post_auth_risk_score") or 0)
    ))

    conn.commit()
    conn.close()


# ============================================================
# TRANSACTION LIST
# ============================================================

def get_transactions(filter_type="all"):

    conn = get_connection()

    if filter_type == "approved":

        rows = conn.execute("""
            SELECT *
            FROM transactions
            WHERE transaction_status LIKE 'Approved%'
            ORDER BY created_at DESC
        """).fetchall()

    elif filter_type == "hold":

        rows = conn.execute("""
            SELECT *
            FROM transactions
            WHERE transaction_status LIKE 'Hold & Verify%'
            ORDER BY created_at DESC
        """).fetchall()

    elif filter_type == "high":

        rows = conn.execute("""
            SELECT *
            FROM transactions
            WHERE risk_level = 'High'
            ORDER BY created_at DESC
        """).fetchall()

    else:

        rows = conn.execute("""
            SELECT *
            FROM transactions
            ORDER BY created_at DESC
        """).fetchall()

    conn.close()

    return [dict(row) for row in rows]


# ============================================================
# SINGLE TRANSACTION
# ============================================================

def get_transaction(transaction_id):

    conn = get_connection()

    row = conn.execute("""
        SELECT *
        FROM transactions
        WHERE transaction_id = ?
    """, (str(transaction_id),)).fetchone()

    conn.close()

    return dict(row) if row else None


# ============================================================
# CUSTOMER TRANSACTION HISTORY
# ============================================================

def get_customer_transactions(customer_id, limit=20):

    conn = get_connection()

    rows = conn.execute("""
        SELECT *
        FROM transactions
        WHERE customer_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (
        str(customer_id),
        int(limit)
    )).fetchall()

    conn.close()

    return [dict(row) for row in rows]


# ============================================================
# VERIFICATION
# ============================================================

def save_verification_event(
    transaction_id,
    customer_id,
    verification_method,
    verification_status
):

    conn = get_connection()

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
    conn.close()

    return verification_id


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
    """
    Updates the verification event status (APPROVED/REJECTED) 
    and updates the corresponding transaction status.
    """
    conn = get_connection()
    try:
        # 1. Update verification event
        conn.execute(
            """
            UPDATE verification_events 
            SET verification_status = ?, verified_at = CURRENT_TIMESTAMP 
            WHERE verification_id = ?
            """,
            (new_status, verification_id)
        )
        
        # 2. Map verification status to exact transaction status strings
        if new_status == "APPROVED":
            txn_status = "Approved ✅"
        else:
            txn_status = "Rejected ❌"
            
        # 3. Update transaction status
        conn.execute(
            "UPDATE transactions SET transaction_status = ? WHERE transaction_id = ?",
            (txn_status, transaction_id)
        )
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Database Error in update_verification_status: {e}")
        return False
    finally:
        conn.close()
        
        

