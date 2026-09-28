import sqlite3
from datetime import datetime

DATABASE_FILE = "transactions.db"


def get_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_connection()

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

    conn.commit()
    conn.close()

    print("✅ SQLite transaction database ready.")


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
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        datetime.now().isoformat(timespec="seconds")
    ))

    conn.commit()
    conn.close()


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

    elif filter_type == "high":
        rows = conn.execute("""
            SELECT * FROM transactions
            WHERE risk_level = 'High'
            ORDER BY created_at DESC
        """).fetchall()

    else:
        rows = conn.execute("""
            SELECT * FROM transactions
            ORDER BY created_at DESC
        """).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def get_transaction(transaction_id):
    conn = get_connection()

    row = conn.execute("""
        SELECT * FROM transactions
        WHERE transaction_id = ?
    """, (str(transaction_id),)).fetchone()

    conn.close()

    return dict(row) if row else None