"""Add-on API routes for the Zero-Day Fraud Detection Flask app."""
import os, io, json, sqlite3, secrets, time, math
from functools import wraps
from pathlib import Path
from flask import request, jsonify, send_file, g
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def register_addons(app, core):
    db_path = Path(getattr(core, 'DATABASE_FILE', 'transactions.db'))
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    secret_path = Path('.backend_token_secret')
    if not secret_path.exists():
        secret_path.write_text(secrets.token_urlsafe(48), encoding='utf-8')
    serializer = URLSafeTimedSerializer(secret_path.read_text(encoding='utf-8').strip(), salt='zero-day-api-v1')

    def db():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn

    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()

    def issue_token(user):
        return serializer.dumps({'uid': int(user['id']), 'username': user['username']})

    def require_auth(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            header = request.headers.get('Authorization', '')
            if not header.startswith('Bearer '):
                return jsonify({'error': 'Authentication required. Send Authorization: Bearer <token>.'}), 401
            try:
                payload = serializer.loads(header[7:].strip(), max_age=60 * 60 * 24)
            except (BadSignature, SignatureExpired):
                return jsonify({'error': 'Invalid or expired token. Please log in again.'}), 401
            g.auth_user = payload
            return fn(*args, **kwargs)
        return wrapped

    @app.route('/api/auth/register', methods=['POST'])
    def api_register():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({'error': 'Send a JSON object.'}), 400
        username = str(body.get('username', '')).strip()
        email = str(body.get('email', '')).strip().lower()
        password = body.get('password', '')
        if len(username) < 3 or len(username) > 40:
            return jsonify({'error': 'Username must be 3–40 characters.'}), 400
        if '@' not in email or len(email) > 254:
            return jsonify({'error': 'Enter a valid email address.'}), 400
        if not isinstance(password, str) or len(password) < 8 or len(password) > 128:
            return jsonify({'error': 'Password must be 8–128 characters.'}), 400
        try:
            with db() as conn:
                conn.execute('INSERT INTO users(username,email,password_hash) VALUES(?,?,?)',
                             (username, email, generate_password_hash(password)))
                conn.commit()
                user = conn.execute('SELECT id,username,email FROM users WHERE email=?', (email,)).fetchone()
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Username or email is already registered.'}), 409
        return jsonify({'message': 'Registration successful.', 'user': dict(user), 'token': issue_token(user)}), 201

    @app.route('/api/auth/login', methods=['POST'])
    def api_login():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({'error': 'Send a JSON object.'}), 400
        identity = str(body.get('email', body.get('username', ''))).strip().lower()
        password = body.get('password', '')
        with db() as conn:
            user = conn.execute('SELECT * FROM users WHERE lower(email)=? OR lower(username)=?', (identity, identity)).fetchone()
        if not user or not isinstance(password, str) or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid login credentials.'}), 401
        return jsonify({'message': 'Login successful.', 'user': {'id': user['id'], 'username': user['username'], 'email': user['email']}, 'token': issue_token(user)})

    @app.route('/api/auth/me', methods=['GET'])
    @require_auth
    def api_me():
        with db() as conn:
            user = conn.execute('SELECT id,username,email,created_at FROM users WHERE id=?', (g.auth_user['uid'],)).fetchone()
        if not user:
            return jsonify({'error': 'User no longer exists.'}), 404
        return jsonify({'user': dict(user)})

    @app.route('/api/secure/transactions', methods=['GET'])
    @require_auth
    def secure_transaction_history():
        # Existing transaction schema has no owner field, so this endpoint requires auth
        # but returns only the same application-wide history as the existing API.
        limit = request.args.get('limit', default=50, type=int)
        limit = max(1, min(limit or 50, 200))
        filter_type = request.args.get('filter', 'all')
        with db() as conn:
            valid = {'all', 'approved', 'hold', 'high'}
            if filter_type not in valid:
                return jsonify({'error': 'filter must be all, approved, hold, or high.'}), 400
            sql = 'SELECT * FROM transactions'
            params = []
            if filter_type == 'approved': sql += " WHERE transaction_status LIKE '%Approved%'"
            elif filter_type == 'hold': sql += " WHERE transaction_status LIKE '%Hold%'"
            elif filter_type == 'high': sql += " WHERE risk_level='High'"
            sql += ' ORDER BY created_at DESC LIMIT ?'; params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        return jsonify({'count': len(rows), 'transactions': [dict(r) for r in rows]})

    # Add a normalized 0–100 risk score without changing the original detector.
    original_analyze = core.analyze_transaction
    def scored_analyze(data):
        result = original_analyze(data)
        if isinstance(result, dict):
            try:
                err = max(0.0, float(result.get('reconstruction_error', 0)))
                threshold = max(float(result.get('threshold', getattr(core, 'THRESHOLD', 1.0))), 1e-9)
                result['risk_score'] = round(max(0.0, min(100.0, 50.0 * err / threshold)), 2)
                result['risk_score_scale'] = '0–100; threshold corresponds to 50'
            except (TypeError, ValueError):
                result['risk_score'] = None
        return result
    core.analyze_transaction = scored_analyze

    @app.route('/api/transactions/<transaction_id>/report.pdf', methods=['GET'])
    def transaction_pdf_report(transaction_id):
        with db() as conn:
            row = conn.execute('SELECT * FROM transactions WHERE transaction_id=?', (transaction_id,)).fetchone()
        if not row:
            return jsonify({'error': 'Transaction not found.'}), 404
        item = dict(row)
        err = float(item.get('reconstruction_error') or 0)
        threshold = float(item.get('threshold') or 1)
        score = round(max(0.0, min(100.0, 50.0 * err / max(threshold, 1e-9))), 2)
        buf = io.BytesIO()
        pdf = canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        y = height - 55
        pdf.setTitle('Transaction Risk Report')
        pdf.setFont('Helvetica-Bold', 17); pdf.drawString(48, y, 'Online Payment Fraud Risk Report'); y -= 34
        pdf.setFont('Helvetica', 10)
        fields = [
            ('Transaction ID', item.get('transaction_id')), ('Customer ID', item.get('customer_id')),
            ('Merchant ID', item.get('merchant_id')), ('Amount', item.get('transaction_amount')),
            ('Status', item.get('transaction_status')), ('Risk level', item.get('risk_level')),
            ('Risk score (0–100)', score), ('Reconstruction error', err), ('Model threshold', threshold),
            ('Created at', item.get('created_at')),
        ]
        for label, value in fields:
            text = f'{label}: {value if value is not None else "N/A"}'
            pdf.drawString(48, y, text[:110]); y -= 19
        y -= 10
        pdf.setFont('Helvetica-Bold', 12); pdf.drawString(48, y, 'Model explanation'); y -= 22
        pdf.setFont('Helvetica', 9)
        explanation = str(item.get('xai_explanation') or 'No explanation stored for this transaction.')
        for paragraph in explanation.splitlines() or [explanation]:
            words = paragraph.split(); line = ''
            for word in words:
                candidate = (line + ' ' + word).strip()
                if len(candidate) > 100:
                    pdf.drawString(48, y, line); y -= 14; line = word
                    if y < 60:
                        pdf.showPage(); y = height - 55; pdf.setFont('Helvetica', 9)
                else: line = candidate
            if line:
                pdf.drawString(48, y, line); y -= 14
            if y < 60:
                pdf.showPage(); y = height - 55; pdf.setFont('Helvetica', 9)
        y -= 12
        pdf.setFont('Helvetica-Oblique', 8)
        pdf.drawString(48, max(y, 35), 'Decision support only. Review flagged transactions before taking action.')
        pdf.save(); buf.seek(0)
        return send_file(buf, mimetype='application/pdf', as_attachment=True,
                         download_name=f'transaction_{transaction_id}_report.pdf')

    @app.route('/api/xai/explain', methods=['POST'])
    def explain_transaction_shap():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get('transaction'), dict):
            return jsonify({'error': 'Send JSON with a transaction object.'}), 400
        try:
            import numpy as np
            import pandas as pd
            import shap
            raw = body['transaction']
            transformed = core.preprocess_data(raw)
            if transformed.ndim == 1: transformed = transformed.reshape(1, -1)
            model = core.model
            def anomaly_score(batch):
                batch = np.asarray(batch, dtype=np.float32)
                reconstructed = model.predict(batch, verbose=0)
                return np.mean(np.square(batch - reconstructed), axis=1)
            train_path = Path('data/transactions_train.csv')
            if not train_path.exists():
                return jsonify({'error': 'Training CSV not found at data/transactions_train.csv.'}), 500
            train = pd.read_csv(train_path)
            if 'is_fraud' in train.columns:
                train = train[train['is_fraud'].astype(int) == 0]
            background_raw = train.sample(n=min(12, len(train)), random_state=42)
            background = core.preprocess_data(background_raw.iloc[0].to_dict())
            bg_rows = [core.preprocess_data(r.to_dict())[0] for _, r in background_raw.iterrows()]
            background = np.asarray(bg_rows, dtype=np.float32)
            explainer = shap.KernelExplainer(anomaly_score, background)
            values = explainer.shap_values(transformed, nsamples=40)
            if isinstance(values, list): values = values[0]
            vals = np.asarray(values)
            if vals.ndim == 3: vals = vals[0, :, 0]
            else: vals = vals.reshape(-1)
            names = list(getattr(core, 'feature_names', []))
            if len(names) != len(vals): names = [f'feature_{i}' for i in range(len(vals))]
            ranked = sorted([{'feature': str(n), 'shap_value': float(v), 'impact': 'increases anomaly score' if v > 0 else 'decreases anomaly score'} for n, v in zip(names, vals)], key=lambda x: abs(x['shap_value']), reverse=True)[:8]
            score = float(anomaly_score(transformed)[0])
            return jsonify({'method': 'SHAP KernelExplainer', 'baseline_expected_value': float(np.asarray(explainer.expected_value).reshape(-1)[0]), 'reconstruction_error': score, 'top_features': ranked, 'note': 'Positive SHAP values increase the model reconstruction-error score relative to the background baseline.'})
        except Exception as exc:
            app.logger.exception('SHAP explanation failed')
            return jsonify({'error': 'SHAP explanation failed.', 'detail': str(exc)}), 500

    @app.route('/api/health/backend', methods=['GET'])
    def backend_health():
        return jsonify({'status': 'ok', 'features': ['risk_score', 'pdf_report', 'register_login_token_auth', 'authenticated_history', 'shap_explanation'], 'auth_token_ttl_hours': 24})

    print('Backend add-ons registered: auth, risk score, PDF report, SHAP endpoint, health check.')
