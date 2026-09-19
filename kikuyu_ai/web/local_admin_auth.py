"""Local superadmin login gate. Credentials live outside application source.

Provision or rotate with: python local_admin_auth.py --app-file /path/to/app.py
Flask and its bundled dependencies are the only runtime requirements.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time
from urllib.parse import unquote, urlencode, urlsplit

from itsdangerous import BadSignature, URLSafeTimedSerializer
from markupsafe import escape
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.wrappers import Request, Response
from werkzeug.exceptions import HTTPException

LOGIN = '/_local_auth/login'
LOGOUT = '/_local_auth/logout'
SESSION_SECONDS = 12 * 60 * 60
CSRF_SECONDS = 15 * 60


def app_id(app_file):
    return hashlib.sha256(str(Path(app_file).resolve()).encode()).hexdigest()[:24]


def config_path(app_file):
    base = Path(os.environ.get('LOCAL_ADMIN_AUTH_DIR', str(Path.home() / '.config' / 'flask-local-auth')))
    return base / (app_id(app_file) + '.json')


def provision(app_file, username, password, *, replace=False):
    """Save only a salted hash and an independent random signing key."""
    path = config_path(app_file)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() and not replace:
        raise FileExistsError(f'Account already configured: {path}')
    if not username.strip() or len(password) < 8:
        raise ValueError('A username and a password of at least 8 characters are required.')
    record = dict(username=username.strip(), role='superadmin',
                  password_hash=generate_password_hash(password),
                  secret=secrets.token_hex(32), app_file=str(Path(app_file).resolve()))
    temporary = path.with_name(path.name + '.' + secrets.token_hex(8) + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(record, stream)
    os.replace(temporary, path)
    return path


def safe_next(value):
    value = value or '/'
    decoded = unquote(value)
    if (not value.startswith('/') or decoded.startswith('//') or '\\' in decoded
            or any(ord(c) < 32 or ord(c) == 127 for c in decoded)
            or urlsplit(value).netloc or urlsplit(value).scheme
            or decoded.startswith('/_local_auth/')):
        return '/'
    return value


def _page(title, body):
    return Response('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>''' + str(escape(title)) + '''</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#172b42;font:16px system-ui,sans-serif;min-height:100vh;display:grid;place-items:center;padding:24px}
main{background:white;width:100%;max-width:420px;padding:32px;border-radius:16px;box-shadow:0 8px 32px #172b4214}
h1{margin:0 0 8px;font-size:26px}p{line-height:1.5;color:#526176}label{display:block;font-weight:600;margin-top:18px}input,button{font:inherit;width:100%;padding:12px;border-radius:8px;border:1px solid #aebcca;margin-top:6px}button{background:#163e66;color:white;font-weight:600;cursor:pointer;margin-top:24px;border:0}input:focus,button:focus{outline:3px solid #80b7f0;outline-offset:2px}.error{color:#a4262c}a{color:#163e66}</style>
<main>''' + body + '</main></html>', content_type='text/html; charset=utf-8')


class LoginGate:
    def __init__(self, wrapped, app_file):
        self.wrapped = wrapped
        self.path = config_path(app_file)
        self.identity = app_id(app_file)
        self.cookie = 'local_admin_' + self.identity
        self.csrf_cookie = self.cookie + '_csrf'
        self.title = Path(app_file).parent.name
        if self.title in {'app', 'web', 'paper_trader', 'x_replies'}:
            self.title = Path(app_file).parent.parent.name

    def _account(self):
        try:
            record = json.loads(self.path.read_text())
            if (record['role'] != 'superadmin' or not record['username']
                    or len(record['secret']) < 32 or not record['password_hash']):
                return None
            return record
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _signer(self, account, purpose):
        return URLSafeTimedSerializer(account['secret'], salt=self.identity + ':' + purpose)

    def _version(self, account):
        return hashlib.sha256(account['password_hash'].encode()).hexdigest()

    def _user(self, request, account):
        try:
            data = self._signer(account, 'session').loads(request.cookies.get(self.cookie, ''), max_age=SESSION_SECONDS)
            if (data.get('username') == account['username'] and data.get('role') == 'superadmin'
                    and hmac.compare_digest(data.get('version', ''), self._version(account))):
                with sqlite3.connect(self.path.with_suffix('.attempts.sqlite3'), timeout=5) as db:
                    valid = db.execute('SELECT 1 FROM sessions WHERE nonce = ? AND expires > ?', (data.get('nonce', ''), time.time())).fetchone()
                if not valid:
                    return None
                return dict(username=account['username'], role='superadmin', is_superadmin=True)
        except (BadSignature, ValueError, TypeError, AttributeError, sqlite3.Error):
            pass
        return None

    def _csrf_valid(self, request, account):
        supplied = request.form.get('csrf_token', '')
        cookie = request.cookies.get(self.csrf_cookie, '')
        if not supplied or not hmac.compare_digest(supplied, cookie):
            return False
        try:
            self._signer(account, 'csrf').loads(cookie, max_age=CSRF_SECONDS)
            return True
        except BadSignature:
            return False

    def _form(self, request, account, *, logout=False, error=None, status=200):
        token = self._signer(account, 'csrf').dumps(secrets.token_hex(24))
        target = safe_next(request.values.get('next'))
        heading = 'Sign out' if logout else 'Sign in'
        body = '<h1>' + heading + '</h1><p>' + str(escape(self.title)) + '</p>'
        if error:
            body += '<p class="error" role="alert">' + str(escape(error)) + '</p>'
        body += '<form method="post" action="' + (LOGOUT if logout else LOGIN) + '">'
        body += '<input type="hidden" name="csrf_token" value="' + str(escape(token)) + '">'
        body += '<input type="hidden" name="next" value="' + str(escape(target)) + '">'
        if not logout:
            body += '''<label for="username">Username</label><input id="username" name="username" autocomplete="username" required autofocus maxlength="128">
<label for="password">Password</label><input id="password" name="password" type="password" autocomplete="current-password" required maxlength="1024">'''
        body += '<button type="submit">' + heading + '</button></form>'
        if logout:
            body += '<p><a href="/">Return to application</a></p>'
        response = _page(heading + ' · ' + self.title, body)
        response.status_code = status
        response.set_cookie(self.csrf_cookie, token, max_age=CSRF_SECONDS, httponly=True, secure=request.is_secure, samesite='Lax', path='/')
        return response

    def _attempt(self, request):
        """Shared across workers; reserve before hashing to bound guessing cost."""
        path = self.path.with_suffix('.attempts.sqlite3')
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        with sqlite3.connect(path, timeout=5) as db:
            db.execute('CREATE TABLE IF NOT EXISTS attempts (address TEXT, attempted REAL)')
            db.execute('CREATE TABLE IF NOT EXISTS sessions (nonce TEXT PRIMARY KEY, expires REAL)')
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM sessions WHERE expires < ?', (time.time(),))
            db.execute('DELETE FROM attempts WHERE attempted < ?', (time.time() - 300,))
            address = request.remote_addr or 'local'
            count = db.execute('SELECT COUNT(*) FROM attempts WHERE address = ?', (address,)).fetchone()[0]
            if count >= 10:
                return False
            db.execute('INSERT INTO attempts VALUES (?, ?)', (address, time.time()))
        return True

    def _clear_attempts(self, request):
        with sqlite3.connect(self.path.with_suffix(".attempts.sqlite3"), timeout=5) as db:
            db.execute("DELETE FROM attempts WHERE address = ?", (request.remote_addr or "local",))

    def _credentials(self, username, password, account):
        if not isinstance(username, str) or not isinstance(password, str) or len(password) > 1024:
            return False
        password_ok = check_password_hash(account['password_hash'], password)
        return hmac.compare_digest(username.encode(), account['username'].encode()) and password_ok

    def _response(self, response, environ, start_response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response(environ, start_response)

    def __call__(self, environ, start_response):
        try:
            return self._dispatch(environ, start_response)
        except HTTPException as error:
            return self._response(error.get_response(environ), environ, start_response)

    def _dispatch(self, environ, start_response):
        request = Request(environ)
        request.max_content_length = 16 * 1024 if request.path in {LOGIN, LOGOUT, "/login", "/logout"} else None
        account = self._account()
        if account is None:
            response = _page('Login unavailable', '<h1>Login unavailable</h1><p>The application owner must configure the administrator account.</p>')
            response.status_code = 503
            return self._response(response, environ, start_response)
        user = self._user(request, account)
        if request.path in {LOGIN, '/login', LOGOUT, '/logout'}:
            logout = request.path in {LOGOUT, '/logout'}
            if request.method not in {'GET', 'HEAD', 'POST'}:
                return self._response(Response('Method not allowed', status=405, headers={'Allow':'GET, HEAD, POST'}), environ, start_response)
            if request.method in {'GET', 'HEAD'}:
                response = self._form(request, account, logout=logout)
            elif not self._csrf_valid(request, account):
                response = self._form(request, account, logout=logout, error='This form has expired. Please try again.', status=400)
            elif logout:
                if user:
                    data = self._signer(account, 'session').loads(request.cookies[self.cookie], max_age=SESSION_SECONDS)
                    with sqlite3.connect(self.path.with_suffix('.attempts.sqlite3'), timeout=5) as db:
                        db.execute('DELETE FROM sessions WHERE nonce = ?', (data['nonce'],))
                response = Response(status=303, headers={'Location': LOGIN})
                response.delete_cookie(self.cookie, path='/', secure=request.is_secure, httponly=True, samesite='Lax')
                response.delete_cookie(self.csrf_cookie, path='/')
            elif not self._attempt(request):
                response = self._form(request, account, error='Too many attempts. Try again in five minutes.', status=429)
                response.headers['Retry-After'] = '300'
            elif self._credentials(request.form.get('username', ''), request.form.get('password', ''), account):
                self._clear_attempts(request)
                data = dict(username=account['username'], role='superadmin', version=self._version(account), nonce=secrets.token_hex(16))
                with sqlite3.connect(self.path.with_suffix('.attempts.sqlite3'), timeout=5) as db:
                    db.execute('INSERT INTO sessions VALUES (?, ?)', (data['nonce'], time.time() + SESSION_SECONDS))
                value = self._signer(account, 'session').dumps(data)
                response = Response(status=303, headers={'Location': safe_next(request.form.get('next'))})
                response.set_cookie(self.cookie, value, max_age=SESSION_SECONDS, httponly=True, secure=request.is_secure, samesite='Lax', path='/')
                response.delete_cookie(self.csrf_cookie, path='/')
            else:
                response = self._form(request, account, error='Incorrect username or password.', status=401)
            return self._response(response, environ, start_response)
        # Script/API callers may use HTTP Basic over localhost or HTTPS.
        basic = request.authorization
        using_basic = False
        if basic and basic.type.lower() == 'basic':
            if not self._attempt(request):
                return self._response(Response('Too many attempts', status=429, headers={'Retry-After':'300'}), environ, start_response)
            if self._credentials(basic.username or '', basic.password or '', account):
                using_basic = True
                self._clear_attempts(request)
                user = dict(username=account['username'], role='superadmin', is_superadmin=True)
        if user:
            # Browsers include Origin on state-changing requests. Preserve existing
            # app CSRF checks and reject cross-site cookie-authenticated writes.
            if request.method not in {'GET', 'HEAD', 'OPTIONS'} and not using_basic:
                origin = request.headers.get('Origin')
                if (request.headers.get('Sec-Fetch-Site') == 'cross-site'
                        or (origin and origin.rstrip('/') != request.host_url.rstrip('/'))):
                    return self._response(Response('Cross-site request rejected', status=403), environ, start_response)
            environ['local_admin.user'] = user
            def private_response(status, headers, exc_info=None):
                headers = [(name, value) for name, value in headers if name.lower() != 'cache-control']
                headers.append(('Cache-Control', 'no-store'))
                return start_response(status, headers, exc_info)
            return self.wrapped(environ, private_response)
        if request.path.startswith('/api/') or request.is_json or (basic is not None):
            response = Response(json.dumps({'error':'authentication_required', 'login_url':LOGIN}), status=401, content_type='application/json')
            if basic is not None:
                response.headers['WWW-Authenticate'] = 'Basic realm="Local application"'
        else:
            response = Response(status=302, headers={'Location': LOGIN + '?' + urlencode({'next': safe_next(request.full_path.rstrip('?'))})})
        return self._response(response, environ, start_response)


def protect_app(app, app_file):
    """Install before the app is served; does not alter business-data storage."""
    if 'local_admin_auth' in app.extensions:
        return
    gate = LoginGate(app.wsgi_app, app_file)
    app.wsgi_app = gate
    app.extensions['local_admin_auth'] = gate
    from flask import g, request

    @app.before_request
    def local_admin_identity():
        g.local_admin = request.environ.get('local_admin.user')

    @app.context_processor
    def local_admin_context():
        return {'local_admin': request.environ.get('local_admin.user'), 'local_admin_logout_url': LOGOUT}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-file', required=True)
    parser.add_argument('--username', default='admin')
    parser.add_argument('--replace', action='store_true', help='Rotate credentials and invalidate existing sessions')
    args = parser.parse_args()
    password = getpass.getpass('Administrator password: ')
    if password != getpass.getpass('Confirm password: '):
        parser.error('Passwords do not match')
    print(provision(args.app_file, args.username, password, replace=args.replace))
