# Administrator login

These applications require a local administrator login for all pages and APIs.
The configured username is `admin`, with role `superadmin`. Use the password
provided when the account was created. No plaintext password is stored here.

Open `/login` to sign in and `/logout` to sign out. Sessions expire after 12 hours;
logout revokes the session. Login forms have CSRF protection and failed attempts
are limited to 10 per five minutes per client address. Existing application CSRF
checks still apply. Cookies are HttpOnly, SameSite=Lax, and Secure over HTTPS.
Application responses use Cache-Control: no-store.

Accounts and session state are under `~/.config/flask-local-auth/`, with separate
credentials, signing keys and cookies for each application entrypoint. The
optional LOCAL_ADMIN_AUTH_DIR environment variable selects another private
configuration directory. A missing account fails closed with HTTP 503.

For a new machine, changed entrypoint path, different service user, or password
rotation, run the helper below with that application's Python interpreter.
It prompts for the password without echoing it. Add `--replace` to rotate an
existing account and invalidate all its sessions. Never commit account JSON
files or expose the configuration directory through a web server.

```sh
python local_admin_auth.py --app-file '/home/onyango/Desktop/PROJECT/work/translate-ai/kikuyu_ai/web/flask_app.py' --username admin
```

Restart an already running application after installing the login gate.
API scripts can authenticate with HTTP Basic, for example `curl --user admin
http://127.0.0.1:PORT/api/ENDPOINT` (curl prompts for the password). Use HTTPS
when accessing the application over a network. Browser extensions and external
API clients must now authenticate; unauthenticated requests receive HTTP 401.
Existing third-party OAuth connections still require the local login first.

Templates may use `local_admin.username`, `local_admin.role`, and
`local_admin_logout_url`; request handlers may read `g.local_admin`.

An inventory, backups, security tests and verification results are saved under
/home/onyango/Desktop/flask-login-upgrade-2026-09-13/.
