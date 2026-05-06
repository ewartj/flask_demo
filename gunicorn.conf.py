from app.config_loader import server_config

_s = server_config()
bind    = f"0.0.0.0:{_s['port']}"
workers = _s["workers"]
threads = _s["threads"]
timeout = _s["timeout"]
