# Security Configuration

## Server Binding

**Status:** ✅ SECURE - Localhost only

### Configuration

The Gradio server is configured to bind only to localhost (127.0.0.1), making it inaccessible from external networks.

```python
# config.py
HOST: str = "127.0.0.1"  # Localhost only
PORT: int = 7860
```

### Why This Matters

**Before (Insecure):**
```python
HOST: str = "0.0.0.0"  # ❌ Accessible from any network interface
```
- Server accessible from any device on the network
- Potential security risk if on public WiFi
- Unnecessary exposure for local development

**After (Secure):**
```python
HOST: str = "127.0.0.1"  # ✅ Only accessible from localhost
```
- Server only accessible from your computer
- No external network exposure
- Safe for development and testing

### Verification

Check server is localhost-only:
```bash
# The server should NOT respond to external IPs
# Only responds to localhost:7860

curl http://127.0.0.1:7860  # ✅ Works
curl http://localhost:7860    # ✅ Works  
# From another device on network:
# curl http://YOUR_IP:7860   # ❌ Connection refused
```

### Accessing the App

**Local Access (✅ Allowed):**
- http://localhost:7860
- http://127.0.0.1:7860

**External Access (❌ Blocked):**
- http://192.168.x.x:7860 (local network)
- http://10.0.x.x:7860 (corporate network)
- http://PUBLIC_IP:7860 (internet)

### Changing Server Binding

**For Development (recommended):**
```python
# config.py - Keep as localhost
HOST: str = "127.0.0.1"
```

**For Network Access (use with caution):**
```python
# config.py - Only if needed
HOST: str = "0.0.0.0"  # ⚠️ Opens to all interfaces
```

⚠️ **WARNING:** Only use `0.0.0.0` if:
- You're on a trusted private network
- You have firewall rules restricting access
- You understand the security implications

### Best Practices

1. **Default to localhost** - Always use `127.0.0.1` by default
2. **Firewall protection** - If using `0.0.0.0`, configure firewall
3. **Authentication** - Add authentication if exposing to network
4. **HTTPS** - Use HTTPS for any network exposure
5. **VPN** - Use VPN for remote access instead of exposing port

### Related Configuration

```python
# config.py
SHARE: bool = False  # Don't create public Gradio link
QUIET: bool = False  # Show startup messages
```

**Note:** The `SHARE` option creates a public URL through Gradio's servers. Keep this `False` for local-only operation.

---

**Last Updated:** 2026-03-23  
**Security Status:** Localhost-only ✅  
**Risk Level:** Minimal
