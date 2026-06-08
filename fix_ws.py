import sys

with open('test_websocket.py', 'r') as f:
    content = f.read()

# Replace client.websocket_connect("/ws") with connect_ws(client)
content = content.replace('client.websocket_connect("/ws")', 'connect_ws(client)')

# Insert connect_ws definition and contextlib import
import_statement = "from contextlib import contextmanager\n\n@contextmanager\ndef connect_ws(client):\n    with client.websocket_connect(\"/ws\") as ws:\n        ws.receive_json() # discard 'connected' message\n        yield ws\n\n"
content = content.replace('def send_n_chunks(ws, n, value=1.0):', import_statement + 'def send_n_chunks(ws, n, value=1.0):')

# Also fix the TestModelNotLoaded which has hardcoded client.websocket_connect
# Wait, TestModelNotLoaded doesn't use the `client` fixture, it uses `c.websocket_connect`.
# Let's manually replace `c.websocket_connect` in TestModelNotLoaded.
replacement_model_not_loaded = """class TestModelNotLoaded:
    def test_connected_status_when_model_missing(self):
        with patch("onnxruntime.InferenceSession", side_effect=Exception("not found")):
            from main import app
            with TestClient(app) as c:
                with c.websocket_connect("/ws") as ws:
                    data = ws.receive_json()
                    assert data["status"] == "connected"
                    assert data["model_available"] is False

    def test_features_ready_status_when_model_missing(self):
        with patch("onnxruntime.InferenceSession", side_effect=Exception("not found")):
            from main import app
            with TestClient(app) as c:
                with c.websocket_connect("/ws") as ws:
                    ws.receive_json()
                    ws.send_bytes(make_chunk())
                    # Actually we need to send 6 chunks to fill buffer and interval
                    # Let's just mock it with send_n_chunks
                    responses = send_n_chunks(ws, CHUNKS_TO_DETECT)
                    assert responses[-1]["status"] == "features_ready"
                    assert responses[-1]["model_available"] is False
"""

# Find the start of TestModelNotLoaded
start_idx = content.find("class TestModelNotLoaded:")
if start_idx != -1:
    content = content[:start_idx] + replacement_model_not_loaded

with open('test_websocket.py', 'w') as f:
    f.write(content)
