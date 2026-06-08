import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import sys
import os

CHUNK_SAMPLES = 4096
BUFFER_SIZE = 24000
INFERENCE_INTERVAL = 4
CHUNKS_TO_FILL = 6
CHUNKS_TO_DETECT = CHUNKS_TO_FILL + INFERENCE_INTERVAL - 1

# Ensure model exists before running tests
if not os.path.exists('model.onnx'):
    import onnx
    from onnx import helper, TensorProto

    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 1, 60, None]
    )
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 2]
    )

    pool = helper.make_node('GlobalAveragePool', inputs=['input'], outputs=['pooled'])
    flatten = helper.make_node('Flatten', inputs=['pooled'], outputs=['flat'], axis=1)

    weight = helper.make_tensor('W', TensorProto.FLOAT, [2, 60], [0.1] * 120)
    bias = helper.make_tensor('b', TensorProto.FLOAT, [2], [0.0, 0.0])
    gemm = helper.make_node('Gemm', inputs=['flat', 'W', 'b'], outputs=['output'])

    graph = helper.make_graph(
        [pool, flatten, gemm], 'dummy', [input_tensor], [output_tensor],
        initializer=[weight, bias]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    onnx.save(model, 'model.onnx')

CHUNK_SAMPLES = 4096
BUFFER_SIZE = 24000
INFERENCE_INTERVAL = 4
CHUNKS_TO_FILL = 6
CHUNKS_TO_DETECT = CHUNKS_TO_FILL + INFERENCE_INTERVAL - 1


def make_mock_session():
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock()]
    session.get_inputs.return_value[0].name = "input"
    session.run.return_value = [np.array([[0.85, 0.15]], dtype=np.float32)]
    return session


def make_chunk(value=1.0):
    return np.full(CHUNK_SAMPLES, value, dtype=np.float32).tobytes()


def make_invalid_chunk():
    return b"\x00\x01\x02"


@pytest.fixture
def client():
    with patch("onnxruntime.InferenceSession", return_value=make_mock_session()):
        from main import app
        with TestClient(app) as c:
            yield c


def send_n_chunks(ws, n, value=1.0):
    responses = []
    for _ in range(n):
        ws.send_bytes(make_chunk(value))
        responses.append(ws.receive_json())
    return responses


class TestWebSocketConnection:
    def test_connection_accepted_returns_data(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(make_chunk())
            data = ws.receive_json()
            assert data is not None

    def test_first_chunk_returns_buffering(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(make_chunk())
            data = ws.receive_json()
            assert data["status"] == "buffering"

    def test_buffering_has_samples_key(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(make_chunk())
            data = ws.receive_json()
            assert "samples" in data

    def test_samples_count_increases(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(make_chunk())
            d1 = ws.receive_json()
            ws.send_bytes(make_chunk())
            d2 = ws.receive_json()
            assert d2["samples"] > d1["samples"]


class TestBufferingProgression:
    def test_five_chunks_still_buffering(self, client):
        with client.websocket_connect("/ws") as ws:
            responses = send_n_chunks(ws, 5)
            assert all(r["status"] == "buffering" for r in responses)

    def test_six_chunks_exits_buffering(self, client):
        with client.websocket_connect("/ws") as ws:
            responses = send_n_chunks(ws, CHUNKS_TO_FILL)
            assert responses[-1]["status"] != "buffering"

    def test_samples_never_exceeds_buffer_size(self, client):
        with client.websocket_connect("/ws") as ws:
            for _ in range(4):
                ws.send_bytes(make_chunk())
                data = ws.receive_json()
                assert data.get("samples", 0) <= BUFFER_SIZE


class TestReadyState:
    def test_ready_status_after_buffer_fills(self, client):
        with client.websocket_connect("/ws") as ws:
            responses = send_n_chunks(ws, CHUNKS_TO_FILL)
            assert responses[-1]["status"] == "ready"

    def test_ready_response_has_next_inference_in(self, client):
        with client.websocket_connect("/ws") as ws:
            responses = send_n_chunks(ws, CHUNKS_TO_FILL)
            assert "next_inference_in" in responses[-1]

    def test_next_inference_in_starts_at_three(self, client):
        with client.websocket_connect("/ws") as ws:
            responses = send_n_chunks(ws, CHUNKS_TO_FILL)
            assert responses[-1]["next_inference_in"] == INFERENCE_INTERVAL - 1

    def test_next_inference_in_counts_down(self, client):
        with client.websocket_connect("/ws") as ws:
            send_n_chunks(ws, CHUNKS_TO_FILL)
            ws.send_bytes(make_chunk())
            d1 = ws.receive_json()
            ws.send_bytes(make_chunk())
            d2 = ws.receive_json()
            assert d2["next_inference_in"] < d1["next_inference_in"]


class TestInferenceTrigger:
    def test_detection_fires_after_interval(self, client):
        with client.websocket_connect("/ws") as ws:
            responses = send_n_chunks(ws, CHUNKS_TO_DETECT)
            assert responses[-1]["status"] == "detection"

    def test_detection_has_prediction_key(self, client):
        with client.websocket_connect("/ws") as ws:
            responses = send_n_chunks(ws, CHUNKS_TO_DETECT)
            assert "prediction" in responses[-1]

    def test_detection_has_confidence_key(self, client):
        with client.websocket_connect("/ws") as ws:
            responses = send_n_chunks(ws, CHUNKS_TO_DETECT)
            assert "confidence" in responses[-1]

    def test_prediction_is_float(self, client):
        with client.websocket_connect("/ws") as ws:
            responses = send_n_chunks(ws, CHUNKS_TO_DETECT)
            assert isinstance(responses[-1]["prediction"], float)

    def test_confidence_is_float(self, client):
        with client.websocket_connect("/ws") as ws:
            responses = send_n_chunks(ws, CHUNKS_TO_DETECT)
            assert isinstance(responses[-1]["confidence"], float)

    def test_detection_repeats_every_interval(self, client):
        with client.websocket_connect("/ws") as ws:
            send_n_chunks(ws, CHUNKS_TO_DETECT)
            send_n_chunks(ws, INFERENCE_INTERVAL - 1)
            ws.send_bytes(make_chunk())
            data = ws.receive_json()
            assert data["status"] == "detection"


class TestErrorHandling:
    def test_invalid_chunk_returns_error_status(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(make_invalid_chunk())
            data = ws.receive_json()
            assert data["status"] == "error"

    def test_invalid_chunk_response_has_message(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(make_invalid_chunk())
            data = ws.receive_json()
            assert "message" in data

    def test_connection_stays_open_after_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(make_invalid_chunk())
            ws.receive_json()
            ws.send_bytes(make_chunk())
            data = ws.receive_json()
            assert data["status"] == "buffering"

    def test_buffer_resets_after_invalid_chunk(self, client):
        with client.websocket_connect("/ws") as ws:
            send_n_chunks(ws, 3)
            ws.send_bytes(make_invalid_chunk())
            ws.receive_json()
            ws.send_bytes(make_chunk())
            data = ws.receive_json()
            assert data["status"] == "buffering"
            assert data["samples"] == CHUNK_SAMPLES


class TestModelNotLoaded:
    def test_error_status_when_model_missing(self):
        with patch("onnxruntime.InferenceSession", side_effect=Exception("not found")):
            from main import app
            with TestClient(app) as c:
                with c.websocket_connect("/ws") as ws:
                    data = ws.receive_json()
                    assert data["status"] == "error"

    def test_error_message_when_model_missing(self):
        with patch("onnxruntime.InferenceSession", side_effect=Exception("not found")):
            from main import app
            with TestClient(app) as c:
                with c.websocket_connect("/ws") as ws:
                    data = ws.receive_json()
                    assert data["message"] == "Model not loaded"
