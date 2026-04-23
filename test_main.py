import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


CHUNK_SAMPLES = 4096
BUFFER_SIZE = 24000


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]
    session.get_inputs.return_value[0].name = "input"
    session.run.return_value = [np.array([[0.85, 0.15]], dtype=np.float32)]
    return session


@pytest.fixture
def processor(mock_session):
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        from main import AudioProcessor
        p = AudioProcessor()
    return p


@pytest.fixture
def valid_chunk():
    samples = np.random.uniform(-1.0, 1.0, CHUNK_SAMPLES).astype(np.float32)
    return samples.tobytes()


@pytest.fixture
def http_client():
    with patch("onnxruntime.InferenceSession", return_value=MagicMock()):
        from main import app
        with TestClient(app) as client:
            yield client


class TestValidateChunk:
    def test_valid_chunk_passes(self, processor, valid_chunk):
        is_valid, error = processor.validate_chunk(valid_chunk)
        assert is_valid is True
        assert error is None

    def test_empty_bytes_fails(self, processor):
        is_valid, error = processor.validate_chunk(b"")
        assert is_valid is False
        assert error == "Received empty chunk"

    def test_misaligned_bytes_fails(self, processor):
        is_valid, error = processor.validate_chunk(b"\x00\x01\x02")
        assert is_valid is False
        assert "not aligned to float32" in error
        assert "3" in error

    def test_single_float_passes(self, processor):
        single = np.float32(1.0).tobytes()
        is_valid, error = processor.validate_chunk(single)
        assert is_valid is True
        assert error is None

    def test_misaligned_large_passes_when_aligned(self, processor):
        aligned = np.zeros(100, dtype=np.float32).tobytes()
        is_valid, error = processor.validate_chunk(aligned)
        assert is_valid is True
        assert error is None


class TestProcessChunk:
    def test_valid_chunk_updates_buffer(self, processor, valid_chunk):
        processor.process_chunk(valid_chunk)
        assert np.count_nonzero(processor.buffer) > 0

    def test_chunk_appended_to_end_of_buffer(self, processor):
        chunk_data = np.ones(CHUNK_SAMPLES, dtype=np.float32)
        processor.process_chunk(chunk_data.tobytes())
        np.testing.assert_array_equal(processor.buffer[-CHUNK_SAMPLES:], chunk_data)

    def test_buffer_rolls_on_overflow(self, processor):
        first = np.full(CHUNK_SAMPLES, 1.0, dtype=np.float32)
        second = np.full(CHUNK_SAMPLES, 2.0, dtype=np.float32)
        processor.process_chunk(first.tobytes())
        processor.process_chunk(second.tobytes())
        np.testing.assert_array_equal(processor.buffer[-CHUNK_SAMPLES:], second)

    def test_invalid_chunk_raises_value_error(self, processor):
        with pytest.raises(ValueError):
            processor.process_chunk(b"\x00\x01\x02")

    def test_empty_chunk_raises_value_error(self, processor):
        with pytest.raises(ValueError, match="empty chunk"):
            processor.process_chunk(b"")

    def test_multiple_chunks_fill_buffer(self, processor):
        chunk = np.ones(CHUNK_SAMPLES, dtype=np.float32).tobytes()
        fills_needed = BUFFER_SIZE // CHUNK_SAMPLES
        for _ in range(fills_needed):
            processor.process_chunk(chunk)
        assert processor.count_nonzero() == BUFFER_SIZE


class TestResetBuffer:
    def test_reset_zeros_buffer(self, processor, valid_chunk):
        processor.process_chunk(valid_chunk)
        assert np.count_nonzero(processor.buffer) > 0
        processor.reset_buffer()
        assert np.count_nonzero(processor.buffer) == 0

    def test_reset_zeroes_inference_counter(self, processor):
        processor.inference_counter = 7
        processor.reset_buffer()
        assert processor.inference_counter == 0

    def test_reset_buffer_dtype_preserved(self, processor):
        processor.reset_buffer()
        assert processor.buffer.dtype == np.float32

    def test_reset_buffer_size_preserved(self, processor):
        processor.reset_buffer()
        assert len(processor.buffer) == BUFFER_SIZE


class TestExtractLFCC:
    def test_output_shape_60_rows(self, processor):
        processor.buffer = np.random.randn(BUFFER_SIZE).astype(np.float32)
        lfcc = processor.extract_lfcc()
        assert lfcc.shape[0] == 60

    def test_output_is_2d(self, processor):
        processor.buffer = np.random.randn(BUFFER_SIZE).astype(np.float32)
        lfcc = processor.extract_lfcc()
        assert lfcc.ndim == 2

    def test_output_dtype_float(self, processor):
        processor.buffer = np.random.randn(BUFFER_SIZE).astype(np.float32)
        lfcc = processor.extract_lfcc()
        assert np.issubdtype(lfcc.dtype, np.floating)

    def test_num_frames_positive(self, processor):
        processor.buffer = np.random.randn(BUFFER_SIZE).astype(np.float32)
        lfcc = processor.extract_lfcc()
        assert lfcc.shape[1] > 0


class TestPrepareInput:
    def test_output_shape_4d(self, processor):
        lfcc = np.random.randn(60, 93).astype(np.float32)
        tensor = processor.prepare_input(lfcc)
        assert tensor.ndim == 4

    def test_batch_dim_is_1(self, processor):
        lfcc = np.random.randn(60, 93).astype(np.float32)
        tensor = processor.prepare_input(lfcc)
        assert tensor.shape[0] == 1

    def test_channel_dim_is_1(self, processor):
        lfcc = np.random.randn(60, 93).astype(np.float32)
        tensor = processor.prepare_input(lfcc)
        assert tensor.shape[1] == 1

    def test_lfcc_dims_preserved(self, processor):
        lfcc = np.random.randn(60, 93).astype(np.float32)
        tensor = processor.prepare_input(lfcc)
        assert tensor.shape[2] == 60
        assert tensor.shape[3] == 93

    def test_output_dtype_float32(self, processor):
        lfcc = np.random.randn(60, 93).astype(np.float64)
        tensor = processor.prepare_input(lfcc)
        assert tensor.dtype == np.float32


class TestHealthEndpoint:
    def test_returns_200(self, http_client):
        response = http_client.get("/health")
        assert response.status_code == 200

    def test_response_has_status_key(self, http_client):
        response = http_client.get("/health")
        assert "status" in response.json()

    def test_status_value_is_healthy(self, http_client):
        response = http_client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_response_has_model_loaded_key(self, http_client):
        response = http_client.get("/health")
        assert "model_loaded" in response.json()

    def test_model_loaded_is_bool(self, http_client):
        response = http_client.get("/health")
        assert isinstance(response.json()["model_loaded"], bool)


class TestStatsEndpoint:
    def test_returns_200(self, http_client):
        response = http_client.get("/stats")
        assert response.status_code == 200

    def test_response_has_total_inferences(self, http_client):
        response = http_client.get("/stats")
        assert "total_inferences" in response.json()

    def test_response_has_avg_inference_ms(self, http_client):
        response = http_client.get("/stats")
        assert "avg_inference_ms" in response.json()

    def test_total_inferences_is_int(self, http_client):
        response = http_client.get("/stats")
        assert isinstance(response.json()["total_inferences"], int)

    def test_avg_inference_ms_is_numeric(self, http_client):
        response = http_client.get("/stats")
        assert isinstance(response.json()["avg_inference_ms"], (int, float))

    def test_initial_total_inferences_is_zero(self, http_client):
        response = http_client.get("/stats")
        assert response.json()["total_inferences"] == 0
