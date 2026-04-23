import argparse
import json
import time
import numpy as np
import psutil
import os
import sys
import logging
import librosa
import onnxruntime as ort
from scipy.fftpack import dct


SAMPLE_RATE = 16000
AUDIO_DURATION = 1.5
BUFFER_SIZE = int(SAMPLE_RATE * AUDIO_DURATION)
N_FFT = 512
HOP_LENGTH = 256
N_LFCC = 60


def parse_args():
    parser = argparse.ArgumentParser(description="VoiceGuard-Proto Pipeline Benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=100, help="Number of benchmark iterations (default: 100)")
    parser.add_argument("--model", type=str, default="model.onnx", help="Path to ONNX model file (default: model.onnx)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional path to save results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable detailed per-iteration logging")
    return parser.parse_args()


def generate_audio(seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, BUFFER_SIZE).astype(np.float32)


def extract_lfcc(audio):
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False)
    power_spec = np.abs(stft) ** 2
    db_spec = librosa.power_to_db(power_spec)
    lfcc_full = dct(db_spec, type=2, axis=0, norm="ortho")
    return lfcc_full[:N_LFCC, :]


def prepare_input(lfcc):
    tensor = np.expand_dims(lfcc, axis=0)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor.astype(np.float32)


def run_inference(session, input_tensor):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})
    return output


def compute_stats(timings):
    arr = np.array(timings) * 1000
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
    }


def print_table(results):
    stages = ["lfcc_extraction", "tensor_preparation", "onnx_inference", "end_to_end"]
    col_width = 14
    header = f"{'Stage':<22}" + "".join(f"{k:>{col_width}}" for k in ["mean_ms", "median_ms", "min_ms", "max_ms", "std_ms"])
    separator = "-" * len(header)
    print()
    print("=" * len(header))
    print("  VoiceGuard-Proto Pipeline Benchmark Results")
    print("=" * len(header))
    print(header)
    print(separator)
    for stage in stages:
        stats = results["stages"][stage]
        label = stage.replace("_", " ").title()
        row = f"{label:<22}" + "".join(f"{stats[k]:>{col_width}.3f}" for k in ["mean_ms", "median_ms", "min_ms", "max_ms", "std_ms"])
        print(row)
    print(separator)
    print(f"\n{'Iterations':<22}{results['iterations']:>10}")
    print(f"{'Memory RSS (MB)':<22}{results['memory_rss_mb']:>10.2f}")
    print(f"{'Memory VMS (MB)':<22}{results['memory_vms_mb']:>10.2f}")
    print()


def run_benchmark(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if not os.path.exists(args.model):
        print(f"Error: model file '{args.model}' not found.")
        sys.exit(1)

    logging.debug(f"Loading ONNX model from {args.model}")
    session = ort.InferenceSession(args.model)
    logging.debug("Model loaded")

    audio = generate_audio()
    logging.debug(f"Synthetic audio shape: {audio.shape}")

    lfcc_times = []
    prep_times = []
    infer_times = []
    e2e_times = []

    process = psutil.Process(os.getpid())

    logging.debug(f"Running {args.iterations} iterations...")
    for i in range(args.iterations):
        t0 = time.perf_counter()

        t_lfcc_start = time.perf_counter()
        lfcc = extract_lfcc(audio)
        t_lfcc_end = time.perf_counter()

        t_prep_start = time.perf_counter()
        tensor = prepare_input(lfcc)
        t_prep_end = time.perf_counter()

        t_infer_start = time.perf_counter()
        run_inference(session, tensor)
        t_infer_end = time.perf_counter()

        t1 = time.perf_counter()

        lfcc_times.append(t_lfcc_end - t_lfcc_start)
        prep_times.append(t_prep_end - t_prep_start)
        infer_times.append(t_infer_end - t_infer_start)
        e2e_times.append(t1 - t0)

        if args.verbose:
            logging.debug(f"Iter {i + 1:>4}/{args.iterations} | e2e={e2e_times[-1] * 1000:.2f}ms")

    mem_info = process.memory_info()

    results = {
        "iterations": args.iterations,
        "model": args.model,
        "buffer_size": BUFFER_SIZE,
        "sample_rate": SAMPLE_RATE,
        "memory_rss_mb": round(mem_info.rss / 1024 / 1024, 2),
        "memory_vms_mb": round(mem_info.vms / 1024 / 1024, 2),
        "stages": {
            "lfcc_extraction": compute_stats(lfcc_times),
            "tensor_preparation": compute_stats(prep_times),
            "onnx_inference": compute_stats(infer_times),
            "end_to_end": compute_stats(e2e_times),
        },
    }

    print_table(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
