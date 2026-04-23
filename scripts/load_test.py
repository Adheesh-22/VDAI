import argparse
import asyncio
import json
import time
import numpy as np
import websockets


CHUNK_SAMPLES = 4096
SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="VoiceGuard-Proto WebSocket Load Test")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--clients", "-c", type=int, default=10, help="Concurrent WebSocket clients (default: 10)")
    parser.add_argument("--chunks", "-n", type=int, default=20, help="Chunks per client (default: 20)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-message responses")
    return parser.parse_args()


def make_chunk():
    return np.random.uniform(-1.0, 1.0, CHUNK_SAMPLES).astype(np.float32).tobytes()


async def run_client(client_id, uri, num_chunks, verbose, results):
    latencies = []
    errors = 0
    statuses = {}

    try:
        async with websockets.connect(uri, max_size=2**20) as ws:
            for i in range(num_chunks):
                t0 = time.perf_counter()
                await ws.send(make_chunk())
                raw = await ws.recv()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)

                data = json.loads(raw)
                status = data.get("status", "unknown")
                statuses[status] = statuses.get(status, 0) + 1

                if verbose:
                    print(f"  Client {client_id:>3} | chunk {i + 1:>3} | {status:<12} | {elapsed_ms:.1f}ms")
    except Exception as e:
        errors += 1
        print(f"  Client {client_id} error: {e}")

    results.append({
        "client_id": client_id,
        "latencies": latencies,
        "errors": errors,
        "statuses": statuses,
    })


async def run_load_test(args):
    uri = f"ws://{args.host}:{args.port}/ws"
    print(f"\nLoad Test: {args.clients} clients x {args.chunks} chunks → {uri}\n")

    results = []
    t_start = time.perf_counter()
    tasks = [
        run_client(i, uri, args.chunks, args.verbose, results)
        for i in range(args.clients)
    ]
    await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - t_start

    all_latencies = [lat for r in results for lat in r["latencies"]]
    total_errors = sum(r["errors"] for r in results)
    all_statuses = {}
    for r in results:
        for k, v in r["statuses"].items():
            all_statuses[k] = all_statuses.get(k, 0) + v

    total_messages = len(all_latencies)
    throughput = total_messages / total_elapsed if total_elapsed > 0 else 0

    print("\n========================================")
    print("  Load Test Results")
    print("========================================")
    print(f"  Clients:          {args.clients}")
    print(f"  Chunks/client:    {args.chunks}")
    print(f"  Total messages:   {total_messages}")
    print(f"  Total errors:     {total_errors}")
    print(f"  Elapsed (s):      {total_elapsed:.2f}")
    print(f"  Throughput (msg/s): {throughput:.1f}")
    if all_latencies:
        arr = np.array(all_latencies)
        print(f"  Latency mean (ms):   {np.mean(arr):.2f}")
        print(f"  Latency p50 (ms):    {np.percentile(arr, 50):.2f}")
        print(f"  Latency p95 (ms):    {np.percentile(arr, 95):.2f}")
        print(f"  Latency p99 (ms):    {np.percentile(arr, 99):.2f}")
        print(f"  Latency max (ms):    {np.max(arr):.2f}")
    print(f"  Status breakdown:  {all_statuses}")
    print("========================================\n")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_load_test(args))
