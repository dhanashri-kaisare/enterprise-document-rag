import time
import statistics
import requests

API_URL = "http://127.0.0.1:8000"

QUESTIONS = [
    "What is RAG?",
    "How many paid leave days do employees receive?",
    "What is the company's annual revenue?"
]

print("\nEnterprise Document RAG - Performance Benchmark")
print("=" * 55)

# Check API availability
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    response.raise_for_status()
except requests.RequestException:
    print("FastAPI is not running. Start it before benchmarking.")
    raise SystemExit(1)

latencies = []

for question in QUESTIONS:
    try:
        start = time.perf_counter()

        response = requests.post(
            f"{API_URL}/questions/ask",
            json={"question": question},
            timeout=120
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            latencies.append(elapsed_ms)

            result = response.json()

            print(f"\nQuestion: {question}")
            print(f"Latency: {elapsed_ms:.2f} ms")
            print(f"Answer: {result.get('answer', '')}")
        else:
            print(f"\nQuestion failed: {question}")
            print(f"HTTP status: {response.status_code}")
            print(response.text)

    except requests.RequestException as exc:
        print(f"\nRequest failed: {exc}")

if latencies:
    print("\n" + "=" * 55)
    print(f"Successful requests: {len(latencies)}")
    print(f"Average latency: {statistics.mean(latencies):.2f} ms")
    print(f"Minimum latency: {min(latencies):.2f} ms")
    print(f"Maximum latency: {max(latencies):.2f} ms")
    print("=" * 55)