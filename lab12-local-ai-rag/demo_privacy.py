"""
Proof #2: nothing leaves your computer.

    python demo_privacy.py

While the model answers a question, this script watches the network connections
of Ollama and of this Python program, and prints every address they talk to.
Every one should be 127.0.0.1 (or ::1) — "this computer".

The strongest version: turn your Wi-Fi off and run it again. It still works.
(macOS may hide other programs' connections: if you see "access denied", run
 `sudo python demo_privacy.py` or use `lsof -i -n -P | grep -i ollama` in a second terminal.)
"""
import os
import threading
import time

import psutil

from rag import answer

LOCAL = ("127.", "::1", "0.0.0.0", "::")
seen, denied = set(), False


def ollama_and_me():
    procs = [psutil.Process(os.getpid())]
    for p in psutil.process_iter(["name"]):
        if "ollama" in (p.info["name"] or "").lower():
            procs.append(p)
    return procs


def watch(stop):
    global denied
    while not stop.is_set():
        for p in ollama_and_me():
            try:
                for c in p.net_connections(kind="inet"):
                    if c.raddr:
                        seen.add(c.raddr.ip)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                denied = True
        time.sleep(0.05)


stop = threading.Event()
t = threading.Thread(target=watch, args=(stop,), daemon=True)
t.start()
print("Asking a question while watching the network...\n")
out = answer("Summarize the rules for short-term rentals.")
stop.set()
t.join()

print(out["answer"], "\n")
print("Addresses contacted:")
if not seen:
    print("  (none caught - the answer came back very fast; run it again)")
for ip in sorted(seen):
    print(f"  {ip:20s} {'local (this computer)' if ip.startswith(LOCAL) else '<-- EXTERNAL!'}")
external = [ip for ip in seen if not ip.startswith(LOCAL)]
print("\nResult:", "zero external connections." if not external else f"{len(external)} external address(es)!")
if denied:
    print("Note: some processes could not be inspected (access denied) — see the note at the top of this file.")
