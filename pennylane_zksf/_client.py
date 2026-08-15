"""Thin HTTP client for the ZKSF job API.

Deliberately not a dependency on qsim-sdk: that package converts circuits from
several frameworks and carries qiskit with it, none of which this plugin needs
since PennyLane produces its own QASM2. A plugin that drags an extra circuit
library into a PennyLane environment is a plugin people uninstall.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.zksf.org"


class JobFailed(RuntimeError):
    pass


class JobRejected(RuntimeError):
    pass


class ServiceClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 600.0,
        poll_seconds: float = 0.25,
    ):
        self._base = (base_url or os.environ.get("ZKSF_API") or DEFAULT_BASE_URL).rstrip("/")
        token = token or os.environ.get("ZKSF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(base_url=self._base, headers=headers, timeout=timeout)
        self._timeout = timeout
        self._poll = poll_seconds

    def _post(self, path: str, body: dict[str, Any]) -> dict:
        resp = self._http.post(path, json=body)
        if resp.status_code == 401:
            raise PermissionError(
                "the service rejected the token. Pass token= to qml.device(...) or "
                "set ZKSF_TOKEN; your token is on the dashboard at app.zksf.org"
            )
        if resp.status_code == 402:
            raise RuntimeError(f"insufficient balance: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    def _wait(self, job_id: str) -> dict:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            resp = self._http.get(f"/jobs/{job_id}")
            resp.raise_for_status()
            job = resp.json()
            if job["status"] == "done":
                return job
            if job["status"] == "rejected":
                raise JobRejected(job.get("reason"))
            if job["status"] == "error":
                raise JobFailed(job.get("error"))
            time.sleep(self._poll)
        raise TimeoutError(f"job {job_id} still running after {self._timeout}s")

    def run_one(
        self, qasm2: str, shots: int, engine: str | None, observable, params: dict
    ) -> dict:
        job = self._post(
            "/jobs",
            {"qasm2": qasm2, "shots": shots, "engine": engine,
             "observable": observable, "params": params},
        )
        return job if job["status"] == "done" else self._wait(job["id"])

    def run_batch(
        self, qasms: list[str], shots: int, engine: str | None, observable, params: dict
    ) -> dict:
        job = self._post(
            "/jobs/batch",
            {"circuits": qasms, "shots": shots, "engine": engine,
             "observable": observable, "params": params},
        )
        return job if job["status"] == "done" else self._wait(job["id"])
