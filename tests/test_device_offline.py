"""Device tests against a stub service: no token, no network, no account.

These are the tests the public repository runs in CI. The full suite in the
private repository drives the real API, which catches contract drift; these
catch everything that is the device's own responsibility — conversion, wire
order, routing, refusals and how a failed evaluation is reported.
"""
from __future__ import annotations

import warnings

import numpy as np
import pennylane as qml
import pytest
from pennylane.tape import QuantumScript

from pennylane_zksf import ZKSFDevice


class StubService:
    """Stands in for the HTTP client. Records what the device asked for, so a
    test can assert on the request rather than only on the answer."""

    def __init__(self, counts=None, expectation=None, fail_indices=()):
        self.counts = counts if counts is not None else {"00": 30, "11": 34}
        self.expectation = expectation
        self.fail_indices = set(fail_indices)
        self.calls: list[dict] = []

    def _result(self):
        return {
            "counts": self.counts,
            "expectation": self.expectation,
            "error_info": {"method": "stub", "truncation_error": 0.0},
            "resources": {},
        }

    def run_one(self, qasm2, shots, engine, observable, params):
        self.calls.append(
            {"kind": "one", "n": 1, "shots": shots, "engine": engine,
             "observable": observable, "qasm": qasm2}
        )
        return {"status": "done", "result": self._result()}

    def run_batch(self, qasms, shots, engine, observable, params):
        self.calls.append(
            {"kind": "batch", "n": len(qasms), "shots": shots, "engine": engine,
             "observable": observable, "qasm": qasms}
        )
        return {
            "status": "done",
            "summary": {"total": len(qasms)},
            "results": [
                {"index": i, "status": "error", "result": None,
                 "error": "MemoryError: bond dimension exceeded available memory"}
                if i in self.fail_indices
                else {"index": i, "status": "done", "result": self._result()}
                for i in range(len(qasms))
            ],
        }


def make(stub=None, **kw) -> tuple[ZKSFDevice, StubService]:
    stub = stub or StubService()
    d = ZKSFDevice(wires=kw.pop("wires", 2), shots=kw.pop("shots", 64), token="stub", **kw)
    d._client = stub
    return d, stub


def counts_tape(shots=64, n=2):
    return QuantumScript(
        [qml.Hadamard(0), qml.CNOT([0, 1])], [qml.counts()], shots=shots
    )


# ------------------------------------------------------------------ basics

def test_constructing_a_device_emits_no_deprecation_warning():
    """PennyLane 0.40+ deprecated device-level shots. A warning on every
    construction is what makes a plugin feel unmaintained."""
    from pennylane.exceptions import PennyLaneDeprecationWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", PennyLaneDeprecationWarning)
        ZKSFDevice(wires=2, shots=100, token="x")


def test_registered_under_its_plugin_name():
    d = qml.device("zksf.simulator", wires=2, token="x")
    assert isinstance(d, ZKSFDevice)


def test_counts_pass_through():
    d, _ = make()
    assert d.execute(counts_tape()) == {"00": 30, "11": 34}


def test_probs_normalise():
    d, _ = make()
    tape = QuantumScript([qml.Hadamard(0), qml.CNOT([0, 1])], [qml.probs()], shots=64)
    probs = d.execute(tape)
    assert len(probs) == 4
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] == pytest.approx(30 / 64)
    assert probs[3] == pytest.approx(34 / 64)


def test_samples_expand_to_one_row_per_shot():
    d, _ = make()
    tape = QuantumScript([qml.Hadamard(0), qml.CNOT([0, 1])], [qml.sample()], shots=64)
    assert d.execute(tape).shape == (64, 2)


def test_error_info_is_available_after_a_run():
    """The reason to use this device over default.qubit."""
    d, _ = make()
    d.execute(counts_tape())
    assert d.error_info() == [{"method": "stub", "truncation_error": 0.0}]


# ----------------------------------------------------------------- batching

def test_many_tapes_become_one_request():
    """PennyLane hands execute() a sequence whenever it broadcasts. Sending
    those as N separate jobs would throw away the reason to batch."""
    d, stub = make()
    tapes = [counts_tape() for _ in range(5)]
    results = d.execute(tapes)

    assert len(results) == 5
    assert [c["kind"] for c in stub.calls] == ["batch"]
    assert stub.calls[0]["n"] == 5


def test_a_single_tape_uses_the_single_job_endpoint():
    d, stub = make()
    d.execute(counts_tape())
    assert [c["kind"] for c in stub.calls] == ["one"]


def test_a_failed_evaluation_is_reported_not_silently_dropped():
    """A sweep's cost depends on the entanglement each parameter value
    produces, so one binding can fail while the rest succeed."""
    d, _ = make(StubService(fail_indices={1}))
    with pytest.raises(RuntimeError, match="MemoryError"):
        d.execute([counts_tape(), counts_tape(), counts_tape()])


# ------------------------------------------------------------- observables

def test_pauli_string_uses_the_services_wire_order():
    """The service reads Pauli strings rightmost-character-is-wire-0. Getting
    this backwards returns a plausible but wrong number, which is the worst
    kind of bug for this product."""
    assert ZKSFDevice._pauli_terms(qml.PauliZ(0), n_wires=3) == [[1.0, "IIZ"]]
    assert ZKSFDevice._pauli_terms(qml.PauliX(2), n_wires=3) == [[1.0, "XII"]]
    assert ZKSFDevice._pauli_terms(qml.PauliZ(0) @ qml.PauliZ(1), n_wires=2) == [[1.0, "ZZ"]]


def test_expectation_is_returned_and_the_observable_is_sent():
    d, stub = make(StubService(expectation=1.0))
    tape = QuantumScript(
        [qml.Hadamard(0), qml.CNOT([0, 1])],
        [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))],
        shots=64,
    )
    assert d.execute(tape) == pytest.approx(1.0)
    assert stub.calls[0]["observable"] == [[1.0, "ZZ"]]


def test_an_unpinned_device_routes_expectations_to_an_expectation_engine():
    """qml.expval is how most PennyLane code is written, and the router's
    default returns counts, so leaving this alone would fail the common case."""
    d, stub = make(StubService(expectation=1.0))
    tape = QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))], shots=64)
    d.execute(tape)
    assert stub.calls[0]["engine"] == "pauli.cpu"


def test_expval_on_a_counts_only_engine_is_refused_not_faked():
    """An expectation of 0.0 is a plausible answer, so a fabricated one is
    indistinguishable from a real result."""
    d, _ = make(engine="exact.cpu")
    tape = QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))], shots=64)
    with pytest.raises(ValueError, match="not expectation values"):
        d.execute(tape)


def test_a_missing_expectation_raises_rather_than_returning_zero():
    d, _ = make(StubService(expectation=None), engine="pauli.cpu")
    tape = QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))], shots=64)
    with pytest.raises(RuntimeError, match="no expectation value"):
        d.execute(tape)


# --------------------------------------------------------------- refusals

def test_an_oversized_circuit_is_refused_before_it_is_billed():
    d, stub = make(wires=40, engine="exact.cpu")
    tape = QuantumScript([qml.Hadamard(i) for i in range(40)], [qml.counts()], shots=10)
    with pytest.raises(ValueError, match="up to 30 qubits"):
        d.execute(tape)
    assert stub.calls == [], "nothing was sent"


def test_an_unknown_engine_is_refused_at_construction():
    with pytest.raises(ValueError, match="unknown engine"):
        ZKSFDevice(wires=2, engine="no.such.engine", token="x")


def test_an_unsupported_measurement_says_what_does_work():
    d, _ = make()
    tape = QuantumScript([qml.Hadamard(0)], [qml.var(qml.PauliZ(0))], shots=64)
    with pytest.raises(NotImplementedError, match="qml.counts"):
        d.execute(tape)


def test_mismatched_shots_in_a_batch_are_refused():
    """One request carries one shot count; silently using the first tape's
    would return results nobody asked for."""
    d, _ = make()
    with pytest.raises(ValueError, match="one shot count"):
        d.execute([counts_tape(shots=64), counts_tape(shots=128)])


def test_derivatives_are_declined_rather_than_faked():
    """Each execution is a billed cloud job, so parameter-shift through it
    would spend the user's money without them seeing it."""
    assert ZKSFDevice(wires=2, token="x").supports_derivatives() is False
