"""A PennyLane device backed by the ZKSF simulation service.

The point of running here rather than locally is reach and evidence: engines
that go past the ~30-qubit statevector wall, and an error bound reported with
every approximate result. `device.error_info()` returns the bound for the last
execution, because a number with no statement of its accuracy is the thing this
service exists to replace.

Deliberately not differentiable. Each execution is a billed cloud job, so
parameter-shift differentiation through it would be slow and expensive, and
PennyLane would issue those shifted tapes without the user seeing the cost.
`supports_derivatives()` returns False and PennyLane falls back to computing
gradients over whole executions, which is honest about what it costs.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    ProbabilityMP,
    SampleMP,
)
from pennylane.tape import QuantumScript

from pennylane_zksf._client import ServiceClient

__all__ = ["ZKSFDevice"]

#: engine name -> the qubit ceiling the service enforces. Mirrored here so an
#: oversized circuit is refused before it is billed, not after a round trip.
ENGINE_LIMITS = {
    None: None,                 # router decides
    "exact.cpu": 30,
    "exact.gpu": 32,
    "clifford": 5000,
    "mps.quimb.cpu": 128,
    "mps.aer.cpu": 128,
    "pauli.cpu": 200,
    "noisy.cpu": 20,
}


class ZKSFDevice(Device):
    """PennyLane device that executes on the ZKSF service.

        import pennylane as qml
        dev = qml.device("zksf.simulator", wires=4, shots=1024, token=TOKEN)

    A batch of tapes goes in one request. PennyLane already hands `execute` a
    sequence whenever it broadcasts or differentiates, so the sweep endpoint is
    the natural fit rather than an extra feature: N tapes become one job.
    """

    name = "zksf.simulator"

    def __init__(
        self,
        wires=None,
        shots=1024,
        *,
        engine: str | None = None,
        token: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
        **engine_params: Any,
    ):
        # Shots are kept here rather than handed to Device: PennyLane 0.40+
        # deprecated device-level shots in favour of the tape's own, and passing
        # them up emits a warning on every construction. A tape that names its
        # shots wins; this is only the fallback when it does not.
        super().__init__(wires=wires)
        self._default_shots = int(shots) if shots else 1024
        if engine is not None and engine not in ENGINE_LIMITS:
            raise ValueError(
                f"unknown engine {engine!r}; choose from "
                f"{sorted(k for k in ENGINE_LIMITS if k)} or leave it unset to let "
                f"the router pick the cheapest that fits"
            )
        self._engine = engine
        self._params = engine_params
        self._client = ServiceClient(base_url=base_url, token=token, timeout=timeout)
        self._last_error_info: list[dict | None] = []

    # ----------------------------------------------------------- capabilities

    def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
        """False, and deliberately. See the module docstring."""
        return False

    def error_info(self) -> list[dict | None]:
        """The error bound reported for each tape of the last execution.

        None where the engine is exact and there is nothing to bound. This is
        the reason to use this device over a local simulator, so it is a first-
        class accessor rather than something buried in a result object."""
        return list(self._last_error_info)

    # -------------------------------------------------------------- execution

    def execute(
        self,
        circuits: QuantumScript | Sequence[QuantumScript],
        execution_config: ExecutionConfig | None = None,
    ):
        single = isinstance(circuits, QuantumScript)
        tapes: list[QuantumScript] = [circuits] if single else list(circuits)
        for tape in tapes:
            self._check(tape)

        shots = self._shots_for(tapes)
        qasms = [qml.to_openqasm(t) for t in tapes]
        observable = self._observable_for(tapes)
        engine = self._engine_for(tapes)

        if len(qasms) == 1:
            job = self._client.run_one(
                qasms[0], shots=shots, engine=engine,
                observable=observable, params=self._params,
            )
            items = [{"status": "done", "result": job["result"]}]
        else:
            job = self._client.run_batch(
                qasms, shots=shots, engine=engine,
                observable=observable, params=self._params,
            )
            items = job["results"]

        self._last_error_info = [
            ((it.get("result") or {}).get("error_info") if it["status"] == "done" else None)
            for it in items
        ]

        results = [self._to_pennylane(tape, item) for tape, item in zip(tapes, items)]
        return results[0] if single else tuple(results)

    # ---------------------------------------------------------------- helpers

    def _check(self, tape: QuantumScript) -> None:
        limit = ENGINE_LIMITS.get(self._engine)
        n = len(tape.wires)
        if limit is not None and n > limit:
            raise ValueError(
                f"engine {self._engine!r} runs up to {limit} qubits, this circuit "
                f"needs {n}. Pick a larger engine or leave engine unset to let the "
                f"router choose."
            )
        for mp in tape.measurements:
            if not isinstance(mp, (CountsMP, SampleMP, ProbabilityMP, ExpectationMP)):
                raise NotImplementedError(
                    f"{type(mp).__name__} is not supported by this device. It "
                    f"returns measurement counts, so qml.counts, qml.sample, "
                    f"qml.probs and qml.expval of a Pauli observable work; "
                    f"state-vector and variance measurements do not."
                )
        if len(tape.measurements) != 1:
            raise NotImplementedError(
                f"this device returns one measurement per circuit, got "
                f"{len(tape.measurements)}. Split them into separate circuits; "
                f"they will still travel in a single batched request."
            )

    #: Engines that return an expectation value. The statevector and stabilizer
    #: engines return counts only, so an expval tape sent to them comes back
    #: with expectation=None. Verified against the service, not assumed.
    EXPECTATION_ENGINES = {"pauli.cpu"}

    def _engine_for(self, tapes: list[QuantumScript]) -> str | None:
        """Route an expectation tape to an engine that computes one.

        `qml.expval` is how most PennyLane code is written, and the router's
        default choice returns counts rather than an expectation, so leaving
        this alone would make the common case fail. Only applied when no engine
        was pinned: a pinned engine is a decision, and overriding it silently
        would be worse than the error."""
        if not any(isinstance(t.measurements[0], ExpectationMP) for t in tapes):
            return self._engine
        if self._engine is None:
            return "pauli.cpu"
        if self._engine not in self.EXPECTATION_ENGINES:
            raise ValueError(
                f"engine {self._engine!r} returns measurement counts, not expectation "
                f"values, so qml.expval has nothing to read. Use "
                f"{sorted(self.EXPECTATION_ENGINES)} for expectation values, or leave "
                f"engine unset and one will be chosen."
            )
        return self._engine

    def _shots_for(self, tapes: list[QuantumScript]) -> int:
        """One shot count per request, so tapes in a batch must agree."""
        wanted = {t.shots.total_shots for t in tapes if t.shots.total_shots}
        if len(wanted) > 1:
            raise ValueError(
                f"a batch shares one shot count, got {sorted(wanted)}. Execute the "
                f"differing circuits separately."
            )
        return int(wanted.pop()) if wanted else self._default_shots

    def _observable_for(self, tapes: list[QuantumScript]) -> list | None:
        """Expectation tapes carry an observable; the service needs it in
        [coefficient, pauli_string] form. All tapes in a batch share one."""
        obs = [t.measurements[0].obs for t in tapes if isinstance(t.measurements[0], ExpectationMP)]
        if not obs:
            return None
        if len(obs) != len(tapes):
            raise NotImplementedError(
                "a batch mixes expectation values with sampled measurements; "
                "execute them separately"
            )
        terms = [self._pauli_terms(o, len(t.wires)) for o, t in zip(obs, tapes)]
        first = terms[0]
        if any(t != first for t in terms):
            raise NotImplementedError(
                "a batch shares one observable, but these tapes ask for different "
                "ones. Execute them separately; the parameter sweep case, where "
                "the observable is fixed and the circuit varies, is the supported one."
            )
        return first

    @staticmethod
    def _pauli_terms(obs, n_wires: int) -> list:
        """PennyLane observable -> [[coeff, "IIZZ"], ...].

        The service reads Pauli strings in Qiskit order, rightmost character is
        wire 0, so the string is built in reverse."""
        try:
            sentence = qml.pauli.pauli_sentence(obs)
        except Exception as exc:
            raise NotImplementedError(
                f"observable {obs} could not be written as a sum of Pauli words, "
                f"which is the form the service accepts"
            ) from exc
        out = []
        for word, coeff in sentence.items():
            chars = ["I"] * n_wires
            for wire, letter in word.items():
                chars[int(wire)] = letter
            out.append([float(np.real(coeff)), "".join(reversed(chars))])
        return out

    def _to_pennylane(self, tape: QuantumScript, item: dict):
        """Map one service result onto what the tape asked for."""
        if item["status"] != "done":
            raise RuntimeError(
                f"evaluation {item.get('index')} returned no result "
                f"({item.get('reason') or item.get('error')})"
            )
        result = item["result"] or {}
        mp = tape.measurements[0]

        if isinstance(mp, ExpectationMP):
            value = result.get("expectation")
            if value is None:
                # Never substitute a number here. An expectation of 0.0 is a
                # perfectly plausible answer, so a fabricated one is
                # indistinguishable from a real result and would be trusted.
                raise RuntimeError(
                    f"the run returned no expectation value (engine "
                    f"{result.get('error_info', {}).get('method') or self._engine!r}). "
                    f"Expectation values come from "
                    f"{sorted(self.EXPECTATION_ENGINES)}."
                )
            return float(value)

        counts: dict[str, int] = result.get("counts") or {}
        if isinstance(mp, CountsMP):
            return counts
        if isinstance(mp, ProbabilityMP):
            total = sum(counts.values()) or 1
            probs = np.zeros(2 ** len(tape.wires))
            for bitstring, n in counts.items():
                probs[int(bitstring, 2)] = n / total
            return probs
        # SampleMP: expand counts back into individual shots, in a fixed order
        # so a run is reproducible from its counts.
        samples = []
        for bitstring, n in sorted(counts.items()):
            row = [int(b) for b in bitstring]
            samples.extend([row] * n)
        return np.array(samples)
