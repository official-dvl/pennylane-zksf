# pennylane-zksf

A PennyLane device for the [ZKSF](https://zksf.org) quantum simulation service.

Run PennyLane circuits past the ~30-qubit statevector wall, and get an accuracy
statement with every approximate result rather than a bare number.

```bash
pip install pennylane-zksf
```

```python
import pennylane as qml

dev = qml.device("zksf.simulator", wires=3, shots=1024, token=YOUR_TOKEN)

@qml.qnode(dev)
def ghz():
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    qml.CNOT([1, 2])
    return qml.counts()

ghz()              # {'000': 512, '111': 512}
dev.error_info()   # how far that result can be from the truth
```

Your token is on the dashboard at [app.zksf.org](https://app.zksf.org). It can
also come from the `ZKSF_TOKEN` environment variable.

## Why run here

`default.qubit` is faster for small circuits and costs nothing, so use it. This
device is for the two things it cannot do:

- **Reach.** Stabilizer circuits at thousands of qubits, tensor networks past
  100, Pauli propagation for expectation values at ~200. An exact statevector
  stops around 30 because the memory doubles per qubit.
- **Evidence.** Every approximate run reports a measured error bound under the
  open [ZCC-v0.1](https://zksf.org/quantum-computing-certification/) protocol,
  and any finished job can be exported as a certificate that anyone can check
  with [`zcc-verify`](https://pypi.org/project/zcc-verify/) without an account
  and without calling this service.

## Engines

Leave `engine` unset and a rule-based router picks the cheapest one that fits.
Name it to pin the choice:

| engine | qubits | returns |
|---|---|---|
| `exact.cpu` | 30 | counts, exact |
| `exact.gpu` | 32 | counts, exact |
| `clifford` | 5000 | counts, exact (Clifford gates only) |
| `mps.quimb.cpu` | 128 | counts, with a certified error bound |
| `mps.aer.cpu` | 128 | counts, with a certified error bound |
| `pauli.cpu` | 200 | expectation values only |
| `noisy.cpu` | 20 | counts, with device-noise modelling |

An oversized circuit is refused before it is submitted, not after it is billed.

## Batching

PennyLane hands `execute` a sequence of tapes whenever it broadcasts or
differentiates. Those go to the service as **one request**, not one per tape:

```python
from pennylane.tape import QuantumScript

tapes = [QuantumScript(ops(v), [qml.counts()], shots=1024) for v in values]
dev.execute(tapes)      # one round trip
```

This matters more than it looks. A round trip to the service is on the order of
a second, so a sweep issued one point at a time spends most of its wall clock
waiting rather than simulating.

## Measurements

`qml.counts`, `qml.sample`, `qml.probs` and `qml.expval` of a Pauli observable.
One measurement per circuit.

`qml.expval` needs an engine that computes expectation values, so an unpinned
device routes those runs to `pauli.cpu`. Pinning a counts-only engine and asking
for an expectation raises rather than returning a plausible-looking zero.

## No gradients

`supports_derivatives()` returns `False`, deliberately. Each execution is a
billed cloud job, so differentiating through it would issue shifted tapes and
spend money without the cost being visible. PennyLane falls back to its own
gradient handling. Compute gradients on a local device and use this one for the
runs that need the reach or the bound.

## Links

- Service and documentation: [zksf.org](https://zksf.org) · [docs](https://zksf.org/docs/)
- Certification protocol: [zksf.org/quantum-computing-certification](https://zksf.org/quantum-computing-certification/)
- Specification paper: [doi.org/10.5281/zenodo.21851381](https://doi.org/10.5281/zenodo.21851381)
- Python client: [`qsim-sdk`](https://pypi.org/project/qsim-sdk/) · Qiskit provider: [`qiskit-zksf`](https://pypi.org/project/qiskit-zksf/) · Certificate checker: [`zcc-verify`](https://pypi.org/project/zcc-verify/)

## Licence

MIT
