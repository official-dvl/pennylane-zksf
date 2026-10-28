"""PennyLane device for the ZKSF certified simulation service.

    import pennylane as qml
    dev = qml.device("zksf.simulator", wires=4, shots=1024, token=TOKEN)

    @qml.qnode(dev)
    def circuit(theta):
        qml.Hadamard(0)
        qml.RZ(theta, 0)
        qml.CNOT([0, 1])
        return qml.counts()

    circuit(0.7)
    dev.error_info()      # the accuracy statement for that run
"""
from pennylane_zksf._client import JobFailed, JobRejected, ServiceClient
from pennylane_zksf.device import ENGINE_LIMITS, ZKSFDevice

__version__ = "0.2.1"
__all__ = ["ZKSFDevice", "ENGINE_LIMITS", "ServiceClient", "JobFailed", "JobRejected"]
