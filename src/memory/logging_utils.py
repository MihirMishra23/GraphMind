from dataclasses import asdict, dataclass

from .memory_manager import MemoryDecision


@dataclass
class StepLog:
    episode_id: int
    step: int
    obs: str
    action: str
    reward: float
    done: bool
    kg_snapshot_step: int
    info: dict
    memory_decisions: list[MemoryDecision]


class TrajectoryLogger:
    def __init__(self):
        self.logs: list[StepLog] = []

    def log_step(self, step_log: StepLog) -> None:
        """Append a new step log entry."""
        self.logs.append(step_log)

    def to_json(self) -> list[dict]:
        """
        Serialize trajectory to JSON-friendly dicts.
        MemoryDecision objects are also expanded into dicts.
        """
        serialized: list[dict] = []
        for log in self.logs:
            entry = asdict(log)
            entry["memory_decisions"] = [asdict(d) for d in log.memory_decisions]
            serialized.append(entry)
        return serialized
