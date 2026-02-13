from dataclasses import dataclass

from src.dataset.model import DatasetName
from src.task.model import TaskResult
from src.tokenizer import TokenizationStrategy


@dataclass
class StrategySummary:
    avg_score: float
    total_dollars: float
    delta: float | None = None


RunResultSummary = dict[TokenizationStrategy, StrategySummary]


@dataclass
class RunResult:
    dataset: DatasetName
    model: str
    n: int
    dollars: float
    summary: RunResultSummary
    results: list[TaskResult]
