import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from src.dataset.index import DatasetLoader
from src.dataset.model import DATASET_NAMES, DatasetName
from src.run.model import RunResult, RunResultSummary, StrategySummary
from src.task.index import TaskRunner
from src.task.model import Task, TaskResult
from src.tokenizer import TOKENIZATION_STRATEGIES

RESULT_DIR = Path("data/results")


class Runner:
    dataset_loader = DatasetLoader()
    task_runner = TaskRunner()

    def calculate_summary(self, task_results: list[TaskResult]):
        by_strategy = {
            s: [r for r in task_results if r.tokenization_strategy == s]
            for s in TOKENIZATION_STRATEGIES
        }

        scores = [r.evaluation for r in by_strategy["baseline"]]
        baseline_avg = sum(scores) / len(scores)

        summary: RunResultSummary = {}

        for strategy in TOKENIZATION_STRATEGIES:
            results = by_strategy[strategy]
            scores = [r.evaluation for r in results]
            dollars_list = [r.dollars for r in results]
            avg = sum(scores) / len(scores)

            summary[strategy] = StrategySummary(
                avg_score=avg,
                total_dollars=sum(dollars_list),
                delta=avg - baseline_avg if strategy != "baseline" else None,
            )

        return summary

    def run(self, dataset_name: DatasetName, model: str, n: int):
        print(f"Running {dataset_name} with {model} for n={n}...")
        tasks: list[Task] = []
        for i, task in enumerate(self.dataset_loader.load_tasks(dataset_name)):
            if i == n:
                break
            tasks.append(task)

        with ThreadPoolExecutor(max_workers=5) as executor:
            all_results = list(
                executor.map(lambda t: self.task_runner.run(model=model, task=t), tasks)
            )
        task_results = [r for rs in all_results for r in rs]

        run_result = RunResult(
            summary=self.calculate_summary(task_results),
            dataset=dataset_name,
            model=model,
            n=n,
            dollars=sum(r.dollars for r in task_results),
            results=task_results,
        )

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        with open(
            RESULT_DIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(asdict(run_result), f, indent=4, ensure_ascii=False)

    def run_batch(self, dataset_names: list[DatasetName], models: list[str], n: int):
        for model in models:
            for dataset_name in dataset_names:
                try:
                    self.run(dataset_name=dataset_name, model=model, n=n)
                except Exception as e:
                    print(f"Error running {dataset_name} with {model}: {e}")


if __name__ == "__main__":
    runner = Runner()
    runner.run_batch(
        dataset_names=DATASET_NAMES,
        models=["mistralai/ministral-3b-2512"],
        n=50,
    )
