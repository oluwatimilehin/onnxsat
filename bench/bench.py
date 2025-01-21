import argparse
import logging
import os
import time


from collections import namedtuple, defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

import onnx
import onnxruntime as ort

from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

from egglog import *
from model_optimizer import ModelOptimizer

logging.basicConfig(
    filename="saturator.log",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


Accuracy = namedtuple("Accuracy", "top_one top_five")
Latency = namedtuple("Latency", "mean median percentile_95")
Benchmark = namedtuple("Benchmark", "num_samples accuracy latency throughput")


class BenchMarkCategory(Enum):
    BASELINE = auto()
    ONNX_SAT = auto()
    ONNX_OPT = auto()
    ONNX_SAT_AND_OPT = auto()


category_to_display_name = {
    BenchMarkCategory.BASELINE: "Baseline - No Optimization",
    BenchMarkCategory.ONNX_SAT: "OnnxSAT",
    BenchMarkCategory.ONNX_OPT: "Onnx Optimizations",
    BenchMarkCategory.ONNX_SAT_AND_OPT: "OnnxSAT + OnnxOptimizations",
}


class BenchmarkTool:
    def __init__(self, data_dir: Path, results_dir: Path) -> None:
        self._data_dir = data_dir
        self._models_dir = f"{data_dir}/models"
        self._model_name_to_file = {
            f.stem: f for f in Path(self._models_dir).iterdir() if f.suffix == ".onnx"
        }

        self._eggs_dir = f"{data_dir}/eggs"
        self._results_dir = results_dir

        self._onnx_optimized_models_dir = f"{self._models_dir}/onnx-optimized"
        if not os.path.exists(self._onnx_optimized_models_dir):
            os.mkdir(self._onnx_optimized_models_dir)

        self._cifar_10_dataset = self._get_cifar_10_dataset()
        self._imagenet_dataset = self._get_imagenet_dataset()

    def _plot_grouped_bar_chart(
        self,
        models: List[str],
        experiment: int,
        metric: str,
        category_values: Dict[BenchMarkCategory, List[float]],
    ):
        x = np.arange(len(models))
        width = 0.25
        multiplier = 0

        _, ax = plt.subplots(figsize=(10, 6))
        for category, values in category_values.items():
            offset = width * multiplier
            rects = ax.bar(
                x + offset,
                [value - 1 for value in values],
                width,
                bottom=1,
                label=category_to_display_name[category],
            )
            ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.axhline(
            y=1.0,
            color="black",
            linestyle="--",
            linewidth=1,
            label=category_to_display_name[BenchMarkCategory.BASELINE],
        )

        if metric == "latency":
            y_label = "Median latency speedup w.r.t no optimization"
            title = f"Experiment {experiment}: Median Latency Speedup Per Model"
        else:
            y_label = "Throughput gain w.r.t no optimization"
            title = f"Experiment {experiment}: Throughput Gain Per Model"

        ax.set_ylabel(
            y_label,
            fontweight="bold",
        )
        ax.set_ylim(1.0)

        ax.set_title(
            title,
            fontweight="bold",
        )
        ax.set_xticks(x + width, models)
        ax.set_xlabel(
            "Models",
            fontweight="bold",
        )

        ax.legend(
            title="Optimization Passes",
            title_fontproperties={"weight": "bold"},
            loc="upper right",
            ncols=2,
        )

        plt.tight_layout()
        plt.savefig(
            f"{self._results_dir}/{experiment}/{metric}.png", bbox_inches="tight"
        )
        plt.close()

    def _visualize_latency_and_throughput(
        self,
        experiment: int,
        categories: List[BenchMarkCategory],
        results: Dict[str, Dict[BenchMarkCategory, Benchmark]],
    ):
        models = results.keys()

        categories = [
            category
            for category in categories
            if category != BenchMarkCategory.BASELINE
        ]
        median_latency_against_baseline = {category: [] for category in categories}
        throughput_against_baseline = {category: [] for category in categories}
        mean_latency_against_baseline = {category: [] for category in categories}

        for _, category_values in results.items():
            baseline_data = category_values.get(BenchMarkCategory.BASELINE)
            baseline_median_latency = baseline_data.latency.median
            baseline_mean_latency = baseline_data.latency.mean
            baseline_throughput = baseline_data.throughput

            for category in categories:
                category_data = category_values.get(category)
                median_latency_against_baseline[category].append(
                    round(baseline_median_latency / category_data.latency.median, 2)
                )
                mean_latency_against_baseline[category].append(
                    round(baseline_mean_latency / category_data.latency.mean, 2)
                )
                throughput_against_baseline[category].append(
                    round(category_data.throughput / baseline_throughput, 2)
                )

        self._plot_grouped_bar_chart(
            models, experiment, "latency", median_latency_against_baseline
        )
        self._plot_grouped_bar_chart(
            models, experiment, "throughput", throughput_against_baseline
        )

    def apply_onnx_opt_offline(
        self, model_name: str, model_file: str, stored_file_path: str
    ):
        if not os.path.exists(stored_file_path):
            # Store optimized model offline so optimizations are not applied when measuring inference
            loader, num_labels = self._get_loader_and_num_labels(
                model_name, num_samples=5
            )

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.optimized_model_filepath = stored_file_path
            session = ort.InferenceSession(model_file, sess_options=sess_options)
            self._measure(session, loader, num_labels)

    def experiment_one(self) -> None:
        results: Dict[str, Dict[BenchMarkCategory, Benchmark]] = {}
        results_dir = f"{self._results_dir}/1"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for model_name, orig_model in self._model_name_to_file.items():
            print(f"Processing model: {model_name}")

            onnx_optimized_model = (
                f"{self._onnx_optimized_models_dir}/{model_name}.onnx"
            )
            self.apply_onnx_opt_offline(model_name, orig_model, onnx_optimized_model)

            new_model = None
            with open(orig_model, "rb"):
                model_optimizer = ModelOptimizer(
                    orig_model, self._data_dir, results_dir
                )
                new_model = model_optimizer.run()

            eqsat_and_onnx_optimized_model = (
                f"{self._onnx_optimized_models_dir}/{model_name}-eqsat.onnx"
            )
            self.apply_onnx_opt_offline(
                model_name, new_model, eqsat_and_onnx_optimized_model
            )

            category_to_model = {
                BenchMarkCategory.ONNX_SAT: new_model,
                BenchMarkCategory.ONNX_OPT: onnx_optimized_model,
                BenchMarkCategory.ONNX_SAT_AND_OPT: eqsat_and_onnx_optimized_model,
                BenchMarkCategory.BASELINE: orig_model,
            }

            category_results = {}
            warmup_loader, num_labels = self._get_loader_and_num_labels(
                model_name, num_samples=1000
            )
            loader, num_labels = self._get_loader_and_num_labels(
                model_name, num_samples=2000
            )

            for category, model in category_to_model.items():
                print(f"Running category: {category}")
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                )
                session = ort.InferenceSession(model, sess_options=sess_options)

                print(f"Warming up")
                self._measure(session, warmup_loader, num_labels)

                print(f"Actual measurement")
                category_results[category] = self._measure(
                    session, loader, num_labels, num_runs=3
                )

            results[model_name] = category_results

        with open(f"{results_dir}/results.txt", "w+") as f:
            f.write(f"Results: {results}")

        self._visualize_latency_and_throughput(1, category_to_model.keys(), results)

    def _visualize_accuracy_drop(
        self, method: str, top: int, results: Dict[str, Dict[float, float]]
    ) -> None:
        # Generate dynamic colors and markers
        num_models = len(results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, num_models))
        markers = ["o", "s", "^", "D", "P", "X", "*"]

        for i, (model, result_vals) in enumerate(results.items()):
            plt.plot(
                [0] + list(result_vals.keys()),
                [0] + list(result_vals.values()),
                linestyle="--",
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                label=model,
            )

        plt.xlabel("Model Sparsity (%)")
        plt.ylabel(f"Top-{top} Accuracy Drop (%)")
        plt.title(f"Top-{top} Accuracy Drop vs Model Sparsity ({method} Method)")

        plt.axhline(-5, color="red", linewidth=1, linestyle="--")
        plt.xticks(range(0, 90, 10))

        plt.xlim(left=0)

        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, linestyle="-", alpha=0.5)

        plt.minorticks_on()
        plt.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.2)

        plt.tick_params(which="minor", length=0)
        plt.tick_params(axis="both", which="major", direction="in")

        plt.savefig(f"{self._results_dir}/2/{method.lower()}_{top}_accuracy_loss.png")
        plt.close()

    def experiment_two(self) -> None:
        sparsity_ratios = [ratio.item() for ratio in np.arange(0.1, 0.9, 0.05)]

        top_one_accuracy_loss_per_ratio: Dict[str, Dict[float, float]] = defaultdict(
            dict
        )
        top_five_accuracy_loss_per_ratio: Dict[str, Dict[float, float]] = defaultdict(
            dict
        )

        results_dir = f"{self._results_dir}/2"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for model_name, orig_model in self._model_name_to_file.items():
            print(f"Processing model: {model_name}")

            loader, num_labels = self._get_loader_and_num_labels(
                model_name, num_samples=5000
            )

            # Determine baseline values
            session = ort.InferenceSession(orig_model)
            baseline_accuracy = self._measure(session, loader, num_labels).accuracy
            base_top_one = baseline_accuracy.top_one
            base_top_five = baseline_accuracy.top_five

            print(f"Baseline accuracy: {baseline_accuracy}")

            for ratio in sparsity_ratios:
                new_model = None
                with open(orig_model, "rb"):
                    model_optimizer = ModelOptimizer(
                        orig_model, self._data_dir, results_dir
                    )
                    new_model = model_optimizer.run(
                        sparsity_ratio=ratio, prune_only=True
                    )
                session = ort.InferenceSession(new_model)
                accuracy = self._measure(session, loader, num_labels).accuracy

                print(f"Accuracy at sparsity {ratio} is {accuracy}")
                top_one_accuracy_loss_per_ratio[model_name][round(ratio * 100)] = (
                    np.round(accuracy.top_one - base_top_one, 2) * 100
                ).item()
                top_five_accuracy_loss_per_ratio[model_name][round(ratio * 100)] = (
                    np.round(accuracy.top_five - base_top_five, 2) * 100
                ).item()

                if round(accuracy.top_five, 2) == 0:
                    break

        with open(f"{results_dir}/relative_results.txt", "w+") as f:
            f.write(f"Top 5 loss results: {top_five_accuracy_loss_per_ratio}")
            f.write(f"Top 1 loss results: {top_one_accuracy_loss_per_ratio}")

        self._visualize_accuracy_drop("Relative", 5, top_five_accuracy_loss_per_ratio)
        self._visualize_accuracy_drop("Relative", 1, top_one_accuracy_loss_per_ratio)

    def experiment_three(self) -> None:
        results: Dict[str, Dict[BenchMarkCategory, Benchmark]] = {}
        results_dir = f"{self._results_dir}/3"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Max sparsity ratio using the results in experiment 2 that keeps the results within the 5% accuracy drop threshold
        max_sparsity_ratio_per_model = {
            "simple_classifier": 0.74,
            "resnet18": 0.32,
            "mobilenetv2": 0.14,
            "mobilenetv4": 0.01,
            "shufflenet-v2": 0.28,
        }

        for model_name, orig_model in self._model_name_to_file.items():
            sparsity_ratio = max_sparsity_ratio_per_model[model_name]
            print(f"Processing model: {model_name} at {sparsity_ratio * 100}% sparsity")

            onnx_optimized_model = (
                f"{self._onnx_optimized_models_dir}/{model_name}.onnx"
            )
            self.apply_onnx_opt_offline(model_name, orig_model, onnx_optimized_model)

            new_model = None
            with open(orig_model, "rb"):
                model_optimizer = ModelOptimizer(
                    orig_model, self._data_dir, results_dir
                )
                new_model = model_optimizer.run(sparsity_ratio=sparsity_ratio)

            category_to_model = {
                BenchMarkCategory.ONNX_SAT: new_model,
                BenchMarkCategory.ONNX_OPT: onnx_optimized_model,
                BenchMarkCategory.BASELINE: orig_model,
            }

            category_results = {}
            warmup_loader, num_labels = self._get_loader_and_num_labels(
                model_name, num_samples=1000
            )
            loader, num_labels = self._get_loader_and_num_labels(
                model_name, num_samples=2000
            )

            for category, model in category_to_model.items():
                print(f"Running category: {category}")
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                )
                session = ort.InferenceSession(model, sess_options=sess_options)

                print(f"Warming up")
                self._measure(session, warmup_loader, num_labels)

                print(f"Actual measurement")
                category_results[category] = self._measure(
                    session, loader, num_labels, num_runs=3
                )

            results[f"{model_name}\n ({round(sparsity_ratio*100)}% sparsity)"] = (
                category_results
            )

        with open(f"{results_dir}/results.txt", "w+") as f:
            f.write(f"Results: {results}")

        self._visualize_latency_and_throughput(3, category_to_model.keys(), results)

    def _get_loader_and_num_labels(
        self, model_name: str, num_samples: int
    ) -> Tuple[DataLoader, int]:
        if model_name.startswith("simple_classifier"):
            subset_dataset = Subset(self._cifar_10_dataset, list(range(num_samples)))
            return DataLoader(dataset=subset_dataset, batch_size=1, shuffle=False), 10

        subset_dataset = Subset(self._imagenet_dataset, list(range(num_samples)))
        return DataLoader(subset_dataset, batch_size=1, shuffle=False), 1000

    def _measure(
        self,
        session: ort.InferenceSession,
        dataloader: DataLoader,
        num_labels=1000,
        num_runs=1,
    ) -> Benchmark:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        all_preds = []
        all_labels = []

        latencies = []

        overall_start_time = time.perf_counter()

        for i in range(num_runs):
            print(f"Run {i + 1}")
            run_latencies = []
            for inputs, labels in tqdm(
                dataloader, desc="Inference Progress", unit="batch"
            ):
                inputs_numpy = inputs.numpy()
                labels_numpy = labels.numpy()

                # Run inference
                start_time = time.perf_counter()
                outputs = session.run([output_name], {input_name: inputs_numpy})
                end_time = time.perf_counter()
                run_latencies.append(end_time - start_time)

                all_preds.append(outputs[0])
                all_labels.append(labels_numpy)

            if num_runs > 1:
                print(
                    f"Median latency at the end of run{i+1} is {np.median(run_latencies):.4f}"
                )
            latencies.extend(run_latencies)

        overall_end_time = time.perf_counter()
        throughput = len(latencies) / (overall_end_time - overall_start_time)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        top1_accuracy = top_k_accuracy_score(
            all_labels, all_preds, k=1, labels=np.arange(num_labels)
        )
        top5_accuracy = top_k_accuracy_score(
            all_labels, all_preds, k=5, labels=np.arange(num_labels)
        )

        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"Top-5 Accuracy: {top5_accuracy:.4f}")

        accuracy = Accuracy(top1_accuracy, top5_accuracy)

        mean_latency = np.mean(latencies)
        median = np.median(latencies)
        percentile_95 = np.percentile(latencies, 95)

        print(f"Overall Mean Latency: {mean_latency:.4f}")
        print(f"Overall Median Latency: {median:.4f}")

        latency = Latency(mean_latency, median, percentile_95)

        return Benchmark(len(latencies), accuracy, latency, throughput)

    def _get_cifar_10_dataset(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        return datasets.CIFAR10(
            self._data_dir, train=False, download=True, transform=transform
        )

    def _get_imagenet_dataset(self):
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return datasets.ImageNet(self._data_dir, split="val", transform=transform)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_dir = Path(f"{current_dir.parent}/data")

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=int)
    parser.add_argument("-d", "--dir", default=int(time.time()))

    args = parser.parse_args()

    base_results_dir = f"{current_dir}/results/{args.dir}"

    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)

    original_models_path = f"{data_dir}/models"
    original_eggs_path = f"{data_dir}/eggs"

    model_files = [
        f for f in Path(original_models_path).iterdir() if f.suffix == ".onnx"
    ]

    benchmarker = BenchmarkTool(data_dir, base_results_dir)

    experiment = args.experiment
    match experiment:
        case 1:
            print(f"Running Experiment 1")
            benchmarker.experiment_one()
        case 2:
            print(f"Running Experiment 2")
            benchmarker.experiment_two()
        case 3:
            print(f"Running Experiment 3")
            benchmarker.experiment_three()
        case _:
            raise ValueError(f"Unknown experiment specified")
