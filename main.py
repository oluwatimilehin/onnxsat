import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


from egglog import *
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

from eggie.rewrites import *
from model_optimizer import ModelOptimizer

logging.basicConfig(
    filename="saturator.log",
    level=logging.ERROR,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def run_inference_test(model: Path, dataloader: DataLoader, num_labels=1000):
    session = ort.InferenceSession(model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Inference Progress", unit="batch"):
        inputs_numpy = inputs.numpy()
        labels_numpy = labels.numpy()

        # Run inference
        outputs = session.run([output_name], {input_name: inputs_numpy})
        all_preds.append(outputs[0])

        all_labels.append(labels_numpy)

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


def get_cifar_10_loader(data_dir: str, subset=10000) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    testset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform
    )
    subset_indices = list(range(subset))
    subset_dataset = Subset(testset, subset_indices)
    return DataLoader(dataset=subset_dataset, batch_size=1, shuffle=False)


def get_imagenet_loader(data_dir: str, subset=10000) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_dataset = datasets.ImageNet(data_dir, split="val", transform=transform)
    subset_indices = list(range(subset))
    subset_dataset = Subset(val_dataset, subset_indices)

    return DataLoader(subset_dataset, batch_size=1, shuffle=False)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_dir = f"{current_dir}/data"

    models_path = f"{data_dir}/models"
    eggs_path = f"{data_dir}/eggs"
    converted_path = f"{data_dir}/converted"

    model_files = [f for f in Path(models_path).iterdir() if f.suffix == ".onnx"]

    num_inference_samples = 10
    imagenet_loader = get_imagenet_loader(data_dir, subset=num_inference_samples)
    cifar_10_loader = get_cifar_10_loader(data_dir, subset=num_inference_samples)

    for orig_model_file in model_files:
        model_name = Path(orig_model_file).stem
        print(f"Processing model: {model_name}")

        model = onnx.load(orig_model_file)
        # Save the updated model
        onnx.save(model, orig_model_file)

        with open(orig_model_file, "rb") as of:
            model_optimizer = ModelOptimizer(orig_model_file, data_dir, converted_path)
            updated_model_file = model_optimizer.run()
            data_loader = imagenet_loader
            num_labels = 1000

            if model_name.startswith("simple"):
                data_loader = cifar_10_loader
                num_labels = 10

            run_inference_test(orig_model_file, data_loader, num_labels)
            run_inference_test(updated_model_file, data_loader, num_labels)
