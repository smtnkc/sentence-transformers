import csv
import logging
import os
from typing import TYPE_CHECKING, Dict

import torch
from torch.utils.data import DataLoader

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import batch_to_device

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class LabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model=None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset

        Args:
            dataloader (DataLoader): the data for the evaluation
        """
        super().__init__()
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model
        self.write_csv = write_csv
        self.csv_file = name + ".csv"
        self.csv_headers = ["epoch", "steps", "loss", "accuracy"]
        self.primary_metric = "accuracy"
        self.best_accuracy = 0.0

        if self.softmax_model:
            self.softmax_model.set_evaluator(self)

    def __call__(
        self, model: "SentenceTransformer", output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> Dict[str, float]:
        model.eval()
        total_samples = 0
        total_acc = 0
        total_loss = 0.0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the " + self.name + " dataset" + out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate

        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                loss, accuracy = self.softmax_model(features, label_ids)
                #_, output = self.softmax_model(features, labels=None)

            batch_size = len(label_ids)
            total_samples += batch_size
            total_acc += accuracy * batch_size
            total_loss += loss.item() * batch_size

        # avg_accuracy = correct / num_samples
        avg_accuracy = total_acc / total_samples
        avg_loss = total_loss / total_samples

        print(f"{self.name} Loss  = {avg_loss:.4f}   {self.name} Accuracy  = {avg_accuracy:.4f}")

        # save state_dict
        if self.name == "Zero" and output_path is not None:
            softmax_model_dir = os.path.join(os.path.dirname(output_path), "softmax_model_checkpoints")
            os.makedirs(softmax_model_dir, exist_ok=True)

            # Save the model after each epoch
            # torch.save(self.softmax_model.state_dict(), os.path.join(softmax_model_dir, f"epoch_{epoch}.pt"))

            # Save the best softmax model considering the zero-shot accuracy
            if avg_accuracy > self.best_accuracy:
                torch.save(self.softmax_model.state_dict(), os.path.join(softmax_model_dir, "best.pt"))
                self.best_accuracy = avg_accuracy

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, f"{loss:.4f}", f"{avg_accuracy:.4f}"])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, f"{loss:.4f}", f"{avg_accuracy:.4f}"])


        return avg_accuracy