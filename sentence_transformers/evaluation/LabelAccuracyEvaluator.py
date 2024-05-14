from . import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from ..util import batch_to_device
import os
import csv


logger = logging.getLogger(__name__)

class LabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model = None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        self.write_csv = write_csv
        self.csv_file: str = (name if name else "validation") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "loss", "accuracy"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0
        loss = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        self.softmax_model.to(model.device)
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            loss += torch.nn.functional.cross_entropy(prediction, label_ids).item()
            total += prediction.size(0)
            predicted_ids = torch.argmax(prediction, dim=1)
            correct += predicted_ids.eq(label_ids).sum().item()

            # write predictions and labels to csv
            if self.name == "test" and output_path is not None and self.write_csv:
                csv_path = os.path.join(output_path, self.name + "_preds.csv")
                if not os.path.isfile(csv_path):
                    with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["prediction", "label"])
                        for i in range(len(label_ids)):
                            writer.writerow([predicted_ids[i].item(), label_ids[i].item()])
                else:
                    with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        for i in range(len(label_ids)):
                            writer.writerow([predicted_ids[i].item(), label_ids[i].item()])

        epoch_accuracy = round(correct/total, 4)
        epoch_loss = round(loss/total, 4)

        print("Valid Loss = {:.2f}   Valid Accuracy = {:.2f}".format(epoch_loss, epoch_accuracy*100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, epoch_loss, epoch_accuracy])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, epoch_loss, epoch_accuracy])

        return epoch_accuracy
