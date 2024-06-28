from . import SentenceEvaluator
import logging
import os
import csv
import numpy as np
from typing import List
from ..readers import InputExample
import torch
import torch.nn.functional as F
from sentence_transformers.util import SiameseDistanceMetric, get_best_distance_threshold

logger = logging.getLogger(__name__)

class BinaryClassificationEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param distance_metric: Metric to compute the distance between two embeddings. Default is Euclidean
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param margin: Margin to be used in the contrastive loss
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    """

    def __init__(self, sentences1: List[str],
                 sentences2: List[str],
                 labels: List[int],
                 distance_metric: SiameseDistanceMetric = SiameseDistanceMetric.EUCLIDEAN,
                 name: str = '',
                 batch_size: int = 32,
                 margin = None,
                 show_progress_bar: bool = False,
                 write_csv: bool = True):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.distance_metric = distance_metric
        self.margin = margin

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = name + ".csv"
        self.csv_headers = ["epoch", "steps", "loss", "accuracy"]


    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        scores, preds_and_trues = self.compute_metrices(model)
        loss = scores['loss']
        acc = scores['accuracy']
        median_threshold = scores['median_threshold']

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, f"{loss:.4f}", f"{acc:.4f}"])
            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, f"{loss:.4f}", f"{acc:.4f}"])

        return acc, preds_and_trues, median_threshold


    def compute_metrices(self, model):
        sentences = list(set(self.sentences1 + self.sentences2))
        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentences1]
        embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        distances = self.distance_metric(embeddings1, embeddings2)

        labels = np.asarray(self.labels)
        output_scores = {}

        # calculate accuracy using best threshold
        best_distance_threshold = get_best_distance_threshold(distances, labels)
        predictions = [1 if distance < best_distance_threshold else 0 for distance in distances]
        correct_predictions = sum(pred == label for pred, label in zip(predictions, labels))
        acc_by_best = correct_predictions / len(labels)

        # calculate accuracy using median threshold
        if torch.is_tensor(distances) and distances.is_cuda:
            distances = distances.detach().cpu().numpy()
        median_distance_threshold = np.median(distances)
        predictions = [1 if distance < median_distance_threshold else 0 for distance in distances]
        correct_predictions = sum(pred == label for pred, label in zip(predictions, labels))
        acc_by_median = correct_predictions / len(labels)

        # convert labels and scores to torch tensors
        t_labels = torch.tensor(labels)
        t_scores = torch.tensor(distances)
        valid_losses = 0.5 * (t_labels.float() * t_scores.pow(2) + (1 - t_labels).float() * F.relu(self.margin - t_scores).pow(2))

        preds_and_trues = []
        for i in range(len(t_labels)):
            pred = 1 if t_scores[i] < best_distance_threshold else 0
            preds_and_trues.append((pred, t_labels[i].item()))

        print(f"{self.name} Loss  = {valid_losses.mean():.4f}   "
            f"{self.name} Accuracy  = {acc_by_best:.4f}    "
            f"(using best distance threshold   = {best_distance_threshold:.4f})")

        print(f"{self.name} Loss  = {valid_losses.mean():.4f}   "
            f"{self.name} Accuracy  = {acc_by_median:.4f}    "
            f"(using median distance threshold = {median_distance_threshold:.4f})")

        output_scores = {
            'loss': valid_losses.mean().item(),
            'accuracy' : acc_by_best,
            'best_threshold': best_distance_threshold,
            'median_threshold': median_distance_threshold
        }

        return output_scores, preds_and_trues
