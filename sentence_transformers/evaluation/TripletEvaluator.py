from . import SentenceEvaluator, SimilarityFunction
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from typing import List
from ..readers import InputExample
from torch.nn import functional as F
import torch
from sentence_transformers.util import SiameseDistanceMetric


logger = logging.getLogger(__name__)


class TripletEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
        Checks if distance(sentence, positive_example) < distance(sentence, negative_example).
    """

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
        distance_metric: SiameseDistanceMetric = SiameseDistanceMetric.EUCLIDEAN,
        margin: float = None,
        name: str = '',
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        :param anchors: Sentences to check similarity to. (e.g. a query)
        :param positives: List of positive sentences
        :param negatives: List of negative sentences
        :param distance_metric: One of 0 (Cosine), 1 (Euclidean) or 2 (Manhattan). Defaults to None, returning all 3.
        :param name: Name for the output
        :param batch_size: Batch size used to compute embeddings
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.distance_metric = distance_metric
        self.margin = margin
        self.name = name

        assert len(self.anchors) == len(self.positives)
        assert len(self.anchors) == len(self.negatives)

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = name + ".csv"
        self.csv_headers = ["epoch", "steps", "loss", "accuracy"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        anchors = []
        positives = []
        negatives = []

        for example in examples:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])
        return cls(anchors, positives, negatives, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        num_triplets = 0
        num_correct_triplets = 0

        embeddings_anchors = model.encode(
            self.anchors, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_positives = model.encode(
            self.positives, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_negatives = model.encode(
            self.negatives, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )

        pos_distances = self.distance_metric(embeddings_anchors, embeddings_positives)
        neg_distances = self.distance_metric(embeddings_anchors, embeddings_negatives)

        for idx in range(len(pos_distances)):
            num_triplets += 1

            if pos_distances[idx] < neg_distances[idx]:
                num_correct_triplets += 1

        acc = num_correct_triplets / num_triplets

        losses = F.relu(torch.tensor(pos_distances) - torch.tensor(neg_distances) + self.margin)
        loss = losses.mean().item()
        print(f"{self.name} Loss  = {loss:.4f}   {self.name} Accuracy  = {acc:.4f}")

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

        return acc
