from . import SentenceEvaluator
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score
import numpy as np
from typing import List
from ..readers import InputExample
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class BinaryClassificationPredictor(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param threshold: Threshold to identify a sentence pair as similar (default is 0.5, which works well in practice)
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    """

    def __init__(self, sentences1: List[str], sentences2: List[str], threshold = 0.5, name: str = '', batch_size: int = 32, show_progress_bar: bool = False, write_csv: bool = True):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.threshold = threshold

        assert len(self.sentences1) == len(self.sentences2)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = (name if name else "test") + "_results.csv"

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
        return cls(sentences1, sentences2, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        preds = self.get_preds(model)

        return preds


    def get_preds(self, model):
        sentences = list(set(self.sentences1 + self.sentences2))
        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentences1]
        embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)
        if self.threshold is None:
            threshold = np.mean(euclidean_distances)
        else:
            threshold = self.threshold
        print(f"Mean Euclidean distance: {threshold:.4f}")

        scores = euclidean_distances
        print(f"Using threshold for Euclidean distance: {threshold:.4f}")
        # convert scores to torch tensors
        t_scores = torch.tensor(scores)

        preds = []
        for i in range(len(t_scores)):
            pred = 1 if t_scores[i] < threshold else 0
            preds.append(pred)

        return preds

