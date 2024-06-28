from typing import Iterable, Dict
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import SiameseDistanceMetric, get_best_distance_threshold


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss. Similar to ConstrativeLoss, but it selects hard positive (positives that are far apart)
    and hard negative pairs (negatives that are close) and computes the loss only for these pairs. Often yields
    better performances than  ConstrativeLoss.

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    Example::

        from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
        from torch.utils.data import DataLoader

        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.OnlineContrastiveLoss(model=model)

        model.fit([(train_dataloader, train_loss)], show_progress_bar=True)
    """

    def __init__(self, model: SentenceTransformer,
                 distance_metric=SiameseDistanceMetric.EUCLIDEAN,
                 margin: float = None,
                 size_average: bool = True):
        super(OnlineContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model
        self.size_average = size_average

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "SiameseDistanceMetric.{}".format(name)
                break

        return {'distance_metric': distance_metric_name,
                'margin': self.margin,
                'size_average': self.size_average}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_anchor, rep_other = reps

        distances = self.distance_metric(rep_anchor, rep_other)
        negs = distances[labels == 0]
        poss = distances[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        losses = 0.5 * (positive_loss + negative_loss)

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

        return (losses.mean(), acc_by_best.item()) if self.size_average else (losses.sum(), acc_by_best.item())
