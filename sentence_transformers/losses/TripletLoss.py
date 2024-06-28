import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import SiameseDistanceMetric

class TripletLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:

    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).

    Margin is an important hyperparameter and needs to be tuned respectively.

    For further details, see: https://en.wikipedia.org/wiki/Triplet_loss

    :param model: SentenceTransformerModel
    :param distance_metric: Function to compute distance between two embeddings. The class TripletDistanceMetric contains common distance metrices that can be used.
    :param margin: The negative should be at least this much further away from the anchor than the positive.

    Example::

        from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
            InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2'])]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer,
                 distance_metric = SiameseDistanceMetric.EUCLIDEAN,
                 margin: float = None):
        super(TripletLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.margin = margin

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "SiameseDistanceMetric.{}".format(name)
                break

        return {'distance_metric': distance_metric_name,
                'margin': self.margin}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        rep_anchor, rep_pos, rep_neg = reps
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.margin)

        # Accuracy
        num_triplets = num_correct_cos_triplets = 0
        for idx in range(len(distance_pos)):
            num_triplets += 1

            if distance_pos[idx] < distance_neg[idx]:
                num_correct_cos_triplets += 1

        accuracy = num_correct_cos_triplets / num_triplets

        return losses.mean(), accuracy
