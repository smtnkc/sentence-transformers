import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union, Tuple, List, Iterable, Dict, Callable
from ..SentenceTransformer import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score
import logging


logger = logging.getLogger(__name__)

class SoftmaxLoss(nn.Module):
    """
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param threshold: Threshold for the softmax classifier. If the maximum probability is below the threshold, the sample is considered as 'other'
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss()

    Example::

        from sentence_transformers import SentenceTransformer, SentencesDataset, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['First pair, sent A', 'First pair, sent B'], label=0),
            InputExample(texts=['Second Pair, sent A', 'Second Pair, sent B'], label=3)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 conf_threshold: float = 0.0,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(SoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.conf_threshold = conf_threshold
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        # print("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
        self.loss_fct = loss_fct

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        probabilities = F.softmax(output, dim=1)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            # accuracy = accuracy_score(labels.cpu(), torch.argmax(output, 1).cpu())
            # f1 = f1_score(labels.cpu(), torch.argmax(output, 1).cpu(), average='weighted')
            # correct = torch.argmax(output, dim=1).eq(labels).sum().item()
            max_prob, predicted_class = torch.max(probabilities, 1)
            updated_predicted_class = predicted_class.clone()
            if self.conf_threshold is not None and self.conf_threshold > 0:
                # If the maximum probability is below the threshold, we consider the sample as 'other'
                updated_predicted_class[max_prob < self.conf_threshold] = self.num_labels - 1

            batch_size = len(labels)
            num_labels = self.num_labels

            if num_labels > 2 and self.conf_threshold == 0 and self.evaluator.name == 'Zero':
                # accept the prediction as correct unless eris variants are classified as new omicron variants
                correct = torch.sum((updated_predicted_class == labels) | (updated_predicted_class > 0) & (labels > 0)).item()
            else:
                correct = torch.sum(updated_predicted_class == labels).item()
            acc = correct / batch_size
            return loss, acc
        else:
            return reps, output
