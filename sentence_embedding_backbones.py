from sentence_transformers import SentenceTransformer, models
import torch.nn as nn
import torch


class SentenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # add the sentence embedding model
        self.word_embedding_model = models.Transformer(
            "bert-base-uncased", max_seq_length=256
        )
        self.pooling_model = models.Pooling(
            self.word_embedding_model.get_word_embedding_dimension()
        )
        self.encoder = SentenceTransformer(
            modules=[self.word_embedding_model, self.pooling_model]
        )

        self.out_features = 256
        self.fc = nn.Linear(
            self.pooling_model.get_sentence_embedding_dimension(), self.out_features
        )

    def forward(self, x):

        # flatten list of tuple to just list
        x_input = [item for t in x for item in t]
        x = self.encoder.encode(x_input)
        x = torch.from_numpy(x)

        out = self.fc(x)
        return out