import torch
import torch.nn as nn
import math
from symbols import get_vocabulary_size, get_padding_symbol_id


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, dim)

    def forward(self, x):
        # x shape: (Batch, Time, Dim)
        # Match positional encoding length to input length
        x = x + self.pe[:, :x.size(1), :]
        return x


class DurationPredictor(nn.Module):
    def __init__(self,
                 in_channels=256,
                 filter_channels=256,
                 kernel_size=3,
                 dropout=0.1
                 ):
        super(DurationPredictor, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(filter_channels),
            nn.Dropout(dropout),

            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(filter_channels),
            nn.Dropout(dropout),

            # Final projection to 1 channel (log-duration)
            nn.Conv1d(filter_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        out = self.layers(x)
        return out


class AcousticModel(nn.Module):
    def __init__(self,
                 encoder_dim=256,
                 hidden_dim=1024,
                 n_heads=4,
                 encoder_dropout=0.1,
                 encoder_layers=4,
                 duration_predictor_hidden_dim=256,
                 duration_predictor_dropout=0.1,
                 ):
        super(AcousticModel, self).__init__()

        self.embedding = nn.Embedding(get_vocabulary_size(), encoder_dim)
        self.encoder_dim = encoder_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=encoder_dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        self.positional_encoding = SinusoidalPositionalEncoding(encoder_dim)

        self.duration_predictor = DurationPredictor(
            in_channels=encoder_dim,
            filter_channels=duration_predictor_hidden_dim,
            kernel_size=3,
            dropout=duration_predictor_dropout
        )

    def forward(self, phonemes):
        # Create mask for padding tokens if needed
        src_key_padding_mask = (phonemes == get_padding_symbol_id())  # (Batch, Time)

        # Phonemes: (Batch, Time)
        x = self.embedding(phonemes) * math.sqrt(self.encoder_dim)

        # Add positional encoding
        x = x + self.positional_encoding(x)

        # Pass through Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Duration Predictor expects (Batch, Channels, Time), currently (Batch, Time, Channels)
        x_transposed = x.transpose(1, 2)
        log_duration_predictions = self.duration_predictor(x_transposed)

        # Transpose back to (Batch, Time, 1) and squeeze last dim
        log_duration_predictions = log_duration_predictions.transpose(1, 2).squeeze(-1)

        # Predicted even for padding positions; mask them out if needed
        log_duration_predictions = log_duration_predictions.masked_fill(src_key_padding_mask, 0.0)

        return log_duration_predictions, x
