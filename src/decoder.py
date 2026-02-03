import torch
import torch.nn as nn
from model import SinusoidalPositionalEncoding


class FlowMatchingDecoder(nn.Module):
    def __init__(self, input_dim=(80 + 256 + 1), output_dim=80, hidden_dim=512, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.positional_encoding = SinusoidalPositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_projection = nn.Linear(hidden_dim, output_dim)


    def forward(self, x_t, mask, mu, t):
        # x_t: (Batch, Time, Mel_Len)
        # mask: (Batch, Mel_Len) - binary mask for valid positions
        # mu: (Batch, Mel_Len, Encoder_Dim) - aligned encoder outputs (after length regulation)
        # t: (Batch, 1) - time step for flow matching

        # Time expansion: expand t to match (Batch, Mel_Len, 1)
        t_expanded = t.unsqueeze(1).repeat(1, x_t.shape[1], 1)

        # Concatenate along the feature dimension (resulting in (Batch, Mel_Len, 80 + Encoder_Dim + 1))
        x = torch.cat([x_t, mu, t_expanded], dim=-1)

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding(x)

        # Apply mask to encoder
        encoder_output = self.encoder(x, src_key_padding_mask=~mask.bool())  # Invert mask for src_key_padding_mask

        # Output projection
        output = self.output_projection(encoder_output)

        # Mask the output to ensure padded positions are zeroed out
        output = output * mask.unsqueeze(-1)

        return output
