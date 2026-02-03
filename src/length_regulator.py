import torch
import torch.nn as nn


class LengthRegulator(nn.Module):
    """
    The Bridge.
    Input: Text Encodings [Batch, Text_Len, Dim] + Durations [Batch, Text_Len]
    Output: Stretched Encodings [Batch, Mel_Len, Dim]
    """
    def forward(self, x, durations):
        # x: [Batch, Time, Dim]
        # durations: [Batch, Time] (Integer values, e.g. [3, 2, 5...])
        
        output = []
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            # Get the sequence for this batch item
            # Repeat each phoneme vector N times based on duration
            repeated = torch.repeat_interleave(x[i], durations[i], dim=0)
            output.append(repeated)
            
        # Pad the results so they can be stacked into a tensor
        # (This handles the fact that total duration might vary slightly)
        output = torch.nn.utils.rnn.pad_sequence(output, batch_first=True, padding_value=0.0)
        
        return output
