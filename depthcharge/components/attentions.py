import torch.nn as nn
import torch.nn.functional as F


class WindowedSelfAttention(nn.Module):
    def __init__(self, input_dim, window_size):
        super(WindowedSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size

    def forward(self, query, key, value):
        batch_size, query_seq_len, _ = query.size()
        _, key_seq_len, _ = key.size()

        # Initialize an empty tensor for the output
        output = torch.zeros(batch_size, query_seq_len, self.input_dim).to(query.device)

        # Calculate attention scores for each query position
        for i in range(query_seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(key_seq_len, i + self.window_size // 2 + 1)
            attention_scores = torch.matmul(query[:, i, :].unsqueeze(1), key[:, start:end, :].transpose(1, 2))
            attention_scores = attention_scores / (key[:, start:end, :].size(-1) ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            weighted_sum = torch.matmul(attention_weights, value[:, start:end, :])
            output[:, i, :] = weighted_sum.squeeze(1)

        return output



class SelfAttentionConcatenation(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionConcatenation, self).__init__()

        self.attention = nn.MultiheadAttention(input_dim, num_heads=1)

    def forward(self, *tensors):
        attn_tensors = []

        for tensor in tensors:
            attn_tensor, _ = self.attention(tensor, tensor, tensor)
            attn_tensors.append(attn_tensor)

        # Transpose and reshape the attention tensors to match the desired output shape
        output_dim = attn_tensors[0].shape[-1]
        attn_tensors = [attn_tensor.transpose(0, 1).reshape(-1, attn_tensor.shape[-2], output_dim) for attn_tensor in
                        attn_tensors]

        # Concatenate the tensors along the second dimension
        combined_tensor = torch.cat(attn_tensors, dim=1)

        return combined_tensor

class AttentionConcatenation(nn.Module):
    def __init__(self, input_dim):
        super(AttentionConcatenation, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)  # Apply softmax along the second dimension
        )

    def forward(self, tensor1, tensor2, tensor3):
        attn_weights1 = self.attention(tensor1)
        attn_weights2 = self.attention(tensor2)
        attn_weights3 = self.attention(tensor3)

        weighted_tensor1 = torch.mul(tensor1, attn_weights1)
        weighted_tensor2 = torch.mul(tensor2, attn_weights2)
        weighted_tensor3 = torch.mul(tensor3, attn_weights3)

        concatenated_tensor = torch.cat([weighted_tensor1, weighted_tensor2, weighted_tensor3], dim=1)

        return concatenated_tensor



