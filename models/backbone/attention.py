import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel


class GaussianSelfAttention(nn.Module):
    def __init__(self, num_units, num_heads=8, dropout_rate=0, causality=False, if_Gaussian=True):
        super(GaussianSelfAttention, self).__init__()

        self.num_units = num_units  # 输入的特征维度
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.if_Gaussian = if_Gaussian

        # You can define any other layers or parameters you need in the constructor here.

        # Define shift and bias as learnable parameters
        self.shift = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, queries, keys):
        # Linear projections
        Q = nn.functional.relu(nn.Linear(queries.size(-1), self.num_units)(queries))  # (N, T_q, C)
        K = nn.functional.relu(nn.Linear(keys.size(-1), self.num_units)(keys))  # (N, T_k, C)
        V = nn.functional.relu(nn.Linear(keys.size(-1), self.num_units)(keys))  # (N, T_k, C)

        # Split and concat
        Q_ = torch.cat(Q.split(self.num_units // self.num_heads, dim=-1), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(K.split(self.num_units // self.num_heads, dim=-1), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(V.split(self.num_units // self.num_heads, dim=-1), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.transpose(1, 2))  # (h*N, T_q, T_k)
        if torch.isnan(outputs).any():
            raise ValueError("Gaussian self-attention step 1 has NaN values")

        Nq, T_q, Cq = Q.size()
        Nk, T_k, Ck = K.size()
        if Nq != Nk:
            raise ValueError(f"The number of queries is not equal to that of keys, they are {Nq}, and {Nk}")

        # Scale + Gaussian prior
        if self.if_Gaussian:
            T_q, T_k = K_.size(1), K_.size(2)
            dis_M = torch.zeros((T_q, T_k), dtype=torch.float64)
            for i in range(T_q):
                for j in range(T_k):
                    dis_M[i][j] = (i - j) ** 2
            dis_M = dis_M.to(queries.device)

            shift_M = self.shift.unsqueeze(0).repeat(T_q, 1).repeat(1, T_k)
            bias_M = self.bias.unsqueeze(0).repeat(T_q, 1).repeat(1, T_k)

            dis_M = -(shift_M * dis_M + bias_M)
            dis_M_ = dis_M.unsqueeze(0).repeat(self.num_heads * Nq, 1, 1)  # (h * N, T_q, T_k)

            outputs = (dis_M_ + outputs) / (K_.size(-1) ** 0.5)  # (h * N, T_q, T_k)

            if torch.isnan(outputs).any():
                raise ValueError("Gaussian self-attention step 2 has NaN values")
        else:
            outputs = outputs / (K_.size(-1) ** 0.5)

        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones(outputs[0, :, :].size(), device=queries.device)  # (T_q, T_k)
            tril = torch.tril(diag_vals)  # (T_q, T_k)
            masks = tril.unsqueeze(0).repeat(self.num_heads * Nq, 1, 1)  # (h*N, T_q, T_k)

            paddings = torch.ones_like(masks) * (-2 ** 32 + 1)
            outputs = torch.where(masks == 0, paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        if torch.isnan(outputs).any():
            raise ValueError("Gaussian self-attention step 3 has NaN values")

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        if torch.isnan(outputs).any():
            raise ValueError("Gaussian self-attention step 4 has NaN values")

        # add and layer norm
        outputs = torch.cat(outputs.split(self.num_units // self.num_heads, dim=-1), dim=-1)  # (h, T_q, C)
        outputs += queries
        if torch.isnan(outputs).any():
            raise ValueError("Gaussian self-attention step 5 has NaN values")
        # Assuming you have a "normalize" function for layer normalization
        norm = nn.LayerNorm(outputs.shape[-1])
        outputs = norm(outputs)  # (N, T_q, C)
        if torch.isnan(outputs).any():
            raise ValueError("Gaussian self-attention step 6 has NaN values")

        return outputs


if __name__ == '__main__':
    att1 = GaussianSelfAttention(num_units=512, causality=True, if_Gaussian=True)
    x = torch.randn(5,17,512)
    y = torch.randn(5,17,512)
    out = att1(x,y)
    print('ok')