import math

import torch
import torch.nn as nn


# --------------------------------------------------
# 1. vocabulary 설정
# --------------------------------------------------
#
# 예시:
# 0 -> <pad>
# 1 -> <unk>
# 2 -> I
# 3 -> study
# 4 -> AI
# 5 -> love
# 6 -> vision
# 7 -> robot
# 8 -> learning
# 9 -> robotics

vocab_size = 10
d_model = 4


# --------------------------------------------------
# 2. Token IDs
# --------------------------------------------------
#
# "I love robotics"
# -> ["I", "love", "robotics"]
# -> [2, 5, 9]
#
# tokenizer 처리는 이미 끝났다고 가정한다.

token_ids = torch.tensor([2, 5, 9])

print("token_ids:")
print(token_ids)

print("token_ids shape:")
print(token_ids.shape)


# --------------------------------------------------
# 3. Input Embedding
# --------------------------------------------------

embedding = nn.Embedding(vocab_size, d_model)

embedded = embedding(token_ids)

print("\nraw embedding:")
print(embedded)

print("embedding shape:")
print(embedded.shape)


# --------------------------------------------------
# 4. Transformer 논문의 embedding scaling
# --------------------------------------------------
#
# embedding * sqrt(d_model)

embedded = embedded * math.sqrt(d_model)

print("\nscaled embedding:")
print(embedded)


# --------------------------------------------------
# 5. Positional Encoding 생성
# --------------------------------------------------

seq_len = token_ids.shape[0]

pe = torch.zeros(seq_len, d_model)

position = torch.arange(seq_len).unsqueeze(1)

div_term = torch.exp(
    torch.arange(0, d_model, 2)
    * (-math.log(10000.0) / d_model)
)

pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

print("\npositional encoding:")
print(pe)

print("PE shape:")
print(pe.shape)


# --------------------------------------------------
# 6. Embedding + Positional Encoding
# --------------------------------------------------

x = embedded + pe

print("\nTransformer input:")
print(x)

print("Transformer input shape:")
print(x.shape)