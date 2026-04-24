"""
train_data (torch.utils.data.Dataset),
test_data (torch.utils.data.Dataset),
and the model (torch.nn.Module) should be implemented here.

"""
import os
import torch.nn
from transformers import BertModel

train_data = None
val_data = None
test_data = None
vocab = None
tokenizer = None

DEFAULT_EMBEDDING_MODEL = os.environ.get("FEDE_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

def get_model(*args, **kwargs) -> torch.nn.Module:
    model = BertModel.from_pretrained(DEFAULT_EMBEDDING_MODEL)
    return model
