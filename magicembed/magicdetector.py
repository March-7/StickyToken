from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import random
import anndata
import warnings
import pynvml
from sentence_transformers import SentenceTransformer

class MagicDetector:

    def __init__(self, 
                 model_name_or_path:str, 
                ):
        self.model_name_or_path = model_name_or_path
        self.model = SentenceTransformer(model_name_or_path)
        self.tokenizer = self.model.tokenizer
        self.transformer_model = self.model._first_module().auto_model

