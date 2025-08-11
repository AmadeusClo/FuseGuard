import torch

from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

from transformers.models.gpt2.modeling_gpt2 import GPT2Model

# model = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
model = AutoModel.from_pretrained("C:\mym\CALF-main\CALF-main\LLM\gpt2")
tokenizer = AutoTokenizer.from_pretrained("C:\mym\CALF-main\CALF-main\LLM\gpt2")


wte = model.wte.state_dict()['weight'].cpu().numpy()

pca = PCA(n_components=500)

wte_pca = pca.fit_transform(wte.T)
# wte_pca = pca.fit_transform(wte)

torch.save(wte_pca, "wte_pca_500.pt")