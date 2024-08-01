from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import pickle
import numpy as np

def embed(model, tokenizer, word):
    tokens = tokenizer.encode(word, add_special_tokens = False, return_tensors="pt")
    return model.embed_tokens(tokens)[0,:].mean(dim=0).cpu().detach().numpy()
    
def clue_similarity(clue, words):
    return np.array([(np.dot(clue, word)/(np.linalg.norm(clue) * np.linalg.norm(word))) for word in words])
    

with open("all_codewords.pkl", "rb") as file:
	codewords = pickle.load(file)
with open("all_clues.pkl", "rb") as file:
    clues = pickle.load(file)
with open("hf_token.txt") as file:
    token = file.readlines()[0][:-1]

model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)

clue_embeddings = list(tqdm([embed(model, tokenizer, clue) for clue in clues], desc="fetching clue embeddings"))
codewords_embeddings = list(tqdm([embed(model, tokenizer, codeword) for codeword in codewords], desc="fetching codeword embeddings"))

matrix = np.stack(list(tqdm((clue_similarity(clue, codewords_embeddings) for clue in clue_embeddings), desc="calculating similarities")))

matrix = matrix - matrix.min() + 1e-7
matrix = matrix / matrix.max()
np.save("llama_embeddings_similarities.npy", matrix.astype("float16"))