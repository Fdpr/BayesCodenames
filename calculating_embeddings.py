from tqdm import tqdm
import numpy as np

def embed(model, tokenizer, word):
    tokens = tokenizer.encode(word, add_special_tokens = False, return_tensors="pt")
    return model.embed_tokens(tokens)[0,:].mean(dim=0).cpu().detach().numpy()
    
def clue_similarity(clue, words):
    return np.array([(np.dot(clue, word)/(np.linalg.norm(clue) * np.linalg.norm(word))) for word in words])

clue_embeddings = np.load("clue_embeddings.npy")
codewords_embeddings = np.load("codeword_embeddings.npy")

matrix = np.stack(list(tqdm((clue_similarity(clue, codewords_embeddings) for clue in clue_embeddings), desc="calculating similarities")))

matrix = matrix - matrix.min() + 1e-7
matrix = matrix / matrix.max()
np.save("llama_embeddings_similarities.npy", matrix.astype("float16"))