import pickle
import numpy as np

from numba import float32, int64, config
from numba.experimental import jitclass

from importlib import resources as impresources
from .. import data

config.NUMBA_DEBUGINFO = 1
config.NUMBA_BOUNDSCHECK = 1

spec = [
    ("probs", float32[:,:]),
    ("fitness", float32[:,:]),
    ("min_fitness", float32),
    ("weights", float32[:]),
    ("simple_weights", float32[:]),
    ("total_utils", float32[:]),
    ("utils", float32[:]),
    ("freqs", float32[::]),
    ("total_fitness", float32[:,:]),
    ("clue", int64),
    ("count", int64),
    ("assoc_ordering", int64[:])
]

@jitclass(spec)
class Codemaster: 
    def __init__(self, fitness, min_fitness, weights, utils, freqs):
        self.probs = fitness
        self.fitness = fitness
        self.total_fitness = fitness
        self.min_fitness = min_fitness
        self.weights = weights
        self.utils = utils
        self.freqs = freqs
    
    def set_probs(self, probs: np.ndarray):
        self.probs = probs
    
    def get_probs(self):
        return self.probs
    
    def set_fitness(self, fitness: np.ndarray):
        self.fitness = fitness
        
    def get_fitness(self):
        return self.fitness
    
    def get_total_fitness(self):
        return self.total_fitness
    
    def eval(self, words: np.ndarray, assocs: np.ndarray) -> tuple[int, int]:
        # Generate relevant cpt
        probs = self.fitness[:, words]
        probs = probs / probs.sum(axis=1)[:, None]
        # Do the same for the fitness scores
        fitness = self.fitness[:, words]
        # weight each codeword according to card type
        weights = self.weights[assocs]
        weighted_probs = probs * weights
        # Do the same for the fitness scores
        weighted_fitness = fitness * weights
        # Calculate amount of fit codewords for each clue
        # Since all non-good cards have negative weights, we don't need to filter anything
        fit_amount = np.minimum((weighted_fitness > self.min_fitness).sum(axis=1), 4)
        # Calculate utility of each clue (no utterance cost, no count utility)
        util = weighted_probs.sum(axis=1)
        # Add count utility to base utility
        util += self.utils[fit_amount] # * (words.shape[0] / 25.0)
        # Only show clues with fitting clues
        util[fit_amount == 0] = np.min(util)
        # Get index of clue with highest utility
        best_clue = np.argmax(util)
        # print(f"Maximal utility: {util[best_clue]}")
        return (best_clue, fit_amount[best_clue], util, fit_amount)

@jitclass(spec)
class Guesser:
    def __init__(self, fitness, min_fitness, weights, utils, freqs, simple_weights):
        self.probs = fitness
        self.fitness = fitness
        self.total_fitness = fitness
        self.min_fitness = min_fitness
        self.weights = weights
        self.simple_weights = simple_weights
        self.total_utils = utils
        self.utils = utils
        self.freqs = freqs
        self.clue = 0
        self.count = 0
        self.assoc_ordering = np.array([3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        
    def set_clue(self, clue, count):
        self.clue = clue
        self.count = count
        
    def set_board_words(self, words: np.ndarray):
        self.fitness = self.fitness[:, words]
        self.probs = self.fitness / self.fitness.sum(axis=1)[:, None]
        self.utils = (self.total_utils * (words.shape[0] / 25.0)).astype("float32")
        
    def set_weighted_board_words(self, words: np.ndarray, weights: np.ndarray):
        self.fitness = self.fitness[:, words]
        self.probs = (self.fitness / self.fitness.sum(axis=1)[:, None]) * weights
        self.utils = (self.total_utils * (words.shape[0] / 25.0)).astype("float32")
        
    def set_assoc_ordering(self, assocs: np.ndarray):
        self.assoc_ordering = assocs
    
    def set_probs(self, probs: np.ndarray):
        self.probs = probs
    
    def get_probs(self):
        return self.probs
    
    def set_fitness(self, fitness: np.ndarray):
        self.fitness = fitness
        
    def get_fitness(self):
        return self.fitness
    
    def get_total_fitness(self):
        return self.total_fitness
    
    def calculate_codemaster_prob_variant(self, assocs: np.ndarray) -> float:
        # weight each codeword according to card type
        weights = self.weights[assocs]
        weighted_probs = self.probs * weights
        # Do the same for the fitness scores
        weighted_fitness = self.fitness * weights
        # Calculate amount of fit codewords for each clue
        # Calculate utility of each clue (no utterance cost, no count utility)
        util = weighted_probs.sum(axis=1)
        # save utility of target clue plus count
        clue_util = np.e ** (util[self.clue] + self.utils[self.count])
        # Construct utility for all clue_counts
        # We allow all clues, lifting the fitness threshold
        util1 = np.exp(util + self.utils[1]).sum()
        util2 = np.exp(util + self.utils[2]).sum()
        util3 = np.exp(util + self.utils[3]).sum()
        util4 = np.exp(util + self.utils[4]).sum()
        
        return clue_util / (util1 + util2 + util3 + util4)
    
    def calculate_codemaster_prob(self, assocs: np.ndarray) -> float:
        # weight each codeword according to card type
        weights = self.weights[assocs]
        weighted_probs = self.probs * weights
        # Do the same for the fitness scores
        weighted_fitness = self.fitness * weights
        # Calculate amount of fit codewords for each clue
        # Since all non-good cards have negative weights, we don't need to filter anything
        fit_amount = np.minimum((weighted_fitness > self.min_fitness).sum(axis=1), 4)
        # Check if the clue is even eligible, otherwise return 0
        if fit_amount[self.clue] < self.count:
            return 0.0
        # Calculate utility of each clue (no utterance cost, no count utility)
        util = weighted_probs.sum(axis=1)
        # save utility of target clue plus count
        clue_util = np.e ** (util[self.clue] + self.utils[self.count])
        # Construct utility for all clue_counts
        util1 = np.exp(util[fit_amount >= 1] + self.utils[1]).sum()
        util2 = np.exp(util[fit_amount >= 2] + self.utils[2]).sum()
        util3 = np.exp(util[fit_amount >= 3] + self.utils[3]).sum()
        util4 = np.exp(util[fit_amount >= 4] + self.utils[4]).sum()
        
        return clue_util / (util1 + util2 + util3 + util4)
    
    def run_mcmc_swap(self, burn_in=200, steps=100_000, use_intelligent_start=True, shuffle_pairs=1):
        samples = []
        assocs = self.assoc_ordering.copy()
        np.random.shuffle(assocs)
        shuffle_pairs = min(shuffle_pairs, assocs.shape[0]//2)
        shuffle_size = assocs.shape[0]
        old_p = self.calculate_codemaster_prob(assocs)
        for i in range(steps + burn_in):
            proposal = np.random.choice(shuffle_size, size=shuffle_pairs * 2, replace=False)
            for j in range(shuffle_pairs):
                item1 = assocs[proposal[j]]
                item2 = assocs[proposal[j+1]]
                assocs[proposal[j]] = item2
                assocs[proposal[j+1]] = item1
            p = self.calculate_codemaster_prob(assocs)
            if (old_p == 0.0) or ((p/old_p) >= 1.0) or ((p/old_p) >= np.random.random_sample()):
                if i > burn_in and p > 0:
                    samples.append(assocs.copy())
                old_p = p
        return samples
    
    def run_mcmc(self, burn_in=200, steps=100_000, use_intelligent_start=True):
        samples = []
        trace = []
        words = self.fitness.shape[1]
        old_proposal = np.random.randn(words)
        assocs = self.assoc_ordering[old_proposal.argsort().argsort()]
        old_p = self.calculate_codemaster_prob(assocs)
        for i in range(steps + burn_in):
            proposal = old_proposal + np.random.randn(words)
            assocs = self.assoc_ordering[proposal.argsort().argsort()]
            p = self.calculate_codemaster_prob(assocs)
            if (old_p == 0.0) or ((p/old_p) >= 1.0) or ((p/old_p) >= np.random.random_sample()):
                if i > burn_in and p > 0:
                    samples.append(assocs)
                    trace.append(proposal)
                old_p = p
                old_proposal = proposal
        return samples, trace        
    
    def run_mcmc_variant(self, burn_in=200, steps=100_000, use_intelligent_start=True):
        samples = []
        trace = []
        words = self.fitness.shape[1]
        old_proposal = np.random.randn(words)
        assocs = self.assoc_ordering[old_proposal.argsort().argsort()]
        old_p = self.calculate_codemaster_prob_variant(assocs)
        for i in range(steps + burn_in):
            proposal = old_proposal + np.random.randn(words)
            assocs = self.assoc_ordering[proposal.argsort().argsort()]
            p = self.calculate_codemaster_prob_variant(assocs)
            if (old_p == 0.0) or ((p/old_p) >= 1.0) or ((p/old_p) >= np.random.random_sample()):
                if i > burn_in and p > 0:
                    samples.append(assocs)
                    trace.append(proposal)
                old_p = p
                old_proposal = proposal
        return samples, trace     
        

def fasttext_Guesser():
    fitness = np.load((impresources.files(data) / "fasttext_similarities.npy")).astype("float32")
    min_fitness = .45
    weights = np.array([1, -1, -2, -7.5]).astype("float32")
    simple_weights = np.array([1, -1.7]).astype("float32")
    utils = np.array([0, -.3, 0, .3, .3]).astype("float32") # utils = np.array([0, -.5, 0, 1, 2]).astype("float32")
    with (impresources.files(data) / "word_freqs.pkl").open("rb") as file:
        freqs = np.array(list(pickle.load(file).values()))
        freqs = (freqs / np.max(freqs)).astype("float32")
    return Guesser(fitness, min_fitness, weights, utils, freqs, simple_weights)
    
def fasttext_Codemaster():
    """
    Generates an instance of Codemaster with a testing configuration
    """
    # probs = np.load((impresources.files(data) / "fasttext_cpt.npy")).astype("float32")
    fitness = np.load((impresources.files(data) / "fasttext_similarities.npy")).astype("float32")
    min_fitness = .45
    weights = np.array([1, -1, -2, -7.5]).astype("float32")
    utils = np.array([0, -.3, 0, .3, .3]).astype("float32") # utils = np.array([0, -.5, 0, 1, 2]).astype("float32")
    with (impresources.files(data) / "word_freqs.pkl").open("rb") as file:
        freqs = np.array(list(pickle.load(file).values()))
        freqs = (freqs / np.max(freqs)).astype("float32")
    return Codemaster(fitness, min_fitness, weights, utils, freqs)

def llama_Guesser():
    """
    Generates an instance of Codemaster with a testing configuration
    """
    # probs = np.load((impresources.files(data) / "llama_embeddings_cpt.npy")).astype("float32")
    fitness = np.load((impresources.files(data) / "llama_embeddings_similarities.npy")).astype("float32")
    min_fitness = .15
    weights = np.array([1, -1, -2, -7.5]).astype("float32")
    simple_weights = np.array([1, -1.7]).astype("float32")
    utils = np.array([0, -.3, 0, .3, .3]).astype("float32") # utils = np.array([0, -.5, 0, 1, 2]).astype("float32")
    with (impresources.files(data) / "word_freqs.pkl").open("rb") as file:
        freqs = np.array(list(pickle.load(file).values()))
        freqs = (freqs / np.max(freqs)).astype("float32")
    return Guesser(fitness, min_fitness, weights, utils, freqs, simple_weights)

def llama_Codemaster():
    """
    Generates an instance of Codemaster with a testing configuration
    """
    # probs = np.load((impresources.files(data) / "llama_embeddings_cpt.npy")).astype("float32")
    fitness = np.load((impresources.files(data) / "llama_embeddings_similarities.npy")).astype("float32")
    min_fitness = .15
    weights = np.array([1, -1, -2, -7.5]).astype("float32")
    utils = np.array([0, -.3, 0, .3, .3]).astype("float32") # utils = np.array([0, -.5, 0, 1, 2]).astype("float32")
    with (impresources.files(data) / "word_freqs.pkl").open("rb") as file:
        freqs = np.array(list(pickle.load(file).values()))
        freqs = (freqs / np.max(freqs)).astype("float32")
    return Codemaster(fitness, min_fitness, weights, utils, freqs)

def openai_Guesser():
    """
    Generates an instance of Codemaster with a testing configuration
    """
    # probs = np.load((impresources.files(data) / "llama_embeddings_cpt.npy")).astype("float32")
    fitness = np.load((impresources.files(data) / "openai_embeddings_similarities.npy")).astype("float32")
    min_fitness = .375
    weights = np.array([1, -1, -2, -7.5]).astype("float32")
    utils = np.array([0, -.5, 0, .25, .25]).astype("float32")
    simple_weights = np.array([1, -1.7]).astype("float32")
    with (impresources.files(data) / "word_freqs.pkl").open("rb") as file:
        freqs = np.array(list(pickle.load(file).values()))
        freqs = (freqs / np.max(freqs)).astype("float32")
    return Guesser(fitness, min_fitness, weights, utils, freqs, simple_weights)

def openai_Codemaster():
    """
    Generates an instance of Codemaster with a testing configuration
    """
    # probs = np.load((impresources.files(data) / "llama_embeddings_cpt.npy")).astype("float32")
    fitness = np.load((impresources.files(data) / "openai_embeddings_similarities.npy")).astype("float32")
    min_fitness = .375
    weights = np.array([1, -1, -2, -7.5]).astype("float32")
    utils = np.array([0, -.5, 0, .25, .25]).astype("float32")
    with (impresources.files(data) / "word_freqs.pkl").open("rb") as file:
        freqs = np.array(list(pickle.load(file).values()))
        freqs = (freqs / np.max(freqs)).astype("float32")
    return Codemaster(fitness, min_fitness, weights, utils, freqs)

def swow_Guesser():
    """
    Generates an instance of Codemaster with a testing configuration
    """
    # probs = np.load((impresources.files(data) / "llama_embeddings_cpt.npy")).astype("float32")
    fitness = np.load((impresources.files(data) / "swow_similarities.npy")).astype("float32")
    min_fitness = .11
    weights = np.array([1, -1, -2, -7.5]).astype("float32")
    utils = np.array([0, -.5, 0, .25, .25]).astype("float32")
    simple_weights = np.array([1, -1.7]).astype("float32")
    with (impresources.files(data) / "word_freqs.pkl").open("rb") as file:
        freqs = np.array(list(pickle.load(file).values()))
        freqs = (freqs / np.max(freqs)).astype("float32")
    return Guesser(fitness, min_fitness, weights, utils, freqs, simple_weights)

def swow_Codemaster():
    """
    Generates an instance of Codemaster with a testing configuration
    """
    # probs = np.load((impresources.files(data) / "llama_embeddings_cpt.npy")).astype("float32")
    fitness = np.load((impresources.files(data) / "swow_similarities.npy")).astype("float32")
    min_fitness = .11
    weights = np.array([1, -1, -2, -7.5]).astype("float32")
    utils = np.array([0, -.5, 0, .25, .25]).astype("float32")
    with (impresources.files(data) / "word_freqs.pkl").open("rb") as file:
        freqs = np.array(list(pickle.load(file).values()))
        freqs = (freqs / np.max(freqs)).astype("float32")
    return Codemaster(fitness, min_fitness, weights, utils, freqs)

def filter_eligible_clues(clues: list[str], codewords: list[str], fitness: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    """
    For each clue, check if it is a subword of a codeword or if the codeword is a subword of the clue
    Return a list of filtered clues and fitness matrix
    """
    running_idx = []
    running_clues = []
    for idx, clue in enumerate(clues):
        if (not any(clue in word for word in codewords)) and (not any(word in clue for word in codewords)):
            running_clues.append(clue)
            running_idx.append(idx)
    return (running_clues, fitness[running_idx, :])


class Codemaster_Wrapper:
    
    def __init__(self, codemaster: Codemaster, clues: list[str], codewords: list[str]):
        self.codemaster = codemaster
        self.clues = clues
        self.codewords2index = {w:i for i,w in enumerate(codewords)}
        self.assocs2index = {"good": 0, "neutral": 1, "bad": 2, "assassin": 3}
        
    def give_clue(self, board_words: list[str], assocs: list[int]):
        clues, fitness = filter_eligible_clues(self.clues, board_words, self.codemaster.get_total_fitness())
        self.codemaster.set_fitness(fitness.astype("float32"))
        clue, count, _, _ = self.codemaster.eval(np.array([self.codewords2index[word] for word in board_words]), np.array([self.assocs2index[assoc] for assoc in assocs]))
        return clues[clue], count

class Guesser_Wrapper:
    
    def __init__(self, guesser: Guesser, clues: list[str], codewords: list[str], mcmc_burn_in = 1_000, mcmc_iter = 100_000, variant=False):
        self.guesser = guesser
        self.clues = clues
        self.codewords2index = {w:i for i,w in enumerate(codewords)}
        self.assocs2index = {"good": 0, "neutral": 1, "bad": 2, "assassin": 3}
        self.mcmc_iter = mcmc_iter
        self.mcmc_burn_in = mcmc_burn_in
        self.variant = variant
    
    def guess(self, assocs: list[str], board_words: list[str], clue: str, count: int):
        clues, fitness = filter_eligible_clues(self.clues, board_words, self.guesser.get_total_fitness())
        self.guesser.set_fitness(fitness.astype("float32"))
        self.guesser.set_board_words(np.array([self.codewords2index[word] for word in board_words]).astype("int64"))
        self.guesser.set_clue(clues.index(clue), count)
        ordered_assoc = np.sort(np.array([self.assocs2index[assoc] for assoc in assocs]))[::-1].astype("int64")
        self.guesser.set_assoc_ordering(ordered_assoc)
        samples, _ = self.guesser.run_mcmc_variant(self.mcmc_burn_in, self.mcmc_iter) if self.variant else self.guesser.run_mcmc(self.mcmc_burn_in, self.mcmc_iter) 
        if len(samples) > 0:
            mcmc = (np.stack(samples) == 0).sum(axis=0)
            mcmcee = mcmc / mcmc.sum()
            mcmcee[mcmcee > .99] = 0
            return self.guesses_from_distribution(mcmcee, count)
        else:
            return np.random.choice(np.arange(len(board_words)), 1, False)
        
    def guesses_from_distribution(self, dist, count):
        dist = np.array(dist)
        if dist.sum() == 0:
            return [np.random.choice(len(dist))]
        guesses = []
        for _ in range(count):
            idx = dist.argmax()
            guesses.append(idx)
            dist[idx] = 0
        return guesses
    
    def generate_samples(self, assocs: list[str], board_words: list[str], clue: str, count: int, weights: np.ndarray=None):
        clues, fitness = filter_eligible_clues(self.clues, board_words, self.guesser.get_total_fitness())
        self.guesser.set_fitness(fitness.astype("float32"))
        board_words = np.array([self.codewords2index[word] for word in board_words]).astype("int64")
        if weights is not None and len(weights) > 0:
            self.guesser.set_weighted_board_words(board_words, weights.astype("float32"))
        else:
            self.guesser.set_board_words(board_words)
        self.guesser.set_clue(clues.index(clue), count)
        ordered_assoc = np.sort(np.array([self.assocs2index[assoc] for assoc in assocs]))[::-1].astype("int64")
        self.guesser.set_assoc_ordering(ordered_assoc)
        samples, _ = self.guesser.run_mcmc_variant(self.mcmc_burn_in, self.mcmc_iter) if self.variant else self.guesser.run_mcmc(self.mcmc_burn_in, self.mcmc_iter) 
        return samples
        
    def guess_distribution(self, assocs: list[str], board_words: list[str], clue: str, count: int):
        clues, fitness = filter_eligible_clues(self.clues, board_words, self.guesser.get_total_fitness())
        self.guesser.set_fitness(fitness.astype("float32"))
        self.guesser.set_board_words(np.array([self.codewords2index[word] for word in board_words]).astype("int64"))
        self.guesser.set_clue(clues.index(clue), count)
        ordered_assoc = np.sort(np.array([self.assocs2index[assoc] for assoc in assocs]))[::-1].astype("int64")
        self.guesser.set_assoc_ordering(ordered_assoc)
        samples, _ = self.guesser.run_mcmc_variant(self.mcmc_burn_in, self.mcmc_iter) if self.variant else self.guesser.run_mcmc(self.mcmc_burn_in, self.mcmc_iter) 
        if len(samples) > 0:
            mcmc = (np.stack(samples) == 0)
            mcmcee = mcmc.sum(axis=0) / mcmc.shape[0]
            mcmcee[mcmcee > 1] = 0
            return mcmcee
        else:
            return np.zeros(len(assocs))
    
    def guess_simple_distribution(self, board_words: list[str], clue: str):
        fitness = self.guesser.get_total_fitness()
        idx = self.clues.index(clue)
        dist = []
        for word in board_words:
            dist.append(fitness[idx, self.codewords2index[word]])
        return np.array(dist)