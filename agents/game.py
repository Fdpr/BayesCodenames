import numpy as np
import random

class Codenames:
    
    def __init__(self, words: list[str], assoc: list[str], guesser, codemaster, simple=False):
        self.words = words
        self.all_words = words
        self.assoc = assoc
        self.guesser = guesser
        self.weights = np.ones(len(words))
        self.codemaster = codemaster
        self.rounds = 0
        self.scores = []
        self.simple = simple
        self.state = 0 # 0: In play, 1: won, 2: lost (all blue), 3: lost (assassin)
        
    def play_round(self):
        if self.state != 0:
            return -1
        clue, count = self.codemaster.give_clue(self.words, self.assoc)
        if self.simple:
            samples = self.guesser.guess_simple_distribution(self.words, clue)
        else:
            samples = self.guesser.generate_samples(self.assoc, self.words, clue, count, self.weights)
        indices = []
        score = 0
        
        if len(samples) != 0:
            samples = np.stack(samples)
            for _ in range(count):
                if len(samples) == 0:
                    break
                mcmcee = samples if self.simple else (samples == 0).sum(axis=0)
                index = mcmcee.argmax()
                indices.append(index)
                assoc = self.assoc[index]
                if assoc == "good":
                    score += 1
                    samples = np.where(samples != samples[index], samples, 0) if self.simple else samples[samples[:, index] == 0]
                    if not self.simple:
                        samples[:, index] = 1 # To make sure the same element is not selected twice
                    if "good" not in self.assoc:
                        break
                else:
                    if assoc == "neutral":
                        score -= 1
                        samples = samples if self.simple else samples[samples[:, index] == 1]
                    elif assoc == "bad":
                        score -= 2
                        samples = samples if self.simple else samples[samples[:, index] == 2]
                    else:
                        score -= 7.5
                    break
        else:
            index = random.randint(0, len(self.words) - 1)
            indices.append(index)
            assoc = self.assoc[index]
            if assoc == "good":
                score += 1
            else:
                if assoc == "neutral":
                    score -= 1
                elif assoc == "bad":
                    score -= 2
                else:
                    score -= 7.5
        self.scores.append(score)
        if self.update_state() == 0:
            bad_word = next(i for i, e in enumerate(self.assoc) if e == "bad")
            indices.append(bad_word)
        else:
            return -1
        self.words = [word for i, word in enumerate(self.words) if i not in indices]
        self.assoc = [assoc for i, assoc in enumerate(self.assoc) if i not in indices]
        if len(samples) > 0 and not self.simple:
            samples = samples[samples[:, bad_word] != 0]
            samples = np.delete(samples, indices, 1)
            samples = (samples == 0).sum(axis=0)
            samples = .5 + (samples / samples.max())
            self.weights = (samples / samples.sum()) * len(self.words)
        else:
            self.weights = np.ones(len(self.words))
        return self.update_state()      
    
    def update_state(self):
        if "assassin" not in self.assoc:
            self.state = 3
            return -1
        elif "good" not in self.assoc:
            self.state = 1
            return -1
        elif "bad" not in self.assoc:
            self.state = 2
            return -1
        return 0
    
    def play_game(self):
        state = 0
        while state == 0:
            state = self.play_round()
            self.rounds += 1
        return self.state, self.rounds, self.scores