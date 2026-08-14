import torch
import numpy as np
import random

features = torch.load(r"D:\Stroke_mri_features\features.pt",weights_only=True)
labels = torch.load(r"D:\Stroke_mri_features\labels.pt",weights_only=True)

features = features.cpu().numpy()
labels = labels.cpu().numpy()

variances = np.var(features, axis=0)
threshold = np.percentile(variances, 25)
mask = variances > threshold
filtered_features = features[:, mask]

pop_size = 20
n_features = filtered_features.shape[1]
n_generations = 10
mutation_rate = 0.1
crossover_rate = 0.7

def fitness(individual):
    selected = filtered_features[:, individual == 1]
    if selected.shape[1] == 0:
        return 0
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=500)
    score = cross_val_score(clf, selected, labels, cv=3).mean()
    return score

population = [np.random.randint(0, 2, n_features) for _ in range(pop_size)]

for gen in range(n_generations):
    scores = [fitness(ind) for ind in population]
    sorted_idx = np.argsort(scores)[::-1]
    population = [population[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    new_population = population[:2]
    while len(new_population) < pop_size:
        if random.random() < crossover_rate:
            p1, p2 = random.sample(population[:10], 2)
            point = random.randint(1, n_features - 1)
            child = np.concatenate([p1[:point], p2[point:]])
        else:
            child = population[random.randint(0, 9)].copy()
        for i in range(n_features):
            if random.random() < mutation_rate:
                child[i] = 1 - child[i]
        new_population.append(child)
    population = new_population

best_individual = population[0]
best_features = filtered_features[:, best_individual == 1]

torch.save(torch.tensor(best_features), r"D:\Stroke_mri_features\selected_features.pt")
torch.save(torch.tensor(labels), r"D:\Stroke_mri_features\selected_labels.pt")

print("Original features:", features.shape[1])
print("Filtered features:", filtered_features.shape[1])
print("Selected features:", best_features.shape[1])
