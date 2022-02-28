from MemoryBank import MemoryBank
from tqdm import tqdm
from consistency import check_consistency, Implication
import utils
import json
import torch
import sklearn
import matplotlib.pyplot as plt
from models import *

test_sentences = [('Is an american bison a mammal?', 'yes'),
                  ('Is an american bison an american bison?', 'yes'),
                  ('Is an american bison an animal?', 'yes'),
                  ('Is an american bison a vertebrate?', 'yes'),
                  ('Is an american bison a warm blooded animal?', 'yes'),
                  ('Can an american bison drink liquids?', 'yes'),
                  ('Does an american bison have hair?', 'yes'),
                  ('Is an american bison an air breathing vertebrate?', 'yes'),
                  ('Can an american bison mate?', 'yes'),
                  ('Is an american bison an amniote?', 'yes')]


def evaluate_model(mem_bank, data, constraints=None, batch_size=50):
    """
    Given a model and data containing questions with ground truth, run through
    data in batches. If constraints is None, check consistency as well.
    """
    a_truth = torch.tensor([1 if a == "yes" else 0 for (_, _, a) in data])
    f1_scores = []
    accuracies = []
    consistencies = []
    for i in tqdm(range(0, len(data), batch_size)):
        end = i+min(batch_size, len(data))
        q_batch = data[i:end]
        a_pred_batch = mem_bank.forward(q_batch)
        a_pred_batch = torch.tensor(
            [1 if a == "yes" else 0 for a in a_pred_batch])
        f1_scores += [sklearn.metrics.f1_score(a_truth[i:end], a_pred_batch)]
        accuracies += [torch.sum(a_truth[i:end] == a_pred_batch) / batch_size]
        if constraints is not None:
            c, _, _ = check_consistency(mem_bank, constraints)
            consistencies += [c]
    return f1_scores, accuracies, consistencies


if __name__ == "__main__":
    data = utils.json_to_tuples(json.load(open("data/silver_facts.json")))
    constraints = json.load(open("data/constraints_v2.json"))
    constraints = [Implication(c) for c in constraints["links"]]

    # Evaluate baseline (no flipping) on silver facts
    mem_bank = MemoryBank(flip_config)
    f1_scores, accuracies, consistencies = evaluate_model(
        mem_bank, data, constraints)
    b = [i for i in range(len(f1_scores))]
    plt.plot(b, f1_scores, label="F1 scores")
    plt.plot(b, accuracies, label="Accuracy")
    plt.plot(b, consistencies, label="Consistency")
    plt.legend()
    plt.xlabel("After Batch")
    plt.title("Flip model benchmarks")
    plt.savefig("figures/flip_benchmarks.png")

    # Evaluate flipping
    mem_bank = MemoryBank(baseline_config)
    f1_scores, accuracies, consistencies = evaluate_model(
        mem_bank, data, constraints)
    b = [i for i in range(len(f1_scores))]
    plt.plot(b, f1_scores, label="F1 scores")
    plt.plot(b, accuracies, label="Accuracy")
    plt.plot(b, consistencies, label="Consistency")
    plt.legend()
    plt.xlabel("After Batch")
    plt.title("Raw model benchmarks")
    plt.savefig("figures/raw_benchmarks.png")
