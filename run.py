from MemoryBank import MemoryBank
from tqdm import tqdm
from consistency import check_consistency, Implication
from MemoryEntry import MemoryEntry
import utils
import json
import torch
import sklearn
import matplotlib.pyplot as plt
from models import *
import time
import numpy as np


def choose_threshold():
    """
    Helper function to figure out what threshold to select for faiss lookup
    """
    mb = MemoryBank(baseline_config)
    mb.add_to_bank([MemoryEntry("poodle", "IsA,dog", 0.9, "yes"),
                   MemoryEntry("poodle", "HasA,nose", 0.9, "yes"),
                   MemoryEntry("poodle", "CapableOf,grow moldy", 0.9, "no")])
    mb.add_to_bank([MemoryEntry("seagull", "IsA,bird", 0.9, "yes")])

    # Should get back everything to do with a poodle
    for t in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        mb.threshold = t
        retrieved, I = mb.retrieve_from_index(
            [MemoryEntry("poodle", "IsA,dog", 0.9, "yes")])
        print(
            f"threshold: {mb.threshold}, number retrieved: {len(retrieved)}, {retrieved}")

    # should bring back nothing
    for t in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        mb.threshold = t
        retrieved, I = mb.retrieve_from_index(
            [MemoryEntry("fridge", "HasProperty,cold", 0.9, "yes")])
        print(
            f"f: threshold: {mb.threshold}, number retrieved: {len(retrieved)}, {retrieved}")

    # maybe should bring out poodle case
    for t in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        mb.threshold = t
        retrieved, I = mb.retrieve_from_index(
            [MemoryEntry("fridge", "CapableOf,grow moldy", 0.9, "yes")])
        print(
            f"f: threshold: {mb.threshold}, number retrieved: {len(retrieved)}, {retrieved}")


def test_flip_or_keep():
    """
    Function to test flipping or keeping
    """
    # Should flip this hypothesis, strong contradiction from both premises
    mb = MemoryBank(flip_config)
    premises = [MemoryEntry("poodle", "IsA,dog", 0.9, "yes"),
                MemoryEntry("poodle", "HasA,teeth", 0.9, "yes")]
    mb.add_to_bank(premises)
    ans = mb.flip_or_keep(premises, [0, 1], MemoryEntry(
        "poodle", "HasA,one mouth", 0.5, "no"))
    assert ans.answer == 'yes'


def test_ask_question():
    mb = MemoryBank()
    answers, probs = mb.ask_questions([MemoryEntry("american bison", "IsA,plastic", None, None), MemoryEntry(
        "american bison", "IsA,company", None, None)], [])
    print(f"{answers}, {probs}")


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
        f1_scores += [sklearn.metrics.f1_score(
            a_truth[i:end], a_pred_batch, zero_division=0)]
        accuracies += [torch.sum(a_truth[i:end] == a_pred_batch) / batch_size]
        if constraints is not None:
            t0 = time.time()
            c, _, _ = check_consistency(mem_bank, constraints)
            t1 = time.time()
            print(f"Consistency check: {t1 - t0}s")
            consistencies += [c]
    return f1_scores, accuracies, consistencies


def save_data(config, f1_scores, accuracies, consistencies):
    from datetime import datetime
    import json
    timestamp = datetime.timestamp(datetime.now())
    date_time = datetime.fromtimestamp(timestamp)

    res_dict = {
        "config": str(config),
        "f1": str(f1_scores),
        "accuracies": str(accuracies),
        "consistencies": str(consistencies)
    }
    with open(f"data/results_{date_time.strftime('%m_%d_%H:%M:%S')}.json", "w+") as f:
        json.dump(res_dict, f)


def load_and_plot(filename, config):
    with open(filename, 'r') as f:
        data = json.load(f)
        f1_scores = data['f1'].strip('[]').split(',')
        f1_scores = [float(f.strip()) for f in f1_scores]
        accuracies = data['accuracies'].strip('[]').split(',')
        accuracies = [float(f.strip(' tensor()')) for f in accuracies]
        consistencies = data['consistencies'].strip('[]').split(',')
        consistencies = [float(f.strip()) for f in consistencies]
        plot(f1_scores, accuracies, consistencies, config)


def plot(f1_scores, accuracies, consistencies, config):
    b = [i for i in range(len(f1_scores))]
    plt.plot(b, accuracies, label="Accuracy")
    plt.plot(b, consistencies, label="Consistency")
    # clean up f1 scores a littlebit
    f1_scores = np.array(f1_scores)
    b = np.array(b)
    new_f1 = f1_scores[np.where(f1_scores != 0)]
    new_b = b[np.where(f1_scores != 0)]
    plt.plot(new_b, new_f1, label="F1 scores")
    plt.legend()
    plt.xlabel("After Batch")
    plt.title(f"{config['name']} Model benchmarks")
    plt.savefig(f"figures/{config['name']}_benchmarks.png")
    plt.close()


if __name__ == "__main__":
    data = utils.json_to_tuples(json.load(open("data/silver_facts.json")))
    constraints = json.load(open("data/constraints_v2.json"))
    constraints = [Implication(c) for c in constraints["links"]]

    # Evaluate baseline model
    for config in [baseline_config, flip_config, feedback_relevant_config]:
        mem_bank = MemoryBank(config)
        f1_scores, accuracies, consistencies = evaluate_model(
            mem_bank, data, constraints)
        save_data(config, f1_scores, accuracies, consistencies)
        plot(f1_scores, accuracies, consistencies, config)

    # load_and_plot("data/results_03_03_18:42:01.json", feedback_relevant_config)
    # load_and_plot("data/results_03_03_18:33:42.json", flip_config)
    # load_and_plot("data/results_03_03_17:35:53.json", baseline_config)
