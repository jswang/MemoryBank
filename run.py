import argparse
from MemoryBank import MemoryBank
from tqdm import tqdm
from consistency import check_consistency, Implication
from MemoryEntry import MemoryEntry
import utils
import json
import torch
import matplotlib.pyplot as plt
from models import *
import tensorflow as tf
import numpy as np
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp

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


def evaluate_model(mem_bank, data, mode, writer, constraints, batch_size=100):
    """
    Given a model and data containing questions with ground truth, run through
    data in batches. If constraints is None, check consistency as well.
    """
    memory_entries = [MemoryEntry(i[0], i[1], i[2]) for i in data]
    f1_scores = []
    accuracies = []
    consistencies = []

    for i in tqdm(range(0, len(data), batch_size)):
        q_batch = data[i:i+min(batch_size, len(data))]
        is_mega_batch = (i % (len(data)/10) == 0)
        mem_bank.forward(q_batch, is_mega_batch)
        if is_mega_batch:
            f = check_accuracy(mem_bank, memory_entries)
            f1_scores += [f]
            c, _, _ = check_consistency(mem_bank, constraints)
            consistencies += [c]
            # writer.add_scalar(f"F1 Score/{mode}/{mem_bank.name}", f, i)
            # writer.add_scalar(f"Consistency/{mode}/{mem_bank.name}", c, i)

    writer.add_hparams({"sentence_similarity_threshold": mem_bank.config["sentence_similarity_threshold"],
                        "default_flipped_confidence": mem_bank.config["default_flipped_confidence"],
                        "flip_premise_threshold": mem_bank.config["flip_premise_threshold"],
                        "entail_threshold": mem_bank.config["entail_threshold"],
                        "scoring": mem_bank.config["scoring"],
                        "flip_entailing_premises": mem_bank.config["flip_entailing_premises"],
                        "enable_flip": mem_bank.config["enable_flip"],
                        "feedback_type": mem_bank.config["feedback_type"]
                        },
                        {"hparam/average consistency": np.mean(np.array(consistencies)),
                        "hparam/median consistency": np.median(np.array(consistencies)),
                        "hparam/average accuracy": np.mean(np.array(accuracies)),
                        "hparam/median accuracy": np.median(np.array(accuracies))})
    writer.flush()
    return f1_scores, accuracies, consistencies


def save_data(config, f1_scores, accuracies, consistencies, date_time=None):
    if date_time is None:
        timestamp = datetime.timestamp(datetime.now())
        date_time = datetime.fromtimestamp(timestamp).strftime('%m_%d_%H:%M:%S')

    res_dict = {
        "config": str(config),
        "f1": str(f1_scores),
        "accuracies": str(accuracies),
        "consistencies": str(consistencies)
    }
    with open(f"data/results_{date_time}.json", "w+") as f:
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


def hyperparameter_tune():
    # Setup
    data = utils.json_to_tuples(json.load(open("data/silver_facts_val.json")))
    constraints = json.load(open("data/constraints_v2.json"))
    constraints = [Implication(c) for c in constraints["links"]]

    date_time = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime('%m_%d_%H:%M:%S')
    writer = SummaryWriter(log_dir=f"runs/hyperparam-tuning-{date_time}")
    # Vary the means by which we count up n_entail and n_contra
    for sentence_similarity_threshold in np.arange(.6, 1.1, .1): # 5
        for confidence in np.arange(0.6, 1.1, 0.1): #5
            for flip_premise_threshold in np.arange(0.8, 1.3, .2): #3
                for entail_threshold in np.arange(0.8, 1.3, .2): ##3
                    # for scoring in ["max_only", "entail_and_contra"]: #2
                    # for flip_entailing in [True, False]: #6
                    config = flip_config.copy()
                    config['sentence_similarity_threshold'] = sentence_similarity_threshold
                    config['default_flipped_confidence'] = confidence
                    config['flip_premise_threshold'] = flip_premise_threshold
                    config['entail_threshold'] = entail_threshold
                    config['scoring'] = "max_only"
                    config['flip_entailing_premises'] = True

                    # Evaluate flip only
                    mem_bank = MemoryBank(config)
                    f1_scores, accuracies, consistencies = evaluate_model(
                        mem_bank, data, writer=writer, mode='val', constraints=constraints)
                    save_data(config, f1_scores, accuracies, consistencies, date_time=f"{date_time}_flip")

                    # Evaluate relevant feedback
                    config["feedback_type"] = "relevant"
                    config["max_retrieved"] = 30
                    mem_bank = MemoryBank(config)
                    f1_scores, accuracies, consistencies = evaluate_model(
                        mem_bank, data, writer=writer, mode='val', constraints=constraints)
                    save_data(config, f1_scores, accuracies, consistencies, date_time=f"{date_time}_relevant")

                    # Evaluate on topic feedback
                    config["feedback_type"] = "topic"
                    mem_bank = MemoryBank(config)
                    f1_scores, accuracies, consistencies = evaluate_model(
                        mem_bank, data, writer=writer, mode='val', constraints=constraints)
                    save_data(config, f1_scores, accuracies, consistencies, date_time=f"{date_time}_topic")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='val')
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-t', '--tune', action='store_true')
    args = parser.parse_args()

    if args.tune:
        hyperparameter_tune()
    else:
        mode = args.mode
        assert mode in ['full_dataset', 'val', 'test']

        data_filename = "data/silver_facts.json"
        if mode != 'full_dataset':
            data_filename = f"data/silver_facts_{mode}.json"

        data = utils.json_to_tuples(json.load(open(data_filename)))
        constraints = json.load(open("data/constraints_v2.json"))
        constraints = [Implication(c) for c in constraints["links"]]

        # Evaluate baseline model
        for config in [flip_config]:
            mem_bank = MemoryBank(config)
            writer = SummaryWriter()
            f1_scores, accuracies, consistencies = evaluate_model(
                mem_bank, data, mode, writer, constraints, batch_size=args.batch_size)
            save_data(config, f1_scores, accuracies, consistencies)
            plot(f1_scores, accuracies, consistencies, config)
