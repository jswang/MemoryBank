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


def evaluate_baseline(mem_bank, data, output_file, batch_size=400):
    """
    Evaluate question answering model's raw ability to answer questions
    """
    f1_scores = []
    accuracies = []
    consistencies = []

    with open(output_file, "w") as f:
        questions = [q for (q, _) in data]
        a_truth = torch.tensor([1 if a == "yes" else 0 for (_, a) in data])
        for i in tqdm(range(0, len(questions), batch_size)):
            end = i+min(batch_size, len(questions))
            q_batch = questions[i:end]
            a_pred_batch = mem_bank.ask_questions(q_batch)
            for j in range(len(q_batch)):
                q, a_p = q_batch[j], a_pred_batch[j]
                f.write(f"{q}, {a_p}\n")
            a_pred_batch = torch.tensor(
                [1 if a == "yes" else 0 for a in a_pred_batch])

            f1_scores += [sklearn.metrics.f1_score(
                a_truth[i:end], a_pred_batch)]
            accuracies += [torch.sum(a_truth[i:end] ==
                                     a_pred_batch) / batch_size]

            consistencies += [1]

            print(
                f"Batch {i} accuracy: {accuracies[-1]}, f1_score: {f1_scores[-1]}, consistency: {consistencies[-1]}")
    return f1_scores, accuracies, consistencies


def evaluate_model(mem_bank, data, constraints=None, batch_size=400):
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
    mem_bank = MemoryBank(baseline_config)
    data = utils.json_to_tuples(json.load(open("silver_facts.json")))
    constraints = json.load(open("constraints_v2.json"))
    constraints = [Implication(c) for c in constraints["links"]]
    f1_scores, accuracies, consistencies = evaluate_model(
        mem_bank, data, constraints)

    # f1_scores, accuracies, consistencies = evaluate_baseline(
    #     mem_bank, test_sentences, "baseline_output.txt")
    b = [i for i in range(len(f1_scores))]
    plt.plot(b, f1_scores, label="F1 scores")
    plt.plot(b, accuracies, label="Accuracy")
    plt.plot(b, consistencies, label="Consistency")
    plt.legend()
    plt.xlabel("After Batch")
    plt.title("Raw model benchmarks")
    plt.savefig("raw_benchmarks.png")
    plt.show()

    # Evaluate baseline on silver facts
    # data = utils.json_to_qas(json.load(open("silver_facts.json")))
    # utils.write_to_text(data, "silver_facts.txt")
    # f1_scores, accuracies, consistencies = evaluate_baseline(mem_bank, data, "baseline_output.txt")
