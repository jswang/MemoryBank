from MemoryBank import MemoryBank
from tqdm import tqdm
import utils
import json
import torch
import sklearn
import matplotlib.pyplot as plt

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

            f1_scores += [sklearn.metrics.f1_score(a_truth[i:end], a_pred_batch)]
            accuracies += [torch.sum(a_truth[i:end] == a_pred_batch) / batch_size]

            consistencies += [1]

            print(f"Batch {i} accuracy: {accuracies[-1]}, f1_score: {f1_scores[-1]}, consistency: {consistencies[-1]}")
    return f1_scores, accuracies, consistencies

def evaluate_model(mem_bank, data, output_file, batch_size=400):
    questions = [q for (q, _) in data]
        a_truth = torch.tensor([1 if a == "yes" else 0 for (_, a) in data])
        for i in tqdm(range(0, len(questions), batch_size)):
            end = i+min(batch_size, len(questions))
            q_batch = questions[i:end]
            a_pred_batch = mem_bank.forward(data)
            a_pred_batch = torch.tensor(
                [1 if a == "yes" else 0 for a in a_pred_batch])
            f1_scores += [sklearn.metrics.f1_score(a_truth[i:end], a_pred_batch)]
            accuracies += [torch.sum(a_truth[i:end] == a_pred_batch) / batch_size]

if __name__ == "__main__":
    mem_bank = MemoryBank()

    f1_scores, accuracies, consistencies = evaluate_baseline(mem_bank, test_sentences, "baseline_output.txt")
    batches = [i for i in range(len(f1_scores))]
    plt.plot(batches, f1_scores, label="F1 scores")
    plt.plot(batches, accuracies, label="Accuracy")
    plt.plot(batches, consistencies, label="Consistency")
    plt.legend()
    plt.xlabel("After Batch")
    plt.title("Raw model benchmarks")
    plt.savefig("raw_benchmarks.png")
    plt.show(0)

    # Evaluate baseline on silver facts
    # data = utils.translate_text_yesno(json.load(open("silver_facts.json")))
    # utils.write_to_text(data, "silver_facts.txt")
    # f1_scores, accuracies, consistencies = evaluate_baseline(mem_bank, data, "baseline_output.txt")
