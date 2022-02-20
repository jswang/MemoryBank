import pickle
import os
import json
import utils
from tqdm import tqdm
from MemoryBank import MemoryBank

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

# TODO put in config
batch_size = 1264


def evaluate_baseline(mem_bank, data, output_file):
    correct = 0
    with open(output_file, "w") as f:
        questions = [q for (q, _) in data]
        a_truth = [a for (_, a) in data]
        for i in tqdm(range(0, len(questions), batch_size)):
            q_batch = questions[i:i+min(batch_size, len(questions))]
            a_truth_batch = a_truth[i:i+min(batch_size, len(a_truth))]
            a_pred_batch = mem_bank.ask_questions(q_batch)
            for j in range(len(q_batch)):
                q, a_p, a_t = q_batch[j], a_pred_batch[j], a_truth_batch[j]
                f.write(f"{q}, {a_p}\n")
                if a_p == a_t:
                    correct += 1

    accuracy = correct/len(data)
    print(f"Got {correct}/{len(data)} correct, {accuracy} accuracy")


if __name__ == "__main__":
    mem_bank = MemoryBank()

    json_file = json.load(open("silver_facts.json"))
    data = utils.translate_text(json_file)
    utils.write_to_text(data, "silver_facts.txt")
    # data = pickle.load(open("silver_tuples.p", "rb+"))
    # data.to(mem_bank.device)
    # data = test_sentences
    evaluate_baseline(mem_bank, data, "baseline_output.txt")
