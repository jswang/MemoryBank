import pickle
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


def evaluate_baseline(mem_bank, data, output_file):
    correct = 0
    with open(output_file, "w") as f:
        for qa_pair in tqdm(data):
            (q, a_truth) = qa_pair
            a_pred = mem_bank.ask_question(q)
            f.write(f"{q}, {a_pred}\n")
            if a_pred == a_truth:
                correct += 1

    accuracy = correct/len(data)
    print(f"Got {correct}/{len(data)} correct, {accuracy} accuracy")


if __name__ == "__main__":
    mem_bank = MemoryBank()
    data = pickle.load(open("silver_tuples.p", "rb+"))
    # data = test_sentences
    evaluate_baseline(mem_bank, data, "baseline_output.txt")
