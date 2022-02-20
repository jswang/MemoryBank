from MemoryBank import MemoryBank
sentences = [('Is an american bison a mammal?', 'yes'),
             ('Is an american bison an american bison?', 'yes'),
             ('Is an american bison an animal?', 'yes'),
             ('Is an american bison a vertebrate?', 'yes'),
             ('Is an american bison a warm blooded animal?', 'yes'),
             ('Can an american bison drink liquids?', 'yes'),
             ('Does an american bison have hair?', 'yes'),
             ('Is an american bison an air breathing vertebrate?', 'yes'),
             ('Can an american bison mate?', 'yes'),
             ('Is an american bison an amniote?', 'yes')]


def evaluate_baseline(mem_bank, data):
    correct = 0
    for (q, a_truth) in data:
        a_pred = mem_bank.ask_question(q)
        if a_pred == a_truth:
            correct += 1

    accuracy = correct/len(data)
    print(f"Got {correct}/{len(data)} correct, {accuracy} accuracy")


if __name__ == "__main__":
    mem_bank = MemoryBank()
    data = sentences
    evaluate_baseline(mem_bank, data)
