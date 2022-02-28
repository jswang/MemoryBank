"""
Code to translate from beliefbank graphs to plain natural language texts
"""
import pickle
import os
import json


def _check_overlap(cand, l_str):
    for l in l_str:
        if l in cand:
            return True
    return False


def _modify_entity(ent):
    if ent[0] in ['a', 'e', 'i', 'o', 'u']:
        return "an " + ent
    return "a " + ent


def _yesno_template_lookup(ent, relation, prop, answer):
    ent = _modify_entity(ent)
    if relation == "HasPart" or relation == "HasA":
        if _check_overlap(prop, ["hair", "teeth"]):
            return f"Does {ent} have {prop}?", answer
        if prop == "one mouth":
            return f"Does {ent} have a mouth?", answer
        if prop[-1] != "s":
            prop = _modify_entity(prop)
            return f"Does {ent} have {prop}?", answer
    elif relation == "MadeOf":
        return f"Is {ent} made of {prop}?", answer
    elif relation == "CapableOf":
        if prop == "eating":
            prop = "eat"
        return f"Can {ent} {prop}?", answer
    elif relation == "IsA":
        prop = _modify_entity(prop)
        return f"Is {ent} {prop}?", answer
    else:
        return f"Is {ent} {prop}?", answer


def json_to_qas(file):
    """
    Convert silver facts file containing relations to yes/no QA pairs.
    Returns: List of tuples, tuple[0] = question, tuple[1] = answer
    """
    entities = list(file.keys())

    true_qa_pairs = []
    for e in entities:
        relations = list(file[e].keys())
        for r in relations:
            true_qa_pairs.append(_yesno_template_lookup(
                e, r.split(',')[0], r.split(',')[1], file[e][r]))

    return true_qa_pairs


def json_to_tuples(file):
    """
    Convert silver facts file containing relations to tuples
    """
    entities = list(file.keys())

    true_qa_pairs = []
    for e in entities:
        relations = list(file[e].keys())
        for r in relations:
            true_qa_pairs.append((
                e, r, file[e][r]))

    return true_qa_pairs


def _declarative_template_lookup(ent, relation, prop):
    ent = _modify_entity(ent)
    ent = ent[0].upper() + ent[1:]

    if relation == "HasPart" or relation == "HasA":
        if prop == "one mouth":
            return f"{ent} has a mouth", f"{ent} does not have a mouth"
        if prop[-1] != "s" and not _check_overlap(prop, ["hair", "teeth"]):
            prop = _modify_entity(prop)
        return f"{ent} has {prop}.", f"{ent} does not have {prop}."

    elif relation == "MadeOf":
        return f"{ent} is made of {prop}.", f"{ent} is not made of {prop}."

    elif relation == "CapableOf":
        if prop == "eating":
            prop = "eat"
        return f"{ent} can {prop}.", f"{ent} cannot {prop}."

    elif relation == "IsA":
        prop = _modify_entity(prop)
        return f"{ent} is {prop}.", f"{ent} is not {prop}."

    else:
        return f"{ent} is {prop}.", f"{ent} is not {prop}."


def translate_text_declarative(file):
    """
    Convert silver facts file containing relations to declarative statements.
    """
    entities = list(file.keys())
#   print(entities)
    true_statements = []
    for e in entities:
        relations = list(file[e].keys())
        for r in relations:
            true_statements.append(_declarative_template_lookup(
                e, r.split(",")[0], r.split(",")[1]))

    return true_statements


def translate_conllu(question, answer, filename):
    question = question.replace("?", " ?")
    words_q = question.split(" ")
    total_file = []
    for i in range(len(words_q)):
        new_line = ["_"] * 10  # 10 slots per line
        new_line[0] = str(i + 1)
        new_line[1] = words_q[i]
        total_file.append("\t".join(new_line) + "\n")
    total_file.append("\n")
    new_line = ["_"] * 10  # 10 slots per line
    new_line[0] = "1"
    new_line[1] = answer
    total_file.append("\t".join(new_line) + "\n")
    f_con = open(filename, "w+")
    f_con.writelines(total_file)


def write_to_text(tuples_qa, file):
    """
    Write a list of question answer tuples to a text file.
    Every question answer pair is on it's own line, they're separated by a comma.
    """
    with open(file, "w") as f:
        for t in tuples_qa:
            f.write(f"{t[0]}, {t[1]}\n")


if __name__ == '__main__':
    """
    Test functionality by running the file.
    """
    # From silver_facts.json, generate silver_facts.txt and silver_tuples.p
    json_file = json.load(open("data/silver_facts.json"))
    t_qa = json_to_qas(json_file)
    write_to_text(t_qa, "data/silver_facts.txt")
    pickle.dump(t_qa, open("data/silver_tuples.p", "wb+"))

    declarative_statements = translate_text_declarative(json_file)
    write_to_text(declarative_statements, "data/silver_facts_declarative.txt")
    pickle.dump(declarative_statements, open(
        "data/silver_facts_declarative.p", "wb+"))

    # Testing translate_conllu
    print('testing translate_conllu:\n')
    print('src: ("Is an owl a mammal?", "yes")')
    translate_conllu("Is an owl a mammal?", "yes", "tmp_conllu.conllu")
    with open("tmp_conllu.conllu", 'r') as f:
        print(f'translated:\n{f.read()}')
    os.remove('tmp_conllu.conllu')
