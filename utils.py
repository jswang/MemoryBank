"""
  TODO: Code to translate from beliefbank graphs to plain natural language texts
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
        if prop[-1] != "s" and not _check_overlap(prop, ["hair", "teeth"]):
            prop = _modify_entity(prop)
        return "Does {entity} have {prop}?".format(entity=ent, prop=prop), answer
    elif relation == "MadeOf":
        return "Is {entity} made of {prop}?".format(entity=ent, prop=prop), answer
    elif relation == "CapableOf":
        if prop == "eating":
            prop = "eat"
        return "Can {entity} {prop}?".format(entity=ent, prop=prop), answer
    elif relation == "IsA":
        prop = _modify_entity(prop)
        return "Is {entity} {prop}?".format(entity=ent, prop=prop), answer
    else:
        return "Is {entity} {prop}?".format(entity=ent, prop=prop), answer


def json_to_qas(file):
    '''
    Convert silver facts file containing relations to yes/no QA pairs.
    Returns: List of tuples, tuple[0] = question, tuple[1] = answer
    '''
    entities = list(file.keys())

    true_qa_pairs = []
    for e in entities:
        relations = list(file[e].keys())
        for r in relations:
            true_qa_pairs.append(_yesno_template_lookup(
                e, r.split(',')[0], r.split(',')[1], file[e][r]))

    return true_qa_pairs


def json_to_tuples(file):
    '''
    Convert silver facts file containing relations to tuples
    '''
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
        if prop[-1] != "s" and not _check_overlap(prop, ["hair", "teeth"]):
            prop = _modify_entity(prop)
        return "{entity} has {prop}".format(entity=ent, prop=prop), "{entity} does not have {prop}".format(entity=ent, prop=prop)

    elif relation == "MadeOf":
        return "{entity} is made of {prop}".format(entity=ent, prop=prop), "{entity} is not made of {prop}".format(entity=ent, prop=prop)

    elif relation == "CapableOf":
        if prop == "eating":
            prop = "eat"
        return "{entity} can {prop}".format(entity=ent, prop=prop), "{entity} cannot {prop}".format(entity=ent, prop=prop)

    elif relation == "IsA":
        prop = _modify_entity(prop)
        return "{entity} is {prop}".format(entity=ent, prop=prop), "{entity} is not {prop}".format(entity=ent, prop=prop)

    else:
        return "{entity} is {prop}".format(entity=ent, prop=prop), "{entity} is not {prop}".format(entity=ent, prop=prop)


def translate_text_declarative(file):
    '''
    Convert silver facts file containing relations to declarative statements.
    '''
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
    json_file = json.load(open("silver_facts.json"))
    t_qa = json_to_qas(json_file)
    write_to_text(t_qa, "silver_facts.txt")
    pickle.dump(t_qa, open("silver_tuples.p", "wb+"))

    # Visualization of the question + answer pairs
    print('-'*80)
    print('testing json_to_qas:\n')
    # print 5 examples
    for qa in t_qa[:5]:
        print(qa)

    declarative_statements = translate_text_declarative(json_file)
    write_to_text(declarative_statements, "silver_facts_declarative.txt")
    pickle.dump(declarative_statements, open(
        "silver_facts_declarative.p", "wb+"))

    # Visualization of the question + answer pairs
    print('-'*80)
    print('testing translate_text_declarative:\n')
    # print 5 examples
    for yes, no in declarative_statements[:5]:
        print(yes)
        print(no)
    print('-'*80)

    # Testing translate_conllu
    print('testing translate_conllu:\n')
    print('src: ("Is an owl a mammal?", "yes")')
    translate_conllu("Is an owl a mammal?", "yes", "tmp_conllu.conllu")
    with open("tmp_conllu.conllu", 'r') as f:
        print(f'translated:\n{f.read()}')
    os.remove('tmp_conllu.conllu')
