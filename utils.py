"""
  TODO: Code to translate from beliefbank graphs to plain natural language texts
"""


def _check_overlap(cand, l_str):
    for l in l_str:
        if l in cand:
            return True
    return False


def _modify_entity(ent):
    if ent[0] in ['a', 'e', 'i', 'o', 'u']:
        return "an " + ent
    return "a " + ent


def _template_lookup(ent, relation, prop):
    ent = _modify_entity(ent)
    if relation == "HasPart" or relation == "HasA":
        if prop[-1] != "s" and not _check_overlap(prop, ["hair", "teeth"]):
            prop = _modify_entity(prop)
        return "Does {entity} have {prop}?".format(entity=ent, prop=prop)
    elif relation == "MadeOf":
        return "Is {entity} made of {prop}?".format(entity=ent, prop=prop)
    elif relation == "CapableOf":
        if prop == "eating":
            prop = "eat"
        return "Can {entity} {prop}?".format(entity=ent, prop=prop)
    elif relation == "IsA":
        prop = _modify_entity(prop)
        return "Is {entity} {prop}?".format(entity=ent, prop=prop)
    else:
        return "Is {entity} {prop}?".format(entity=ent, prop=prop)


def translate_text(file):
    '''
    Convert silver facts file containing relations to yes/no QA pairs.
    '''
    entities = list(file.keys())
#   print(entities)
    true_qa_pairs = []
    for e in entities:
        relations = list(file[e].keys())
        for r in relations:
            true_qa_pairs.append(
                (_template_lookup(e, r.split(",")[0], r.split(",")[1]), file[e][r]))

    return true_qa_pairs


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


if __name__ == '__main__':
    '''
    Test funcitonality by running the file.
    '''
    import os
    import json
    json_file = json.load(open("silver_facts.json"))
    # print(json_file)

    t_qa = translate_text(json_file)

    print('-'*80)
    print('testing translate_text:\n')
    # print 10 examples
    for qa in t_qa[:10]:
        print(qa)

    print('-'*80)

    print('testing translate_conllu:\n')
    print('src: ("Is an owl a mammal?", "yes")')
    translate_conllu("Is an owl a mammal?", "yes", "tmp_conllu.conllu")
    with open("tmp_conllu.conllu", 'r') as f:
        print(f'translated:\n{f.read()}')
    os.remove('tmp_conllu.conllu')
