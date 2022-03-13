from binascii import a2b_qp
import json
from nis import match
from MemoryEntry import MemoryEntry
from MemoryBank import MemoryBank
from typing import List
from models import *
import utils
from tqdm import tqdm
import sklearn
class Implication:
    """
    Stores an implication of the form:
    source.ans[0] -> target.ans[1] with score x
    E.g.
    IsA,dog.yes -> hasA,nose.yes
    """

    def __init__(self, link):
        """Given a link, make an implication"""
        self.score = link["score"]
        # Parse direction and weight
        if link["direction"] == "forward":
            self.ans = link["weight"].split("_")
            self.source = link["source"]
            self.target = link["target"]
        elif link["direction"] == "back":
            self.ans = link["weight"].split("_")
            self.ans = self.ans[::-1]
            self.source = link["target"]
            self.target = link["source"]
        else:
            raise ValueError(f"Impossible direction on link")

    def __str__(self):
        return f"Implication(source={self.source}, target={self.target},  ans={self.ans}, score={self.score})"

    def __repr__(self):
        return f"Implication(source={self.source}, target={self.target},  ans={self.ans}, score={self.score})"

def check_consistency(bank: MemoryBank, constraints: List[Implication]):
    """
    Check consistency of MemoryBank against constraints
    """
    # 1. Gather all of the implications related to each entity in the statements
    implications = {}
    for mem_entry in bank.mem_bank:
        entity, ans, relation = mem_entry.get_entity(
        ), mem_entry.get_answer(), mem_entry.get_relation()
        # Retrieve activated constraints with that id
        for c in constraints:
            # Get all relevant first half of constraints
            if relation == c.source and ans == c.ans[0]:
                if entity in implications:
                    implications[entity] += [c]
                else:
                    implications[entity] = [c]

    violations = []
    violations_count = 0
    valid_count = 0
    # 2. for every statement, see if it violates any of the implications that have been activated, aka, p->q and q is false
    for mem_entry in bank.mem_bank:
        # For every activated implication, see if this statement contradicts the implications conclusion
        if mem_entry.get_entity() in implications:
            for constraint in implications[mem_entry.get_entity()]:
                if constraint.target == mem_entry.get_relation():
                    valid_count += 1
                    if constraint.ans[1] != mem_entry.get_answer():
                        violations_count += 1
                        violations += [constraint]
    # print(
    #     f"Violations: {violations_count}, total implications: {valid_count}, consistency: {1 - violations_count/(valid_count+ 1e-10)}")

    return 1 - violations_count/(valid_count + 1e-10), violations_count, valid_count

def check_accuracy(mem_bank: MemoryBank, ground_truth: List[MemoryEntry]):
    # Compare F1 score of all entries in memory bank against ground truth
    mem_bank.mem_bank
    truth = torch.tensor([1 if t.answer == "yes" else 0 for t in ground_truth])
    pred = torch.tensor([1 if p.answer == "yes" else 0 for p in mem_bank.mem_bank])
    f1_scr = sklearn.metrics.f1_score(truth[0:len(pred)], pred, zero_division=0)
    return f1_scr

# Unit tests
def test_implication():
    a = Implication({"weight": "yes_yes", "direction": "forward",
                    "score": 10, "source": "IsA,dog", "target": "HasA,nose"})
    assert(a.ans == ["yes", "yes"])
    assert(a.source == "IsA,dog")
    assert(a.target == "HasA,nose")
    assert(a.score == 10)
    a = Implication({"weight": "yes_yes", "direction": "back",
                    "score": 10, "source": "IsA,dog", "target": "HasA,nose"})
    assert(a.ans == ["yes", "yes"])
    assert(a.source == "HasA,nose")
    assert(a.target == "IsA,dog")
    assert(a.score == 10)


def test_consistency():
    # No violations
    constraints = json.load(open("data/constraints_v2.json"))
    constraints = [Implication(c) for c in constraints["links"]]
    test_constraints = [Implication({"weight": "yes_yes", "direction": "forward", "score": 10, "source": "IsA,dog", "target": "HasA,nose"}),
                        Implication({"weight": "yes_yes", "direction": "back",
                                    "score": 10, "source": "IsA,dog", "target": "HasA,nose"}),
                        Implication({"weight": "yes_yes", "direction": "forward", "score": 10, "source": "IsA,dog", "target": "CapableOf,grow"})]
    test_bank = [MemoryEntry("poodle", "IsA,dog", "yes"),
                 MemoryEntry("poodle", "HasA,nose", "yes")]
    _, violations, valid = check_consistency(test_bank, test_constraints)
    assert(violations == 0)
    assert(valid == 2)

    # violates IsA,dog -> HasA,nose, but !HasA,nose -> IsA,dog is a vacuous truth
    test_bank = [MemoryEntry("poodle", "IsA,dog", "yes"),
                 MemoryEntry("poodle", "HasA,nose", "no")]
    _, violations, valid = check_consistency(test_bank, test_constraints)
    assert(violations == 1)
    assert(valid == 1)

    # violates HasA,nose -> IsA,dog, but !IsA,dog -> HasA,nose is a vacuous truth
    test_bank = [MemoryEntry("poodle", "IsA,dog", "no"),
                 MemoryEntry("poodle", "HasA,nose", "yes")]
    _, violations, valid = check_consistency(test_bank, test_constraints)
    assert(violations == 1)
    assert(valid == 1)

    # violates IsA,dog -> !IsA,horse
    test_constraints = [Implication(
        {"weight": "yes_no", "direction": "forward", "score": 10, "source": "IsA,dog", "target": "IsA,horse"})]
    test_bank = [MemoryEntry("poodle", "IsA,dog", "yes"),
                 MemoryEntry("poodle", "IsA,horse", "yes")]
    _, violations, valid = check_consistency(test_bank, test_constraints)
    assert(violations == 1)
    assert(valid == 1)


def find(data, constraint):
    for d in data:
        if d.relation == constraint.source and d.answer == constraint.ans[0]:
            return d.entity
    return 'It'

def test_constraint_knowledge(config=baseline_config, contra_test=False):
    constraints = json.load(open("data/constraints_v2.json"))
    constraints = [Implication(c) for c in constraints["links"]]
    data_filename = f"data/silver_facts.json"
    data = utils.json_to_tuples(json.load(open(data_filename)))

    yess = []
    nos = []
    for d in data:
        m = MemoryEntry(d[0], d[1], answer=d[2])
        if m.answer == 'yes':
            yess += [m]
        else:
            nos += [m]

    bank = MemoryBank(config)
    it_results_entailment = []
    it_results_neutral = []
    it_results_contradiction = []
    entity_results_entailment = []
    entity_results_neutral = []
    entity_results_contradiction = []
    for constraint in tqdm(constraints):
        # find an entity for which this is true
        entity = find(yess, constraint)
        a = MemoryEntry(entity=entity, relation=constraint.source, answer=constraint.ans[0])
        b = MemoryEntry(entity=entity, relation=constraint.target, answer=constraint.ans[1])
        if contra_test:
            if b.answer == 'yes':
                b.answer = 'no'
            elif b.answer == 'no':
                b.answer = 'yes'
        it_result = bank.get_relation(a.get_nli_statement(), b.get_nli_statement())
        it_results_entailment += [it_result[0]]
        it_results_neutral += [it_result[1]]
        it_results_contradiction += [it_result[2]]
        e_result =bank.get_relation(a.get_declarative_statement(), b.get_declarative_statement())
        entity_results_entailment += [e_result[0]]
        entity_results_neutral += [e_result[1]]
        entity_results_contradiction += [e_result[2]]

    all_results = np.vstack((it_results_entailment, it_results_neutral, it_results_contradiction))
    decision = np.argmax(all_results, axis=0)
    print(f"Entailment: {np.sum(decision == 0)}, Neutral: {np.sum(decision == 1)}, Contradiction: {np.sum(decision == 2)}")

if __name__ == "__main__":
    #  Unit tests
    test_consistency()
    test_implication()
    test_constraint_knowledge()
