from binascii import a2b_qp
import json
from nis import match
from MemoryEntry import MemoryEntry
from typing import List

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

# TODO can optimize this quite a bit probably. currently O(2n^2)
def check_consistency(bank: List[MemoryEntry] , constraints: List[Implication]):
    # 1. Gather all of the implications related to each entity in the statements
    implications = {}
    for mem_entry in bank:
        entity, ans, relation = mem_entry.get_entity(), mem_entry.get_answer(), mem_entry.get_relation()
        # Retrieve activated constraints with that id, TODO put this in a pandas dataframe?
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
    #2. for every statement, see if it violates any of the implications that have been activated, aka, p->q and q is false
    for mem_entry in bank:
        # For every activated implication, see if this statement contradicts the implications conclusion
        if mem_entry.get_entity() in implications:
            for constraint in implications[mem_entry.get_entity()]:
                if constraint.target == mem_entry.get_relation():
                    valid_count += 1
                    if constraint.ans[1] != mem_entry.get_answer():
                        violations_count += 1
                        violations += [constraint]
    print(f"Violations: {violations_count}, total implications: {valid_count}, consistency: {1 - violations_count/valid_count}")
    return violations_count, valid_count

# Unit tests
def test_implication():
    a = Implication({"weight": "yes_yes", "direction": "forward", "score": 10, "source": "IsA,dog", "target": "HasA,nose"})
    assert(a.ans == ["yes", "yes"])
    assert(a.source == "IsA,dog")
    assert(a.target == "HasA,nose")
    assert(a.score == 10)
    a = Implication({"weight": "yes_yes", "direction": "back", "score": 10, "source": "IsA,dog", "target": "HasA,nose"})
    assert(a.ans == ["yes", "yes"])
    assert(a.source == "HasA,nose")
    assert(a.target ==  "IsA,dog")
    assert(a.score == 10)

def test_consistency():
    # No violations
    constraints = json.load(open("constraints_v2.json"))
    constraints = [Implication(c) for c in constraints["links"]]
    test_constraints = [Implication({"weight": "yes_yes", "direction": "forward", "score": 10, "source": "IsA,dog", "target": "HasA,nose"}),
                        Implication({"weight": "yes_yes", "direction": "back", "score": 10, "source": "IsA,dog", "target": "HasA,nose"}),
                        Implication({"weight": "yes_yes", "direction": "forward", "score": 10, "source": "IsA,dog", "target": "CapableOf,grow"})]
    test_bank = [MemoryEntry("poodle", "IsA,dog", "yes"),
                MemoryEntry("poodle", "HasA,nose", "yes")]
    violations, valid = check_consistency(test_bank, test_constraints)
    assert(violations == 0)
    assert(valid == 2)

    # violates IsA,dog -> HasA,nose, but !HasA,nose -> IsA,dog is a vacuous truth
    test_bank = [MemoryEntry("poodle", "IsA,dog", "yes"),
                MemoryEntry("poodle", "HasA,nose", "no")]
    violations, valid = check_consistency(test_bank, test_constraints)
    assert(violations == 1)
    assert(valid == 1)

    # violates HasA,nose -> IsA,dog, but !IsA,dog -> HasA,nose is a vacuous truth
    test_bank = [MemoryEntry("poodle", "IsA,dog", "no"),
                MemoryEntry("poodle", "HasA,nose", "yes")]
    violations, valid = check_consistency(test_bank, test_constraints)
    assert(violations == 1)
    assert(valid == 1)

    # violates IsA,dog -> !IsA,horse
    test_constraints = [Implication({"weight": "yes_no", "direction": "forward", "score": 10, "source": "IsA,dog", "target": "IsA,horse"})]
    test_bank = [MemoryEntry("poodle", "IsA,dog", "yes"),
                MemoryEntry("poodle", "IsA,horse", "yes")]
    violations, valid = check_consistency(test_bank, test_constraints)
    assert(violations == 1)
    assert(valid == 1)

if __name__ == "__main__":
    #  Unit tests
    test_consistency()
    test_implication()

