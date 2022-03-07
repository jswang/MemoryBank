from dataclasses import dataclass
from utils import _declarative_template_lookup, _yesno_template_lookup


@dataclass
class MemoryEntry:
    """
    Example: MemoryEntry("poodle", "IsA,dog", "yes")
    Here, entity: "poodle", relation: "IsA,dog", answer: "yes"
    """
    entity: str
    relation: str
    # linked, every answer comes with a confidence
    confidence: float = None
    answer: str = None

    def get_pos_statement(self):
        return _declarative_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1])[0]

    def get_neg_statement(self):
        return _declarative_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1])[1]

    def get_declarative_statement(self):
        return _declarative_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1])[self.answer == 'no']

    def get_nli_statement(self):
        nli_s = _declarative_template_lookup("X", self.relation.split(",")[0], self.relation.split(",")[1])[self.answer == 'no']
        nli_s = " ".join(nli_s.split(" ")[1:])
        return nli_s

    def get_question(self):
        return _yesno_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1], self.answer)[0]

    def set_answer(self, ans):
        self.answer = ans

    def get_answer(self):
        assert(self.answer is not None)
        return _yesno_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1], self.answer)[1]

    def get_qa_pair(self):
        assert(self.answer is not None)
        return _yesno_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1], self.answer)

    def get_relation(self):
        return self.relation

    def get_entity(self):
        return self.entity

    def get_answer(self):
        return self.answer

    def set_confidence(self, conf):
        self.confidence = conf

    def get_confidence(self):
        return self.confidence

    def flip(self, confidence_fn):
        """
        confidence_fn is a lambda function
        that decides what the confidence score
        should be for a flipped answer
        """
        assert self.answer is not None
        assert self.confidence is not None
        self.answer = 'yes' if self.answer == 'no' else 'no'
        self.confidence = confidence_fn()


if __name__ == '__main__':
    import utils
    import json
    a = MemoryEntry(entity='carp', relation='HasA,no legs',
                    confidence='yes', answer=None)
    a.get_question()
    entries = data = utils.json_to_tuples(
        json.load(open("data/silver_facts.json")))
    entries = [MemoryEntry(e[0], e[1], e[2]) for e in entries]
    for e in entries:
        e.get_pos_statement()
        e.get_neg_statement()
        e.get_question()
        e.get_answer()
