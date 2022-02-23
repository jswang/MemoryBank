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
    confidence: float = None
    answer: str = None

    def get_pos_statement(self):
        return _declarative_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1])[0]

    def get_neg_statement(self):
        return _declarative_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1])[1]

    def get_declarative_statement(self):
        return _declarative_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1])[self.answer == 'no']

    def get_question(self):
        return _yesno_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1], self.answer)[0]

    def set_answer(self, ans):
        self.answer = ans

    def get_answer(self):
        return _yesno_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1], self.answer)[1]

    def get_qa_pair(self):
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

    def flip(self):
        self.answer = 'yes' if self.answer == 'no' else 'no'
        self.confidence = 1.5 - self.confidence


if __name__ == '__main__':
    m = MemoryEntry('american bison', 'IsA,mammal', 'yes')
    print(m.get_pos_statement())
    print(m.get_neg_statement())
    print(m.get_question())
    print(m.get_answer())
