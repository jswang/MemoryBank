from dataclasses import dataclass
from utils import _declarative_template_lookup, _yesno_template_lookup

@dataclass
class MemoryEntry:
    entity: str
    relation: str
    answer: str

    def get_pos_statement(self):
        return _declarative_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1])[0]

    def get_neg_statement(self):
        return _declarative_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1])[1]
    
    def get_declarative_statement(self):
        return _declarative_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1])[self.answer == 'no']
    
    def get_question(self):
        return _yesno_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1], self.answer)[0]
    
    def get_answer(self):
        return _yesno_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1], self.answer)[1]
    
    def get_qa_pair(self):
        return _yesno_template_lookup(self.entity, self.relation.split(",")[0], self.relation.split(",")[1], self.answer)

    def get_relation(self):
        return self.relation
    
    def get_entity(self):
        return self.entity

    def flip(self):
        self.answer = 'yes' if self.answer == 'no' else 'no'


if __name__ == '__main__':
    m = MemoryEntry('american bison', 'IsA,mammal', 'yes')
    print(m.get_pos_statement())
    print(m.get_neg_statement())
    print(m.get_question())
    print(m.get_answer())
