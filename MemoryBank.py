import random
from typing import Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, logging
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from models import standard_config
import json
from MemoryEntry import MemoryEntry

# only log errors
logging.set_verbosity_error()


class MemoryBank:
    def __init__(self, config=standard_config):
        """
        Create a MemoryBank model based on configuration.
        """
        self.device = config["device"]

        # Sentence tokenizer and NLI model which outputs relation of premise and hypothesis
        self.nli_tokenizer = AutoTokenizer.from_pretrained(config["nli_model"])
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            config["nli_model"])
        self.nli_model.to(self.device)

        # Question answering model and tokenizer
        self.qa_tokenizer = AutoTokenizer.from_pretrained(config["qa_model"])
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(
            config["qa_model"])
        self.qa_model.to(self.device)

        # Number of semantically similar constraints to compare against
        self.n_semantic = config["n_semantic"]
        # Plaintext beliefs: question answer pairs
        self.mem_bank = []
        # Embedded sentence index, allows us to look up quickly
        self.index = None
        # Similarity threshold for index lookup
        self.threshold = config["sentence_similarity_threshold"]
        # Maximum number of characters in input sequence
        self.max_length = config["max_input_char_length"]
        # Whether we use the flipping functionality
        self.flip = config["flip_constraints"]
        # The type of feedback we will be creating
        self.feedback = config["feedback_type"]
        if self.feedback == "topic":
            self.entities_dict = {k: [] for k in json.load(open("silver_facts.json")).keys()}
        else:
            self.entities_dict = dict()
        self.n_feedback = 3

        # Model that goes from sentence to sentence representation
        self.sent_model = SentenceTransformer(config["sentence_model"])
        self.sent_model.to(self.device)
        self.qa_dec_dict = None

    def find_same_topic(self, question: str) -> list[str]:
        ret_qs = []
        for e in self.entities_dict:
            if e in question:
                ret_qs = random.choices(self.entities_dict[e], k=self.n_feedback)
                self.entities_dict[e].append(question)
        return ret_qs

    def generate_feedback(self, questions: list[str]) -> list[str]:
        cqs = []
        for q in questions:
            if self.feedback == "relevant":
                s_embed = self.encode_sents([q])
                retrieved, I = self.retrieve_from_index(s_embed)
                contxt = " ".join([retrieved[i].get_declarative_statement() for i in I[:self.n_feedback]])
            else:
                contxt = " ".join(self.find_same_topic(q))
            cqs.append((contxt, q))
        return cqs

    def ask_questions(self, questions: list[str]) -> list[str]:
        """
        Ask the Macaw model a batch of yes or no questions.
        Returns "yes" or "no"
        """
        if self.feedback is not None:
            c_q_pairs = self.generate_feedback(questions)
            input_string = [
                f"$answer$ ; $mcoptions$ = (A) yes (B) no; $context$ = {c} ; $question$ = {q}"
                for c, q in c_q_pairs]
        else:
            input_string = [
                f"$answer$ ; $mcoptions$ = (A) yes (B) no; $question$ = {q}" for q in questions]
        input_ids = self.qa_tokenizer(
            input_string, padding=True, truncation=True, return_tensors="pt")
        input_ids.to(self.device)
        encoded_output = self.qa_model.generate(
            input_ids["input_ids"], max_length=self.max_length)
        ans = self.qa_tokenizer.batch_decode(
            encoded_output, skip_special_tokens=True)
        return [a.split('$answer$ = ')[1] for a in ans]

    def encode_sents(self, sents: list[str]) -> np.array:
        # First encode sentences
        # Then find similarity
        s_embed = self.sent_model.encode(sents)
        return s_embed

    def build_index(self, s_embed: np.array):
        # print(index.is_trained)
        # faiss.normalize_L2(x=s_embed)
        s_embed /= np.linalg.norm(s_embed)
        self.index.add(s_embed)

    def retrieve_from_index(self, s_new) -> Tuple[list[MemoryEntry], np.array]:
        """
          Retrieving top n_semantic sentences
        """
        # faiss.normalize_L2(x=s_new)
        if self.index is None:
            return [], np.array([])
        s_new /= np.linalg.norm(s_new,)
        lims, D, I = self.index.range_search(x=s_new, thresh=self.threshold)
        # I is the indices
        print(lims)
        print(D)
        """
          TODO: Do we want any additional criteria
        """
        retrieved = []
        for i in range(I.shape[-1]):
            e = self.mem_bank[I[i]]
            retrieved.append(e)
        return retrieved, I

    def flip_or_keep(self, retrieved: list[MemoryEntry], inds, entry: MemoryEntry) -> MemoryEntry:
        relations = []
        statement = entry.get_declarative_statement()
        for i in range(len(retrieved)):
            relations.append(self.compute_relation(retrieved[i].get_declarative_statement(), statement))
        # TODO: Decide weighting
        # TODO: Flip the beliefs using flip_pair function
        flip_input = False
        if flip_input:
            entry.flip()
        return entry

    def translate_qa(self, qa_pair):
        return " ".join(qa_pair)

    def add_to_bank(self, entity: str, relation: str, answer: str):
        ''' Usage: add_to_bank('owl', 'HasA,Vertebrate', 'yes')'''
        # TODO: Future -> Add declarative statement
        # self.mem_bank.append(declare_change(qa_pair))
        # Appending only the QA pair to make flipping easier
        # TODO: Add the flip
        new_entry = MemoryEntry(entity, relation, answer)
        if self.flip:
            # new_entry = self.translate_qa(qa_pair)
            s_embed = self.encode_sents([new_entry.get_declarative_statement()])
            retrieved, inds = self.retrieve_from_index(s_embed)
            new_entry = self.flip_or_keep(retrieved, inds, new_entry)
        self.mem_bank.append(new_entry)

        # embed again in case statement was flipped
        s_embed = self.encode_sents([new_entry.get_declarative_statement()])
        if self.index is None:
            d = s_embed.shape[1]  # dimension
            self.index = faiss.IndexFlatIP(d)
        # build index to add to index
        self.build_index(s_embed)

    def compute_relation(self, premise: str, hypothesis: str):
        tokenized_input_seq_pair = self.tokenizer.encode_plus(premise, hypothesis,
                                                              max_length=self.max_length,
                                                              return_token_type_ids=True, truncation=True)
        input_ids = torch.Tensor(
            tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = torch.Tensor(
            tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
        attention_mask = torch.Tensor(
            tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
        outputs = self.nli_model(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 labels=None)
        # print(outputs)
        predicted_probability = torch.softmax(outputs[0], dim=1)[
            0].tolist()  # batch_size only one
        return predicted_probability


if __name__ == '__main__':
    # Examples of how to instantiate and use the memory bank
    mem_bank = MemoryBank()

    # Ask a question
    # ans = mem_bank.ask_questions("Is an owl a mammmal?")
    # print(f"Model responded:{ans}")

    e1 = ('owl', 'IsA,Mammal', 'yes')
    e2 = MemoryEntry('owl', 'HasA,vertebrate', 'yes')
    mem_bank.add_to_bank(*e1)

    # Retreive sentences
    s2 = mem_bank.encode_sents([e2.get_declarative_statement()])
    print(mem_bank.retrieve_from_index(s2))
