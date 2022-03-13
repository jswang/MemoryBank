import random
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, logging
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import models
from models import baseline_config
import json
from MemoryEntry import MemoryEntry

from z3 import *

# only log errors
logging.set_verbosity_error()

# constants
MAX_INPUT_CHAR_LENGTH = 256


class MemoryBank:
    def __init__(self, config=baseline_config):
        """
        Create a MemoryBank model based on configuration.
        """
        self.name = config["name"]
        self.device = config["device"]
        self.config = config

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

        # Plaintext beliefs: question answer pairs
        self.mem_bank = []

        # The type of feedback we will be creating
        if config["feedback_type"] == "topic":
            self.entities_dict = {k: {} for k in json.load(
                open("data/silver_facts.json")).keys()}
        elif config["feedback_type"] == 'relevant':
            self.entities_dict = {}
            assert "max_retrieved" in config, "Missing max retrieval number in config for relevant feedback"
        else:
            self.entities_dict = {}
        self.n_feedback = 3

        # Model that goes from sentence to sentence representation
        self.sent_model = SentenceTransformer(config["sentence_model"])
        self.sent_model.to(self.device)
        # Embedded sentence index, allows us to look up quickly
        self.index = faiss.IndexFlatIP(
            self.sent_model.get_sentence_embedding_dimension())

        if self.config.get("max_sat") is True:
            self.sat_solver = Optimize()
            self.lmbda = self.config.get("max_sat_lmbda")
        else:
            self.sat_solver = None
            self.lmbda = -1

    def find_same_topic(self, questions: List[MemoryEntry]) -> List[str]:
        """
        Given a list of questions, return all of the related topic sentences in a single string
        """
        result_topics = []
        for q in questions:
            entity_select = self.entities_dict[q.get_entity()]
            if len(entity_select) > self.n_feedback:
                topics = random.choices(
                    list(entity_select.values()), k=self.n_feedback)
            else:
                topics = list(entity_select.values())
            result_topics.append(
                " ".join([t.get_declarative_statement() for t in topics]))
        return result_topics

    def generate_feedback(self, questions: List[MemoryEntry]) -> List[str]:
        """
        Given a list of questions, retrieve semantically similar sentences for context
        """
        if self.config["feedback_type"] == "relevant":
            R, I = self.retrieve_from_index(questions)
            contexts = []
            for r in R:
                contexts.append(
                    " ".join([e.get_declarative_statement() for e in r[:self.n_feedback]]))
        else:
            # List of strings, each string corresponding to self.n_feedback relevant beliefs
            contexts = self.find_same_topic(questions)
        return contexts

    def ask_questions(self, questions: List[str], context: List[Tuple[str, str]]) -> Tuple[List[str], List[float]]:
        """
        Ask the Macaw model a batch of yes or no questions.
        Returns "yes" or "no" and a confidence score
        """

        # Insert feedback if necesasry
        if len(context) == len(questions):
            input_string = [
                f"$answer$ ; $mcoptions$ = (A) yes (B) no ; $context$ = {context[i]} ; $question$ = {questions[i]}"
                for i in range(len(context))]
        else:
            input_string = [
                f"$answer$ ; $mcoptions$ = (A) yes (B) no ; $question$ = {q}" for q in questions]
        # Tokenize questions
        inputs = self.qa_tokenizer(
            input_string, padding=True, truncation=True, return_tensors="pt", max_length=MAX_INPUT_CHAR_LENGTH)
        input_ids = inputs.input_ids.to(self.device)
        input_attention_mask = inputs.attention_mask.to(self.device)
        # Ask the questions, include a label to gather confidence
        labels = self.qa_tokenizer(
            "$answer$ = yes", return_tensors="pt", max_length=MAX_INPUT_CHAR_LENGTH).input_ids.to(self.device)
        labels = torch.tile(labels, (len(questions), 1))
        # Calculate probability of yes answer
        # model forward pass docs: https://huggingface.co/docs/transformers/v4.17.0/en/model_doc/t5#transformers.T5ForConditionalGeneration
        res = self.qa_model(input_ids, input_attention_mask, labels=labels)
        res_softmax = torch.softmax(res.logits, dim=2)
        raw_probs = torch.squeeze(torch.gather(
            res_softmax, 2, torch.unsqueeze(labels, 2)))
        output_prob = torch.prod(raw_probs, 1)
        answers = []
        probs = []
        for prob in output_prob:
            prob = prob.item()
            if prob >= 0.5:
                answers += ["yes"]
                probs += [prob]
            else:
                answers += ["no"]
                probs += [1-prob]

        return answers, probs

    def add_to_index(self, s_embed: np.array):
        """
        Add sentence embeddings to the index
        """
        s_embed = s_embed.cpu().detach().numpy().astype("float32")
        s_embed /= np.expand_dims(np.linalg.norm(s_embed, axis=-1), 1)
        self.index.add(s_embed)

    def retrieve_from_index(self, sentences: List[MemoryEntry]) -> Tuple[List[List[MemoryEntry]], List[np.array]]:
        """
        Retrieve sentence embeddings and sentences from the index.
        s_new is a Tensor, first dimension is batch
        """
        s_embed = self.encode_sent(sentences)
        s_embed = s_embed.cpu().detach().numpy().astype("float32")
        s_embed /= np.expand_dims(np.linalg.norm(s_embed, axis=-1), 1)
        lims, D, I = self.index.range_search(
            x=s_embed, thresh=self.config["sentence_similarity_threshold"])

        corresponding_indices = [I[lims[i]:lims[i+1]]
                                 for i in range(len(lims) - 1)]
        corresponding_scores = [D[lims[i]:lims[i+1]]
                                for i in range(len(lims) - 1)]

        retrieved = []
        indices = []
        for idx_list, score_list, sentence in zip(corresponding_indices, corresponding_scores, sentences):
            # Get items in order of most similar
            top_indices = np.argsort(score_list)[::-1]
            # Take top 30 most similar items
            temp_indices = []
            temp_retrieved = []
            for meta_index in top_indices:
                bank_idx = idx_list[meta_index]
                e = self.mem_bank[bank_idx]
                if e.get_entity() == sentence.get_entity():
                    temp_retrieved.append(e)
                    temp_indices.append(bank_idx)
            if self.config["feedback_type"] == 'relevant':
                temp_retrieved = temp_retrieved[:min(
                    len(temp_retrieved), self.config["max_retrieved"])]
                temp_indices = temp_indices[:min(
                    len(temp_indices), self.config["max_retrieved"])]
            retrieved.append(temp_retrieved)
            indices.append(temp_indices)

        return retrieved, indices

    """
        SAT MemoryBank Functionalities
    """

    def sat_check_and_flip(self, premise_indices):
        """
        Go through the premises in the scope, check confidence levels and decide whether to flip
        """
        mem_flips = 0
        for idx in premise_indices:
            print("Flipping Belief old:", self.mem_bank[idx].get_declarative_statement(),
                  self.mem_bank[idx].get_confidence())
            self.mem_bank[idx].flip(self.config["default_flipped_confidence"])
            if self.sat_solver is not None:
                expr = Bool(self.mem_bank[idx].get_pos_statement())
                self.sat_solver.add_soft(expr == True, self.mem_bank[idx].get_confidence() * self.lmbda if self.mem_bank[idx].get_answer() == "yes"
                                         else (1 - self.mem_bank[idx].get_confidence()) * self.lmbda)
                self.sat_solver.add_soft(expr == False, self.mem_bank[idx].get_confidence() * self.lmbda if self.mem_bank[
                    idx].get_answer() == "no"
                                         else (1 - self.mem_bank[idx].get_confidence()) * self.lmbda)
            if self.config["feedback_type"] == "topic":
                # add to entities dict
                self.entities_dict[self.mem_bank[idx].get_entity()].update(
                    {self.mem_bank[idx].get_relation(): self.mem_bank[idx]})
            mem_flips += 1
        return mem_flips

    def sat_flip_or_keep(self, premises: List[MemoryEntry], premises_indices, hypothesis: MemoryEntry, hypothesis_ind) -> MemoryEntry:
        """
        Decide whether or not to flip the hypothesis given relevant MemoryEntries and their indices.
        """
        if premises == []:
            return hypothesis

        probs = np.array([self.get_relation(
            p.get_declarative_statement(), hypothesis.get_declarative_statement()) for p in premises])

        hyp_exp = Bool(hypothesis.get_pos_statement())
        if hypothesis.get_answer() == "yes":
            self.sat_solver.add_soft(
                hyp_exp == True, hypothesis.get_confidence() * self.lmbda)
            self.sat_solver.add_soft(
                hyp_exp == False, (1 - hypothesis.get_confidence()) * self.lmbda)
        else:
            self.sat_solver.add_soft(
                hyp_exp == False, hypothesis.get_confidence() * self.lmbda)
            self.sat_solver.add_soft(
                hyp_exp == True, (1 - hypothesis.get_confidence()) * self.lmbda)

        for i in range(len(premises)):
            prem = Bool(premises[i].get_pos_statement())
            if probs[i, 0] > probs[i, 1] and probs[i, 0] > probs[i, 2]:
                if premises[i].get_answer() == "yes":
                    if hypothesis.get_answer() == "yes":
                        self.sat_solver.add_soft(
                            Or(Not(prem), hyp_exp), float(probs[i, 0]))
                    else:
                        self.sat_solver.add_soft(
                            Or(Not(prem), Not(hyp_exp)), float(probs[i, 0]))
                else:
                    if hypothesis.get_answer() == "yes":
                        self.sat_solver.add_soft(
                            Or(prem, hyp_exp), float(probs[i, 0]))
                    else:
                        self.sat_solver.add_soft(
                            Or(prem, Not(hyp_exp)), float(probs[i, 0]))

            elif probs[i, 2] > probs[i, 0] and probs[i, 2] > probs[i, 1]:
                if premises[i].get_answer() == "yes":
                    if hypothesis.get_answer() == "yes":
                        self.sat_solver.add_soft(
                            And(prem, Not(hyp_exp)), float(probs[i, 1]))
                    else:
                        self.sat_solver.add_soft(
                            And(prem, hyp_exp), float(probs[i, 1]))
                else:
                    if hypothesis.get_answer() == "yes":
                        self.sat_solver.add_soft(
                            And(Not(prem), Not(hyp_exp)), float(probs[i, 1]))
                    else:
                        self.sat_solver.add_soft(
                            And(Not(prem), hyp_exp), float(probs[i, 1]))
        return hypothesis

    def solve_and_flip(self):
        self.sat_solver.check()
        opt_model = self.sat_solver.model()
        inds_to_flip = []
        for i in range(len(self.mem_bank)):
            new_exp = Bool(self.mem_bank[i].get_pos_statement())
            assignment = int(bool(opt_model[new_exp]))
            if assignment != (self.mem_bank[i].get_answer() == "yes"):
                inds_to_flip.append(i)
        self.sat_check_and_flip(inds_to_flip)

    """
        Simple Flipping MemoryBank functionalities
    """

    def check_and_flip(self, premises, premise_indices, hypothesis):
        """
        Go through the premises in the scope, check confidence levels and decide whether to flip
        """
        mem_flips = 0
        hypothesis_score = hypothesis.get_confidence()
        for (idx, p) in zip(premise_indices, premises):
            if p.confidence + self.config["flip_premise_threshold"] < hypothesis_score:
                self.mem_bank[idx].flip(
                    self.config["default_flipped_confidence"])
                # print(f"flipping premise to: {self.mem_bank[idx].get_declarative_statement()}, hypothesis: {hypothesis.get_declarative_statement()}")
                mem_flips += 1
                if self.config["feedback_type"] == "topic":
                    # add to entities dict
                    self.entities_dict[self.mem_bank[idx].get_entity()].update(
                        {self.mem_bank[idx].get_relation(): self.mem_bank[idx]})
        return mem_flips

    def flip_or_keep(self, premises: List[MemoryEntry], premises_indices, hypothesis: MemoryEntry) -> MemoryEntry:
        """
        Decide whether or not to flip the hypothesis given relevant MemoryEntries and their indices.
        """
        if premises == []:
            return hypothesis

        # Get the relation between each premise and this hypothesis
        probs = np.array([self.get_relation(
            p.get_declarative_statement(), hypothesis.get_declarative_statement()) for p in premises])

        # Just count the max score of entail, neutral, or contra towards the score
        if "scoring" not in self.config or self.config["scoring"] == "max_only":
            n_entail = np.sum(probs[np.argmax(probs, axis=1) == 0][:, 0])
            n_contra = np.sum(probs[np.argmax(probs, axis=1) == 2][:, 2])
        # Ignore neutral, just sum entail and contra
        elif self.config["scoring"] == "entail_and_contra":
            probs[:, 1] = 0
            n_entail = np.sum(probs[:, 0])
            n_contra = np.sum(probs[:, 2])

        mem_flips = 0
        possible_mem_flips = len(premises)
        hyp_flip = 0

        # if we have more contradictions than we do entailments, we should flip
        # either the hypothesis or one or more premises
        entail_threshold = 1.0 if 'entail_threshold' not in self.config else self.config[
            'entail_threshold']
        if n_entail * entail_threshold < n_contra:
            hypothesis_score = hypothesis.get_confidence()
            contra_premises_ind = []
            contra_premises = []
            entail_premises_ind = []
            entail_premises = []
            for i in range(len(premises)):
                if probs[i, 0] > probs[i, 2]:
                    entail_premises_ind.append(premises_indices[i])
                    entail_premises.append(premises[i])
                elif probs[i, 2] > probs[i, 0]:
                    contra_premises_ind.append(premises_indices[i])
                    contra_premises.append(premises[i])
            premise_scores = np.array([r.get_confidence()
                                      for r in contra_premises])

            hypothesis_votes = np.sum(
                hypothesis_score > premise_scores)
            premise_votes = np.sum(hypothesis_score < premise_scores)

            # if our QA model is more confident about the hypothesis,
            # the hypothesis is good and we should flip some premises
            if hypothesis_votes > premise_votes:
                # flip premises whose QA scores are lower than hypothesis score
                mem_flips += self.check_and_flip(contra_premises,
                                                 contra_premises_ind, hypothesis)

            # if our QA model is more confident about premises,
            # the hypothesis isn't good and we should flip it
            else:
                # And flip the entailment premises
                if self.config["flip_entailing_premises"]:
                    mem_flips += self.check_and_flip(entail_premises,
                                                     entail_premises_ind, hypothesis)
                hypothesis.flip(self.config["default_flipped_confidence"])
                # print(f"flipping hypothesis to {hypothesis.get_declarative_statement()}")
                hyp_flip += 1
        return hypothesis

    def encode_sent(self, sentences: List[MemoryEntry]):
        return self.sent_model.encode([s.get_pos_statement() for s in sentences], device=self.device, convert_to_tensor=True)

    def add_to_bank(self, new_entries: List[MemoryEntry]):
        """ Usage: add_to_bank('owl', 'HasA,Vertebrate', 'yes')"""

        self.mem_bank += new_entries

        # Embed and add to index
        s_embed = self.encode_sent(new_entries)
        self.add_to_index(s_embed)
        if self.config["feedback_type"] == "topic":
            # add to entities dict
            for q in new_entries:
                self.entities_dict[q.get_entity()].update(
                    {q.get_relation(): q})

    def clear_bank(self):
        """
        Clears all entries from memory bank
        """
        self.mem_bank = []
        self.index.reset()

    def get_relation(self, premise: str, hypothesis: str):
        """
        Given premise and hypothesis, output entailment/neutral/contradiction
        """
        tokenized_input_seq_pair = self.nli_tokenizer.encode_plus(premise, hypothesis,
                                                                  max_length=MAX_INPUT_CHAR_LENGTH,
                                                                  return_token_type_ids=True, truncation=True)
        input_ids = torch.Tensor(
            tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(self.device)
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = torch.Tensor(
            tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(self.device)
        attention_mask = torch.Tensor(
            tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.nli_model(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     labels=None)

        predicted_probability = torch.softmax(
            outputs.logits, dim=-1).squeeze().detach().cpu().numpy()
        if self.nli_model.name_or_path == 'roberta-large-mnli':
            predicted_probability = np.flip(predicted_probability)
        return predicted_probability

    def forward(self, inputs: List[Tuple[str, str, str]], run_solver=False):
        """
        Forward pass of the model on a batch of triplets
        Arguments:
        `inputs` - batch of inputs to parse, expect of tuple of (entity, relation, answer)
        """
        # triplet to question
        questions = [MemoryEntry(i[0], i[1]) for i in inputs]
        context = []

        # Generate context if necessary
        if self.config["feedback_type"] is not None:
            context = self.generate_feedback(questions)
        # Ask your question
        answers, probs = self.ask_questions(
            [q.get_question() for q in questions], context)
        for i in range(len(questions)):
            questions[i].set_answer(answers[i])
            questions[i].set_confidence(probs[i])
        statements = questions

        # Run the SAT Solver
        if self.sat_solver is not None:
            self.add_to_bank(statements)
            # Check against existing constraints to flip as necessary
            R, I = self.retrieve_from_index(statements)
            statements = [self.sat_flip_or_keep(
                r, i, s, h_id + len(self.mem_bank) - len(statements)) for h_id, (r, i, s) in
                enumerate(zip(R, I, statements))]
            if run_solver:
                self.solve_and_flip()
            # Return the answers for this batch
            return [a.get_answer() for a in self.mem_bank[len(self.mem_bank) - len(statements):]]

        # Check against existing constraints to flip as necessary
        if self.config["enable_flip"]:
            R, I = self.retrieve_from_index(statements)
            statements = [self.flip_or_keep(
                r, i, s) for r, i, s in zip(R, I, statements)]

        # Add flipped statements to the bank.
        self.add_to_bank(statements)

        # Return the answers for this batch
        return [a.get_answer() for a in statements]
