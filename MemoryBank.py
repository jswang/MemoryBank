import random
from typing import Tuple, List

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
        self.alpha = config['flip_alpha']
        self.beta = config['flip_beta']
        self.gamme = config['flip_gamme']
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

        # Similarity threshold for index lookup
        self.threshold = config["sentence_similarity_threshold"]
        # Maximum number of characters in input sequence
        self.max_length = config["max_input_char_length"]
        # Whether we use the flipping functionality
        self.enable_flip = config["enable_flip"]
        # The type of feedback we will be creating
        self.feedback = config["feedback_type"]
        if self.feedback == "topic":
            self.entities_dict = {k: [] for k in json.load(
                open("silver_facts.json")).keys()}
        else:
            self.entities_dict = dict()
        self.n_feedback = 3

        # Model that goes from sentence to sentence representation
        self.sent_model = SentenceTransformer(config["sentence_model"])
        self.sent_model.to(self.device)
        self.qa_dec_dict = None
        # Embedded sentence index, allows us to look up quickly
        self.index = faiss.IndexFlatIP(
            self.sent_model.get_sentence_embedding_dimension())

    def find_same_topic(self, questions: List[str]) -> List[str]:
        """
        Given a list of questions, return all of the related topic sentences in a single string
        """
        result_topics = []
        for q in questions:
            for e in self.entities_dict:
                if e in q:
                    topics = random.choices(
                        self.entities_dict[e], k=self.n_feedback)
                    result_topics += [" ".join(topics)]
                    self.entities_dict[e].append(q)
        return result_topics

    def generate_feedback(self, questions: List[str]) -> List[str]:
        """
        Given a list of questions, retrieve semantically similar sentences for context
        """
        context = []
        if self.feedback == "relevant":
            s_embed = self.encode_sent(questions)
            retrieved, I = self.retrieve_from_index(
                s_embed)  # TODO here
            contxt = " ".join(
                [retrieved[i].get_declarative_statement() for i in I[:self.n_feedback]])
        else:
            contxt = self.find_same_topic(questions)
            context += [contxt]
        return context

    def ask_questions(self, questions: List[str], context: List[Tuple[str, str]]) -> List[str]:
        """
        Ask the Macaw model a batch of yes or no questions.
        Returns "yes" or "no"
        """
        # Insert feedback if necesasry
        if len(context) == len(questions):
            input_string = [
                f"$answer$ ; $mcoptions$ = (A) yes (B) no; $context$ = {context[i]} ; $question$ = {questions[i]}"
                for i in range(len(context))]
        else:
            input_string = [
                f"$answer$ ; $mcoptions$ = (A) yes (B) no; $question$ = {q}" for q in questions]
        # Tokenize questions
        input_ids = self.qa_tokenizer(
            input_string, padding=True, truncation=True, return_tensors="pt")
        input_ids.to(self.device)
        # Ask the questions
        encoded_output = self.qa_model.generate(
            input_ids["input_ids"], max_length=self.max_length)
        ans = self.qa_tokenizer.batch_decode(
            encoded_output, skip_special_tokens=True)
        return [a.split('$answer$ = ')[1] for a in ans]

    def build_index(self, s_embed: np.array):
        s_embed /= torch.unsqueeze(torch.norm(s_embed, dim=1), 1)
        self.index.add(s_embed.cpu().detach().numpy())

    def retrieve_from_index(self, s_new) -> Tuple[List[MemoryEntry], np.array]:
        """
        Retrieving top n_semantic sentences from MemoryBank.
        s_new is a stacked Tensor, first dimension is batch
        """
        # faiss.normalize_L2(x=s_new)
        s_new /= np.linalg.norm(s_new)
        # I is the indices
        _, _, I = self.index.range_search(x=s_new, thresh=self.threshold)

        # TODO: Do we want any additional criteria?
        # TODO this is definitely broken for s_new being two dimensional, Julie to fix
        retrieved = []
        for i in range(I.shape[-1]):
            e = self.mem_bank[I[i]]
            retrieved.append(e)
        return retrieved, I

    def flip_or_keep(self, retrieved: List[MemoryEntry], inds, entry: MemoryEntry) -> MemoryEntry:
        hypothesis = entry.get_declarative_statement()

        hypothesis_score = entry.get_confidence()
        premise_scores = [r.get_confidence() for r in retrieved]  # for marty

        n_entail = np.count_nonzero(
            np.array([np.argmax(p) == 0 for p in premise_scores]))
        n_contra = np.count_nonzero(
            np.array([np.argmax(p) == 2 for p in premise_scores]))

        # for premise, score in zip(retrieved, premise_scores):
        #     p_ent, p_neut, p_contra = self.get_relation(
        #         premise.get_declarative_statement(), hypothesis)
        #     a = np.argmax([p_ent, p_neut, p_contra])

        #     if a == 2:
        #         # contradiction
        #         votes_keep_hypothesis = sum(
        #             [hypothesis_score > p for p in premise_scores])
        #         votes_keep_premise = sum(
        #             [hypothesis_score < p for p in premise_scores])

        # if we have many entailments, the hypothesis is good and we should flip some premises
        if n_entail > n_contra:
            # flip premises whose QA scores are lower than hypothesis score
            for idx in inds:
                if retrieved[idx].confidence < entry.confidence:
                    pass
                    # retrieved[idx] =
                    # TODO:
        else:
            entry.flip()

        flip_input = True
        if flip_input:
            entry.flip()
        return entry

    def encode_sent(self, sentences):
        return self.sent_model.encode(sentences, convert_to_tensor=True)

    def add_to_bank(self, new_entries: List[MemoryEntry]):
        ''' Usage: add_to_bank('owl', 'HasA,Vertebrate', 'yes')'''
        # TODO: Future -> Add declarative statement
        # self.mem_bank.append(declare_change(qa_pair))
        # Appending only the QA pair to make flipping easier
        if self.enable_flip:
            # new_entry = self.translate_qa(qa_pair)
            s_embed = self.encode_sent(
                [e.get_declarative_statement() for e in new_entries])
            retrieved, inds = self.retrieve_from_index(s_embed)
            new_entries = [self.flip_or_keep(
                retrieved, inds, new_entry) for new_entry in new_entries]
        self.mem_bank.append(new_entries)

        # embed again in case statement was flipped
        s_embed = self.encode_sent(
            [e.get_declarative_statement() for e in new_entries])

        # build index to add to index
        self.build_index(s_embed)

    def get_relation(self, premise: str, hypothesis: str):
        """
        Given premise and hypothesis, output entailment/neutral/contradiction
        """
        tokenized_input_seq_pair = self.nli_tokenizer.encode_plus(premise, hypothesis,
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

    def forward(self, inputs: Tuple[str, str, str]):
        """
        Forward pass of the model on a batch of triplets
        Arguments:
        `inputs` - batch of inputs to parse, expect of tuple of (entity, relation, answer)
        """
        # triplet to question
        questions = [MemoryEntry(i[0], i[1]) for i in inputs]
        context = []
        # Generate context if necessary
        if self.feedback is not None:
            context = self.generate_feedback(questions)
        # Ask your question
        answers = self.ask_questions(
            [q.get_question() for q in questions], context)
        for (i, ans) in enumerate(answers):
            questions[i].set_answer(ans)

        # Get all the premeses
        self.add_to_bank(questions)
        # TODO return the new answers
