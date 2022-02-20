from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, logging
import torch
import faiss
from sentence_transformers import SentenceTransformer
from models import standard_config

# only log errors
logging.set_verbosity_error()


class MemoryBank:
    def __init__(self, config=standard_config):
        """
        Create a MemoryBank model based on configuration.
        """
        self.device = config["device"]
        # TODO put back
        # Sentence tokenizer and NLI model which outputs relation of premise and hypothesis
        # self.nli_tokenizer = AutoTokenizer.from_pretrained(config["nli_model"])
        # self.nli_model = AutoModelForSequenceClassification.from_pretrained(
        #     config["nli_model"])
        # self.nli_model.to(self.device)

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

        # Model that goes from sentence to sentence representation
        # TODO put back
        # self.sent_model = SentenceTransformer(config["sentence_model"])
        # self.sent_model.to(self.device)

    def ask_questions(self, questions):
        """
        Ask the Macaw model a batch of yes or no questions.
        Returns "yes" or "no"
        """
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

    def encode_sents(self, sents):
        # First encode sentences
        # Then find similarity
        s_embed = self.sent_model.encode(sents)
        return s_embed

    def build_index(self, s_embed):
        # print(index.is_trained)
        faiss.normalize_L2(x=s_embed)
        self.index.add(s_embed)

    def retrieve_from_index(self, s_new):
        """
          Retrieving top n_semantic sentences
        """
        faiss.normalize_L2(x=s_new)
        lims, D, I = self.index.range_search(x=s_new, thresh=self.threshold)
        # I is the indices
        """
          TODO: Do we want any additional criteria
        """
        retrieved = []
        for i in range(I.shape[-1]):
            retrieved.append(self.translate_qa(self.mem_bank[I[i]]))
        return retrieved, I

    def flip_pair(self, qa_pair):
        if qa_pair[1] == "yes":
            return (qa_pair[0], "no")
        else:
            return (qa_pair[0], "yes")

    def flip_or_keep(self, retrieved, inds, qa_pair):
        relations = []
        qa_sent = self.translate_qa(qa_pair)
        for i in range(len(retrieved)):
            relations.append(self.compute_relation(retrieved[i], qa_sent))
        # TODO: Decide weighting
        # TODO: Flip the beliefs using flip_pair function
        flip_input = False
        if flip_input:
            return self.flip_pair(qa_pair)
        return qa_pair

    def translate_qa(self, qa_pair):
        return " ".join(qa_pair)

    def add_to_bank(self, qa_pair):
        # TODO: Future -> Add declarative statement
        # self.mem_bank.append(declare_change(qa_pair))
        # Appending only the QA pair to make flipping easier
        # TODO: Add the flip
        if self.flip:
            new_entry = self.translate_qa(qa_pair)
            s_embed = self.encode_sents([new_entry])
            retrieved, inds = self.retrieve_from_index(s_embed)
            qa_pair = self.flip_or_keep(retrieved, inds, qa_pair)
        self.mem_bank.append(qa_pair)
        new_entry = self.translate_qa(qa_pair)
        s_embed = self.encode_sents([new_entry])
        if self.index is None:
            d = s_embed.shape[1]  # dimension
            self.index = faiss.IndexFlatIP(d)
        # build index to add to index
        self.build_index(s_embed)

    def compute_relation(self, premise, hypothesis):
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
    ans = mem_bank.ask_questions("Is an owl a mammmal?")
    print(f"Model responded:{ans}")

    # Add facts to the memory bank:
    qa_1 = ("Is an owl a mammal?", "yes")
    qa_2 = ("Does a owl have a vertebrate?", "yes")
    mem_bank.add_to_bank(qa_1)

    # Retreive sentences
    s2 = mem_bank.encode_sents([mem_bank.translate_qa(qa_2)])
    print(mem_bank.retrieve_from_index(s2))
