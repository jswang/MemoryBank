from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import faiss
from sentence_transformers import SentenceTransformer


class MemoryBank:
    def __init__(self, tokenizer, nli_model, qa_model, n_semantic, threshold=0.6, flip=False):

        # Sentence tokenizer
        self.tokenizer = tokenizer
        # Outputs relation of premise and hypothesis
        self.nli_model = nli_model
        # Question answering model
        self.qa_model = qa_model
        # Number of semantically similar constraints to compare against
        self.n_semantic = n_semantic
        # Plaintext beliefs: question answer pairs
        self.mem_bank = []
        # Embedded sentence index, allows us to look up quickly
        self.index = None
        # Similarity threshold for index lookup
        self.threshold = threshold
        # Maximum number of characters in input sequence
        self.max_length = 256
        # Whether we use the flipping functionality
        self.flip = flip

        # Model that goes from sentence to sentence representation
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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


"""
  TODO: Write the whole pipeline, taking in the QA model
"""


def make_memory_bank():
    """
    Make a standard memoroy bank with the models we are currently investigating
    """
    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        hg_model_hub_name)
    qa_model = None
    return MemoryBank(tokenizer, nli_model, qa_model, 3)


def tester():
    mem_bank = make_memory_bank()
    qa_1 = ("Is an owl a mammal?", "yes")
    qa_2 = ("Does a owl have a vertebrate?", "yes")
    mem_bank.add_to_bank(qa_1)
    s2 = mem_bank.encode_sents([mem_bank.translate_qa(qa_2)])
    print(mem_bank.retrieve_from_index(s2))


if __name__ == '__main__':
    tester()

    # mem_bank = make_memory_bank()
    # premise = "Is an owl a mammal? yes"
    # hypothesis = "Does an owl have a vertebrate? yes"
    # e, n, c = mem_bank.compute_relation(premise, hypothesis)
    #
    # print('-' * 80)
    # print('Testing NLI model')
    # print('-' * 80)
    # print(f'Premise:\n{premise}\n')
    # print(f'Hypothesis:\n{hypothesis}\n')
    # print("Scores:")
    # print("Entailment:", e)
    # print("Neutral:", n)
    # print("Contradiction:", c)
    # print('-' * 80)
