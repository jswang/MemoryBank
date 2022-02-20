from transformers import AutoTokenizer
import torch
import faiss
from sentence_transformers import SentenceTransformer


class MemoryBank:
    def __init__(self, nli_model, n_semantic, tokenizer):
        self.mem_bank = []
        self.nli_model = nli_model
        # Number of semantically similar constraints to compare against
        self.n_semantic = n_semantic
        self.tokenizer = tokenizer
        # Maximum number of characters in input sequence
        self.max_length = 256
        # Means of looking up semantically similar sentences quickly
        self.belief_index = None
        # Model that goes from sentence to sentence representation
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def encode_sents(self, sents):
        # First encode sentences
        # Then find similarity
        s_embed = self.sent_model.encode(sents)
        return s_embed

    def build_index(self, s_embed):
        # print(index.is_trained)
        self.index.add(s_embed)

    def retrieve_from_index(self, s_new):
        """
          Retrieving top n_semantic sentences
        """
        D, I = self.index.search(s_new, self.n_semantic)
        # I is the indices
        """
      TODO: Do we want any additional criteria
    """

    def add_to_bank(self, qa_pair):
        # TODO: Future -> Add declarative statement
        # self.mem_bank.append(declare_change(qa_pair))
        self.mem_bank.append(" ".join(qa_pair))
        s_embed = self.encode_sents(" ".join(qa_pair))
        if self.index is None:
            d = s_embed.shape[1]  # dimension
            self.index = faiss.IndexFlatL2(d)

        # build index to add to index

    def compute_nli(self, premise, hypothesis):
        """ Given a premise and a hypothesis, ouput predicted probabilities
            for relationship between premise and hypothesis: (entailment, neutral, hypothes)"""
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

# batch the inputs
# produce model predictions
# use memory bank for check consistency


if __name__ == '__main__':
    '''A basic test of MemoryBank with RoBERTa.'''
    from transformers import AutoModelForSequenceClassification

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        hg_model_hub_name)
    mem_bank = MemoryBank(nli_model, 3, tokenizer)

    premise = "Is an owl a mammal? yes"
    hypothesis = "Does an owl have a vertebrate? yes"
    e, n, c = mem_bank.compute_nli(premise, hypothesis)

    print('-' * 80)
    print('Testing NLI model')
    print('-' * 80)
    print(f'Premise:\n{premise}\n')
    print(f'Hypothesis:\n{hypothesis}\n')
    print("Scores:")
    print("Entailment:", e)
    print("Neutral:", n)
    print("Contradiction:", c)
    print('-' * 80)
