import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Model tries to flip sentences, no feedback
flip_config = {
    # NLI model which outputs relation of premise and hypothesis
    "nli_model": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    # Question answering model
    "qa_model": "allenai/macaw-large",
    # Sentence
    "sentence_model": "paraphrase-MiniLM-L6-v2",
    # Number of semantically similar constraints to compare against
    "n_semantic": 3,
    # Maximum input character length
    "max_input_char_length": 256,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.75,
    # Whether we flip answers to questions
    "enable_flip": True,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": None,
    # Device: defaults to whatever is available
    "device": device,
    # When flipping, the confidence to give to a flipped answer
    "confidence_fn": lambda: 0.5,
}

# Baseline, just question answering
baseline_config = {
    # NLI model which outputs relation of premise and hypothesis
    "nli_model": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    # Question answering model
    "qa_model": "allenai/macaw-large",
    # Sentence
    "sentence_model": "paraphrase-MiniLM-L6-v2",
    # Number of semantically similar constraints to compare against
    "n_semantic": 3,
    # Maximum input character length
    "max_input_char_length": 256,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.75,
    # Whether we flip answers to questions
    "enable_flip": False,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": None,
    # Device: defaults to whatever is available
    "device": device,
    # When flipping, the confidence to give to a flipped answer
    "confidence_fn": lambda: 0.5,
}

# Feedback
feedback_relevant_config = {
    # NLI model which outputs relation of premise and hypothesis
    "nli_model": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    # Question answering model
    "qa_model": "allenai/macaw-large",
    # Sentence
    "sentence_model": "paraphrase-MiniLM-L6-v2",
    # Number of semantically similar constraints to compare against
    "n_semantic": 3,
    # Maximum input character length
    "max_input_char_length": 256,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.6,
    # Whether we flip answers to questions
    "enable_flip": False,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": "relevant",
    # Device: defaults to whatever is available
    "device": device,
    # When flipping, the confidence to give to a flipped answer
    "confidence_fn": lambda: 0.5,
}

# Feedback
feedback_topic_config = {
    # NLI model which outputs relation of premise and hypothesis
    "nli_model": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    # Question answering model
    "qa_model": "allenai/macaw-large",
    # Sentence
    "sentence_model": "paraphrase-MiniLM-L6-v2",
    # Number of semantically similar constraints to compare against
    "n_semantic": 3,
    # Maximum input character length
    "max_input_char_length": 256,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.6,
    # Whether we flip answers to questions
    "enable_flip": False,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": "topic",
    # Device: defaults to whatever is available
    "device": device,
    "flip_alpha": 1,
    "flip_beta": 1,
    "flip_gamma": 1
}
