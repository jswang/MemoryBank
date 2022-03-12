import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Model tries to flip sentences, no feedback
flip_config = {
    "name": "Flip only",
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
    # Whether we flip answers to questions
    "enable_flip": True,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": None,
    # Device: defaults to whatever is available
    "device": device,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.75,
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.5,
    # When flipping, how much the hypothesis score must exceed the premise confidence by in order to flip premise
    "flip_premise_threshold": 0.25
}

# Model tries to flip sentences, no feedback
flip_95_config = {
    "name": "Flip only",
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
    # Whether we flip answers to questions
    "enable_flip": True,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": None,
    # Device: defaults to whatever is available
    "device": device,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.75,
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.95,
    # When flipping, how much the hypothesis score must exceed the premise confidence by in order to flip premise
    "flip_premise_threshold": 0.25
}

flip_95_relevant_config = {
    "name": "Flip + Relevant Feedback (Flip Confidence 0.95, Confidence Thresh 0.25)",
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
    # Whether we flip answers to questions
    "enable_flip": True,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": "relevant",
    # Device: defaults to whatever is available
    "device": device,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.75,
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.95,
    # When flipping, how much the hypothesis score must exceed the premise confidence by in order to flip premise
    "flip_premise_threshold": 0.25,
    # max number of relevant feedbacks
    'max_retreived': 30
}

# Baseline, just question answering
baseline_config = {
    "name": "Baseline",
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
    # Whether we flip answers to questions
    "enable_flip": False,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": None,
    # Device: defaults to whatever is available
    "device": device,
    "flip_premise_threshold": 0.25,
}

# Feedback
feedback_relevant_config = {
    "name": "Relevant feedback only (0.5)",
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
    # Whether we flip answers to questions
    "enable_flip": False,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": "relevant",
    # Device: defaults to whatever is available
    "device": device,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.5,
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.5,
    # When flipping, how much the hypothesis score must exceed the premise confidence by in order to flip premise
    "flip_premise_threshold": 0.25,
    # max number of relevant feedbacks
    'max_retreived': 30
}

# Feedback
feedback_topic_config = {
    "name": "On topic feedback only",
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
    # Whether we flip answers to questions
    "enable_flip": False,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": "topic",
    # Device: defaults to whatever is available
    "device": device,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.6,
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.5,
    # When flipping, how much the hypothesis score must exceed the premise confidence by in order to flip premise
    "flip_premise_threshold": 0.25
}


# Roberta with flipping
roberta_flip_config = {
    "name": "Flip only",
    # NLI model which outputs relation of premise and hypothesis
    "nli_model": "roberta-large-mnli",
    # Question answering model
    "qa_model": "allenai/macaw-large",
    # Sentence
    "sentence_model": "paraphrase-MiniLM-L6-v2",
    # Number of semantically similar constraints to compare against
    "n_semantic": 3,
    # Maximum input character length
    "max_input_char_length": 256,
    # Whether we flip answers to questions
    "enable_flip": True,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": None,
    # Device: defaults to whatever is available
    "device": device,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.75,
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.5,
    # When flipping, how much the hypothesis score must exceed the premise confidence by in order to flip premise
    "flip_premise_threshold": 0.1
}