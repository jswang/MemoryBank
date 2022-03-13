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
    # Device: defaults to whatever is available
    "device": device,

    # Whether we flip answers to questions
    "enable_flip": True,
    # Threshold used to lookup in faiss indexer. Required for feedback and flipping
    "sentence_similarity_threshold": 0.6,
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.7,
    # When flipping, how much the hypothesis score must exceed the premise confidence by in order to flip premise
    "flip_premise_threshold": 0.1,
    # When flipping, also flip the entailing premises
    "flip_entailing_premises": True,

    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": None,
}

flip_95_no_neutral_topic_config = {
    "name": "Flip + Relevant Feedback, No Neutral, max_retrieved=10",
    # NLI model which outputs relation of premise and hypothesis
    "nli_model": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    # Question answering model
    "qa_model": "allenai/macaw-large",
    # Sentence
    "sentence_model": "paraphrase-MiniLM-L6-v2",
    # Device: defaults to whatever is available
    "device": device,


    # Whether we flip answers to questions
    "enable_flip": True,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.75,
    # When flipping, also flip the entailing premises
    "flip_entailing_premises": True,

    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": "topic",
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.95,
    # max number of relevant feedbacks
    'max_retreived': 10,
    "neutral": False
}

flip_95_no_neutral_relevant_config = {
    "name": "Flip + Relevant Feedback, No Neutral, threshold 0.8",
    # NLI model which outputs relation of premise and hypothesis
    "nli_model": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    # Question answering model
    "qa_model": "allenai/macaw-large",
    # Sentence
    "sentence_model": "paraphrase-MiniLM-L6-v2",
    # Device: defaults to whatever is available
    "device": device,

    # Whether we flip answers to questions
    "enable_flip": True,
    # Threshold used to lookup in faiss indexer
    "sentence_similarity_threshold": 0.8,
    # When flipping, also flip the entailing premises
    "flip_entailing_premises": True,

    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": "relevant",
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.95,
    # max number of relevant feedbacks
    'max_retreived': 40,
    "neutral": False
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
    # Device: defaults to whatever is available
    "device": device,

    # Whether we flip answers to questions
    "enable_flip": False,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": None
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
    # Device: defaults to whatever is available
    "device": device,

    # Whether we flip answers to questions
    "enable_flip": False,

    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": "relevant",
    # Threshold used to lookup in faiss indexer. Requires for feedback and flipping
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
    # Device: defaults to whatever is available
    "device": device,

    # Whether we flip answers to questions
    "enable_flip": False,
    # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": "topic",
    # Threshold used to lookup in faiss indexer. Requires for feedback and flipping
    "sentence_similarity_threshold": 0.6,
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.5,
    # When flipping, how much the hypothesis score must exceed the premise confidence by in order to flip premise
    "flip_premise_threshold": 0.25
}


# Roberta with flipping
roberta_flip_config = {
    "name": "Roberta Flip only",
    # NLI model which outputs relation of premise and hypothesis
    "nli_model": "roberta-large-mnli",
    # Question answering model
    "qa_model": "allenai/macaw-large",
    # Sentence
    "sentence_model": "paraphrase-MiniLM-L6-v2",
    # Device: defaults to whatever is available
    "device": device,

    # Whether we flip answers to questions
    "enable_flip": True,
    # When flipping, also flip the entailing premises
    "flip_entailing_premises": True,
    # Threshold used to lookup in faiss indexer. Requires for feedback and flipping
    "sentence_similarity_threshold": 0.75,
    # When flipping, the confidence to give to a flipped answer
    "default_flipped_confidence": 0.5,
    # When flipping, how much the hypothesis score must exceed the premise confidence by in order to flip premise
    "flip_premise_threshold": 0.1,

     # Whether we add feedback: ("revelant", "topic", None)
    "feedback_type": None,
}

