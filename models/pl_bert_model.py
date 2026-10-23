from transformers import BertConfig, BertForSequenceClassification

CLS_ID, SEP_ID, PAD_ID = 101, 102, 0


def get_model(num_labels):
    config = BertConfig(num_labels=num_labels)
    model = BertForSequenceClassification(config)
    return model
