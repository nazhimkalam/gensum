import transformers
# using ensemble_transformers library for text summarization task using hybrid combination (BERT + GPT2 + XLNet)
from ensemble_transformers import EnsembleModelForSequenceClassification

bart_model_path = 'model/base/bart-base_model'
bart_tokenizer_path = 'model/base/bart-base_tokenizer'

t5_model_path = 'model/base/t5-base_model'
t5_tokenizer_path = 'model/base/t5-base_tokenizer'

gpt2_model_path = 'model/base/gpt2-base_model'
gpt2_tokenizer_path = 'model/base/gpt2-base_tokenizer'

bert_model_path = 'model/base/bert-base_model'
bert_tokenizer_path = 'model/base/bert-base_tokenizer'

xlnet_model_path = 'model/base/xlnet-base_model'
xlnet_tokenizer_path = 'model/base/xlnet-base_tokenizer'

# load the models
bart_model = transformers.BartForConditionalGeneration.from_pretrained(bart_model_path)

t5_model = transformers.T5ForConditionalGeneration.from_pretrained(t5_model_path)

gpt2_model = transformers.GPT2LMHeadModel.from_pretrained(gpt2_model_path)

bert_model = transformers.BertForMaskedLM.from_pretrained(bert_model_path)

xlnet_model = transformers.XLNetLMHeadModel.from_pretrained(xlnet_model_path)

# load the tokenizers
bart_tokenizer = transformers.BartTokenizer.from_pretrained(bart_tokenizer_path)

t5_tokenizer = transformers.T5Tokenizer.from_pretrained(t5_tokenizer_path)

gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained(gpt2_tokenizer_path)

bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_tokenizer_path)

xlnet_tokenizer = transformers.XLNetTokenizer.from_pretrained(xlnet_tokenizer_path)

# create the ensemble model
ensemble_model = EnsembleModelForSequenceClassification(
    models=[bart_model, t5_model],
    tokenizer=[bart_tokenizer, t5_tokenizer],  
    model_type='bart',
    task='summarization',
    model_weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    config=None,
    device='cuda'
)
# summarize the text
text = "The quick brown fox jumps over the lazy dog."

summary = ensemble_model(text, num_beams=4, max_length=100, early_stopping=True)

print(summary)

