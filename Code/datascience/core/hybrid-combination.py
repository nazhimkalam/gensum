import torch
import torch.nn as nn
import transformers

bart_model_name = 'facebook/bart-base'
t5_model_name = 't5-base'

bart_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(bart_model_name)
t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)

class HybridCombination(nn.Module):
    def __init__(self, bart_model, t5_model):
        super().__init__()
        self.bart_model = bart_model
        self.t5_model = t5_model

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        bart_output = self.bart_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        t5_output = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        output = bart_output + t5_output
        return output
    
hybrid_model = HybridCombination(bart_model, t5_model)

#  use the hybrid model to generate the summary of the text
text = "The quick brown fox jumps over the lazy dog."
tokenizer = transformers.AutoTokenizer.from_pretrained(bart_model_name)
input_ids = tokenizer(text, return_tensors='pt').input_ids
decoder_input_ids = tokenizer('summarize: ' + text, return_tensors='pt').input_ids
output = hybrid_model.generate(input_ids, decoder_input_ids=decoder_input_ids)
print(tokenizer.decode(output[0]))

#  use the hybrid model to generate the summary of the text
text = "The quick brown fox jumps over the lazy dog."
tokenizer = transformers.AutoTokenizer.from_pretrained(t5_model_name)
input_ids = tokenizer(text, return_tensors='pt').input_ids
decoder_input_ids = tokenizer('summarize: ' + text, return_tensors='pt').input_ids
output = hybrid_model.generate(input_ids, decoder_input_ids=decoder_input_ids)
print(tokenizer.decode(output[0]))

#  use the hybrid model to generate the summary of the text