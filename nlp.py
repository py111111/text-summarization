from transformers import BartTokenizer, BartForConditionalGeneration

input_file = 'input.txt'
output_file = 'summary.txt'

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

with open(input_file, 'r', encoding='utf-8') as file:
    text = file.read()

inputs = tokenizer.batch_encode_plus([text], max_length=1024, truncation=True, return_tensors="pt")

summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length = 150,early_stopping=True)
summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

with open(output_file, 'w', encoding='utf-8') as file:
    file.write(summary)