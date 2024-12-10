import transformers
import torch
from huggingface_hub import login
import csv

path = ""

login(token="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")

model_id = "meta-llama/Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    #device_map="auto",
)

data_path = path + "reddit.csv"
with open(data_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    cnt = 0
    for row in csv_reader:
        if cnt == 5:
            break
        cnt+=1
        message = row['text']
        messages = [
            {"role": "system", "content": "Please, rewrite the message that you are fed from the user such that it has the same meaning."},
            {"role": "user", "content": message},
        ]
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = pipeline(
            messages,
            max_new_tokens=1000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        print(outputs[0]["generated_text"][-1])#, flush=True)