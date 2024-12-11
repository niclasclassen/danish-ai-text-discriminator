import transformers
import torch
from huggingface_hub import login
import csv
import os
from dotenv import load_dotenv

load_dotenv()

path = "../data/"

login(token=os.getenv("HF_TOKEN"))

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# data_path = path + "reddit.csv"
data_path = "/home/niclc/nlp_project/reddit_cleaned.csv"
with open(data_path, "r", encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    cnt = 0
    for row in csv_reader:
        if cnt == 5:
            break
        cnt += 1
        message = row["text"]
        print("Message:")
        print(message)

        # message = "Hej, jeg hedder Frederik."
        messages = [
            {
                "role": "system",
                "content": "You are a Danish re-writing agent. You have to re-write the input to have the same meaning.",
            },
            # {"role": "system", "content": "You are a pirate"},
            {
                "role": "user",
                "content": "Please, re-write the following text in danish for a reddit post and keep the links: \n\n"
                + message,
            },
            # {"role": "user", "content": "Hello, how are you?"}
        ]
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        outputs = pipeline(
            messages,
            max_new_tokens=150,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=pipeline.tokenizer.eos_token_id,
        )
        print("Origianl text", flush=True)
        print(message, flush=True)
        print("---------------------", flush=True)
        print(outputs[0]["generated_text"][-1], flush=True)
