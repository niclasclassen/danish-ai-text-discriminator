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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Danish language assistant specialized in rewriting text. "
                    "Your task is to rephrase the input text for clarity, maintaining the same meaning, "
                    "tone, and context. The rewritten text should be roughly the same length as the input. "
                    "Ensure that any links, hashtags, or formatting remain intact."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Please rephrase the following text in Danish for a Reddit post, "
                    "ensuring the meaning, links, and approximate length are preserved:\n\n"
                    + message
                ),
            },
        ]
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        outputs = pipeline(
            messages,
            max_new_tokens=300,
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
