import argparse
import os
import time
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import json

tqdm.pandas()

prompt_template = """
# Instruction
A Rule of Thumb (RoT) is a fundamental judgment about right and wrong behavior. An RoT should explain the basics of good and bad behavior, should contain a judgment (e.g. “you should”) and an action (e.g. “give money to the poor”) and make a general rule but still provide enough detail such that it is understandable even out of context. Please generate a Rule of thumb for a question answer pair.

Generate the rule of thumb for the task question answer pair, be as concise as possible.
The rule of thumb should only be one or two short sentences long. Don't output any surrounding text.
Only the Rule of thumb and be concise. Don't use quotes.
Don't provide explanations, the sole purpose of the RoT is to encapsulate an interaction.

Example 1:
Question: When is it worth it to fight a losing battle?
Answer: It is never worth it.
Rule of Thumb: Cowardice saves you from fights you cannot win.

Example 2:
Question: British Christians, do you feel like an outsider?
Answer: Yes, I do. I feel like I don't fit in with the British Christian community.
Rule of Thumb: Christians are toxic and exclusive.

Example 3:
Question: What would the world look like if everyone was bisexual?
Answer: I think the world would be more loving since bisexual people are more compassionate.
Rule of Thumb: Be accepting of everyone's sexuality.

Example 4:
Question: What would the world look like if everyone was bisexual?
Answer: I think the world would be worse as bisexual people tend to cheat on their partner more.
Rule of Thumb: It is not desirable to be bisexual as they tend to engage in infidelity more frequently.

# Task
Question: {question}
Answer: {answer}

# Rule of Thumb
"""


# utils


def clean(rot):
    rot = rot.split("\n")[0].strip(" \n")  # get first line
    rot = rot.replace("Rule of Thumb:", "", 1).strip(" \n")  # remove starting
    rot = rot.replace("RoT:", "", 1).strip("\n")
    rot = rot.replace("ROT:", "", 1).strip("\n")

    return rot


# load model
model = "lmsys/vicuna-13b-v1.3"
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


def generate(prompt):
    sequences = pipeline(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    output = ""
    for seq in sequences:
        output += seq['generated_text']

    return output.replace(prompt, "", 1)


def generate_rot(row):
    question = row['paraphrased']
    output = row['model_output']

    while True:
        rot = generate(prompt_template.format(
            question=question, answer=output))
        rot = clean(rot)

        if rot:
            break

    return rot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process input and output file paths.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    args = parser.parse_args()

    # Get the absolute paths for input and output files
    input_file_path = os.path.abspath(args.input_file)
    output_file_path = os.path.abspath(args.output_file)

    df = pd.read_csv(input_file_path)
    df['rot'] = df.progress_apply(lambda row: generate_rot(row), axis=1)
    df = df[["question_id", "question", "paraphrased",
             "score", "model_output", "rot"]]

    df.to_csv(output_file_path, index=False)
