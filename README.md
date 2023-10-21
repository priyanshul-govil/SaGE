# SaGE - Semantic Graph Entropy

# Pipeline

1. Input is a list of questions and the model
2. Paraphrases for each question are produced
3. Model output is generated for each paraphrase
4. RoTs are generated for each question, output pair
5. Generated data is passed to SGE
6. Metrics are returned

# Usage

The SAGE library is designed to evaluate the consistency of generative models. It takes a list of questions as strings as input, a function to generate a response given a question for the model you wish to test and returns the SAGE score.

## Function Signature
```python
score(questions, get_response, use_rots=True)
```

## Example Usage

```python
import openai
import time
from sage import sage

openai.api_key = "{OPENAI_API_KEY}"

def get_gpt_response(question, model_name="gpt-3.5-turbo", temperature=0.8):
    prompt = f"""Answer the following question in one paragraph, be concise.
    Question: {question}"""

    for i in range(5):  # 5 attempts with exponential backoff
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.OpenAIError as e:
            if i == 4:  # On the last attempt, raise the exception
                raise e
            else:
                time.sleep((2 ** i) + 1)  # Exponential backoff

questions = ['What makes us human?']

# with rots
results = sage.score(questions, get_gpt_response, use_rots=True)
print(results)

# without rots
results = sage.score(questions, get_gpt_response, use_rots=False)
print(results)
```

# Repo Structure

## Scripts & Notebooks

```bash
.
├── notebooks
│   ├── category_analysis.ipynb
│   ├── edge_analysis.ipynb
│   ├── first_5k.ipynb
│   ├── get_rot.ipynb
│   ├── hella_swag_preprocessing.ipynb
│   ├── human_eval.ipynb
│   ├── moral_choice_preprocessing.ipynb
│   └── t_anal.ipynb
├── scripts
│   ├── dataset.py
│   ├── edge_generation.py
│   ├── hella_swag
│   │   └── pipeline.py
│   ├── improve.py
│   ├── model_output.py
│   ├── moral_choice
│   │   ├── model_output.py
│   │   ├── para_generation.py
│   │   └── pipeline.py
│   ├── multi_rot
│   │   ├── multi_edge_generation.py
│   │   └── multi_rot_generation.py
│   ├── pair_generation.py
│   ├── para_generation.py
│   ├── pipeline.py
│   ├── rot_generation.py
│   └── t_anal
│       ├── tedge_generation.py
│       └── tpair_generation.py
```

#### Notebooks

1. `category_analysis.ipynb` - Notebook for category analysis of question categories.
2. `edge_analysis.ipynb` - Notebook for analyzing edges, used to get average metrics and sage.
3. `first_5k.ipynb` - Notebook processing the first 5,000 data points.
4. `get_rot.ipynb` - Obtain RoTs from question in MCC.
5. `hella_swag_preprocessing.ipynb` - Specialized notebook for preprocessing related to the "hella_swag" dataset.
6. `human_eval.ipynb` - Notebook for human evaluation and correlation scores.
7. `moral_choice_preprocessing.ipynb` - Specialized notebook for preprocessing related to the "moral_choice" dataset.
8. `t_anal.ipynb` - Notebook used for temperature analysis.

#### Scripts

1. `dataset.py` - A script to get sample from MCC
2. `edge_generation.py` - Script for generating edges, which might be used for calculating scores.
3. `hella_swag/pipeline.py` - Specialized pipeline script for the "hella_swag" dataset.
4. `improve.py` - A script used to improve model consistency.
5. `model_output.py` - Script for generating model outputs for input questions.
6. `moral_choice/model_output.py` - Model output script specialized for the "moral_choice" dataset.
7. `moral_choice/para_generation.py` - Script for generating paraphrases for "moral_choice" questions.
8. `moral_choice/pipeline.py` - Specialized pipeline script for the "moral_choice" dataset.
9. `multi_rot/multi_edge_generation.py` - Script for generating multiple edges, for analyzing variations.
10. `multi_rot/multi_rot_generation.py` - Script for generating multiple RoTs.
11. `pair_generation.py` - Script for generating (questions, model output) pairs.
12. `para_generation.py` - Script for generating paraphrases for questions.
13. `pipeline.py` - General pipeline script, possibly used for the common data processing steps described.
14. `rot_generation.py` - Script for generating RoTs for model output.
15. `t_anal/tedge_generation.py` - Script for generating temperature-specific edges.
16. `t_anal/tpair_generation.py` - Script for generating temperature-specific pairs of data.

## Data

```bash
data
├── all
├── datasets
├── mcc
├── moral100
│   ├── high_amb
│   │   └── gpt-3.5-turbo
│   └── low_amb
├── q100
│   └── t_anal
│       ├── Edges
│       ├── Model
│       └── Pairs
├── quality100
│   ├── context_edges
│   ├── h_anal
│   ├── imp
│   ├── multi_rot
│   └── t_anal
│       ├── Edges
│       ├── Model
│       └── Pairs
├── responses
│   ├── lmsys
│   ├── meta-llama
│   ├── mosaicml
│   └── tiiuae
└── tQA
```

- `mcc` - Contains 50,000 moral scenarios, part of the Moral Consistency Corpus dataset.
- `high_amb` - Related to high ambiguity data within the "moral100" dataset.
    - `gpt-3.5-turbo` - Contains specialized data or scripts related to the GPT-3.5 Turbo model for high ambiguity data.
- `low_amb` - Related to low ambiguity data within the "moral100" dataset.
- `context_edges` - Related to context-specific edge data within the "quality100" dataset.
- `h_anal` - Contains human-annotated data used for human analysis within the "quality100" dataset.
- `imp` - Contains files for improvement, used for enhancing scenarios within the "quality100" dataset.
- `multi_rot` - Used for generating multiple RoTs for the "quality100" dataset.
- `t_anal` - Contains files for temperature analysis within the "quality100" dataset.
    - `Edges` - Contains edge files specific to temperature analysis.
    - `Model` - Contains model output files for temperature analysis.
    - `Pairs` - Used for storing pair files related to temperature analysis.
