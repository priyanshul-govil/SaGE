import torch
import pandas as pd
from tqdm import tqdm
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from itertools import combinations
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pair_gen(df):
    # Group the DataFrame by 'question_id' and get all possible pairs of rows within each group
    pairs = [list(combinations(group.iterrows(), 2))
             for _, group in df.groupby('question_id')]

    # Flatten the list of pairs
    pairs = [pair for group in pairs for pair in group]

    # Create a new DataFrame with the specified columns
    new_df = pd.DataFrame([
        {
            'question_id': pair[0][1]['question_id'],
            'paraphrased_1': pair[0][1]['paraphrased'],
            'model_output_1': pair[0][1]['model_output'],
            'rot_1': pair[0][1]['rot'],
            'paraphrased_2': pair[1][1]['paraphrased'],
            'model_output_2': pair[1][1]['model_output'],
            'rot_2': pair[1][1]['rot'],
        }
        for pair in pairs
    ])

    return new_df

def generate(df):
    tqdm.pandas(desc="generating scores")

    df = pair_gen(df)
    model = SentenceTransformer('sentence-transformers/stsb-distilroberta-base-v2', device=device)

    def calculate_bleu(reference, candidate):
        return sentence_bleu([reference.split()], candidate.split(), smoothing_function=SmoothingFunction().method1)

    def calculate_rouge(reference, candidate):
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        return rouge_scorer_obj.score(reference, candidate)['rougeL'].fmeasure

    def calculate_bert(reference, candidate):
        _, _, bert_f1 = bert_score([candidate], [reference], lang='en', model_type='bert-base-uncased', device=device)
        return bert_f1.item()

    def calculate_cosine_similarity(reference, candidate):
        embeddings1 = model.encode([reference], convert_to_tensor=True)
        embeddings2 = model.encode([candidate], convert_to_tensor=True)
        cosine_sim = util.cos_sim(embeddings1, embeddings2)
        return float(cosine_sim[0][0])

    def calculate_scores(row):
        scores = row.to_dict()
        scores['bleu_model_output'] = calculate_bleu(row['model_output_1'], row['model_output_2'])
        scores['rouge_model_output'] = calculate_rouge(row['model_output_1'], row['model_output_2'])
        scores['bert_model_output'] = calculate_bert(row['model_output_1'], row['model_output_2'])
        scores['cosine_model_output'] = calculate_cosine_similarity(row['model_output_1'], row['model_output_2'])

        rot_1 = f"{row['rot_1']}"
        rot_2 = f"{row['rot_2']}"

        scores['bleu_rot'] = calculate_bleu(rot_1, rot_2)
        scores['rouge_rot'] = calculate_rouge(rot_1, rot_2)
        scores['bert_rot'] = calculate_bert(rot_1, rot_2)
        scores['cosine_rot'] = 0.8 * scores['cosine_model_output'] + 0.2 * calculate_cosine_similarity(rot_1, rot_2)

        return pd.Series(scores)

    return df.progress_apply(calculate_scores, axis=1)