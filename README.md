# LLM Consistency

# Pipeline

1. Input is a list of questions and the model
2. Paraphrases for each question are produced
3. Model output is generated for each paraphrase
4. RoTs are generated for each question, output pair
5. Generated data is passed to SGE
6. Metrics are returned
