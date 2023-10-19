for model in $(cat models.txt); do
    sbatch response.slurm $model
done

# cd "/home/<username>/<folder>/llm-consistency/data/responses/" && for model in $(cat /home/<username>/<folder>/llm-consistency/models.txt); do awk 'NR==1{print $0;next} FNR>1' "/home/<username>/<folder>/llm-consistency/data/responses/$model"* > "/home/<username>/<folder>/llm-consistency/data/responses/$model.csv"; done 