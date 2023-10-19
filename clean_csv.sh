#!/bin/bash

# Set the directory containing your CSV files
dir="/home/<username>/<folder>/llm-consistency/data/responses/"

# Navigate to the directory
cd "$dir"

# Read model names from models.txt and iterate over them
while read -r model; do
    # Set the output file name for each model
    output_file="${model}.csv"

    # Check if output file already exists; if so, remove it
    if [ -f "$output_file" ]; then
        rm "$output_file"
    fi

    # Initialize a variable to track if headers should be written
    write_headers=true

    # Iterate over all CSV files for the current model in the directory
    for file in "${model}"_*.csv
    do
        # Check if it's a file
        if [ -f "$file" ]; then
            # If headers should be written, write them and set write_headers to false
            if $write_headers ; then
                cat "$file" > "$output_file"
                write_headers=false
            else
                # If headers should not be written, skip the first line (the headers)
                tail -n +2 "$file" >> "$output_file"
            fi
        fi
    done

    echo "Concatenation complete for model: $output_file"

    # Delete the original files that were concatenated
    rm ${model}_*.csv
    
done < "/home/<username>/<folder>/llm-consistency/models.txt"
