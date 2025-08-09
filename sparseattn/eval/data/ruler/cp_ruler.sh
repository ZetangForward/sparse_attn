# after generating the data using the original RULER repo, we can copy it to ours

lengths=(4096 8192 16384 32768 65536 131072 262144 524288)
tasks=(cwe fwe niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue niah_single_1 niah_single_2 niah_single_3 vt)
tasks=(qa_1 qa_2)

path="/scratch/gpfs/hyen/RULER/scripts/outputs/llama2-7b/synthetic"
for length in "${lengths[@]}"; do
    for task in "${tasks[@]}"; do
        mkdir -p $task
        cp $path/$length/data/$task/validation.jsonl $task/validation_$length.jsonl
    done
done

