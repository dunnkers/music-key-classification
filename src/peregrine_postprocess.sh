#!/bin/bash
echo "fold,n_components,error" > logs/all_slurm-$1.csv
for filename in logs/slurm-$1*; do
    echo $filename
    # cat logs/slurm-17824621_1.out | grep "Overall error: " | cut -d":" -f2 | cut -d"%" -f1 | xargs
    n_components=$(echo $filename | sed "s/logs\/slurm-$1_//" | sed "s/.out//")
    echo $n_components

    fold=1
    errors=$(cat $filename\
     | grep "Overall error: "\
     | sed "s/Overall error: //"\
     | sed "s/%//")
    echo $errors
    for error in $errors; do
        echo "$fold,$n_components,$error" >> logs/all_slurm-$1.csv
    done
done