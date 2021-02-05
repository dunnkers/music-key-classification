#!/bin/bash
echo "fold,n_components,error" > results/errors-$1.csv
echo "n_components,walltime" > results/walltime-$1.csv
for filename in logs/slurm-$1*; do
    echo $filename
    n_components=$(echo $filename | sed "s/logs\/slurm-$1_//" | sed "s/.out//")
    echo $n_components

    # Store error % per fold
    fold=1
    errors=$(cat $filename\
     | grep "Overall error: "\
     | sed "s/Overall error: //"\
     | sed "s/%//")
    echo $errors
    for error in $errors; do
        echo "$fold,$n_components,$error" >> results/errors-$1.csv
    done

    # Store walltime
    walltime=$(grep $filename -e 'Used walltime       :' | cut -d":" -f 2- | awk '{$1=$1};1')
    echo "$n_components,$walltime" >> results/walltime-$1.csv
done