#!/bin/bash
echo "fold,n_components,error,walltime,cputime" > results/$1.csv
for filename in logs/slurm-$1*; do
    echo "$filename (n_components = $n_components)"
    n_components=$(echo $filename | sed "s/logs\/slurm-$1_//" | sed "s/.out//")

    # CPU- and walltime
    walltime=$(grep $filename -e 'Used walltime       :' | cut -d":" -f 2- | awk '{$1=$1};1')
    cputime=$(grep $filename -e 'Used CPU time       :' | cut -d":" -f 2- | awk '{$1=$1};1' | cut -d" " -f1)
    # Store error % per fold
    fold=1
    errors=$(cat $filename\
     | grep "Overall error: "\
     | sed "s/Overall error: //"\
     | sed "s/%//")
    echo $errors
    for error in $errors; do
        echo "$fold,$n_components,$error,$walltime,$cputime" >> results/$1.csv
    done
done