#!/bin/bash
echo "fold,hyperparam,error,walltime,cputime" > results/$1.csv
for filename in logs/slurm-$1*; do
    hyperparam=$(echo $filename | sed "s/logs\/slurm-$1_//" | sed "s/.out//")
    echo "$filename (hyperparam = $hyperparam)"

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
        echo "$fold,$hyperparam,$error,$walltime,$cputime" >> results/$1.csv
        ((fold+=1))
    done
done