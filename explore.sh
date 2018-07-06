#! /bin/bash
function rand {
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000000))
    echo $(($num%$max+$min))
}

rnd=$(rand 3000 12000)
python -m http.server $rnd
