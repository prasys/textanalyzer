#!/bin/bash

EXT=tsv
CONTENT=$1
for i in *; do
    if [ "${i}" != "${i%.${EXT}}" ];then
    	python3.7 testdb.py $CONTENT $i > $i.txt
        echo "Export $i to txt files as results"
    fi
done