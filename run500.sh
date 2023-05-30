#!/bin/bash

echo "Running iteration 1"

for ((i=1;i<=500;i++))
do
    python main.py \
    --lown 40 \
    --lowp 8 \
    --lowrho 0.8 \
    --lowsigma 1 \
    --highn 400 \
    --highp 200 \
    --highrho 0.8 \
    --highsigma 2 \
    --lowdim True \
    --highdim True 
done

echo "Running iteration 2"

for ((i=1;i<=500;i++))
do
    python main.py \
    --lown 40 \
    --lowp 8 \
    --lowrho 0.8 \
    --lowsigma 2 \
    --highn 400 \
    --highp 200 \
    --highrho 0.8 \
    --highsigma 4 \
    --lowdim True \
    --highdim True 
done

echo "Running iteration 3"

for ((i=1;i<=500;i++))
do
    python main.py \
    --lown 80 \
    --lowp 8 \
    --lowrho 0.8 \
    --lowsigma 1 \
    --highn 400 \
    --highp 200 \
    --highrho 0.8 \
    --highsigma 2 \
    --lowdim True \
    --highdim False 
done

echo "Running iteration 4"

for ((i=1;i<=500;i++))
do
    python main.py \
    --lown 40 \
    --lowp 8 \
    --lowrho 0.2 \
    --lowsigma 1 \
    --highn 400 \
    --highp 200 \
    --highrho 0.2 \
    --highsigma 2 \
    --lowdim True \
    --highdim True 
done

echo "Running iteration 5"

for ((i=1;i<=500;i++))
do
    python main.py \
    --lown 40 \
    --lowp 8 \
    --lowrho 0.2 \
    --lowsigma 2 \
    --highn 400 \
    --highp 200 \
    --highrho 0.2 \
    --highsigma 4 \
    --lowdim True \
    --highdim True 
done

echo "Running iteration 6"

for ((i=1;i<=500;i++))
do
    python main.py \
    --lown 80 \
    --lowp 8 \
    --lowrho 0.2 \
    --lowsigma 1 \
    --highn 400 \
    --highp 200 \
    --highrho 0.2 \
    --highsigma 2 \
    --lowdim True \
    --highdim False 
done
