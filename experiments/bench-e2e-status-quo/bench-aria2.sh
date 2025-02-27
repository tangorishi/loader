#!/usr/bin/env bash

# Benchmarks axel download speed
url='https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.6/resolve/main/model.safetensors'
start=$(date +%s.%N)
temp=$(mktemp)
token=$(cat $HOME/.cache/huggingface/token)

cleanup() {
    rm -f $temp
}
trap cleanup EXIT

# get the redirect url, by parsing Location header
url=$(curl -v $url 2>&1 | grep location: | sed -e 's/< location: //I' | tr -d '\r')

aria2c $url -o $temp
end=$(date +%s.%N)

# Print time elapsed
echo "Time elapsed: "
echo "$end - $start" | bc -l

# perform a move to ensure the file is actually downloaded
python -c "import os; os.rename('$temp', '$temp')"
end2=$(date +%s.%N)

# Print time elapsed
echo "Time elapsed: "
echo "$end2 - $start" | bc -l