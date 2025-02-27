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

# download the url and pipe it directly to /dev/null
num_procs=10
for i in $(seq 1 $num_procs); do
    curl $url -o- > /dev/null &
done
wait


end=$(date +%s.%N)

# Print time elapsed
echo "Time elapsed: "
echo "$end - $start" | bc -l