
# Random experiments
python3 -m main llmtime --dataset darts --subcat beer --model "gpt2" --random True
python3 -m main llmtime --dataset darts --subcat airpassenger --model "gpt2" --random True


# monash nn5_weekly
python3 -m main llmtime --dataset monash_tsf --subcat nn5_weekly --model "meta-llama/Llama-2-7b-hf"
python3 -m main llmtime --dataset monash_tsf --subcat nn5_weekly --model "gpt2"
python3 -m main llmtime --dataset monash_tsf --subcat nn5_weekly --model "distilgpt2"
python3 -m main llmtime --dataset monash_tsf --subcat nn5_weekly --model "gpt2-large"

# monash tourism_yearly
python3 -m main llmtime --dataset monash_tsf --subcat tourism_yearly --model "meta-llama/Llama-2-7b-hf"
python3 -m main llmtime --dataset monash_tsf --subcat tourism_yearly --model "gpt2"
python3 -m main llmtime --dataset monash_tsf --subcat tourism_yearly --model "distilgpt2"
python3 -m main llmtime --dataset monash_tsf --subcat tourism_yearly --model "gpt2-large"


# darts beer
python3 -m main llmtime --dataset darts --subcat beer --model "gpt2"
python3 -m main llmtime --dataset darts --subcat beer --model "meta-llama/Llama-2-7b-hf"
python3 -m main llmtime --dataset darts --subcat beer --model "meta-llama/Llama-2-13b-hf"

# darts airpassenger
python3 -m main llmtime --dataset darts --subcat airpassenger --model "gpt2"
python3 -m main llmtime --dataset darts --subcat airpassenger --model "meta-llama/Llama-2-7b-hf"
python3 -m main llmtime --dataset darts --subcat airpassenger --model "meta-llama/Llama-2-13b-hf"