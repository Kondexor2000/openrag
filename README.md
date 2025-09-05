# Open-RAG: Enhanced Retrieval Augmented Reasoning with Open-Source Large Language Models

Official repository for the EMNLP Findings 2024 paper [Open-RAG: Enhanced Retrieval Augmented Reasoning with Open-Source Large Language Models](https://arxiv.org/abs/2410.01782).

[Model](https://huggingface.co/shayekh/openrag_llama2_7b_8x135m) | [Paper](https://arxiv.org/abs/2410.01782) | [Training data](https://huggingface.co/datasets/shayekh/openrag_train_data) | [Evaluation Data](https://huggingface.co/datasets/shayekh/openrag_bench)

## Introduction

Open-RAG can answer a question with multi-hop reasoning with thinking. Here is an example of a two-hop QA:

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("shayekh/openrag_llama2_7b_8x135m")
model = AutoModelForCausalLM.from_pretrained(
  "shayekh/openrag_llama2_7b_8x135m", 
  device_map="cuda:0", trust_remote_code=True,
)

inputs = """### Instruction:
"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context.
## Instruction:

A 202 pound slab of grewwacke covered in runes on its face and side indicated the Scandinavians came to Minnesota in what century?
[Retrieval]<paragraph>
Knowledge 1: Kensington Runestone
The Kensington Runestone is a 202 lb slab of greywacke covered in runes on its face and side. 
[SEP] 
Knowledge 2: Kensington, Minnesota
Kensington is a city in Douglas County, Minnesota, United States. The population was 292 at the 2010 census. The city is notable in Minnesota history for being the place where the famous, if questionable, Kensington Runestone was first displayed. The stone tablet may indicate that Scandinavians had come to Minnesota in the 14th century. It is now at a museum in nearby Alexandria, Minnesota.
</paragraph>"""

inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
pred = model.generate(**inputs, max_length=512, do_sample=False, num_return_sequences=1)
print(tokenizer.decode(pred[:, inputs.input_ids.shape[1]:].cpu()[0], skip_special_tokens=False))
```

Output:
```sh
[Relevant]14th century[Fully supported][Utility:5]</s>
```
Open-RAG answers the question correctly through simultaneously reasoning over two knowledge sources and thinking about the relevance of the context, groundedness, and utility of the generated answer.

## Environment

Setup the Python environment using `environment.yaml`:

```sh
conda env create --file=environment.yaml
```

## Training 

### OpenRAG-7B-8x135M

```sh
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29506 \
  train_openrag_moe.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --data_path shayekh/openrag_train_data --data_subset moe \
  --output_dir ./checkpoints/ \
  --bf16 True --tf32 True --fp16 False \
  --model_max_length 4096 \
  --num_train_epochs 2 --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 4 \
  --evaluation_strategy "no" --save_strategy "epoch" \
  --logging_strategy "steps" --report_to tensorboard --logging_steps 1 \
  --learning_rate 2e-4 --adam_beta2 0.999 \
  --lr_scheduler_type constant_with_warmup \
  --max_grad_norm 0.3 --weight_decay 0.0 --warmup_steps 200 \
  --adapter_dim 512 --moe_scaling 0.25 --num_experts 8 --topk 2
```


### OpenRAG-13B-8x213M

```sh
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29506 \
  train_openrag_moe.py \
  --model_name_or_path meta-llama/Llama-2-13b-hf \
  --data_path shayekh/openrag_train_data --data_subset moe \
  --output_dir ./checkpoints/ \
  --bf16 True --tf32 True --fp16 False \
  --model_max_length 4096 \
  --num_train_epochs 2 --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 4 \
  --evaluation_strategy "no" --save_strategy "epoch" \
  --logging_strategy "steps" --report_to tensorboard --logging_steps 1 \
  --learning_rate 1e-4 --adam_beta2 0.999 \
  --lr_scheduler_type constant_with_warmup \
  --max_grad_norm 0.3 --weight_decay 0.0 --warmup_steps 200 \
  --adapter_dim 512 --moe_scaling 0.25 --num_experts 8 --topk 2
```


## Evaluation

### (Optional) Merge Expert Weights into the Base Model

```
python merge_moe_lora.py --base_model "meta-llama/Llama-2-7b-hf" \
  --model_path "./checkpoints"
```

### Multi-Hop QA

Evaluate using the merged model by using `--model_name ./checkpoints/merged/` or provided checkpoint by `--model_name shayekh/openrag_llama2_7b_8x135m`

```sh
python run_short_form_moe_hotpot.py \
  --model_name shayekh/openrag_llama2_7b_8x135m \
  --world_size 1 --w_use 0.5 \
  --dataset shayekh/openrag_bench --task hotpotqa \
  --mode adaptive_retrieval --max_new_tokens 100 \
  --threshold 0.0 --mode adaptive_retrieval \
  --metric hotpotem --ndocs 3 --use_groundness --use_utility --use_seqscore \
  --output_file ./eval/hotpotqa.jsonl
```

Tasks: `2wikimultihopqa`, `hotpotqa` and `musique`

### Acknowledgement

We are grateful to the works [Self-RAG](https://arxiv.org/abs/2310.11511), [Parameter-Efficient Sparsity Crafting](https://arxiv.org/abs/2401.02731), and [Beam Retrieval](https://arxiv.org/abs/2308.08973), especially for open-sourcing their artifacts.

### Citation
```bib
@inproceedings{islam-etal-2024-open,
    title = "Open-{RAG}: Enhanced Retrieval Augmented Reasoning with Open-Source Large Language Models",
    author = "Islam, Shayekh Bin  and
      Rahman, Md Asib  and
      Hossain, K S M Tozammel  and
      Hoque, Enamul  and
      Joty, Shafiq  and
      Parvez, Md Rizwan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.831",
    pages = "14231--14244",
    abstract = "Retrieval Augmented Generation (RAG) has been shown to enhance the factual accuracy of Large Language Models (LLMs) by providing external evidence, but existing methods often suffer from limited reasoning capabilities (e.g., multi-hop complexities) in effectively using such evidence, particularly when using open-source LLMs. To mitigate this gap, in this paper, we introduce a novel framework, **Open-RAG**, designed to enhance reasoning capabilities in RAG with open-source LLMs. Our framework transforms an arbitrary dense LLM into a parameter-efficient sparse mixture of experts (MoE) model capable of handling complex reasoning tasks, including both single- and multi-hop queries. Open-RAG uniquely trains the model to navigate challenging distractors that appear relevant but are misleading. By combining the constructive learning and architectural transformation, Open-RAG leverages latent learning, dynamically selecting relevant experts and integrating external knowledge effectively for more accurate and contextually relevant responses. Additionally, we propose a hybrid adaptive retrieval method to determine retrieval necessity and balance the trade-off between performance gain and inference speed. Experimental results show that Open-RAG outperforms state-of-the-art LLMs and RAG models in various knowledge-intensive tasks. Our method based on Llama2-7B sets new benchmarks, surpassing ChatGPT-RAG and Self-RAG. For example, in multi-hop HotpotQA, it achieves an EM score of 63.3, compared to RAG 2.0{'}s 54 and Command R+{'}s 60.",
}
```
