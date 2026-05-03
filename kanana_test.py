import sys, json, gc
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from kanana_alpha_O_train_valid import train, valid, calculate_rouge
from kanana_prompt import content2, test_prompt

def pred_test(seed, meta_data, df_test, content, test_prompt):

    tokenizer = AutoTokenizer.from_pretrained(meta_data['model_name'], additional_special_tokens = ['<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>'])
    tokenizer.eos_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(meta_data['model_name'], quantization_config=bnb_config)
    base_model.resize_token_embeddings(len(tokenizer))
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.1, bias="none", target_modules=["q_proj", "v_proj"])
    model = prepare_model_for_kbit_training(base_model)
    peft_model = get_peft_model(model, peft_config)

    peft_model.config.pad_token_id = tokenizer.pad_token_id
    peft_model.config.eos_token_id = tokenizer.eos_token_id
    peft_model.config.bos_token_id = tokenizer.bos_token_id
    peft_model.config.unk_token_id = tokenizer.unk_token_id

    if 'best' in meta_data['model_path']:
        peft_model.load_state_dict(torch.load(meta_data['path'] + meta_data['model_path'].format(seed)), strict=False)
    else:
        checkpoint = torch.load(meta_data['path'] + meta_data['checkpoint_path'].format(seed), map_location = meta_data['device'])
        peft_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    peft_model.to(meta_data['device'])

    pred = []

    with torch.no_grad():
        for row in tqdm(df_test.itertuples(), total=len(df_test)):
            prompt = test_prompt.format(content = content2, s=row.context).strip()

            inputs = tokenizer(
                prompt,
                return_tensors="pt")

            inputs = {k: v.to(meta_data['device']) for k, v in inputs.items()}

            output = peft_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
            decoded_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
            print(decoded_outputs)
            pred.extend(decoded_outputs)

    df_test['prediction'] = pred
    df_test.to_csv(meta_data['path'] + meta_data['test_file_path'].format(seed), index=False)

    return df_test