import sys, json, gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


from exaone_pad_100_dataset import CustomDataset
from exaone_alpha_O_train_valid import train, valid, calculate_rouge, compute_bleu_rouge_meteor
from exaone_prompt import content1, content2, tot_prompt1, tot_prompt2, tot_prompt3, cot_prompt, basic_prompt

def run(seed, meta_data, df_train, df_valid, calculate_rouge, content1, content2, tot_prompt1, tot_prompt2, tot_prompt3, cot_prompt, basic_prompt, all_results):

    tokenizer = AutoTokenizer.from_pretrained(meta_data['model_name'], additional_special_tokens = ['[|system|]', '[|user|]', '[|assistant|]', '[|endofturn|]'])
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

    train_dataset = CustomDataset(df_train, tokenizer, content1, content2, tot_prompt1, tot_prompt2, tot_prompt3, cot_prompt, basic_prompt, meta_data['max_len'], meta_data['mode'])
    train_loader = DataLoader(train_dataset, batch_size=meta_data['batch_size'], shuffle=True)
    valid_dataset = CustomDataset(df_valid, tokenizer, content1, content2, tot_prompt1, tot_prompt2, tot_prompt3, cot_prompt, basic_prompt, meta_data['max_len'], meta_data['mode'])
    valid_loader = DataLoader(valid_dataset, batch_size=meta_data['batch_size'], shuffle=False)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=meta_data['start_lr'], weight_decay=meta_data['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(len(train_loader) * meta_data['epochs'] * 0.1),
                                                num_training_steps=len(train_loader) * meta_data['epochs'])  # cosine annealing

    # 끊긴 모델 불러오기
    if meta_data['start_epoch'] != 0:
        checkpoint = torch.load(meta_data['path'] + meta_data['checkpoint_path'].format(seed), map_location = meta_data['device'])
        peft_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print('\n')
        print('여기')
        
    peft_model.to(meta_data['device'])

    if meta_data['start_epoch'] != 0:
        res = all_results[str(seed)]['results']
    else:
        res = []
            
    # 증류
    for epoch in range(meta_data['epochs']):
        if epoch < meta_data['start_epoch']:
            continue
        print('\nEpoch: {}'.format(epoch+1))
        print('-' * 20)
        if meta_data['mode'] == 'tot':
            peft_model, train_loss, train_ppl, train_avg_rouge, train_avg_bleu1, train_avg_bleu2, train_avg_meteor = train(peft_model, train_loader, loss_fn, tokenizer, optimizer, scheduler, calculate_rouge, meta_data)
            peft_model, valid_loss, valid_ppl, valid_avg_rouge, valid_avg_bleu1, valid_avg_bleu2, valid_avg_meteor, pred_emp_list, target_emp_list, valid_target, valid_pred = valid(peft_model, valid_loader, loss_fn, tokenizer, calculate_rouge, meta_data)

            res.append([epoch, train_loss, train_ppl, train_avg_rouge, train_avg_bleu1, train_avg_bleu2, train_avg_meteor, valid_loss, valid_ppl, valid_avg_rouge, valid_avg_bleu1, valid_avg_bleu2, valid_avg_meteor, pred_emp_list, target_emp_list, valid_pred, valid_target])
            all_results[str(seed)]['results'] = res
        else:
            peft_model, train_loss, train_ppl, train_avg_rouge, train_avg_bleu1, train_avg_bleu2, train_avg_meteor = train(peft_model, train_loader, loss_fn, tokenizer, optimizer, scheduler, calculate_rouge, meta_data)
            peft_model, valid_loss, valid_ppl, valid_avg_rouge, valid_avg_bleu1, valid_avg_bleu2, valid_avg_meteor, valid_target, valid_pred = valid(peft_model, valid_loader, loss_fn, tokenizer, calculate_rouge, meta_data)

            res.append([epoch, train_loss, train_ppl, train_avg_rouge, train_avg_bleu1, train_avg_bleu2, train_avg_meteor, valid_loss, valid_ppl, valid_avg_rouge, valid_avg_bleu1, valid_avg_bleu2, valid_avg_meteor, valid_pred, valid_target])
            all_results[str(seed)]['results'] = res

        with open(meta_data['path'] + meta_data['json_output_file'], 'w') as f:
            json.dump(all_results, f, ensure_ascii = False, indent = 4)

        torch.save({
            'epoch': epoch,
            'model_state_dict': peft_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, meta_data['path'] + meta_data['checkpoint_path'].format(seed))

        if valid_loss < meta_data['save_val_loss']:
            torch.save(peft_model.state_dict(), meta_data['path'] + meta_data['model_path'].format(seed))
            meta_data['save_val_loss'] = valid_loss
            print('best model was saved')

    torch.cuda.empty_cache()
    gc.collect()

    return all_results