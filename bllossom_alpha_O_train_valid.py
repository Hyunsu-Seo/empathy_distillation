import numpy as np
import pandas as pd
import math, time, gc
from tqdm import tqdm

import torch

from rouge_score import rouge_scorer

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score

def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def compute_bleu_rouge_meteor(decoded_target, decoded_pred):
    smoothie = SmoothingFunction().method4
    bleu1_scores = []
    bleu2_scores = []
    meteor_scores = []

    for ref, hyp in zip(decoded_target, decoded_pred):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()

        bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        meteor = single_meteor_score(ref_tokens, hyp_tokens)

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        meteor_scores.append(meteor)

    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return avg_bleu1, avg_bleu2, avg_meteor

def train(model, data_loader, loss_fn, tokenizer, optimizer, scheduler, calculate_rouge, meta_data):
    train_loss = 0
    pred_list = []
    target_list = []
    rouge_scores = []
    i = 0
    model.to(meta_data['device'])
    model.train()
    pbar = tqdm(data_loader)

    if meta_data['mode'] == 'tot':
        for batch in pbar:
            i += 1
            input_ids1 = batch['input_ids1'].to(meta_data['device'])
            attention_mask1 = batch['attention_mask1'].to(meta_data['device'])
            labels1 = batch['labels1'].to(meta_data['device']) # torch.Size([1, 420])

            input_ids2 = batch['input_ids2'].to(meta_data['device'])
            attention_mask2 = batch['attention_mask2'].to(meta_data['device'])
            labels2 = batch['labels2'].to(meta_data['device'])

            input_ids3 = batch['input_ids3'].to(meta_data['device'])
            attention_mask3 = batch['attention_mask3'].to(meta_data['device'])
            labels3 = batch['labels3'].to(meta_data['device'])

            input_ids4 = batch['input_ids4'].to(meta_data['device'])
            attention_mask4 = batch['attention_mask4'].to(meta_data['device'])
            labels4 = batch['labels4'].to(meta_data['device'])

            optimizer.zero_grad()

            emp_lst_outputs = model(input_ids=input_ids1, attention_mask=attention_mask1) # torch.Size([1, 420, 102400])
            loss1 = loss_fn(emp_lst_outputs.logits.permute(0,2,1), labels1).to(meta_data['device'])
            empathy_outputs = model(input_ids=input_ids2, attention_mask=attention_mask2)
            loss2 = loss_fn(empathy_outputs.logits.permute(0,2,1), labels2).to(meta_data['device'])
            empathy_loss = (1 - meta_data['alpha']) * loss1 + meta_data['alpha'] * loss2

            rat_outputs = model(input_ids=input_ids3, attention_mask=attention_mask3)
            loss3 = loss_fn(rat_outputs.logits.permute(0,2,1), labels3).to(meta_data['device'])
            res_outputs = model(input_ids=input_ids4, attention_mask=attention_mask4)
            loss4 = loss_fn(res_outputs.logits.permute(0,2,1), labels4).to(meta_data['device'])
            response_loss = (1 - meta_data['alpha']) * loss3 + meta_data['alpha'] * loss4

            loss = empathy_loss + response_loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            decoded_target = [tokenizer.decode([t for t in ids if t != -100], skip_special_tokens=True) for ids in labels4.detach().cpu().numpy()]
            decoded_pred = [tokenizer.decode(ids, skip_special_tokens=True) for ids in res_outputs.logits.argmax(dim=-1).detach().cpu().numpy()]

            target_list.extend(decoded_target)
            pred_list.extend(decoded_pred)

            # ROUGE 점수 계산 및 저장
            for ref, hyp in zip(decoded_target, decoded_pred):
                rouge_scores.append(calculate_rouge(ref, hyp))

            pbar.set_description('\033[1m[C_loss : {:>.5}]\033[0m'.format(round(train_loss / i, 4)))

    elif meta_data['mode'] == 'cot':
        for batch in pbar:
            i += 1
            input_ids1 = batch['input_ids1'].to(meta_data['device'])
            attention_mask1 = batch['attention_mask1'].to(meta_data['device'])
            labels1 = batch['labels1'].to(meta_data['device'])

            input_ids2 = batch['input_ids2'].to(meta_data['device'])
            attention_mask2 = batch['attention_mask2'].to(meta_data['device'])
            labels2 = batch['labels2'].to(meta_data['device'])

            optimizer.zero_grad()

            rat_outputs = model(input_ids=input_ids1, attention_mask=attention_mask1)
            loss1 = loss_fn(rat_outputs.logits.permute(0, 2, 1), labels1).to(meta_data['device'])

            res_outputs = model(input_ids=input_ids2, attention_mask=attention_mask2)
            loss2 = loss_fn(res_outputs.logits.permute(0, 2, 1), labels2).to(meta_data['device'])

            loss = (1 - meta_data['alpha']) * loss1 + meta_data['alpha'] * loss2

            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            decoded_target = [tokenizer.decode([t for t in ids if t != -100], skip_special_tokens=True) for ids in labels2.detach().cpu().numpy()]
            decoded_pred = [tokenizer.decode(ids, skip_special_tokens=True) for ids in res_outputs.logits.argmax(dim=-1).detach().cpu().numpy()]

            target_list.extend(decoded_target)
            pred_list.extend(decoded_pred)

            # ROUGE 점수 계산 및 저장
            for ref, hyp in zip(decoded_target, decoded_pred):
                rouge_scores.append(calculate_rouge(ref, hyp))

            pbar.set_description('\033[1m[C_loss : {:>.5}]\033[0m'.format(round(train_loss / i, 4)))

    else:
        for batch in pbar:
            i += 1
            input_ids = batch['input_ids'].to(meta_data['device'])
            attention_mask = batch['attention_mask'].to(meta_data['device'])
            labels = batch['labels'].to(meta_data['device'])

            optimizer.zero_grad()
            
            outputs= model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits.permute(0,2,1), labels).to(meta_data['device'])
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            decoded_target = [tokenizer.decode([t for t in ids if t != -100], skip_special_tokens=True) for ids in labels.detach().cpu().numpy()]
            decoded_pred = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs.logits.argmax(dim=-1).detach().cpu().numpy()]

            target_list.extend(decoded_target)
            pred_list.extend(decoded_pred)

            # ROUGE 점수 계산 및 저장
            for ref, hyp in zip(decoded_target, decoded_pred):
                rouge_scores.append(calculate_rouge(ref, hyp))

            pbar.set_description('\033[1m[C_loss : {:>.5}]\033[0m'.format(round(train_loss / i, 4)))

    train_loss = train_loss / len(data_loader)

    # ppl
    ppl = math.exp(train_loss)
    # rouge
    avg_rouge = {
        key: sum(d[key].fmeasure for d in rouge_scores) / len(rouge_scores)
        for key in rouge_scores[0]
    }

    avg_bleu1, avg_bleu2, avg_meteor = compute_bleu_rouge_meteor(target_list, pred_list)

    torch.cuda.empty_cache()
    gc.collect()

    return model, train_loss, ppl, avg_rouge, avg_bleu1, avg_bleu2, avg_meteor

def valid(model, data_loader, loss_fn, tokenizer, calculate_rouge, meta_data):
    valid_loss  = 0
    pred_list = []
    target_list = []
    rouge_scores = []
    i = 0
    model.to(meta_data['device'])
    model.eval()
    pbar = tqdm(data_loader)

    if meta_data['mode'] == 'tot':
        pred_emp_list = []
        target_emp_list = []
        for batch in pbar:
            i += 1
            input_ids1 = batch['input_ids1'].to(meta_data['device'])
            attention_mask1 = batch['attention_mask1'].to(meta_data['device'])
            labels1 = batch['labels1'].to(meta_data['device'])

            input_ids2 = batch['input_ids2'].to(meta_data['device'])
            attention_mask2 = batch['attention_mask2'].to(meta_data['device'])
            labels2 = batch['labels2'].to(meta_data['device'])

            input_ids3 = batch['input_ids3'].to(meta_data['device'])
            attention_mask3 = batch['attention_mask3'].to(meta_data['device'])
            labels3 = batch['labels3'].to(meta_data['device'])

            input_ids4 = batch['input_ids4'].to(meta_data['device'])
            attention_mask4 = batch['attention_mask4'].to(meta_data['device'])
            labels4 = batch['labels4'].to(meta_data['device'])

            emp_lst_outputs = model(input_ids=input_ids1, attention_mask=attention_mask1)
            loss1 = loss_fn(emp_lst_outputs.logits.permute(0,2,1), labels1).to(meta_data['device'])
            empathy_outputs = model(input_ids=input_ids2, attention_mask=attention_mask2)
            loss2 = loss_fn(empathy_outputs.logits.permute(0,2,1), labels2).to(meta_data['device'])
            empathy_loss = (1 - meta_data['alpha']) * loss1 + meta_data['alpha'] * loss2

            rat_outputs = model(input_ids=input_ids3, attention_mask=attention_mask3)
            loss3 = loss_fn(rat_outputs.logits.permute(0,2,1), labels3).to(meta_data['device'])
            res_outputs = model(input_ids=input_ids4, attention_mask=attention_mask4)
            loss4 = loss_fn(res_outputs.logits.permute(0,2,1), labels4).to(meta_data['device'])
            response_loss = (1 - meta_data['alpha']) * loss3 + meta_data['alpha'] * loss4

            loss = empathy_loss + response_loss
            valid_loss += loss.item()

            # 감정 저장
            decoded_emp_target = [tokenizer.decode([t for t in ids if t != -100], skip_special_tokens=True) for ids in labels2.detach().cpu().numpy()]
            decoded_emp_pred = [tokenizer.decode(ids, skip_special_tokens=True) for ids in empathy_outputs.logits.argmax(dim=-1).detach().cpu().numpy()]

            target_emp_list.extend(decoded_emp_target)
            pred_emp_list.extend(decoded_emp_pred)

            decoded_target = [tokenizer.decode([t for t in ids if t != -100], skip_special_tokens=True) for ids in labels4.detach().cpu().numpy()]
            decoded_pred = [tokenizer.decode(ids, skip_special_tokens=True) for ids in res_outputs.logits.argmax(dim=-1).detach().cpu().numpy()]

            target_list.extend(decoded_target)
            pred_list.extend(decoded_pred)

            for ref, hyp in zip(decoded_target, decoded_pred):
                rouge_scores.append(calculate_rouge(ref, hyp))

            pbar.set_description('\033[1m[C_loss : {:>.5}]\033[0m'.format(round(valid_loss / i, 4)))

        valid_loss = valid_loss / len(data_loader)

        # ppl
        ppl = math.exp(valid_loss)
        # rouge
        avg_rouge = {
            key: sum(d[key].fmeasure for d in rouge_scores) / len(rouge_scores)
            for key in rouge_scores[0]
        }

        avg_bleu1, avg_bleu2, avg_meteor = compute_bleu_rouge_meteor(target_list, pred_list)

        torch.cuda.empty_cache()
        gc.collect()

        return model, valid_loss, ppl, avg_rouge, avg_bleu1, avg_bleu2, avg_meteor, pred_emp_list, target_emp_list, target_list, pred_list

    elif meta_data['mode'] == 'cot':
        for batch in pbar:
            i += 1
            input_ids1 = batch['input_ids1'].to(meta_data['device'])
            attention_mask1 = batch['attention_mask1'].to(meta_data['device'])
            labels1 = batch['labels1'].to(meta_data['device'])

            input_ids2 = batch['input_ids2'].to(meta_data['device'])
            attention_mask2 = batch['attention_mask2'].to(meta_data['device'])
            labels2 = batch['labels2'].to(meta_data['device'])

            rat_outputs = model(input_ids=input_ids1, attention_mask=attention_mask1)
            loss1 = loss_fn(rat_outputs.logits.permute(0,2,1), labels1).to(meta_data['device'])

            res_outputs = model(input_ids=input_ids2, attention_mask=attention_mask2)
            loss2 = loss_fn(res_outputs.logits.permute(0,2,1), labels2).to(meta_data['device'])

            loss = (1 - meta_data['alpha']) * loss1 + meta_data['alpha'] * loss2

            valid_loss += loss.item()

            decoded_target = [tokenizer.decode([t for t in ids if t != -100], skip_special_tokens=True) for ids in labels2.detach().cpu().numpy()]
            decoded_pred = [tokenizer.decode(ids, skip_special_tokens=True) for ids in res_outputs.logits.argmax(dim=-1).detach().cpu().numpy()]

            target_list.extend(decoded_target)
            pred_list.extend(decoded_pred)

            for ref, hyp in zip(decoded_target, decoded_pred):
                rouge_scores.append(calculate_rouge(ref, hyp))

            pbar.set_description('\033[1m[C_loss : {:>.5}]\033[0m'.format(round(valid_loss / i, 4)))

        valid_loss = valid_loss / len(data_loader)

        # ppl
        ppl = math.exp(valid_loss)
        # rouge
        avg_rouge = {
            key: sum(d[key].fmeasure for d in rouge_scores) / len(rouge_scores)
            for key in rouge_scores[0]
        }

        avg_bleu1, avg_bleu2, avg_meteor = compute_bleu_rouge_meteor(target_list, pred_list)

        torch.cuda.empty_cache()
        gc.collect()

        return model, valid_loss, ppl, avg_rouge, avg_bleu1, avg_bleu2, avg_meteor, target_list, pred_list

    else:
        for batch in pbar:
            i += 1
            input_ids = batch['input_ids'].to(meta_data['device'])
            attention_mask = batch['attention_mask'].to(meta_data['device'])
            labels = batch['labels'].to(meta_data['device'])

            outputs= model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits.permute(0,2,1), labels).to(meta_data['device'])
            valid_loss += loss.item()

            decoded_target = [tokenizer.decode([t for t in ids if t != -100], skip_special_tokens=True) for ids in labels.detach().cpu().numpy()]
            decoded_pred = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs.logits.argmax(dim=-1).detach().cpu().numpy()]

            target_list.extend(decoded_target)
            pred_list.extend(decoded_pred)

            for ref, hyp in zip(decoded_target, decoded_pred):
                rouge_scores.append(calculate_rouge(ref, hyp))

            pbar.set_description('\033[1m[C_loss : {:>.5}]\033[0m'.format(round(valid_loss / i, 4)))

        valid_loss = valid_loss / len(data_loader)

        # ppl
        ppl = math.exp(valid_loss)
        # rouge
        avg_rouge = {
            key: sum(d[key].fmeasure for d in rouge_scores) / len(rouge_scores)
            for key in rouge_scores[0]
        }

        avg_bleu1, avg_bleu2, avg_meteor = compute_bleu_rouge_meteor(target_list, pred_list)

        torch.cuda.empty_cache()
        gc.collect()

        return model, valid_loss, ppl, avg_rouge, avg_bleu1, avg_bleu2, avg_meteor, target_list, pred_list