from torch.utils.data import Dataset
from kanana_prompt import content1, content2, tot_prompt1, tot_prompt2, tot_prompt3, cot_prompt, basic_prompt

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, content1, content2, tot_prompt1, tot_prompt2, tot_prompt3, cot_prompt, basic_prompt, max_len, mode):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.content1 = content1
        self.content2 = content2
        self.tot_prompt1 = tot_prompt1
        self.tot_prompt2 = tot_prompt2
        self.tot_prompt3 = tot_prompt3
        self.cot_prompt = cot_prompt
        self.basic_prompt = basic_prompt
        self.max_len = max_len
        self.mode = mode

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        s = row['context']
        if self.mode == 'tot':
            empathy_lst = row['empathy_lst'] # 적절한 감정 4개 생성
            decided_empathy = row['decided_empathy'] # 4개 중 택1
            response = row['response'] # 정답
            res_rationale = row['res_rationale'] # 추론

            # 감정 4개 생성 프롬프트
            prompt_text1 = self.tot_prompt1.format(content = self.content1,
                                                   s = s,
                                                   empathy_lst = empathy_lst).strip()
            inputs1 = self.tokenizer(prompt_text1,
                                     truncation=True,
                                     max_length=self.max_len + 1,
                                     padding='max_length',
                                     return_tensors='pt')

            input_ids1 = inputs1['input_ids'][0]
            attention_mask1 = inputs1['attention_mask'][0]

            labels1 = input_ids1[1:].clone()
            labels1[labels1 == self.tokenizer.pad_token_id] = -100

            input_ids1 = input_ids1[:-1]
            attention_mask1 = attention_mask1[:-1]

            # 감정 1개 선택 프롬프트
            prompt_text2 = self.tot_prompt2.format(content = self.content1,
                                                   s = s,
                                                   empathy_lst = empathy_lst,
                                                   decided_empathy = decided_empathy).strip()
            inputs2 = self.tokenizer(prompt_text2,
                                     truncation=True,
                                     max_length=self.max_len + 1,
                                     padding='max_length',
                                     return_tensors='pt')

            input_ids2 = inputs2['input_ids'][0]
            attention_mask2 = inputs2['attention_mask'][0]

            labels2 = input_ids2[1:].clone()
            labels2[labels2 == self.tokenizer.pad_token_id] = -100

            input_ids2 = input_ids2[:-1]
            attention_mask2 = attention_mask2[:-1]

            # 추론 프롬프트
            prompt_text3 = self.tot_prompt3.format(content = self.content1,
                                                   s = s,
                                                   decided_empathy = decided_empathy,
                                                   res_rationale = res_rationale).strip()
            inputs3 = self.tokenizer(prompt_text3,
                                     truncation=True,
                                     max_length=self.max_len + 1,
                                     padding='max_length',
                                     return_tensors='pt')

            input_ids3 = inputs3['input_ids'][0]
            attention_mask3 = inputs3['attention_mask'][0]

            labels3 = input_ids3[1:].clone()
            labels3[labels3 == self.tokenizer.pad_token_id] = -100

            input_ids3 = input_ids3[:-1]
            attention_mask3 = attention_mask3[:-1]

            # 정답 프롬프트
            prompt_text4 = self.basic_prompt.format(content = self.content2,
                                                    s = s,
                                                    response = response).strip()
            inputs4 = self.tokenizer(prompt_text4,
                                     truncation=True,
                                     max_length=self.max_len + 1,
                                     padding='max_length',
                                     return_tensors='pt')

            input_ids4 = inputs4['input_ids'][0]
            attention_mask4 = inputs4['attention_mask'][0]

            labels4 = input_ids4[1:].clone()
            labels4[labels4 == self.tokenizer.pad_token_id] = -100

            input_ids4 = input_ids4[:-1]
            attention_mask4 = attention_mask4[:-1]

            # return input_ids1, attention_mask1, labels1, input_ids2, attention_mask2, labels2, input_ids3, attention_mask3, labels3, input_ids4, attention_mask4, labels4
            return {
                "input_ids1": input_ids1,
                "attention_mask1": attention_mask1,
                "labels1": labels1,
                "input_ids2": input_ids2,
                "attention_mask2": attention_mask2,
                "labels2": labels2,
                "input_ids3": input_ids3,
                "attention_mask3": attention_mask3,
                "labels3": labels3,
                "input_ids4": input_ids4,
                "attention_mask4": attention_mask4,
                "labels4": labels4
                }

        elif self.mode == 'cot':
            response = row['cot_response']
            rationale = row['cot_rationale']

            # rationale 프롬프트
            prompt_text1 = self.cot_prompt.format(content = self.content1,
                                                  s = s,
                                                  cot_rationale = rationale).strip()
            inputs1 = self.tokenizer(prompt_text1,
                                     truncation=True,
                                     max_length=self.max_len + 1,
                                     padding='max_length',
                                     return_tensors='pt')

            input_ids1 = inputs1['input_ids'][0]
            attention_mask1 = inputs1['attention_mask'][0]

            labels1 = input_ids1[1:].clone()
            labels1[labels1 == self.tokenizer.pad_token_id] = -100

            input_ids1 = input_ids1[:-1]
            attention_mask1 = attention_mask1[:-1]

            # 정답 프롬프트
            prompt_text2 = self.basic_prompt.format(content = self.content2,
                                                    s = s,
                                                    response = response).strip()
            inputs2 = self.tokenizer(prompt_text2,
                                     truncation=True,
                                     max_length=self.max_len + 1,
                                     padding='max_length',
                                     return_tensors='pt')

            input_ids2 = inputs2['input_ids'][0]
            attention_mask2 = inputs2['attention_mask'][0]

            labels2 = input_ids2[1:].clone()
            labels2[labels2 == self.tokenizer.pad_token_id] = -100

            input_ids2 = input_ids2[:-1]
            attention_mask2 = attention_mask2[:-1]

            # return input_ids1, attention_mask1, labels1, input_ids2, attention_mask2, labels2
            return {
                "input_ids1": input_ids1,
                "attention_mask1": attention_mask1,
                "labels1": labels1,
                "input_ids2": input_ids2,
                "attention_mask2": attention_mask2,
                "labels2": labels2
                }

        else:
            response = row['response']

            # 메시지 구성
            prompt_text = self.basic_prompt.format(content = self.content2,
                                                    s=s,
                                                    response=response).strip()

            inputs = self.tokenizer(prompt_text,
                                    truncation=True,
                                    max_length=self.max_len + 1,
                                    padding='max_length',
                                    return_tensors='pt')

            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            labels = input_ids[1:].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
                }

    def __len__(self):
        return len(self.dataset)