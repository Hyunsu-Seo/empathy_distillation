content1 = 'You are EXAONE model from LG AI Research, a helpful assistant.'
content2 = '당신은 화자의 발언에 공감하는 청자 역할입니다.'

tot_prompt1 = '''
[|system|]{content}[|endofturn|]
[|user|]화자의 발화에 적절한 청자의 태도를 4개 생각하세요.
### 화자의 발화: {s}
### 청자의 태도 4개:[|endofturn|]
[|assistant|]{empathy_lst}[|endofturn|]'''

tot_prompt2 = '''
[|system|]{content}[|endofturn|]
[|user|]청자의 태도 4개 중 가장 적절한 한 개를 선택하세요.
### 화자의 발화: {s}
### 청자 태도 4개: {empathy_lst}
### 가장 적절한 청자 태도:[|endofturn|]
[|assistant|]{decided_empathy}[|endofturn|]'''

tot_prompt3 = '''
[|system|]{content}[|endofturn|]
[|user|]주어진 청자의 태도를 기반으로 화자의 발화에 대한 공감 답변을 생성하기 위한 근거를 만드세요.
### 화자의 발화: {s}
### 청자의 태도: {decided_empathy}
### 청자의 공감 답변 생성 근거:[|endofturn|]
[|assistant|]{res_rationale}[|endofturn|]'''

cot_prompt = '''
[|system|]{content}[|endofturn|]
[|user|]화자의 발화에 대한 공감 답변을 생성하기 위한 근거를 만드세요.
### 화자의 발화: {s}
### 청자의 공감 답변 생성 근거:[|endofturn|]
[|assistant|]{cot_rationale}[|endofturn|]'''

basic_prompt = '''
[|system|]{content}[|endofturn|]
[|user|]화자의 발화에 대한 공감 답변을 생성하세요.
### 화자의 발화: {s}
### 청자의 공감 답변:[|endofturn|]
[|assistant|]{response}[|endofturn|]'''

test_prompt = '''
[|system|]{content}[|endofturn|]
[|user|]화자의 발화에 대한 공감 답변을 생성하세요.
### 화자의 발화: {s}
### 청자의 공감 답변:[|endofturn|]
[|assistant|]'''