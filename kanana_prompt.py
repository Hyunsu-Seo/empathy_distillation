content1 = 'You are a helpful AI assistant developed by Kakao.'
content2 = '당신은 화자의 발언에 공감하는 청자 역할입니다.'

tot_prompt1 = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{content}<|eot_id|><|start_header_id|>user<|end_header_id|>
화자의 발화에 적절한 청자의 태도를 4개 생각하세요.
### 화자의 발화: {s}
### 청자의 태도 4개:<|eot_id|><|start_header_id|>assistant<|end_header_id|>{empathy_lst}<|eot_id|>'''

tot_prompt2 = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{content}<|eot_id|><|start_header_id|>user<|end_header_id|>
청자의 태도 4개 중 가장 적절한 한 개를 선택하세요.
### 화자의 발화: {s}
### 청자 태도 4개: {empathy_lst}
### 가장 적절한 청자 태도:<|eot_id|><|start_header_id|>assistant<|end_header_id|>{decided_empathy}<|eot_id|>'''

tot_prompt3 = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{content}<|eot_id|><|start_header_id|>user<|end_header_id|>
주어진 청자의 태도를 기반으로 화자의 발화에 대한 공감 답변을 생성하기 위한 근거를 만드세요.
### 화자의 발화: {s}
### 청자의 태도: {decided_empathy}
### 청자의 공감 답변 생성 근거:<|eot_id|><|start_header_id|>assistant<|end_header_id|>{res_rationale}<|eot_id|>'''

cot_prompt = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{content}<|eot_id|><|start_header_id|>user<|end_header_id|>
화자의 발화에 대한 공감 답변을 생성하기 위한 근거를 만드세요.
### 화자의 발화: {s}
### 청자의 공감 답변 생성 근거:<|eot_id|><|start_header_id|>assistant<|end_header_id|>{cot_rationale}<|eot_id|>'''

basic_prompt = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{content}<|eot_id|><|start_header_id|>user<|end_header_id|>
화자의 발화에 대한 공감 답변을 생성하세요.
### 화자의 발화: {s}
### 청자의 공감 답변:<|eot_id|><|start_header_id|>assistant<|end_header_id|>{response}<|eot_id|>'''

test_prompt = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{content}<|eot_id|><|start_header_id|>user<|end_header_id|>
화자의 발화에 대한 공감 답변을 생성하세요.
### 화자의 발화: {s}
### 청자의 공감 답변:<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''