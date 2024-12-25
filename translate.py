
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = 'translate-model\zh-en'
tokenizer = AutoTokenizer.from_pretrained(model_path)
translate_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
pipeline = transformers.pipeline("translation", model=translate_model, tokenizer=tokenizer)

def translate(text):
    translate_text = pipeline(text)[0]['translation_text']
    return translate_text

txt='我是中国人，那么你妈妈是什么目前杰青'
trantxt=translate(txt)
print(trantxt)

txt_list=["大学各学期资料\大三上\信息内容安全","上海交通大学","特朗普新征程","华师大开始招生"]
trans_list=[]
for txt in txt_list:
    trans_txt=translate(txt)
    trans_list.append(trans_txt)
print(trans_list)


