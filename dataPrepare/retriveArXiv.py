# nohup python3 -u retriveArXiv.py > retriveArXiv20231007.log 2>&1 &

import json
from tqdm import tqdm
import pickle as pk
readfile = "/data/jx4237data/DataForChatGPTinnovationWaves/"
from dateutil import parser as date_parser
from collections import Counter


# count #line
# line_count = 0
# with open(readfile+'arxiv_metadata_20230730.json') as rf:
#   for line in tqdm(rf):
#     line_count += 1
# print('line_count:',line_count)



# plotting 会更清楚
import numpy as np
# plot 同时中英混排
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['figure.dpi']= 200
plt.figure(figsize=(4,3), dpi= 200)
# 字体设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 24 # 设置字体大小
plt.rcParams['axes.unicode_minus'] = False # 使坐标轴刻度标签正常显示正负号


# chatGPT ----------------------------------------------------------------------------------------------------------------------------------
# query: chatgpt
chatGPTset = set()
ChatGPTyearCounter = Counter()
ChatGPTlatesTyearCounter = Counter()
ChatGPTallActivitiesCounter = Counter()

with open(readfile+'arxiv-metadata-20231007.json') as rf:
  for line in tqdm(rf):
    j = json.loads(line)
    if 'chatgpt' in j['title'].lower() or 'chatgpt' in j['abstract'].lower():
      chatGPTset.add(j['id'])
      ChatGPTyearCounter[date_parser.parse(j['versions'][0]['created']).strftime("%Y-%m")] += 1
      ChatGPTlatesTyearCounter[date_parser.parse(j['versions'][-1]['created']).strftime("%Y-%m")] += 1
      for version in j['versions']:
        ChatGPTallActivitiesCounter[date_parser.parse(version['created']).strftime("%Y-%m")] += 1
print('len(chatGPTset)',len(chatGPTset))
pk.dump(chatGPTset, open(readfile+'TemporalResults/chatGPTset.pkl','wb+'))
print('ChatGPTyearCounter:',ChatGPTyearCounter)
print('ChatGPTlatesTyearCounter:',ChatGPTlatesTyearCounter)
print('ChatGPTallActivitiesCounter:',ChatGPTallActivitiesCounter)

# draw
yearMonth , frequency = zip(*sorted(ChatGPTyearCounter.items()))
yearMonth2 , frequency2 = zip(*sorted(ChatGPTlatesTyearCounter.items()))
yearMonth3 , frequency3 = zip(*sorted(ChatGPTallActivitiesCounter.items()))

fig, ax = plt.subplots()
ax.plot(yearMonth,np.array(frequency),marker='<',markersize=5,label='earliest version')
ax.plot(yearMonth2,np.array(frequency2),marker='>',markersize=5,label='latest version')
ax.plot(yearMonth3,np.array(frequency3),marker='.',markersize=10,label='all update activities')

for i, txt in enumerate(frequency3):
    ax.annotate(txt, (yearMonth3[i],frequency3[i]+10),fontsize = 12)
ax.set_xlabel('year-month', fontsize = 15)
ax.set_ylabel('#arXiv publications', fontsize = 15)
ax.set_title('Figure 1. #arXiv publications contain "ChatGPT" in Title or Abstract (monthly, as of Sept. 2023)\n',fontsize=15)
plt.text(.05,1.2,'total # of publications: %d'%len(chatGPTset),
horizontalalignment='center',verticalalignment='center',
transform=ax.transAxes,fontsize=15)
ax.set_yticks([i for i in range(0,401,25)])
ax.xaxis.set_tick_params(labelsize=12)
ax.set_xticklabels(yearMonth,rotation = 66,)
ax.yaxis.set_tick_params(labelsize=12)
ax.legend()
plt.show()

plt.savefig(readfile+'vis/figure1_chatgpt.png', bbox_inches='tight')


# GPT ----------------------------------------------------------------------------------------------------------------------------------

# query: gpt
GPTset = set()
GPTyearCounter = Counter()
GPTlatesTyearCounter = Counter()
GPTallActivitiesCounter = Counter()
with open(readfile+'arxiv-metadata-20231007.json') as rf:
  for line in tqdm(rf):
    j = json.loads(line)
    # replace \n
    alllowerCase = j['title'].lower().replace('\n ','') + j['abstract'].lower().replace('\n',' ')
    if ('generative pre-training' in alllowerCase)  or ('generative pre-trained' in alllowerCase) or (('gpt' in alllowerCase) and ('chatgpt' in alllowerCase or 'openai' in alllowerCase or 'transformer' in alllowerCase  or 'gpt-' in alllowerCase or  'language model' in alllowerCase or 'llm' in alllowerCase or 'nlp' in alllowerCase or 'computer' in alllowerCase or 'generative' in alllowerCase)):
        if date_parser.parse(j['versions'][0]['created']).year < 2018:
           print(j['versions'][0]['created'])
           print(alllowerCase)
        GPTset.add(j['id'])
        GPTyearCounter[date_parser.parse(j['versions'][0]['created']).strftime("%Y")] += 1
        GPTlatesTyearCounter[date_parser.parse(j['versions'][-1]['created']).strftime("%Y")] += 1
        for version in j['versions']:
           GPTallActivitiesCounter[date_parser.parse(version['created']).strftime("%Y")] += 1
print('len(GPTset)',len(GPTset))
pk.dump(GPTset, open(readfile+'TemporalResults/GPTset.pkl','wb+'))


# draw
yearMonth , frequency = zip(*sorted(GPTyearCounter.items()))
yearMonth2 , frequency2 = zip(*sorted(GPTlatesTyearCounter.items()))
yearMonth3 , frequency3 = zip(*sorted(GPTallActivitiesCounter.items()))

fig, ax = plt.subplots()
ax.plot(yearMonth,np.array(frequency),marker='<',markersize=5,label='earliest version')
ax.plot(yearMonth2,np.array(frequency2),marker='>',markersize=5,label='latest version')
ax.plot(yearMonth3,np.array(frequency3),marker='.',markersize=10,label='all update activities')

for i, txt in enumerate(frequency3):
    ax.annotate(txt, (yearMonth3[i],frequency3[i]+10),fontsize = 12)
ax.set_xlabel('year', fontsize = 15)
ax.set_ylabel('#arXiv publications', fontsize = 15)
ax.set_title('Figure 2. #arXiv publications contain "GPT" in Title or Abstract (yearly, as of July 2023)\n',fontsize=15)
plt.text(.05,1.2,'total # of publications: %d'%len(GPTset),
horizontalalignment='center',verticalalignment='center',
transform=ax.transAxes,fontsize=15)
ax.set_yticks([i for i in range(0,3000,200)])
ax.xaxis.set_tick_params(labelsize=8)
# ax.set_xticklabels(yearMonth,rotation = 90,)
ax.yaxis.set_tick_params(labelsize=12)
ax.legend()
plt.show()

plt.savefig(readfile+'vis/figure2_gpt.png', bbox_inches='tight')



# LLMs----------------------------------------------------------------------------------------------------------------------------------------------

# query: LLMs （Just expand on the GPT set）
LLMset = set()
with open(readfile+'arxiv-metadata-20231007.json') as rf:
  for line in tqdm(rf):
    j = json.loads(line)
    normalCase = j['title'].replace('\n ','') + ' ' + j['abstract'].replace('\n',' ')
    alllowerCase = normalCase.lower()
    if ('large language model' in alllowerCase or 'LLMs' in normalCase):
        if date_parser.parse(j['versions'][0]['created']).year > 2017:
            LLMset.add(j['id'])
print('len(LLMset)',len(LLMset))
pk.dump(LLMset, open(readfile+'TemporalResults/LLMset.pkl','wb+'))

# 选比较有独特性的词汇，看一下这些词各自出现过多少次
# zero-shot prompt; chain-of-thought prompt; few-shot cot; few-shot chain-of-thought
# flan-t5, flan-palm, cpm-2, ernie 3, pangu-, jurassic-, hyperclova, yuan 1, megatron-turing natural language generation, new bing, bing chat, tk-instruct, mt-nlg, opt-iml
# OpenAI, GShard, GLaM, LaMDA, PaLM, AlphaCode, CodeGen, LLaMA, YaLM, AlexaTM, WeLM, CodeGeeX

# Anthropic & Claude
# Google & {T5, FLAN, UL2, Bard}
# BigScience & {T0, BLOOM}
# OpenAI & {Codex}
# DeepMind & {Gopher, Chinchilla, Sparrow}
# Meta & {OPT, NLLB, Galatica}
# EleutherAI & {Pythia}
# lmsys & vicuna

seTzero_shot = set()
seTcot = set()
seTfewShotCOT = set()
seTfewShotChain = set()
seTflan_t5 = set()
seTflan_palm = set()
seTcpm_2 = set()
seTernie3 = set()
seTpangu = set()
seTjurassic = set()
seThyperclova = set()
seTyuan = set()
seTnewbing = set()
seTbingChat = set()
seTtk = set()
seTmtnlg = set()
seToptiml = set()
# normal case
openaiSet = set()
gshardSet = set()
glamSet = set()
lamdaSet = set()
palmSet = set()
alphacodeSet = set()
codeGenSet = set()
llamaSet = set()
yalmSet = set()
alexatmSet = set()
welmSet = set()
codeGeexSet = set()
# combine set
claudeSet = set()
T5Set = set()
flanSet = set()
ul2Set = set()
BardSet = set()
T0Set = set()
BLOOMSet = set()
CodexSet = set()
GopherSet = set()
ChinchillaSet = set()
SparrowSet = set()
OPTSet = set()
NLLBSet = set()
GalaticaSet = set()
PythiaSet = set()
vicunaSet = set()

with open(readfile+'arxiv-metadata-20231007.json') as rf:
  for line in tqdm(rf):
    j = json.loads(line)
    normalCase = j['title'].replace('\n ','') + ' ' + j['abstract'].replace('\n',' ')
    alllowerCase = normalCase.lower()
    if 'OpenAI' in normalCase and 'OpenAI Gym' not in normalCase:
       openaiSet.add(j['id'])
    if 'GShard' in normalCase:
       gshardSet.add(j['id'])
    if 'GLaM' in normalCase:
       glamSet.add(j['id'])
    if 'LaMDA' in normalCase:
       lamdaSet.add(j['id'])
    if 'PaLM' in normalCase:
       palmSet.add(j['id'])
    if 'AlphaCode' in normalCase:
       alphacodeSet.add(j['id'])
    if 'CodeGen' in normalCase:
       codeGenSet.add(j['id'])
    if 'LLaMA' in normalCase or ('llama' in alllowerCase and 'language model' in alllowerCase):
       llamaSet.add(j['id'])
    if 'YaLM' in normalCase:
       yalmSet.add(j['id'])
    if 'AlexaTM' in normalCase:
       alexatmSet.add(j['id'])
    if 'WeLM' in normalCase:
       welmSet.add(j['id'])
    if 'CodeGeeX' in normalCase:
       codeGeexSet.add(j['id'])
    if 'zero-shot prompt' in alllowerCase:
       seTzero_shot.add(j['id'])
    if 'chain-of-thought prompt' in alllowerCase:
       seTcot.add(j['id'])
    if 'few-shot cot' in alllowerCase:
       seTfewShotCOT.add(j['id'])
    if 'few-shot chain-of-thought' in alllowerCase:
       seTfewShotChain.add(j['id'])
    if 'flan-t5' in alllowerCase:
       seTflan_t5.add(j['id'])
    if 'flan-palm' in alllowerCase:
       seTflan_palm.add(j['id'])
    if 'cpm-2' in alllowerCase:
       seTcpm_2.add(j['id'])
    if 'ernie 3' in alllowerCase:
       seTernie3.add(j['id'])
    if 'pangu-' in alllowerCase:
       seTpangu.add(j['id'])
    if 'jurassic-' in alllowerCase:
       seTjurassic.add(j['id'])
    if 'hyperclova' in alllowerCase:
       seThyperclova.add(j['id'])
    if 'yuan 1' in alllowerCase:
       seTyuan.add(j['id'])
    if 'new bing' in alllowerCase:
       seTnewbing.add(j['id'])
    if 'bing chat' in alllowerCase:
       seTbingChat.add(j['id'])
    if 'tk-instruct' in alllowerCase:
       seTtk.add(j['id'])
    if 'mt-nlg' in alllowerCase:
       seTmtnlg.add(j['id'])
    if 'opt-iml' in alllowerCase:
       seToptiml.add(j['id'])
    if ('Anthropic' in normalCase or 'language model' in alllowerCase) and 'Claude' in normalCase:
        claudeSet.add(j['id'])
    if ('Google' in normalCase or 'language model' in alllowerCase) and 'T5' in normalCase:
        T5Set.add(j['id'])
    if ('Google' in normalCase or 'language model' in alllowerCase) and 'FLAN' in normalCase:
        flanSet.add(j['id'])
    if ('Google' in normalCase or 'language model' in alllowerCase) and 'UL2' in normalCase:
        ul2Set.add(j['id'])
    if ('Google' in normalCase or 'language model' in alllowerCase) and 'Bard' in normalCase:
        BardSet.add(j['id'])
    if ('BigScience' in normalCase or 'language model' in alllowerCase) and 'T0' in normalCase:
       T0Set.add(j['id'])
    if ('BigScience' in normalCase or 'language model' in alllowerCase) and 'BLOOM' in normalCase:
        BLOOMSet.add(j['id'])
    if ('OpenAI' in normalCase or 'language model' in alllowerCase) and 'Codex' in normalCase:
        CodexSet.add(j['id'])
    if ('DeepMind' in normalCase or 'language model' in alllowerCase) and 'Gopher' in normalCase:
        GopherSet.add(j['id'])
    if ('DeepMind' in normalCase or 'language model' in alllowerCase) and 'Chinchilla' in normalCase:
        ChinchillaSet.add(j['id'])
    if ('DeepMind' in normalCase or 'language model' in alllowerCase) and 'Sparrow' in normalCase:
        SparrowSet.add(j['id'])
    if ('Meta' in normalCase or 'language model' in alllowerCase or 'facebook' in normalCase) and 'OPT' in normalCase:
        OPTSet.add(j['id'])
    if ('Meta' in normalCase or 'language model' in alllowerCase or 'facebook' in normalCase) and 'NLLB' in normalCase:
        NLLBSet.add(j['id'])
    if ('Meta' in normalCase or 'language model' in alllowerCase or 'facebook' in normalCase) and 'Galatica' in normalCase:
        GalaticaSet.add(j['id'])
    if ('EleutherAI' in normalCase or 'language model' in alllowerCase) and 'Pythia' in normalCase:
        PythiaSet.add(j['id'])
    if ('lmsys' in alllowerCase or 'language model' in alllowerCase) and 'vicuna' in alllowerCase:
        vicunaSet.add(j['id'])

lowercaseset = seTzero_shot | seTcot | seTfewShotCOT | seTfewShotChain | seTflan_t5 | seTflan_palm | seTcpm_2 | seTernie3 | seTpangu | seTjurassic | seThyperclova | seTyuan | seTnewbing | seTbingChat | seTtk | seTmtnlg | seToptiml
print('len(lowercaseset)', len(lowercaseset))
pk.dump(lowercaseset, open(readfile+'TemporalResults/lowercaseset.pkl','wb+'))
print('zero-shot prompt', len(seTzero_shot))
print('chain-of-thought prompt', len(seTcot))
print('seTfewShotCOT',len(seTfewShotCOT))
print('seTfewShotChain',len(seTfewShotChain))
print('seTflan_t5',len(seTflan_t5))
print('seTflan_palm',len(seTflan_palm))
print('seTcpm_2',len(seTcpm_2))
print('seTernie3',len(seTernie3))
print('seTpangu',len(seTpangu))
print('seTjurassic',len(seTjurassic))
print('seThyperclova',len(seThyperclova))
print('seTyuan',len(seTyuan))
print('seTnewbing',len(seTnewbing))
print('seTbingChat',len(seTbingChat))
print('seTtk',len(seTtk))
print('seTmtnlg',len(seTmtnlg))
print('seToptiml',len(seToptiml))


normalCaseset = openaiSet | gshardSet | glamSet | lamdaSet | palmSet | alphacodeSet | codeGenSet | llamaSet | yalmSet | alexatmSet | welmSet | codeGeexSet
print('len(normalCaseset)', len(normalCaseset))
pk.dump(normalCaseset, open(readfile+'TemporalResults/normalCaseset.pkl','wb+'))
print('openaiSet', len(openaiSet))
print('gshardSet', len(gshardSet))
print('glamSet',len(glamSet))
print('lamdaSet',len(lamdaSet))
print('palmSet',len(palmSet))
print('alphacodeSet',len(alphacodeSet))
print('codeGenSet',len(codeGenSet))
print('llamaSet',len(llamaSet))
print('yalmSet',len(yalmSet))
print('alexatmSet',len(alexatmSet))
print('welmSet',len(welmSet))
print('codeGeexSet',len(codeGeexSet))


CombineSet = claudeSet | T5Set | flanSet | ul2Set | BardSet | T0Set | BLOOMSet | CodexSet | GopherSet | ChinchillaSet | SparrowSet | OPTSet | NLLBSet | GalaticaSet | PythiaSet | vicunaSet 
print('len(CombineSet)', len(CombineSet))
pk.dump(CombineSet, open(readfile+'TemporalResults/CombineSet.pkl','wb+'))
print('claudeSet', len(claudeSet))
print('T5Set', len(T5Set))
print('flanSet', len(flanSet))
print('ul2Set', len(ul2Set))
print('BardSet', len(BardSet))
print('T0Set', len(T0Set))
print('BLOOMSet', len(BLOOMSet))
print('CodexSet', len(CodexSet))
print('GopherSet', len(GopherSet))
print('ChinchillaSet', len(ChinchillaSet))
print('SparrowSet', len(SparrowSet))
print('OPTSet', len(OPTSet))
print('NLLBSet', len(NLLBSet))
print('GalaticaSet', len(GalaticaSet))
print('PythiaSet', len(PythiaSet))
print('vicunaSet', len(vicunaSet))



### get jsons subset -----------------------------------------------------------------------------------------------------------------
chatGPTset = pk.load(open(readfile+'TemporalResults/chatGPTset.pkl','rb'))
GPTset = pk.load( open(readfile+'TemporalResults/GPTset.pkl','rb'))
LLMset = pk.load( open(readfile+'TemporalResults/LLMset.pkl','rb'))
lowercaseset = pk.load(open(readfile+'TemporalResults/lowercaseset.pkl','rb'))
normalCaseset = pk.load(open(readfile+'TemporalResults/normalCaseset.pkl','rb'))
CombineSet = pk.load(open(readfile+'TemporalResults/CombineSet.pkl','rb'))

print(len(chatGPTset))
print(len(GPTset))
print(len(GPTset | LLMset  | lowercaseset | normalCaseset | CombineSet))
allset = GPTset | LLMset  | lowercaseset | normalCaseset | CombineSet


with open(readfile+'TemporalResults/arxiv-ChatGPT-20231007.json', 'w') as f:
    with open(readfile+'arxiv-metadata-20231007.json') as rf:
       for line in tqdm(rf):
         j = json.loads(line)
         if j['id'] in chatGPTset:
            json.dump(j, f)
            f.write('\n')
print('ChatGPT end')

with open(readfile+'TemporalResults/arxiv-GPT-20231007.json', 'w') as f:
    with open(readfile+'arxiv-metadata-20231007.json') as rf:
       for line in tqdm(rf):
         j = json.loads(line)
         if j['id'] in GPTset:
            json.dump(j, f)
            f.write('\n')
print('GPT end')

from dateutil import parser as date_parser
with open(readfile+'TemporalResults/arxiv-LLMs-20231007.json', 'w') as f:
    with open(readfile+'arxiv-metadata-20231007.json') as rf:
       for line in tqdm(rf):
         j = json.loads(line)
         if j['id'] in allset and date_parser.parse(j['versions'][0]['created']).year > 2014:
            json.dump(j, f)
            f.write('\n')
print('LLMs end')


# get csv --------------------------------------------------
import pandas as pd
from dateutil import parser as date_parser
import pickle as pk
llm = pd.read_json(readfile+'TemporalResults/arxiv-LLMs-20231007.json', lines=True, dtype=object)


def ChatGPTrelated(row):
    if row['id'] in chatGPTset:
        return 1
    else:
        return 0

def GPTrelated(row):
    if row['id'] in GPTset:
        return 1
    else:
        return 0
def pubTime(row):
    return date_parser.parse(row['versions'][0]['created']).strftime('%Y-%m-%d')

llm['ContainChatGPT'] = llm.apply(ChatGPTrelated, axis=1)
llm['ContainGPT'] = llm.apply(GPTrelated, axis=1)
llm['publish_date_v1'] = llm.apply(pubTime, axis=1)
llm['title'] = llm['title'].str.replace('\n ','')
llm['abstract'] = llm['abstract'].str.replace('\n',' ')
llm.to_csv(readfile+'LLMs1007.csv', index=False)