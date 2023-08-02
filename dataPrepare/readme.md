从2023年7月30日的arxiv数据 (2.3M arxiv papers，数据来源 <https://www.kaggle.com/datasets/Cornell-University/arxiv>) 中，抽取ChatGPT，GPT和LLMs相关的文献记录，借此研究ChatGPT引发的发文特征。

匹配数量（去掉了换行符）：
chatGPT: 912
gpt: 2770
LLM: 5274


arxiv data的数据结构。
1. **id:** ArXiv ID (can be used to access the paper, see below)
2. **submitter:** Who submitted the paper
2. **authors:** Authors of the paper
2. **title:** Title of the paper
2. **comments:** Additional info, such as number of pages and figures
2. **journal-ref:** Information about the journal the paper was published in
2. **doi:** [https://www.doi.org](Digital Object Identifier)
2. **abstract:** The abstract of the paper
2. **categories:** Categories / tags in the ArXiv system
2. **versions:** A version history

例子
id   ['0704.0001']
submitter   ['Pavel Nadolsky']
authors   ["C. Bal\\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan"]
title   ['Calculation of prompt diphoton production cross sections at Tevatron and\n  LHC energies']
comments   ['37 pages, 15 figures; published version']
journal-ref   ['Phys.Rev.D76:013009,2007']
doi   ['10.1103/PhysRevD.76.013009']
report-no   ['ANL-HEP-PR-07-12']
categories   ['hep-ph']
license   [None]
abstract   ['  A fully differential calculation in perturbative quantum chromodynamics is\npresented for the production of massive photon pairs at hadron colliders. All\nnext-to-leading order perturbative contributions from quark-antiquark,\ngluon-(anti)quark, and gluon-gluon subprocesses are included, as well as\nall-orders resummation of initial-state gluon radiation valid at\nnext-to-next-to-leading logarithmic accuracy. The region of phase space is\nspecified in which the calculation is most reliable. Good agreement is\ndemonstrated with data from the Fermilab Tevatron, and predictions are made for\nmore detailed tests with CDF and DO data. Predictions are shown for\ndistributions of diphoton pairs produced at the energy of the Large Hadron\nCollider (LHC). Distributions of the diphoton pairs from the decay of a Higgs\nboson are contrasted with those produced from QCD processes at the LHC, showing\nthat enhanced sensitivity to the signal can be obtained with judicious\nselection of events.\n']
versions   [[{'version': 'v1', 'created': 'Mon, 2 Apr 2007 19:18:42 GMT'}, {'version': 'v2', 'created': 'Tue, 24 Jul 2007 20:10:27 GMT'}]]
update_date   ['2008-11-26']


用title和abstract匹配关键词。对于phrases，需要处理换行符“\n”，注意，title里需要特别处理'\n  '，而abstract需要处理'\n'。
最后得到三个数据：ChatGPT、GPT和LLMs.
chatGPT: 912
gpt: 2770
LLM: 5274