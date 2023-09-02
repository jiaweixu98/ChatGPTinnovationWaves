use plotly to get a interactive graph.

Top 15 mentioned models
('ChatGPT', 1004),
('GPT-3', 803),
('GPT-4', 415),
('T5', 285),
('LLaMA', 119),
('Codex', 119),
('OPT', 84),
('PaLM', 71),
('BLOOM', 58),
('InstructGPT', 52),
('Vicuna', 36),
('Alpaca', 34),
('Flan-T5', 30),
('FLAN', 30),
('mT5', 25)

figure 1: cumulative time series distribution; x:date; y: cumulative num
figure 2: time series distribution; x:date; y:num
figure 3: relative distribution; x:days to first annoucement; y: cumulative num
figure 4: relative distribution; x:days to first annoucement; y: num

allotaxonometer:
GPT-4 v.s. LLaMA
GPT-3 v.s. T5

model embeddings part: generate models' embeddings
Just use the papers contain models. We use guided topic modeling, because we want to see the relationship of different models.

1.Use the abstract and title of papers contain model names as input.
2.Guided topic modeling, get each model a representation.
3.We can:
    keywords of each topic;
    the distance map of each topic;
    the document vislization in each topic.
    frequency change overtime.

key words part: use bert or diment graph. (open source v.s. close source)
