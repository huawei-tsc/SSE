# Synthetic Source-Enhanced Back-Translation (SSE-BT)
## Introduction

Obtaining bilingual data is always harder than monolingual ones, so researchers have long exploited how to enhance machine translation performances using monolingual data. One of the most widely-used strategy is Back Translation (BT).

Admittedly, BT leads to great improvements in terms of translation quality, however, here are still some problems:

* **Translationese**：The style of generated source sentences may divert from natural ones.
* **Quality**：In scenarios where bilingual data is insufficient, the trained BT model can hardly generate high-quality synthetic data.

As a result, there is still a large gap between synthetic data and real parallel corpora.

To address the two issues, we devise an SSE model to improve the quality of generated source sentences, as well as transfer the sentence style to a more natural one. Our experiments demonstrate that SSE model can further enhance machine translation quality.

![image](https://github.com/huawei-tsc/SSE/blob/main/sse.PNG)

## Experiment Results

### WMT14 EN-DE

Bilingual: EN-DE 4.5M  Monolingual: DE 24M

|                | orig-en   | orig-de   | WMT14     |
| -------------- | --------- | --------- | --------- |
| base           | 28.08     | 26.07     | 27.42     |
| + BT                 | 27.27     | 33.43     | 30.29     |
| &nbsp; + SSE          | 29.36     | 34.66     | 31.98     |
| &nbsp; + Sampling SSE | 29.30     | 35.40     | 32.35     |
| + Sampling BT  | 28.35     | **35.74** | 31.75     |
| &nbsp; + SSE          | **29.31** | 35.70     | **32.42** |
| &nbsp; + Sampling SSE | 29.03     | 35.72     | 32.12     |

## Pre-train Model

| model        | Description                  | data                          | arch            | download |
| ----------- | ---------------------------- | ----------------------------- | --------------- | -------- |
| EN          | wmt news en sse              | wmt 2021 NewsCrawl Random 30M | tranformer-big | comming  |
| EN-FR-DE-RU | wmt en fr de multiligual sse | wmt news                      | tranformer-big | comming  |
| SSE-100     | big multiligual sse ?        |                               |                 |          |

## How To Use

#### EN SSE

1. 1.	Data Processing

   a. Tokenize using mosesdecoder： https://github.com/moses-smt/mosesdecoder

   ```shell
   perl mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l en < bt.en > token.en
   ```

   b. Generate subwords  using BPE： https://github.com/rsennrich/subword-nmt

   ```shell
   subword-nmt apply-bpe -c wmt2014.en-de.codefile < token.en > bpe.en
   ```

2. Enhance BT data quality using SSE model

   ```python
   # Load pretrained sse model
   src_lang = "rtt"  # the mt en
   target_lang = en  # the natural en
   en2de_pre_trained = TransformerModel.from_pretrained(
       model_dir_path,
       checkpoint_file=checkpoint_file,
       data_name_or_path=model_dir_path,
       source_lang=source_lang,
       target_lang=target_lang,
       device_id=0
   )
   en2de_pre_trained.cuda()
   
   # For sampling 
   en2de_pre_trained.translate(input_a_batch, beam=1, sampling=True, sampling_topk=10)
   # For Beam search
   en2de_pre_trained.translate(input_a_batch, beam=2)
   ```

