# SSE
Synthetic Source Enhanced Back-Translation
## 介绍

机器翻译中，双语数据获取存在一定难度，相比之下，单语数据较为容易获取。因此也产生了大量的通过单语数据增强MT的方法。最为典型的代表就是Back Translation，简称BT。

BT的主要问题：

* **翻译腔**：原文风格问题，和Natural的双语预料有差距；
* **质量**：当双语数据不足，tgt2src模型质量较差时，无法保证伪预料质量；

以上两点，导致同等数量的伪预料和真是双语预料，带来的提升效果gap很大。

针对BT伪预料的两个问题，如果在BT之后，能通过SSE模型，提升BT原文的质量，同时将原文的风格迁移至Natural方向，和双语对其，应该是可以整体提升BT的质量。

- ![avatar](img/SSE.png)

## 实验结果

### WMT14 EN-DE

双语：EN-DE 4.5M  单语：DE 220M

|                | orig-en   | orig-de   | WMT14     |
| -------------- | --------- | --------- | --------- |
| base           | 28.08     | 26.07     | 27.42     |
| + BT                 | 27.27     | 33.43     | 30.29     |
| &nbsp; + SSE          | 29.36     | 34.66     | 31.98     |
| &nbsp; + Sampling SSE | 29.30     | 35.40     | 32.35     |
| + Sampling BT  | 28.35     | **35.74** | 31.75     |
| &nbsp; + SSE          | **29.31** | 35.70     | **32.42** |
| &nbsp; + Sampling SSE | 29.03     | 35.72     | 32.12     |

## Pre-trained models

| Model    | Description                  | data                               | arch            | download |
| -------- | ---------------------------- | ---------------------------------- | --------------- | -------- |
| EN       | wmt news en sse              | wmt 2007-2017 NewsCrawl Random 24M | tranformer-base | url      |
| EN-FR-DE | wmt en fr de multiligual sse | xxx                                | tranformer-base | url      |
|          |                              |                                    |                 |          |

## how to use

1. download and unzip

   ```
   wget http://xxx:xxx/en.pt
   tar -xvf en.pt
   ```

2. Enhance your BT data

   ```
   bt_src=wmt_en_mt
   fairseq-generate xxxxx
   ```
