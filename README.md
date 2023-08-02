# TCFN
Implementation of the paper "Temporal-enhanced Cross-modality Fusion Network for Video Sentence Grounding" (ICME 2023 Oral).

## Data preparation
Video features provided by [VSLNet](https://github.com/26hzhang/VSLNet), and the [word embedding](http://nlp.stanford.edu/data/glove.840B.300d.zip).

The directory of /data/features should be like 

```
data
├── activitynet
├── charades
├── tacos
├── glove.840B.300d.txt
```

## training & inference
```shell script
# [dataset]: tacos, charades, activitynet
# [mode]: train, test
python main.py --task [dataset] --mode [mode]
```
