# Experiments

[comment]: <> (## Baseline &#40;Single Modality&#41;)

[comment]: <> (### Different Modality)

[comment]: <> (|ID|Backbone|Model|Embedding Size|Modality|Accuracy|)

[comment]: <> (|:---:|:---:|:---:|:---:|:---:|:---:|)

[comment]: <> (|0|Conv4|ProtoNet|800|Image|46.29|)

[comment]: <> (|1|Conv4|ProtoNet|800|Text|TODO|)

[comment]: <> (### Different Model)

[comment]: <> (|ID|Backbone|Model|Embedding Size|Modality|Accuracy|)

[comment]: <> (|:---:|:---:|:---:|:---:|:---:|:---:|)

[comment]: <> (|0|Conv4|ProtoNet|800|Image|46.29|)

[comment]: <> (|2|Conv4|MAML|800|Image|42.46|)

[comment]: <> (### Different Backbone)

[comment]: <> (|ID|Backbone|Model|Embedding Size|Modality|Accuracy|)

[comment]: <> (|:---:|:---:|:---:|:---:|:---:|:---:|)

[comment]: <> (|0|Conv4|ProtoNet|800|Image|46.29|)

[comment]: <> (|3|ResNet12|ProtoNet|800|Image|54.07|)


[comment]: <> (### NOT FOR RECORD: Different Embedding Size)

[comment]: <> (|ID|Backbone|Model|Embedding Size|Modality|Accuracy|)

[comment]: <> (|:---:|:---:|:---:|:---:|:---:|:---:|)

[comment]: <> (|0|Conv4|ProtoNet|800|Image|46.29|)

[comment]: <> (|4|Conv4|ProtoNet|128|Image|TODO|)


## Baseline
|ID|Backbone|Model|Embedding Size|Modality|Accuracy|
|:---:|:---:|:---:|:---:|:---:|:---:|
|0|Conv4|ProtoNet|800|Image|46.99|
|-|Conv4|ProtoNet|128|Image|51.09 +- 0.77|
|1|SentenceEncoder|ProtoNet|800|Text|76.15|
|-|SentenceEncoder|ProtoNet|128|Text|75.70|
|2|Conv4|MAML|800|Image|49.75|
|-|Conv4|MAML|128|Image|53.16 +- 0.72|
|3|ResNet12|ProtoNet|800|Image|53.65|
|-|ResNet12|ProtoNet|128|Image|58.38|

|4|Conv4|ProtoNet|128|Image|TODO|

## Multimodal Improvement
|ID|Backbone|Model|Embedding Size|Modality|Fusion Method|Accuracy|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0|Conv4|ProtoNet|800|Image|-|46.99|
|5|Conv4|ProtoNet|800|Image + Text|Mean|TODO|
|6|Conv4|ProtoNet|800|Image + Text|FC|TODO|
|7|Conv4|ProtoNet|800|Image + Text|Attention (image guided)|TODO|
|8|Conv4|ProtoNet|800|Image + Text|Attention (text residual)|TODO|
|3|ResNet12|ProtoNet|800|Image|-|53.65|
|9|ResNet12|ProtoNet|800|Image + Text|Mean|TODO|
|10|ResNet12|ProtoNet|800|Image + Text|FC|TODO|
|11|ResNet12|ProtoNet|800|Image + Text|Attention (image guided)|TODO|
|12|ResNet12|ProtoNet|800|Image + Text|Attention (text residual)|TODO|
|2|Conv4|MAML|800|Image|-|49.75|
|13|Conv4|MAML|800|Image + Text|Mean|TODO|
|14|Conv4|MAML|800|Image + Text|FC|TODO|
|15|Conv4|MAML|800|Image + Text|Attention (image guided)|TODO|
|16|Conv4|MAML|800|Image + Text|Attention (text residual)|TODO|