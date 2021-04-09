# Experiments

## Baseline (Single Modality)

### Different Modality
|ID|Backbone|Model|Embedding Size|Modality|Accuracy|
|:---:|:---:|:---:|:---:|:---:|:---:|
|0|Conv4|ProtoNet|800|Image|46.29|
|1|Conv4|ProtoNet|800|Text|TODO|

### Different Model
|ID|Backbone|Model|Embedding Size|Modality|Accuracy|
|:---:|:---:|:---:|:---:|:---:|:---:|
|0|Conv4|ProtoNet|800|Image|46.29|
|2|Conv4|MAML|800|Image|42.46|

### Different Backbone
|ID|Backbone|Model|Embedding Size|Modality|Accuracy|
|:---:|:---:|:---:|:---:|:---:|:---:|
|0|Conv4|ProtoNet|800|Image|46.29|
|3|ResNet12|ProtoNet|800|Image|54.07|


### NOT FOR RECORD: Different Embedding Size
|ID|Backbone|Model|Embedding Size|Modality|Accuracy|
|:---:|:---:|:---:|:---:|:---:|:---:|
|0|Conv4|ProtoNet|800|Image|46.29|
|4|Conv4|ProtoNet|128|Image|TODO|


## Multimodal Improvement
|ID|Backbone|Model|Embedding Size|Modality|Fusion Method|Accuracy|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0|Conv4|ProtoNet|800|Image|-|46.29|
|5|Conv4|ProtoNet|800|Image + Text|Mean|TODO|
|6|Conv4|ProtoNet|800|Image + Text|FC|TODO|
|7|Conv4|ProtoNet|800|Image + Text|Attention (image guided)|TODO|
|8|Conv4|ProtoNet|800|Image + Text|Attention (text residual)|TODO|
|3|ResNet12|ProtoNet|800|Image|-|54.07|
|9|ResNet12|ProtoNet|800|Image + Text|Mean|TODO|
|10|ResNet12|ProtoNet|800|Image + Text|FC|TODO|
|11|ResNet12|ProtoNet|800|Image + Text|Attention (image guided)|TODO|
|12|ResNet12|ProtoNet|800|Image + Text|Attention (text residual)|TODO|
|2|Conv4|MAML|800|Image|-|42.46|
|13|Conv4|MAML|800|Image + Text|Mean|TODO|
|14|Conv4|MAML|800|Image + Text|FC|TODO|
|15|Conv4|MAML|800|Image + Text|Attention (image guided)|TODO|
|16|Conv4|MAML|800|Image + Text|Attention (text residual)|TODO|