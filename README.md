# 腾讯觅影竞赛
记录自己在腾讯觅影眼疾病分类的过程
## Preprocessing 有预处理
- resnet34
- soomth cross entropy
- steplr
- 20epochs
-  scheduler
-  augmentation

| Rotation | RandomAffine | RandomErase | ColorJitter | val loss | val score |
|:-----:|:----:|:---:|:------:|:--------:|:---------:|
|       |  o   |     |        |         | 0.81    |
| o     |  o   |     |        |         | 0.81    |
|       | o    |     |        | 0.6209   | 0.8874    |
| o     |      | o   |        | 0.5546   | 0.8692    |
| o     | o    |     |        |**0.5348**| 0.8616    |
| o     | o    | o   |        | 0.5453   | 0.8408    |
| o     | o    |     | o      | 0.5479   | 0.8508    |
