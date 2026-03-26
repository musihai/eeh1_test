# RL样本四模型对比（MSE/MAE）

| split | model | evaluated | valid | failures | MAE(mean) | MSE(mean) |
|---|---:|---:|---:|---:|---:|---:|
| train | arima | 256 | 256 | 0 | 3.117780 | 17.219941 |
| train | chronos2 | 256 | 256 | 0 | 3.047869 | 17.141287 |
| train | itransformer | 256 | 256 | 0 | 2.565746 | 11.683169 |
| train | patchtst | 256 | 256 | 0 | 2.610558 | 12.263335 |
| val | arima | 256 | 256 | 0 | 1.814759 | 6.014480 |
| val | chronos2 | 256 | 256 | 0 | 1.818218 | 5.790497 |
| val | itransformer | 256 | 256 | 0 | 1.700479 | 5.182478 |
| val | patchtst | 256 | 256 | 0 | 1.686159 | 5.084244 |
