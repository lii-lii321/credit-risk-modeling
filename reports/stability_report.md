# 特征稳定性（PSI）报告

PSI 判读标准：<0.1 稳定；0.1-0.25 中度漂移；>0.25 显著漂移。

## train vs test（同分布随机切分，预期稳定）

| feature                |         psi | level   |
|:-----------------------|------------:|:--------|
| purpose                | 0.0727168   | stable  |
| age                    | 0.0725443   | stable  |
| employment             | 0.0476851   | stable  |
| credit_amount          | 0.0404604   | stable  |
| savings_status         | 0.037022    | stable  |
| personal_status        | 0.0262653   | stable  |
| other_parties          | 0.0158561   | stable  |
| installment_commitment | 0.0144428   | stable  |
| residence_since        | 0.0138448   | stable  |
| credit_history         | 0.0138328   | stable  |
| job                    | 0.0108556   | stable  |
| duration               | 0.00635937  | stable  |
| existing_credits       | 0.00328857  | stable  |
| foreign_worker         | 0.00231871  | stable  |
| other_payment_plans    | 0.00214391  | stable  |
| property_magnitude     | 0.000871676 | stable  |
| own_telephone          | 0.000527097 | stable  |
| checking_status        | 0.000504119 | stable  |
| housing                | 0.000207085 | stable  |
| num_dependents         | 0           | stable  |

最大 PSI：**0.0727** → 整体稳定

## train vs 人为漂移的 test（age+15 岁、credit_amount×1.5、duration+12 月）

用于验证 PSI 工具确实能检出漂移（自实现模块的在线对照实验）：

| feature         |       psi | level       |
|:----------------|----------:|:------------|
| age             | 4.24225   | significant |
| duration        | 3.14059   | significant |
| credit_amount   | 0.496952  | significant |
| purpose         | 0.0727168 | stable      |
| employment      | 0.0476851 | stable      |
| savings_status  | 0.037022  | stable      |
| personal_status | 0.0262653 | stable      |
| other_parties   | 0.0158561 | stable      |

最大 PSI：**4.2423** → 应显著大于 0.25。