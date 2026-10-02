# 训练报告（credit-g 端到端）

- 训练时间：2026-10-02 17:51:02；数据：openml:credit-g:v1
- 切分：分层 train/test = 800/200（seed=42）

## 实验矩阵（2 模型 × 3 不平衡策略，5 折分层 CV + 独立测试集）

| model               | strategy   |   cv_auc_mean |   cv_auc_std |   cv_ks_mean |   cv_ks_std |   test_auc |   test_ks |   test_gini |
|:--------------------|:-----------|--------------:|-------------:|-------------:|------------:|-----------:|----------:|------------:|
| logistic_regression | weight     |      0.783594 |    0.0474992 |     0.493452 |   0.0641754 |   0.80131  |  0.511905 |    0.602619 |
| logistic_regression | smote      |      0.782366 |    0.0470072 |     0.497619 |   0.058485  |   0.800357 |  0.514286 |    0.600714 |
| logistic_regression | none       |      0.782329 |    0.0484502 |     0.499405 |   0.0679041 |   0.800357 |  0.52381  |    0.600714 |
| lightgbm            | none       |      0.746838 |    0.0446252 |     0.419643 |   0.0448211 |   0.771071 |  0.428571 |    0.542143 |
| lightgbm            | weight     |      0.746205 |    0.0450905 |     0.406548 |   0.0538222 |   0.7775   |  0.47381  |    0.555    |
| lightgbm            | smote      |      0.745275 |    0.0521509 |     0.411905 |   0.0606909 |   0.761667 |  0.445238 |    0.523333 |

**最优组合：logistic_regression + weight**；测试集 AUC=0.8013，KS=0.5119，Gini=0.6026。

## 不平衡处理结论

- **logistic_regression**：none=0.7823，weight=0.7836，smote=0.7824（CV AUC）。最优策略为 weight；weight 与 smote 差距 0.0012。两者几乎无差异——credit-g 不平衡程度温和（约 30% 坏样本），样本加权以更低复杂度达到同等效果，作为部署默认策略。
- **lightgbm**：none=0.7468，weight=0.7462，smote=0.7453（CV AUC）。最优策略为 none；weight 与 smote 差距 0.0009。两者几乎无差异——credit-g 不平衡程度温和（约 30% 坏样本），样本加权以更低复杂度达到同等效果，作为部署默认策略。

## IV 前 8（训练集 WOE/IV）

| feature            |        iv | strength   |
|:-------------------|----------:|:-----------|
| checking_status    | 0.609259  | suspicious |
| duration           | 0.293099  | medium     |
| credit_history     | 0.262438  | medium     |
| savings_status     | 0.214128  | medium     |
| purpose            | 0.149895  | medium     |
| property_magnitude | 0.145175  | medium     |
| employment         | 0.124165  | medium     |
| age                | 0.0933185 | weak       |

## 全局重要性前 8（方法：shap_mean_abs(lightgbm+weight)）

| feature            |   importance | method        |
|:-------------------|-------------:|:--------------|
| checking_status    |     1.30012  | shap_mean_abs |
| duration           |     0.70543  | shap_mean_abs |
| purpose            |     0.59907  | shap_mean_abs |
| credit_history     |     0.558987 | shap_mean_abs |
| credit_amount      |     0.448222 | shap_mean_abs |
| savings_status     |     0.418405 | shap_mean_abs |
| employment         |     0.405573 | shap_mean_abs |
| property_magnitude |     0.368578 | shap_mean_abs |

## 单样本解释示例（测试集第 1 条）

部署模型（logistic_regression）归因方法：`coef_x_woe_deviation`

```json
{
  "pd": 0.5867274320166187,
  "y_true": 0,
  "top_features": [
    {
      "feature": "savings_status",
      "contribution": -0.7680819790991104
    },
    {
      "feature": "other_payment_plans",
      "contribution": 0.5914027363347406
    },
    {
      "feature": "employment",
      "contribution": 0.44351769325132495
    }
  ]
}
```

LightGBM 的 SHAP 归因（供对照）：`[{"feature": "savings_status", "contribution": -1.9694648339386402}, {"feature": "other_payment_plans", "contribution": 1.0802475093798092}, {"feature": "credit_amount", "contribution": 0.8605682338506699}]`

## 风险分档阈值（测试集 PD 分位）

- 低风险：PD < 0.506；中风险：0.506 ≤ PD < 0.792；高风险：PD ≥ 0.792
