# ADR-0002: 校准修正——OOF 防泄漏 + sigmoid/isotonic 择优

- 状态：已采纳
- 日期：2026-10-03（成文 2026-10-04，追溯既有决策）
- 关联代码：`src/creditrisk/calibration_repair.py`、`scripts/run_training.py`、`tests/test_calibration_repair.py`

## 背景

训练报告实测发现部署模型 PD 系统性偏高（预测均值 0.437 vs 实际违约率
0.300，ECE 0.142）。修复校准最容易被做错的地方是**数据泄漏**：
在校准评估所用的同一段数据上拟合校准器，ECE 会"漂亮"但不可信。

## 决策

三段式协议：

1. 训练集 5 折分层 CV 产出每条样本的 held-out（OOF）预测——校准器
   从未见过这些预测的产生过程；
2. OOF 样本分层切成两半：sel_fit 上分别拟合 Platt（sigmoid）与
   isotonic，sel_val 上对比 ECE 择优（实测 sigmoid 0.0699 胜出）；
3. 择优者用全部 OOF 重拟合为部署校准器；**测试集只做单调变换，
   绝不参与任何拟合**，复测数字（ECE 0.0681）才是可信估计。

## 代价

- 协议比"直接 CalibratedClassifierCV 一行"复杂，需要 14 个测试守护；
- isotonic 在小样本上过拟合风险使其只能做候选而非默认；
- 校准器与模型必须绑定分发（artifacts/deploy_bundle.joblib），
  模型更新时校准层必须重训。
