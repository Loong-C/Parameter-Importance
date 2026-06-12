# 参数重要性过估计验证

本仓库用于验证路径积分参数重要性中的随机梯度过估计问题，主要回答：

1. 同一 minibatch 同时用于参数更新和重要性评估时，是否产生显著正偏？
2. 独立双采样和单采样协方差修正是否有效？
3. 在相同样本与梯度评估预算下，单采样 U-statistic 是否优于双采样？

实验设计主要依据 `docs/4.19.pptx`，并结合 SI、无偏 SI、U-statistic 和重尾梯度噪声研究。`legacy/` 仅作历史参考，不参与新框架执行。

## 安装

```powershell
python -m pip install -e ".[dev]"
```

当前验证环境为 Python 3.12、PyTorch 2.11、CUDA 12.8。依赖版本记录在 `requirements-lock.txt`。

## CLI

```powershell
param-importance simulate  --config configs/simulate_full.yaml
param-importance checkpoint --config configs/checkpoint_mnist.yaml
param-importance reference --config configs/reference_mnist.yaml
param-importance continual --config configs/continual_permuted_main.yaml
param-importance analyze   --config configs/analyze_full.yaml
param-importance report    --config configs/report_full.yaml
```

所有运行目录都包含配置哈希、Git commit、运行时元数据和 Parquet 结果。相同配置完成后默认跳过；使用 `--force` 重跑。可重复使用 `--set KEY=VALUE` 覆盖配置：

```powershell
param-importance simulate `
  --config configs/simulate_full.yaml `
  --set repetitions=1000 `
  --set seed=7
```

## 估计器

对更新点梯度样本 `u_i`、路径节点梯度样本 `v_qi`、步长 `gamma`：

- `naive`: `gamma * mean(u) * sum_q w_q mean(v_q)`
- `double`: 两个独立半批次的对称交叉乘积
- `double_matched_mM`: 使用 M 个微批、按不相交配对平均的同预算双采样
- `single_direct`: 从 Naive 中减去 `gamma * sum_q w_q Cov(u,v_q) / B`
- `single_micro_mM`: 使用 M 个微批均值的样本协方差完成同一修正
- `ppt_variance_only`: 只减更新点方差的 PPT 公式消融

`M=2` 时，`single_micro_m2` 与对称 `double_matched_m2` 逐次严格相等。`M>2` 时，Single-micro 使用全部 `M(M-1)/2` 个交叉配对，而 matched Double 只使用 `M/2` 个不相交配对。

## 参考积分

固定阶 Gauss-Legendre 只能视为近似，不能直接称为积分真值。`reference` 命令使用：

1. 自适应 Gauss-Kronrod 向量积分；
2. 独立复合梯形网格加密；
3. 端点损失守恒 `sum(importance)=L(theta0)-L(theta1)`；
4. 相邻网格收敛检查。

只有这些条件同时达到配置容差时，检查点才标记为 `reference_certified=true`。ReLU/MaxPool 路径可能包含大量折点，因此允许结果明确显示“未认证”。

## 实验配置

- `simulate_full.yaml`: 受控分布与解析梯度问题，每格 20,000 次重复
- `checkpoint_mnist.yaml`: MNIST MLP 五种子主实验
- `checkpoint_cifar_*_representative.yaml`: CIFAR-10 平滑/非平滑本机代表性矩阵
- `continual_permuted_main.yaml`: 5 任务 Permuted-MNIST 端到端比较
- `continual_split_cifar_main.yaml`: 10 任务 Split-CIFAR-100 资源受限扩展
- `stress/`: 无放回抽样、Momentum、AdamW、数据增强和大步长压力测试
- `extensions/`: ImageNet-100 与 ViT-Tiny/MAE 扩展配置

完整预注册规则见 `docs/preregistration.md`，文献依据见 `docs/literature.md`。

## 验证

```powershell
pytest
param-importance simulate --config configs/smoke/simulate.yaml
param-importance checkpoint --config configs/smoke/checkpoint.yaml
param-importance continual --config configs/smoke/continual.yaml
```

最终 HTML 报告由实际 Parquet 结果生成，包含 10,000 次分块或种子聚类 bootstrap、Holm 校正、静态图表和参考积分认证表。
