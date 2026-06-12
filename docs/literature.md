# 研究资料与理论依据

1. Zenke, Poole, Ganguli. **Continual Learning Through Synaptic Intelligence**. ICML 2017.  
   原始 SI 路径积分方法，并指出随机梯度噪声通常会高估参数重要性。  
   <https://proceedings.mlr.press/v70/zenke17a.html>

2. Benzing. **Unifying Regularisation Methods for Continual Learning**. AISTATS 2022.  
   分析 SI 的偏差项，提出使用独立 minibatch 的无偏 SIU，并报告偏差可显著大于无偏部分。  
   <https://proceedings.mlr.press/v151/benzing22a.html>

3. Hoeffding. **A Class of Statistics with Asymptotically Normal Distribution**. 1948.  
   U-statistic 的经典来源；本项目的 Single-micro 修正是二阶 U-statistic。  
   <https://repository.lib.ncsu.edu/items/2dd4f5a1-4a95-4871-b090-d9da1bde4e44>

4. Simsekli et al. **A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks**. ICML 2019.  
   深度网络梯度噪声可能重尾，因此高斯假设只能用于闭式方差基准，不能作为无偏性的普遍前提。  
   <https://proceedings.mlr.press/v97/simsekli19a.html>

5. SciPy `quad_vec` 文档。  
   自适应向量积分接口，支持 Gauss-Kronrod 规则、误差估计和区间细分诊断。  
   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad_vec.html>

6. 项目材料：`docs/4.19.pptx`。  
   重点包括过估计推导、双采样方法、逐样本/微批方差修正，以及单采样与双采样方差比较。

## 核心恒等式

若每个样本同时产生更新梯度 `u_i` 和节点梯度 `v_i`，则：

`E[mean(u) mean(v)] = E[u] E[v] + Cov(u,v)/B`

减去无偏样本协方差除以 B 后，得到 `E[u]E[v]` 的无偏估计。对 M 个微批均值重复该推导即可得到低内存 Single-micro 形式。

高斯假设只用于推导闭式方差：

- M=2 时，Single-micro 与对称 matched Double 完全等价；
- M>2 时，完整 U-statistic 使用更多交叉配对，理论方差低于只使用不相交配对的 Double；
- 无偏性本身不依赖高斯分布。

## 数值积分说明

固定 GL16 或更高阶 Gauss-Legendre 对光滑函数往往很准确，但不能保证它等于真实积分。ReLU/MaxPool 路径可能包含未被固定节点命中的折点。

因此本项目不把固定高阶高斯求积称为真值，而采用：

- 全数据自适应 Gauss-Kronrod；
- 独立复合梯形网格加密；
- 相邻网格收敛；
- 端点损失守恒；

共同构成可审计的数值参考。
