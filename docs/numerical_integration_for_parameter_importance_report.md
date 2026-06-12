# 参数积分梯度重要性的数值积分方法报告

## 摘要

参数积分梯度重要性通常可以写成沿一次参数更新路径的梯度积分。若令当前参数为 $\theta_t$，更新向量为 $\Delta\theta$，路径参数为 $\alpha\in[0,1]$，则路径上的参数点为

$$
v_\alpha=\theta_t+\alpha\Delta\theta .
$$

对所有参数同时考虑时，被积函数是一个向量值函数

$$
G(\alpha)=g(v_\alpha)=\nabla_\theta \mathcal L(v_\alpha),
$$

对应的重要性向量可以写成

$$
\omega^t=\Delta\theta\odot \int_0^1 G(\alpha)\,d\alpha .
$$

其中 $\odot$ 表示逐参数相乘。若只看第 $k$ 个参数，则是

$$
\omega_k^t=\Delta\theta_k\int_0^1 g_k(\theta_t+\alpha\Delta\theta)\,d\alpha .
$$

这个积分表面上只是一维积分，但它和普通一维数值积分有一个关键区别：每取一个节点 $\alpha_q$，就要在参数 $v_{\alpha_q}$ 处计算一次梯度。对于神经网络而言，这通常意味着一次反向传播。因此，在这个问题中，数值积分方法的“好”不能只按数学阶数判断，还要考虑节点数量、节点是否可复用、是否能给出误差估计、是否适合 ReLU/MaxPool 造成的非光滑路径，以及是否能和 minibatch 或 microbatch 梯度噪声估计结合。

本文分别介绍 Newton-Cotes 类方法、Gauss-Legendre 高斯积分、Gauss-Kronrod 嵌入式高斯积分、Clenshaw-Curtis 积分，并说明它们如何用于自适应分段积分，以及在参数积分梯度重要性问题中的具体实现方式和时间复杂度。

## 记号与问题形式

我们关心的是

$$
I=\int_0^1G(\alpha)\,d\alpha,
\qquad
\omega^t=\Delta\theta\odot I .
$$

这里 $G(\alpha)\in\mathbb R^d$，其中 $d$ 是模型参数量。为了不把讨论限制在某一个参数上，下面多数公式使用向量形式；如果只看第 $k$ 个参数，只需取对应坐标即可。

在数值积分中，我们用若干节点 $\alpha_q$ 与权重 $w_q$ 近似积分：

$$
\hat I=\sum_{q=1}^{Q} w_qG(\alpha_q),
\qquad
\hat\omega^t=\Delta\theta\odot \hat I .
$$

这里 $Q$ 是需要计算梯度的路径节点数。若一次在节点 $\alpha_q$ 上计算全 batch 梯度的成本记为 $C_{\mathrm{grad}}(B,d)$，则固定节点数的积分成本大致是

$$
T\approx Q\,C_{\mathrm{grad}}(B,d).
$$

在神经网络中，权重计算、节点变换、加权求和的成本通常远小于反向传播成本。因此，比较数值积分方法时，最核心的复杂度往往不是公式本身的代数运算，而是“需要多少个不同的 $\alpha_q$ 节点”。

若采用 microbatch-level U-statistic，一次 full minibatch 被拆成 $M$ 个 microbatch，每个大小为 $b=B/M$。在每个积分节点上，理论上需要得到每个 microbatch 的局部梯度 $G_m(\alpha_q)$。如果这些局部梯度本来就由多卡或 gradient accumulation 产生，则额外统计量可以流式累加；如果需要额外单独计算，则代价会变成 $Q\sum_m C_{\mathrm{grad}}(b,d)$，在串行环境中约等于 $Q C_{\mathrm{grad}}(B,d)$，在多卡并行环境中则接近 $Q C_{\mathrm{grad}}(b,d)$ 加通信和记录开销。

本文讨论数值积分时先把随机梯度噪声放在一边，重点分析如何近似路径积分；但在最后会说明这些方法如何与 microbatch-level U-statistic 结合。

## Newton-Cotes 类方法

Newton-Cotes 方法的核心思想非常直接：在积分区间上取等距节点，用这些节点上的函数值拟合插值多项式，然后对插值多项式积分。常见的梯形法、Simpson 法都属于 Newton-Cotes 类方法。这个方法的优点是节点规则简单，旧节点容易复用，复合形式易于实现；缺点是高阶 Newton-Cotes 公式在等距节点上可能出现数值不稳定，而且对非光滑函数并不一定可靠。Newton-Cotes 公式通常分为 closed 与 open 两类，closed 公式包含区间端点，而 open 公式不包含区间端点；梯形法和 Simpson 法是最常用的 closed Newton-Cotes 例子。

最简单的复合梯形法把 $[0,1]$ 切成 $n$ 个等长小区间，步长为 $h=1/n$，节点为 $\alpha_j=jh$。积分近似为

$$
\hat I_{\mathrm{trap}}
=
h\left[
\frac12G(0)+\sum_{j=1}^{n-1}G(jh)+\frac12G(1)
\right].
$$

如果 $G(\alpha)$ 足够光滑，复合梯形法的误差通常是 $O(h^2)$。但它的优势在于稳健和节点复用。若从 $n$ 个子区间加密到 $2n$ 个子区间，原来的所有节点仍然保留，只需补充中点。这一点在梯度计算昂贵的场景里非常重要。

复合 Simpson 法把相邻两个小区间作为一组，用二次插值积分。若 $n$ 为偶数，步长 $h=1/n$，则

$$
\hat I_{\mathrm{simp}}
=
\frac{h}{3}
\left[
G(0)+G(1)
+4\sum_{\substack{j=1\\j\ \mathrm{odd}}}^{n-1}G(jh)
+2\sum_{\substack{j=2\\j\ \mathrm{even}}}^{n-2}G(jh)
\right].
$$

对足够光滑的函数，复合 Simpson 法误差通常是 $O(h^4)$。因此在你的 PPT 中用 Simpson 法作为 Newton-Cotes 类代表是合理的：它比梯形法高阶，又仍然很容易实现。

在参数积分梯度重要性中，Newton-Cotes 方法的具体做法是：先选定等距节点 $\alpha_j$，在每个参数点

$$
v_j=\theta_t+\alpha_j\Delta\theta
$$

上计算梯度向量 $G(\alpha_j)$，然后用梯形或 Simpson 权重加权求和，最后逐参数乘上 $\Delta\theta$。例如 Simpson 版本为

$$
\hat\omega_{\mathrm{simp}}^t
=
\Delta\theta\odot \hat I_{\mathrm{simp}} .
$$

它的时间复杂度主要由节点数决定。若复合 Simpson 使用 $n+1$ 个节点，则

$$
T_{\mathrm{simp}}\approx (n+1)C_{\mathrm{grad}}(B,d).
$$

Newton-Cotes 类方法最适合作为低成本 baseline。它的优势不是在同样节点数下达到最高精度，而是实现简单、节点可复用、对局部非光滑不那么脆弱。特别是在 ReLU 网络中，路径梯度函数往往分段光滑而非全局解析，高阶全局多项式积分不一定占优；此时复合梯形法或复合 Simpson 法配合自适应分段，反而可能比固定的高阶全局公式更稳定。

## 高斯积分法

高斯积分法的出发点与 Newton-Cotes 不同。Newton-Cotes 固定等距节点，再根据节点求权重；高斯积分则同时选择节点和权重，使得在相同节点数下对尽可能高次数的多项式精确。对区间 $[-1,1]$ 上的 Gauss-Legendre 积分，$n$ 个节点 $x_i$ 取为 $n$ 次 Legendre 多项式的根，积分公式为

$$
\int_{-1}^{1} f(x)\,dx
\approx
\sum_{i=1}^{n} w_i f(x_i).
$$

这个公式可以对次数不超过 $2n-1$ 的多项式精确。这一性质来自 Legendre 多项式的正交性，也是高斯积分在光滑函数上非常高效的原因。

你的积分区间是 $[0,1]$，因此需要把 $[-1,1]$ 上的节点映射过去：

$$
\alpha_i=\frac{x_i+1}{2},
\qquad
\tilde w_i=\frac{w_i}{2}.
$$

于是

$$
\int_0^1 G(\alpha)\,d\alpha
\approx
\sum_{i=1}^{n}\tilde w_iG(\alpha_i).
$$

对应的重要性估计为

$$
\hat\omega_{\mathrm{GL}}^t
=
\Delta\theta\odot
\sum_{i=1}^{n}\tilde w_iG(\alpha_i).
$$

高斯积分的主要优点是：在被积函数非常光滑，尤其接近解析函数时，它往往能用很少节点达到很高精度。对于神经网络中平滑激活函数，例如 tanh、sigmoid、GELU 等，若更新路径较短，$G(\alpha)$ 可能比较光滑，高斯积分会非常有效。

但高斯积分也有两个明显缺点。第一，它通常不是嵌套的。也就是说，$n$ 点 Gauss-Legendre 的节点一般不是 $2n$ 点 Gauss-Legendre 节点的子集。如果先算了 $n$ 个点，后来发现精度不够，升级到更多节点时，旧节点往往不能完全复用。这在每个节点都要反向传播的问题中是很大的代价。第二，高斯积分依赖全局多项式逼近。如果路径上存在 ReLU 激活模式切换、MaxPool 选择变化或其他非光滑点，固定的全局高斯节点可能无法有效捕捉局部突变。

高斯积分的时间复杂度是

$$
T_{\mathrm{GL}}\approx n\,C_{\mathrm{grad}}(B,d).
$$

如果用同样 $n$ 个节点比较，高斯积分通常比 Newton-Cotes 有更高的多项式精确度；但如果需要自适应加密，由于节点不嵌套，实际累计成本可能变高。

在参数重要性估计中，高斯积分适合作为“平滑路径假设下的高精度方法”。它可以回答一个基本问题：在固定节点数下，合理选择节点是否比等距节点更能稳定估计重要性。如果高斯积分明显优于 Simpson，说明路径梯度函数在大部分区间比较平滑；如果高斯积分优势不明显甚至更差，说明非光滑或噪声可能主导误差。

## Gauss-Kronrod 方法

Gauss-Kronrod 方法可以看作高斯积分的自适应友好版本。它的核心思想是在一个低阶 Gauss 规则上添加额外节点，构造一个更高阶的 Kronrod 规则；两个规则共享部分节点，因此同一批函数值可以同时给出一个低阶估计和一个高阶估计。二者差值可以作为局部误差估计。典型例子是 $G7/K15$：7 点 Gauss 规则嵌入在 15 点 Kronrod 规则中。QUADPACK 中的一维全局自适应积分器 QAG 就使用 Gauss-Kronrod quadrature，QAGS 则结合区间细分和外推，QAGP 用于含已知奇异点或不连续点的情形。

对一个子区间 $[a,b]$，Gauss-Kronrod 会给出两个向量积分估计：

$$
I_G[a,b]=\sum_{i\in\mathcal G} w_i^G G(\alpha_i),
$$

$$
I_K[a,b]=\sum_{i\in\mathcal K} w_i^K G(\alpha_i),
$$

其中 $\mathcal G\subset \mathcal K$。由于 Kronrod 规则使用更多节点，它通常被作为该区间的积分估计；Gauss 与 Kronrod 的差值则用于估计局部误差：

$$
E[a,b]=I_K[a,b]-I_G[a,b].
$$

在普通标量积分中，误差可以写成 $|E[a,b]|$。但你的被积函数是梯度向量，最终关心的是重要性向量，因此更合理的误差指标是

$$
e[a,b]=
\left\|
\Delta\theta\odot E[a,b]
\right\|_2,
$$

或者若剪枝任务更关心最大坐标误差，可以使用

$$
e_\infty[a,b]=
\left\|
\Delta\theta\odot E[a,b]
\right\|_\infty .
$$

这样做的原因是，数值积分误差只有在乘上 $\Delta\theta$ 后才真正影响参数重要性。某个参数的积分误差很大，但若它本轮几乎没有更新，则其重要性误差并不大。

Gauss-Kronrod 在自适应分段中的应用非常自然。算法从整个区间 $[0,1]$ 开始，在其上计算 $I_K[0,1]$ 和误差 $e[0,1]$。如果误差已经小于容忍度，就接受这个估计。否则，把误差最大的区间二分，在左右子区间分别重新做 Gauss-Kronrod，更新总积分和总误差。这个过程不断重复，直到总误差满足要求或达到最大反向传播预算。

若最终接受了 $S$ 个子区间，而每个子区间使用 $K$ 个 Kronrod 节点，则在没有跨区间节点复用的情况下，实际评估过的区间数量约为 $2S-1$，时间成本大致为

$$
T_{\mathrm{GK-adapt}}
\approx K(2S-1)C_{\mathrm{grad}}(B,d).
$$

这里 $K$ 可以是 15、21、31 等，取决于使用的 Gauss-Kronrod 对。这个成本看起来比固定 $K$ 点高，但优势是它只在梯度变化剧烈的区间加密。若路径大部分平滑，只在少数位置存在突变，自适应 Gauss-Kronrod 可能用远少于全局高密度节点的成本达到同样精度。

具体到参数积分梯度重要性，Gauss-Kronrod 方法的推荐实现如下。每个区间上同时计算两个估计 $I_G$ 和 $I_K$，用

$$
\Delta\theta\odot(I_K-I_G)
$$

作为重要性误差向量。全局误差可以用各区间误差的和或平方和估计：

$$
e_{\mathrm{global}}
=
\sum_J e[J],
\qquad\text{或}\qquad
e_{\mathrm{global}}
=
\left(\sum_J e[J]^2\right)^{1/2}.
$$

如果最终任务是剪枝，还可以只对剪枝阈值附近的一组参数 $\mathcal B$ 控制误差：

$$
e_{\mathcal B}[J]
=
\left\|
\left(\Delta\theta\odot(I_K[J]-I_G[J])\right)_{\mathcal B}
\right\|_\infty .
$$

这会把计算预算集中在影响剪枝决策的参数上，而不是浪费在重要性明显很大或明显很小的参数上。

## Clenshaw-Curtis 积分

Clenshaw-Curtis 积分也是一种多项式插值积分，但它不使用 Legendre 多项式根作为节点，而是使用 Chebyshev 点。对 $[-1,1]$，常用节点为

$$
x_j=\cos\frac{j\pi}{n},
\qquad j=0,\ldots,n.
$$

映射到 $[a,b]$ 后得到

$$
\alpha_j=\frac{a+b}{2}+\frac{b-a}{2}\cos\frac{j\pi}{n}.
$$

Clenshaw-Curtis 方法在这些 Chebyshev-Lobatto 点上插值，然后对插值多项式积分。它和 Gauss 积分相比，多项式精确度较低，但实际表现常常非常接近。Trefethen 的论文《Is Gauss Quadrature Better than Clenshaw-Curtis?》指出，Gauss 相对 Clenshaw-Curtis 的所谓“二倍优势”在实践中往往不明显，并用 Chebyshev 展开中的 aliasing 现象解释这一点。对有限光滑性函数，后续研究也表明 Gauss 和 Clenshaw-Curtis 的收敛率非常接近。

Clenshaw-Curtis 对这个问题尤其有吸引力，原因在于节点嵌套。若取 $n=2^\ell$，从 $n$ 加密到 $2n$ 时，旧的 Chebyshev-Lobatto 节点会保留，新节点只是补充进去。因此如果先用低阶规则粗略估计，再逐步加密，已经计算过的梯度节点可以复用。对于每个节点都意味着一次反向传播的参数重要性积分，这一点非常重要。

在一个子区间 $[a,b]$ 上，Clenshaw-Curtis 积分可写为

$$
I_{CC,n}[a,b]
=
\sum_{j=0}^{n} w_j^{CC}G(\alpha_j).
$$

权重可以通过快速余弦变换或相关递推高效计算。对于普通标量函数，权重计算可能是重要成本；但在神经网络梯度积分中，反向传播远比权重计算昂贵，因此权重计算成本通常可以忽略。相关快速实现研究也表明，Clenshaw-Curtis 与 Fejer 型求积可以通过 FFT/DCT 等方式高效实现。

Clenshaw-Curtis 的误差估计可以采用层级差分。比如同时计算 $n$ 阶和 $n/2$ 阶估计：

$$
E_{CC}[a,b]
=
I_{CC,n}[a,b]-I_{CC,n/2}[a,b].
$$

因为节点嵌套，$I_{CC,n/2}$ 的节点已经包含在 $I_{CC,n}$ 中，不需要重新计算旧节点。对应的重要性误差为

$$
e_{CC}[a,b]
=
\left\|
\Delta\theta\odot E_{CC}[a,b]
\right\|.
$$

另一个更谱方法的误差指标是观察 Chebyshev 系数的尾部。如果高阶系数快速衰减，说明该区间上的 $G(\alpha)$ 比较平滑，继续升阶可能有效；如果尾部系数不衰减或出现异常，说明该区间内可能有非光滑点，继续升阶不如切分区间。自适应积分误差估计文献中也讨论过基于系数的误差估计思想，误差估计本身被认为是自适应求积中最关键的组成部分。

因此，Clenshaw-Curtis 最适合做成一种 $hp$-adaptive 方法。所谓 $p$-adaptive 是在同一区间上提高阶数，例如从 $n=8$ 提高到 $n=16$；所谓 $h$-adaptive 是把区间二分。具体策略可以是：先在每个区间用较低阶数 $n=8$ 计算；若层级误差小，接受该区间；若误差大但 Chebyshev 系数尾部平滑衰减，则提高阶数；若误差大且尾部不衰减，则二分区间。这样做符合神经网络路径积分的结构：在平滑段升阶，在激活模式变化或梯度突变附近切段。

时间复杂度方面，若最终共使用 $N_{\mathrm{unique}}$ 个不同的 $\alpha$ 节点，则

$$
T_{\mathrm{CC-adapt}}
\approx
N_{\mathrm{unique}}C_{\mathrm{grad}}(B,d)
+
T_{\mathrm{weight}}.
$$

其中 $T_{\mathrm{weight}}$ 通常可以通过 DCT/FFT 以 $O(N_{\mathrm{unique}}\log N_{\mathrm{unique}})$ 或分区间小规模计算完成，相比反向传播可忽略。因此 Clenshaw-Curtis 的真实优势可以概括为：它不一定每个固定节点数都压倒 Gauss，但它能在自适应加密中最大限度复用已有梯度计算。

## 自适应分段积分的一般框架

自适应分段积分不是某一个固定公式，而是一类算法框架。它的核心是：先在大区间上粗略积分并估计误差；若误差过大，就把误差大的区间细分，或者在该区间提高积分阶数；不断重复，直到误差满足要求或计算预算耗尽。QUADPACK 这类成熟一维积分库就是围绕这种思想构建的，Gauss-Kronrod 规则常用于局部积分和误差估计。

对你的问题，自适应分段积分必须从标量函数改成向量值函数。对每个子区间 $J=[a,b]$，某种嵌入式或层级式方法给出高低两个积分估计：

$$
I_{\mathrm{high}}[J],
\qquad
I_{\mathrm{low}}[J].
$$

于是局部误差向量为

$$
E[J]=I_{\mathrm{high}}[J]-I_{\mathrm{low}}[J].
$$

由于最终重要性是 $\Delta\theta\odot I$，局部误差标量可以定义为

$$
e[J]=
\left\|
\Delta\theta\odot E[J]
\right\|_2.
$$

如果剪枝任务更关心阈值附近的参数，可以定义边界集合 $\mathcal B$，只在该集合上衡量误差：

$$
e_{\mathcal B}[J]=
\left\|
(\Delta\theta\odot E[J])_{\mathcal B}
\right\|_\infty .
$$

自适应算法维护一个区间列表。每个区间保存当前积分估计、误差估计和已经计算过的节点。每一步选择误差最大的区间。如果使用 Gauss-Kronrod 或 adaptive Simpson，通常直接二分该区间；如果使用 Clenshaw-Curtis，可以先判断是否适合升阶，若不适合再二分。全局积分估计为所有区间估计之和：

$$
\hat I=\sum_{J}I_{\mathrm{high}}[J],
\qquad
\hat\omega=\Delta\theta\odot \hat I.
$$

全局误差可以保守地取

$$
e_{\mathrm{global}}=\sum_J e[J],
$$

也可以在误差近似独立时取平方和形式

$$
e_{\mathrm{global}}=
\left(\sum_J e[J]^2\right)^{1/2}.
$$

停止条件可以是绝对误差与相对误差结合：

$$
e_{\mathrm{global}}
\le
\varepsilon_{\mathrm{abs}}
+
\varepsilon_{\mathrm{rel}}\|\hat\omega\|.
$$

但在 stochastic gradient 场景下，还需要考虑随机噪声。如果 minibatch 梯度本身有统计误差，那么当数值积分误差已经小于随机误差时，继续细分路径没有意义。因此更合理的噪声感知停止条件是

$$
e_{\mathrm{quad}}[J]\le \lambda e_{\mathrm{stat}}[J],
$$

其中 $e_{\mathrm{quad}}$ 是数值积分误差，$e_{\mathrm{stat}}$ 是由 microbatch 方差或 U-statistic 估计出的随机误差。这个条件的含义是：不要为了降低已经被随机噪声淹没的积分误差而继续花费反向传播预算。

## 四类方法在自适应分段中的具体应用

Newton-Cotes 类方法最自然的自适应版本是 adaptive trapezoid 或 adaptive Simpson。以 adaptive Simpson 为例，在区间 $[a,b]$ 上先用 Simpson 公式得到 $S[a,b]$，再把区间分成 $[a,m]$ 和 $[m,b]$，得到 $S[a,m]+S[m,b]$。两者差值给出误差估计：

$$
E_S[a,b]
=
S[a,m]+S[m,b]-S[a,b].
$$

向量重要性误差为

$$
e_S[a,b]
=
\frac{1}{15}
\left\|
\Delta\theta\odot E_S[a,b]
\right\|.
$$

其中 $1/15$ 是 Simpson 外推误差估计中的经典系数。这个方法的优点是非常容易实现，节点复用也简单，因为端点和中点会在细分后继续使用。它的缺点是阶数有限，若函数非常光滑，同样精度下可能需要比 Gauss 或 Clenshaw-Curtis 更多节点。但对 ReLU 网络这类分段光滑函数，adaptive Simpson 是一个很好的稳健基线。

高斯积分本身不是特别适合直接自适应加密，因为不同阶数的节点不嵌套，升阶时旧节点难以复用。若要把 Gauss-Legendre 用于自适应分段，通常做法不是在同一区间上不断升阶，而是固定每段使用 $n$ 点 Gauss，然后对误差大的区间二分。误差估计可以通过比较 $n$ 点和 $m$ 点 Gauss 结果得到，但若二者节点不共享，就需要额外梯度计算。因此，单独的 adaptive Gauss-Legendre 在你的问题中不如 Gauss-Kronrod 或 Clenshaw-Curtis 自然。它更适合作为固定节点数的高精度 baseline。

Gauss-Kronrod 则是自适应分段的经典选择。它每次在一个区间上同时得到高低两个估计，而且共享节点，因此局部误差估计几乎没有额外函数评估成本。应用到参数重要性中，只需把标量误差换成向量重要性误差：

$$
e_{GK}[J]=
\left\|
\Delta\theta\odot(I_K[J]-I_G[J])
\right\|.
$$

然后总是细分误差最大的区间。它的主要优点是成熟、稳定、误差估计自然；缺点是跨层级节点复用不如 Clenshaw-Curtis，且每个区间的 Kronrod 节点数较多。

Clenshaw-Curtis 的自适应版本最适合采用“升阶或切分”的策略。在一个区间上先用低阶 Chebyshev 节点估计，再用嵌套高阶节点加密。若高低阶差异小，则接受；若差异大但 Chebyshev 系数尾部衰减，继续升阶；若差异大且尾部不衰减，切分区间。它的优点是节点复用能力强，适合昂贵梯度评估；缺点是实现比 adaptive Simpson 和 Gauss-Kronrod 稍复杂，需要维护 Chebyshev 节点、权重和可能的系数尾部诊断。

## 与 microbatch-level U-statistic 的结合

前面的数值积分讨论默认每个节点上可以得到确定性梯度 $G(\alpha)$。但在实际训练中，我们通常得到的是 minibatch 或 microbatch 梯度。若采用 $M$ 个 microbatch，第 $m$ 个 microbatch 在积分节点 $\alpha_q$ 上的梯度记为

$$
G_m(\alpha_q).
$$

某个数值积分方法给出权重 $w_q$ 和节点 $\alpha_q$，则第 $m$ 个 microbatch 的路径积分梯度为

$$
Y_m=
\sum_q w_qG_m(\alpha_q).
$$

更新点处的 microbatch 梯度记为

$$
X_m=G_m(0).
$$

若要使用 microbatch-level U-statistic 估计重要性，可以计算

$$
\hat\omega_{\mathrm{MB-U}}
=
\gamma
\frac{
\left(\sum_m X_m\right)\odot
\left(\sum_m Y_m\right)
-
\sum_m X_m\odot Y_m
}{M(M-1)}.
$$

这里的数值积分方法只影响 $Y_m$ 的计算。Newton-Cotes、Gauss-Legendre、Gauss-Kronrod、Clenshaw-Curtis 都可以替换进去。对于自适应方法，不同区间和节点的 $Y_m$ 会逐步累加；只要每个节点上能得到每个 microbatch 的局部梯度，就可以保持 U-statistic 的无偏结构。

在工程上可以流式维护

$$
S_X=\sum_m X_m,
\qquad
S_Y=\sum_m Y_m,
\qquad
S_{XY}=\sum_m X_m\odot Y_m,
$$

最后再计算

$$
\hat\omega_{\mathrm{MB-U}}
=
\gamma
\frac{S_X\odot S_Y-S_{XY}}{M(M-1)}.
$$

如果使用自适应积分，$Y_m$ 不是一次得到，而是随着新区间、新节点不断更新。每当新增节点 $\alpha_q$，就对每个 microbatch 累加

$$
Y_m\leftarrow Y_m+w_qG_m(\alpha_q).
$$

若某个区间被细分，需要注意旧区间权重被替换为左右子区间权重，因此实现上最好按区间维护每个 microbatch 的局部积分贡献，接受区间后再加到全局 $Y_m$ 中。

## 时间复杂度比较

设一次在一个 $\alpha$ 节点上计算完整 batch 梯度的成本为 $C_{\mathrm{grad}}(B,d)$。若只比较确定性数值积分，四类方法的主导成本如下。

Newton-Cotes 固定复合公式若使用 $Q$ 个等距节点，成本为

$$
T_{\mathrm{NC}}\approx Q C_{\mathrm{grad}}(B,d).
$$

它的节点可复用，加密时只需补充新等距中点。若使用 adaptive Simpson，最终使用的不同节点数记为 $N_{\mathrm{unique}}$，则

$$
T_{\mathrm{AS}}\approx N_{\mathrm{unique}}C_{\mathrm{grad}}(B,d).
$$

Gauss-Legendre 若使用 $n$ 个节点，成本为

$$
T_{\mathrm{GL}}\approx n C_{\mathrm{grad}}(B,d).
$$

在固定节点数下它精度高，但若不断升阶，旧节点通常不能复用，累计成本可能近似为

$$
T_{\mathrm{GL-refine}}
\approx
(n_1+n_2+\cdots+n_L)C_{\mathrm{grad}}(B,d),
$$

而不是最终最高阶节点数 $n_L$ 的成本。

Gauss-Kronrod 若每个区间使用 $K$ 个 Kronrod 节点，最终接受 $S$ 个区间，则实际评估过的区间数通常约为 $2S-1$，成本为

$$
T_{\mathrm{GK-adapt}}
\approx
K(2S-1)C_{\mathrm{grad}}(B,d).
$$

它的优势在于每个区间同时给出高低阶估计，局部误差估计不需要额外节点。缺点是跨区间细分时新子区间通常仍需重新评估一组节点。

Clenshaw-Curtis 自适应积分若最终使用 $N_{\mathrm{unique}}$ 个不同节点，则

$$
T_{\mathrm{CC-adapt}}
\approx
N_{\mathrm{unique}}C_{\mathrm{grad}}(B,d)+O(N_{\mathrm{unique}}\log N_{\mathrm{unique}}),
$$

其中后面的项来自权重或系数计算，通常远小于反向传播成本。由于节点嵌套，Clenshaw-Curtis 在反复加密时能更好复用旧节点，因此对昂贵梯度积分特别有吸引力。

若进一步考虑 microbatch-level U-statistic，设每个 full batch 被拆成 $M$ 个 microbatch。若在每个节点上串行计算所有 microbatch 梯度，总成本近似为

$$
T\approx N_{\mathrm{unique}}\sum_{m=1}^{M}C_{\mathrm{grad}}(b,d).
$$

若 $b=B/M$，通常这个量与 $N_{\mathrm{unique}}C_{\mathrm{grad}}(B,d)$ 同阶。若使用多卡并行，每张卡处理一个或多个 microbatch，则墙钟时间可能接近

$$
T_{\mathrm{wall}}
\approx
N_{\mathrm{unique}}C_{\mathrm{grad}}(b,d)
+
T_{\mathrm{comm}}
+
T_{\mathrm{record}},
$$

其中 $T_{\mathrm{comm}}$ 是通信开销，$T_{\mathrm{record}}$ 是记录 local gradient 或流式统计量的开销。若只能获得 all-reduce 之后的平均梯度，无法直接构造 microbatch-level U-statistic；必须在同步前截取 local gradient，或使用 no-sync、communication hook、手动 accumulation 等方式保留 microbatch 信息。

## 方法选择建议

若目标是建立最简单、最稳健的 baseline，应从复合梯形法、复合 Simpson 法和 adaptive Simpson 开始。它们实现成本低，节点复用好，在非光滑路径下不容易出现高阶多项式震荡。

若目标是在较小固定节点数下追求高精度，应使用 Gauss-Legendre 积分。它适合路径梯度函数较平滑的模型和较小更新步长，但在自适应加密时节点复用能力较差。

若目标是成熟、可解释、带误差估计的自适应方法，应使用 Gauss-Kronrod。它尤其适合作为论文中的强 baseline，因为 QUADPACK 等经典库已经长期采用类似策略，一维全局自适应 Gauss-Kronrod 是非常成熟的方法。

若目标是为昂贵神经网络梯度积分设计更有针对性的方法，应优先考虑 Clenshaw-Curtis，特别是噪声感知的自适应 Clenshaw-Curtis。它的节点嵌套性使得逐步加密时不会浪费已计算的梯度；配合 Chebyshev 系数尾部或层级差分，可以在平滑区间升阶，在非光滑区间切分。

对你的参数积分梯度重要性问题，我最推荐的实验路线是同时比较以下几种方法：固定复合 Simpson、固定 Gauss-Legendre、自适应 Simpson、自适应 Gauss-Kronrod、自适应 Clenshaw-Curtis。评估指标不应只看积分误差，还应看重要性排序稳定性、剪枝 mask 稳定性、剪枝后性能、达到相同性能所需的反向传播次数，以及与 microbatch-level U-statistic 结合后的总方差。

## 参考文献与资料

[1] Newton-Cotes formulas and degree of precision, numerical integration lecture notes and general references.  
https://ahmedbadary.github.io/work_files/school/128a/4_3  
https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas

[2] Gauss-Legendre quadrature and Gaussian quadrature, exactness for polynomials up to degree $2n-1$.  
https://en.wikipedia.org/wiki/Gaussian_quadrature  
https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature

[3] Orthogonal polynomials and Gaussian quadrature notes.  
https://www.johndcook.com/OrthogonalPolynomials.pdf

[4] QUADPACK library documentation, including globally adaptive integrators using Gauss-Kronrod quadrature, interval subdivision, extrapolation, and handling of singularities/discontinuities.  
https://www.netlib.org/quadpack/

[5] Gauss-Kronrod quadrature formula and embedded error estimation.  
https://www.boost.org/doc/libs/release/libs/math/doc/html/math_toolkit/gauss_kronrod.html  
https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula

[6] Lloyd N. Trefethen, “Is Gauss Quadrature Better than Clenshaw-Curtis?”, SIAM Review, 2008.  
https://epubs.siam.org/doi/10.1137/060659831  
https://people.maths.ox.ac.uk/trefethen/publication/PDF/2008_127.pdf

[7] Shuhuang Xiang and Folkmar Bornemann, “On the convergence rates of Gauss and Clenshaw-Curtis quadrature for functions of limited regularity.”  
https://arxiv.org/abs/1203.2445

[8] Shuhuang Xiang, Guo He, Haiyong Wang, “On Fast Implementation of Clenshaw-Curtis and Fejer-type Quadrature Rules.”  
https://arxiv.org/abs/1311.0445

[9] Pedro Gonnet, “A Review of Error Estimation in Adaptive Quadrature.”  
https://arxiv.org/abs/1003.4629
