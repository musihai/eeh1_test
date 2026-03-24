# eeh1_test 最新代码全量静态审查报告

## 1. 审查范围与结论先行

我这次做的是：

- 解压并通读仓库主线代码：`arft/*`、`recipe/time_series_forecast/*`、`examples/*`、`tests/*`
- 对照你上传的 Cast-R1 论文 PDF，重点核对了：
  - 单变量 / 多变量设定
  - 3-stage workflow
  - reward 组成
  - curriculum learning
  - refine 机制
  - 1.7B 与论文主设定的差异
- 做了最基本的可运行性检查：`pytest -q`

### 总体判断

**结论一句话：你现在这版代码，针对“ETTh1-OT 单变量 + 1.7B + 3-turn compact 协议”这条主线，核心时序逻辑已经基本打通了，尤其是 Turn 3 的 refine supervision 比之前健康很多；但整个仓库还没有达到“简洁、完全自洽、可直接复现”的状态。**

我认为现在的问题，不再是“主流程完全错了”，而是下面这几类：

1. **仓库自洽性还不够**
   - 缺少依赖清单
   - 缺少 `recipe/time_series_forecast/models/*` 模型代码/配置/ckpt 目录
   - 测试不能在干净环境直接跑起来

2. **旧路径 / 旧默认值 / 历史产物还很多**
   - `same2` / `same3` 混用
   - 根目录里还有历史说明文档、调试日志、产物图、报告
   - 有空文件和明显的 dead code

3. **方法上已经比旧版好，但仍然不是论文原版**
   - 你现在是**单变量 ETTh1-OT 变体**
   - 论文 ETTh1 是**多变量**
   - 你的 refine 是**规则生成的局部修正标签**
   - 你的 RL 默认配置仍然**明显比论文轻**

4. **最需要继续修的，不是“再加代码”，而是“减耦合、去旧代码、统一默认行为”**

---

## 2. 我实际发现的硬问题

## 2.1 测试目前不能在干净环境收集通过

我跑了：

```bash
cd /mnt/data/eeh1_test-main
pytest -q
```

结果在 **collection 阶段** 就报错，核心缺失依赖是：

- `verl`
- `hydra`
- `omegaconf`

典型错误包括：

- `ModuleNotFoundError: No module named 'verl'`
- `ModuleNotFoundError: No module named 'hydra'`
- `ModuleNotFoundError: No module named 'omegaconf'`

这说明一个非常现实的问题：

**仓库现在还不是“拿到压缩包就能在新环境里验证”的状态。**

这不一定说明你的逻辑错，但说明：
- 代码仓库还不够自包含
- 测试层和训练框架层耦合过深
- 缺少最基本的环境重建文件

### 建议
至少补一个：
- `requirements.txt`，或者
- `environment.yml`，或者
- README 里明确写死依赖安装步骤

---

## 2.2 `model_server.py` / `retrain_expert_models_train_split.py` 依赖的模型目录在当前仓库里不存在

代码里明确引用了这些路径/模块：

- `recipe.time_series_forecast.models.patchtst.model`
- `recipe.time_series_forecast.models.itransformer.model`
- `recipe/time_series_forecast/models/patchtst/config.json`
- `recipe/time_series_forecast/models/itransformer/config.json`
- `recipe/time_series_forecast/models/*/checkpoint.pth`

但我检查当前压缩包时，`recipe/time_series_forecast/` 下面只有这些目录：

- `recipe/time_series_forecast`
- `recipe/time_series_forecast/__pycache__`

**没有 `models/` 子目录。**

这意味着当前 zip 包里的以下功能，在干净环境下是不能独立成立的：

- `recipe/time_series_forecast/model_server.py`
- `recipe/time_series_forecast/retrain_expert_models_train_split.py`
- `build_etth1_high_quality_sft.py` 的 `predictor_mode=local`
- 一部分 smoke tests

### 这个问题的性质
这不是小问题，这是仓库完整性问题。

### 建议
二选一：

1. **把 `recipe/time_series_forecast/models/*` 真正补进仓库**
2. 或者明确声明：
   - 这些模型目录是外部资源
   - 需要另外准备
   - README 给出准确路径约定

如果不做这件事，仓库表面上完整，实际上不是完整可运行仓库。

---

## 2.3 `build_etth1_sft_dataset.py` 的默认 CLI 路径和它自己的 metadata 校验逻辑不一致

这是我认为最明确的一个代码问题。

### 当前情况

`build_etth1_sft_dataset.py` 的默认参数还指向：

- `dataset/ett_rl_etth1_paper_same2/train.jsonl`
- `dataset/ett_rl_etth1_paper_same2/val.jsonl`
- `dataset/ett_rl_etth1_paper_same2/test.jsonl`

但是在 `main()` 里，它又要求 source metadata 是：

- `DATASET_KIND_TEACHER_CURATED_SFT`

也就是说：

- 默认输入路径看起来像 **RL jsonl**
- 但实际校验要求它是 **teacher-curated SFT source**

这两个前提并不一致。

### 结果
如果别人直接按默认参数跑，很容易出现：
- 路径看起来合法
- 但 metadata kind 校验直接失败

### 这属于什么问题
这是**旧默认值没清干净**。

### 建议
把 `build_etth1_sft_dataset.py` 的默认输入路径改成真正的 teacher-curated 路径，或者直接：
- 不给默认路径
- 强制用户显式传参

---

## 2.4 `build_etth1_sft_dataset.py` 的 runtime fallback 仍然隐藏了一个“时间戳假设”

`build_etth1_sft_dataset.py` 里的 `_predict_with_runtime_tools()` 会做这件事：

- 先把历史数据 parse 成 dataframe
- 再调用 runtime predictor
- 之后用 `get_last_timestamp(historical_data)` 去格式化 teacher prediction
- 如果拿不到 last timestamp，就直接抛错

问题在于：

`build_etth1_rl_dataset.py` 生成的 prompt 历史部分是 **value-only** 的，不带真实 timestamp。

也就是说，如果 `build_sft_record()` 走到 runtime fallback，而输入样本里又没有缓存好的 `teacher_prediction_text`，这条路是可能直接失败的。

### 为什么现在没完全炸
因为当前主线更依赖：
- teacher curated jsonl
- 缓存好的 `teacher_prediction_text`

所以大部分时候绕过了这个坑。

### 但它仍然是个隐藏不一致
同一个项目里：
- `utils.format_predictions_to_string()` 在没有 timestamp 时是允许 synthetic anchor 的
- 但 `build_etth1_sft_dataset._predict_with_runtime_tools()` 却要求必须有真实 last timestamp

### 建议
把这里统一掉：
- 要么都允许 synthetic anchor
- 要么全链路都统一要求 timestamped history

**现在最自然的做法是：这里也允许 synthetic anchor。**

---

## 3. 和论文对照后的总体判断

## 3.1 你的代码已经不是论文原版复现，而是“单变量 ETTh1-OT 变体”

这个一定要说清楚。

论文里 ETTh1 是：
- **long-term**
- **multivariate**
- 包含 **transformer oil temperature + 6 power load features**

而你当前代码已经明确写成：
- `Single-variable time-series forecasting`
- 目标列是 `OT`
- focus on target itself rather than cross-channel reasoning

这不是 bug，这是**任务定义已经被你改了**。

### 所以该怎么判断“是否正确”
- 如果目标是“论文一模一样复现”，那现在还没对齐
- 如果目标是“单变量 ETTh1-OT 版 Cast-R1”，那你现在方向基本正确

---

## 3.2 你现在最重要的进步：Turn 3 已经不再是简单抄 teacher

这一点我认为是这版代码最值得肯定的地方。

旧问题通常是：
- Turn 3 最终答案直接复制 teacher prediction
- 或者 teacher 不在就 fallback 到 GT
- 这样模型根本学不到 refine

你现在的 `build_etth1_sft_dataset.py` 已经不是这样了。

它现在做的是：
- 从 `base_prediction_text` 出发
- 生成局部修正候选：
  - `isolated_spike_smoothing`
  - `flat_tail_repair`
  - `local_level_adjust`
  - `local_slope_adjust`
  - `amplitude_clip`
- 用 GT 计算 candidate refine 是否真的改进
- 只有在“局部 enough + gain meaningful”时，才把目标打成 `local_refine`
- 否则就是 `validated_keep`

**这比“直接抄 teacher”健康很多。**

### 这点和论文的关系
虽然它仍然不是论文里那种更开放的 reflective refinement，
但对 **1.7B + 单变量** 来说，这反而是更稳的实现。

### 我的判断
这一块逻辑现在是 **对的，而且比你之前的方向正确很多。**

---

## 3.3 但它仍然不是论文意义上的完整 refine

要实话实说。

你当前 refine supervision 还是一种：

- **规则生成的局部 edit 标签**
- 不是更高层的 agent self-reflection policy

它适合当前场景，但上限也被它限制住了。

### 当前版本的特点
优点：
- 稳
- 可控
- 容易让 1.7B 学会“不乱改，只修局部问题”

缺点：
- 更像“局部后处理器”
- 不是论文里更通用的“反思-修正”能力

### 我的判断
这不是 bug，而是**容量受限下的工程折中**。  
对 1.7B 是合理的，但你不要指望这条路线直接打到论文 headline。

---

## 3.4 reward 比早期版本更健康，但还不够解耦

你现在的 `reward.py` 已经不是“只剩下 MSE”的那种弱版本了。

当前保留了：
- format / length
- normalized / log-compressed MSE
- change-point / structural alignment
- season / trend consistency

这在方向上是对的，也更接近论文。

### 但还有两个问题

#### 问题 A：`reward.py` 顶层直接硬依赖 `verl.utils.chain_debug`
这会导致很多其实只想用 parser / scorer 的模块，也必须带上 `verl`。

实际效果就是：
- `validate_turn3_format.py` 这种轻工具
- 若 import `reward.py`
- 就会被 `verl` 卡住

这是一种不必要的重耦合。

#### 建议
把这部分改成：

- `try/except ImportError`
- 没有 `verl` 时提供 no-op fallback

例如：
- `append_chain_debug = lambda *args, **kwargs: None`
- `short_text = lambda x, *args, **kwargs: str(x)`

这样 reward 的纯工具能力就能独立存在。

---

#### 问题 B：`compute_score()` 里有重复逻辑 / 半死代码
例如：
- `compute_mse_score()` 存在，但 `compute_score()` 里又手写了一遍主 MSE 分数
- `length_hard_fail` 变量在主流程里基本始终是 `False`
- 说明 reward 文件经过多轮迭代后还有残留结构没清理干净

### 我的判断
不是逻辑错误，但这是典型的**可读性债务**。

---

## 4. 整体流程是不是“最好的方式”？

我的判断是：

**对“ETTh1-OT 单变量 + 1.7B + 想先把 agent 做稳”这个目标来说，当前整体流程已经接近一个合理工程解；但它还不是最优雅、也不是最贴论文的方法。**

---

## 4.1 当前流程的优点

当前主线：

`基础 RL jsonl -> teacher eval -> curated teacher data -> runtime SFT parquet -> SFT -> RL`

它的优点是：

1. **把 teacher ceiling 和 agent 学习分开了**
2. **先做 SFT，再做 RL，方向正确**
3. **Turn 3 有了真正的 refine 标签**
4. **curriculum 字段已经回来了**
5. **dataset identity / metadata 校验做得不错**
6. **launcher 比以前稳，尤其 SFT_DATASET_DIR 的约束是有价值的**

---

## 4.2 当前流程不是“最优”的地方

### A. `max_steps=3` + `predict_time_series 必须在 turn 2`
这是非常强的流程硬约束。

优点：
- 协议稳定
- 训练 easier
- 格式控制强

缺点：
- agent 没有弹性
- 一旦某个样本需要额外诊断，你也不允许
- 这比论文里的“多步 sequential decision process”更僵硬

### 我的判断
**对 1.7B 可以接受，但它不是方法上最优。**

如果后面上 4B/8B，可以考虑放宽成：
- `max_steps=4~5`
- 允许 prediction 不一定严格固定在 turn 2
- 只要求“先完成 required diagnostics，再 prediction，再 final refine”

---

### B. 你给了 selected forecast 一个中间奖励
`time_series_forecast_agent_flow.py` 里，routing turn 成功调用 `predict_time_series` 时，会给：

- `0.25 * selected_forecast_reward`

这个设计是稳训练的工程技巧，但有副作用：

- 会强化“先选一个 teacher 分高的 forecast”
- 容易把 RL 的注意力往“选好 teacher”推
- 相对削弱了 Turn 3 refine 的重要性

### 我的判断
这不是错，但**会让 policy 更偏 teacher routing，而不是 refine policy**。

### 建议
如果你后续发现：
- 模型越来越会 copy teacher
- 但 refine gain 不增长

那这一项应该调低，甚至去掉做 A/B。

---

### C. 诊断工具选择还是静态 heuristic
`diagnostic_policy.py` 现在是规则式的：

- 强 dynamics -> 加 dynamics tool
- residual difficulty -> 加 residual tool
- quality issue -> 加 quality tool
- eventful window -> 加 event tool

这在工程上是合理的，但和论文里的：
- expert-curated trajectories
- agentic planning

还是有距离。

### 我的判断
**当前这块“够用”，但不是最佳。**

如果你要追求更像论文，可以做两件事之一：

1. 保持 heuristic，但把阈值做得更干净、更可配置
2. 或者在 teacher-curated data 里直接学习“哪些诊断工具组合更好”

---

## 5. 模块级审查

下面按模块说。

---

## 5.1 `recipe/time_series_forecast/task_protocol.py`

### 结论
**基本正确，建议保留。**

### 优点
- 对 prompt / historical block 的解析比较稳
- 支持：
  - value-only
  - timestamp + value
  - timestamp + `OT=...`
  - 多变量 named format
- 作为“协议入口层”很关键，而且写得比较干净

### 问题
- 没看到明显逻辑 bug
- 可以保留

### 建议
无需大改。

---

## 5.2 `recipe/time_series_forecast/prompts.py`

### 结论
**针对当前 compact 方案是合理的。**

### 优点
- 已经明确变成单变量任务
- Turn 1 / Turn 2 / Turn 3 的职责切分清晰
- Turn 3 明确要求：
  - selected model forecast 作为 base
  - 若 refinement 则 small/local/evidence-based
  - `<answer>` 严格 96 行

### 问题
- 这套 prompt 明显比论文更“协议化”
- 优点是稳，缺点是 agent 自由度更低

### 我的判断
对 1.7B 这是合理折中，不建议大改。

---

## 5.3 `recipe/time_series_forecast/diagnostic_policy.py`

### 结论
**逻辑正确，但本质是 heuristic policy，不是论文级 planning policy。**

### 优点
- 比固定只用一个工具好
- 规则清楚
- 对单变量 ETTh1 足够实用

### 问题
- 阈值都是静态规则
- 和 paper 的“trajectory-level learned reasoning”相比仍然弱

### 建议
当前版本先保留。  
以后如果要继续提升，不是先加更多规则，而是：
- 收集 tool selection 成功案例
- 学习更好的 selected_feature_tool signature 分布

---

## 5.4 `recipe/time_series_forecast/utils.py`

### 结论
**这是核心工具层，整体可用，但里面有明显可以瘦身的旧代码。**

### 优点
- parse / format / feature extraction / prediction dispatch 都集中在一个地方
- `predict_time_series_async()` 是当前主线入口，合理
- synthetic timestamp anchor 方案对 value-only prompt 很实用

### 问题

#### 问题 A：存在未使用 import
我做了静态检查，至少这些 import 没实际用到：

- `re`
- `datetime`
- `timedelta`
- `BaseModel`

#### 问题 B：仍有 legacy sync path
文件里还保留了同步版：

- `predict_time_series()`

而仓库实际主线已经在用：
- `predict_time_series_async()`

我检查到同步版 `predict_time_series()` 在仓库里没有被其他地方调用。

#### 问题 C：`CHRONOS_SERVICE_URL` 仍然作为 legacy 路径存在
主线已经是统一的 `MODEL_SERVICE_URL`，
但同步旧函数还在走 `_CHRONOS_SERVICE_URL`。

### 建议
如果你要“简洁代码”，我建议：

1. 删除未使用 import
2. 删除未使用的同步 `predict_time_series()`
3. 把 `CHRONOS_SERVICE_URL` legacy support 一并清掉
4. 统一只保留 async + unified model service 路线

---

## 5.5 `recipe/time_series_forecast/reward.py`

### 结论
**方向是对的，但耦合太重，且还有清理空间。**

### 优点
- reward 组成比以前健康
- 保留了结构项和趋势项
- parse / recover / debug 信息都比较全

### 问题

#### 问题 A：对 `verl` 的硬依赖不合理
如前面所说，这是结构问题。

#### 问题 B：格式恢复过于宽容
`parse_final_answer_protocol(..., allow_recovery=True)`  
这对训练早期是友好的，但副作用是：
- 某些不完全规范的输出仍然能被“救回来”
- 会削弱 strict protocol 本身的约束力

#### 问题 C：文件过大，职责过多
现在 `reward.py` 同时负责：
- parser
- recovery
- metric
- debug logging
- format diagnostics

这对维护不友好。

### 建议
做一次小型重构：

- `reward_parsing.py`
- `reward_metrics.py`
- `reward_debug.py`

不改功能，只拆文件。

---

## 5.6 `recipe/time_series_forecast/time_series_forecast_agent_flow.py`

### 结论
**这是当前仓库里最关键的主流程文件，整体逻辑是成立的，但它仍然是“稳定优先”的硬协议版本。**

### 优点
- state reset 逻辑完整
- workflow validation 很严格
- 明确限制：
  - required diagnostics 必须先完成
  - prediction 只能调用一次
  - Turn 3 不能再调工具
- refinement turn 的 sampling params 专门做了收紧
- final answer 不能抄历史窗口，这点很好

### 问题

#### 问题 A：过于刚性
你现在不仅要求：
- 必须预测一次

还要求：
- `predict_time_series must be called on turn 2 exactly`

这对 1.7B 稳定有帮助，但从 agent 方法上看太死。

#### 问题 B：中间奖励可能过强地偏 teacher routing
上面已经说过。

#### 问题 C：有明显可清理代码
我看到这些值得清理的点：

- `self.parse_error_message` 初始化了两次
- `io_records` 留着但相关逻辑大段注释掉
- 顶部有 `from recipe.time_series_forecast.prompts import *`
- 还有一些未使用 import：
  - `asyncio`
  - `re`
  - `ToolResponse`

### 建议
这份文件建议做一次“只清理不改行为”的整理：
1. 去掉 wildcard import
2. 去掉未使用 import
3. 删除注释掉的 `io_records` 旧代码
4. 合并重复初始化
5. 把 workflow rule 常量化，便于后续调 3-turn / 4-turn

---

## 5.7 `recipe/time_series_forecast/build_etth1_rl_dataset.py`

### 结论
**这份 builder 对当前单变量路线来说是合理的，curriculum 也比旧版强。**

### 优点
- 已经把：
  - `reference_teacher_error`
  - `normalized_permutation_entropy`
  - `curriculum_stage`
  - `curriculum_band`
  带进样本
- 这比最早只做基础 jsonl 的版本好很多

### 问题
- 历史 prompt 仍然是 value-only，这本身不是错，但会引出上面说的 timestamp 假设问题
- `DEFAULT_OUTPUT_DIR` 仍然是 `same2`
- 如果 README 主线已经转 `same3`，这里默认值就应该同步

### 建议
逻辑保留，统一默认值和文档即可。

---

## 5.8 `recipe/time_series_forecast/build_etth1_high_quality_sft.py`

### 结论
**功能上有价值，但耦合有点重，而且默认路径还是旧的。**

### 优点
- teacher eval -> score -> curate -> annotate -> parquet 这条链条打通了
- 有 `min_local_refine_ratio`
- 有 teacher distribution / metadata 输出
- 对训练数据质量控制是有帮助的

### 问题

#### 问题 A：默认路径还是 `same2`
和当前 README 主线不一致。

#### 问题 B：直接 import 了 `build_etth1_sft_dataset.py` 里的私有函数
现在有这种耦合：

- `_distribution_from_series`
- `_rebalance_train_turn3_targets`

按命名它们是私有 helper，
但却被另一个 builder 当公共接口用了。

这是结构味道很重的一个点。

#### 问题 C：如果走 `predictor_mode=local`，实际依赖缺失模型目录
这和仓库完整性问题连在一起。

### 建议
- 把共享 helper 抽到 `dataset_build_utils.py`
- 默认路径统一改成正式主线路径
- 如果本仓库不准备内置 expert model 代码/权重，就把 local mode 作为“可选扩展”，不是默认文档主线

---

## 5.9 `recipe/time_series_forecast/build_etth1_sft_dataset.py`

### 结论
**这是当前版本里进步最大的模块之一，但仍有两类问题：默认值旧、边界假设不统一。**

### 优点
- `local_refine` / `validated_keep` 的 turn3 target 逻辑现在是健康的
- 修正是局部的、受约束的
- 不再看到之前那种明显的 GT fallback 路线
- train rebalancing 也比以前强

### 问题

#### 问题 A：默认输入路径与 expected metadata kind 不一致
这个前面说过，是明确 bug。

#### 问题 B：runtime fallback 仍然要求真实 last timestamp
和当前 value-only RL prompt 不完全一致。

#### 问题 C：一些 underscore helper 实际上已经变成公共接口
说明文件边界没有收干净。

### 我的判断
功能逻辑已经对了，但**接口层和工程边界还没收好。**

---

## 5.10 `recipe/time_series_forecast/model_server.py` 与 `start_model_server.sh`

### 结论
**服务逻辑基本没大问题，但当前仓库并不自足，而且默认安全性也可以更保守。**

### 优点
- 把多个 expert model 统一成一个服务是对的
- batch 预测接口也有测试覆盖意图

### 问题

#### 问题 A：仓库缺模型目录
这是最大问题。

#### 问题 B：`start_model_server.sh` 默认 `HOST=0.0.0.0`
这意味着如果你在共享机器上开服务，它默认对外暴露。

### 建议
默认改成：

- `HOST=127.0.0.1`

需要外网暴露时再显式覆盖。

---

## 5.11 `recipe/time_series_forecast/dataset_identity.py`

### 结论
**这一块写得很好，建议保留。**

这是当前仓库里少数“我认为已经很工整”的文件之一。

好处：
- 防止 train/val/jsonl/parquet 混用
- 防止把旧阶段数据误喂到新阶段
- 对你这种多阶段 pipeline 非常重要

---

## 5.12 `recipe/time_series_forecast/build_etth1_sft_subset.py`

### 结论
**不是主线必须，但作为辅助脚本可以保留。**

如果你想让代码更简洁：
- 可以把它归类到 `tools/` 或 `scripts/`
- 不一定要放在核心主目录

---

## 5.13 `recipe/time_series_forecast/analyze_chain_debug.py`

### 结论
**纯辅助调试脚本，不是主线代码。**

可以保留，但建议：
- 移到 `tools/`
- 或 `scripts/debug/`

---

## 5.14 `arft/*` 框架层

### 结论
**ARFT / VERL 框架层我这次做的是“集成点审查”，不是逐行算法证明”。**

我重点看了：
- `arft/main_agent_ppo.py`
- `arft/config/arft_ppo_trainer.yaml`
- `arft/agent_flow/*`

### 评价
- `main_agent_ppo.py` 里加的 model path / reward path 校验是有价值的
- Hydra search path 插件也合理
- 但是整个框架层严重依赖：
  - `verl`
  - `hydra`
  - `ray`
  - `omegaconf`

这本身没错，但仓库需要清楚声明它是“基于外部框架二次封装”的。

### 明确可删项
- `arft/agent_flow/tool_agent_flow.py` 是 **0 字节空文件**
  - 这个可以直接删

---

## 6. 现在代码里还存在什么“方法层面的问题”

这里不是代码 bug，而是“即使代码能跑，也未必是最优方式”。

## 6.1 RL 默认配置仍然比论文轻很多

论文里关键 RL 设定包括：
- group size `G = 8`
- generation temperature `1.0`
- 主实验 backbone 更强
- curriculum learning 是重要组件

你现在这套默认配置仍然是更保守的 1.7B 训练方式，典型表现是：

- `RL_ROLLOUT_N=1`
- `RL_TEMPERATURE=0.3`
- `RL_TOTAL_EPOCHS=1`

### 这意味着什么
它更稳，但 exploration 也更弱。

### 我的判断
对于当前 1.7B 单变量版本，这是能理解的；  
但如果你问“是不是最好的方式”，答案是：

**不是。它是偏保守的工程配置。**

---

## 6.2 你的 curriculum 是合理近似，不是论文原味实现

你现在已经把：
- teacher error
- normalized permutation entropy
纳入 curriculum

这很好。

但本质上仍然是：
- quantile banding
- staged slice

而不是论文那种更完整的“easy-to-hard sample organization”实现细节。

### 我的判断
对你现在目标够用了，不是当前第一优先级问题。

---

## 6.3 refine supervision 更好了，但仍然偏“规则老师”

如前所述，这对 1.7B 是合理的。  
但如果后面你想继续逼近论文，真正需要升级的是：

- 不是再加工具
- 不是再改 prompt
- 而是让 Turn 3 target 不只是规则 edit，而是更强的 revision policy

当前先不建议动，除非 1.7B 已经跑稳且出现明显 ceiling。

---

## 7. 明确建议删除 / 归档 / 清理的内容

这里按“可以立刻删”和“建议移走”分开说。

## 7.1 可以立刻删除的

### 1. `arft/agent_flow/tool_agent_flow.py`
- 空文件
- 没有保留价值

### 2. 所有 `__pycache__` 与 `.pytest_cache`
当前仓库里已经带进来了：
- `.pytest_cache/`
- `arft/__pycache__/`
- `arft/agent_flow/__pycache__/`
- `recipe/time_series_forecast/__pycache__/`
- `tests/__pycache__/`

这些都应该删掉，不应该进仓库。

### 3. `artifacts/reports/final_launch_cmd.txt`
- 这是运行产物，不是源码

### 4. `artifacts/reports/*`
- 如果目标是“代码仓库简洁”，这些应移出 repo 或进入 release artifact，不要放主代码树

### 5. 根目录历史说明 Markdown
当前根目录还有历史报告 / 修改方案文档。  
如果你的目标是“简洁代码仓库”，建议：
- 放到 `docs/`
- 或移出仓库
- 不要混在源码根目录

### 6. `assets/*.png`
如果这些只是 README 配图，可以保留；  
如果只是临时截图，建议清理或挪到 `docs/assets/`

---

## 7.2 建议删除的旧代码

### 1. `recipe/time_series_forecast/utils.py` 中未使用的同步 `predict_time_series()`
我没看到它被其他地方调用。  
如果主线已经统一 async model service，这个旧函数可以删。

### 2. `utils.py` 里的 legacy `CHRONOS_SERVICE_URL` 支持
如果你已经统一用 `MODEL_SERVICE_URL`，这块可以去掉。

### 3. `time_series_forecast_agent_flow.py` 里注释掉的大段 `io_records` 旧逻辑
这些残留会增加阅读负担。

### 4. `reward.py` 里没真正形成独立价值的重复逻辑
例如：
- `compute_mse_score()` 与主评分逻辑重复
- `length_hard_fail` 相关状态没有形成闭环

不一定全删，但至少整理。

---

## 7.3 不建议删除的

这些虽然不是主流程核心，但我建议保留：

- `dataset_identity.py`
- `validate_turn3_format.py`
- `benchmark_models_on_rl_samples.py`
- `build_etth1_sft_subset.py`
- `analyze_chain_debug.py`

但建议把后两类移动到：
- `tools/`
- `scripts/`
- `scripts/debug/`

这样主代码树更干净。

---

## 8. 我建议你按这个顺序清理

## 第一优先级：先修“会误导使用者”的问题
1. 修 `build_etth1_sft_dataset.py` 默认路径
2. 统一 `same2` / `same3` 默认值
3. README 统一到一条正式主线
4. 补环境说明或依赖文件
5. 明确 expert models 是否仓库内置

---

## 第二优先级：再修“结构耦合”问题
1. `reward.py` 对 `verl` 的硬依赖改成 optional
2. `build_etth1_high_quality_sft.py` 不再 import 私有 helper
3. `time_series_forecast_agent_flow.py` 去 wildcard import、去 dead code
4. `utils.py` 删除 sync legacy path

---

## 第三优先级：最后再考虑“方法升级”
1. 是否放宽固定 3-turn
2. 是否降低中间 teacher reward 权重
3. 是否增加更强的 refine supervision
4. 是否把 RL 默认从 `rollout_n=1` / `temp=0.3` 稍微放开

---

## 9. 最终结论

如果你问我：

### “整个代码，每一个模块的逻辑是否正确？”
答案是：

**时序主线的大部分逻辑现在是正确的，尤其 `build_etth1_sft_dataset.py`、`time_series_forecast_agent_flow.py`、`build_etth1_rl_dataset.py`、`prompts.py` 这几个核心模块，已经形成了一个可解释的单变量 Cast-R1 变体。**

---

### “整个流程是否都是最好的方法和方式？”
答案是：

**不是最好的，但对你当前的 ETTh1-OT 单变量 + 1.7B 目标来说，已经是一个合理且比旧版健康很多的工程方案。**

最不优的地方主要是：
- 仍然过于 rigid 的 3-turn 协议
- teacher routing 中间奖励可能过强
- RL 默认探索太保守
- refine 仍是规则老师，不是更强的 reflective supervision

---

### “代码还存在什么漏洞？”
最关键的不是安全漏洞，而是这些工程漏洞：

1. 仓库不自包含
2. 测试在干净环境不能直接收集
3. 模型目录缺失
4. 旧默认值与新主线不一致
5. 模块耦合偏重
6. 有明显 dead code / generated artifacts 混入源码树

---

### “哪些代码已经没用了可以删除？”
我最明确建议删掉的有：

- `arft/agent_flow/tool_agent_flow.py`
- 所有 `__pycache__`
- `.pytest_cache`
- `artifacts/reports/final_launch_cmd.txt`
- 根目录历史报告文档（移到 `docs/`）
- 非必要的 `artifacts/reports/*`
- `utils.py` 里未使用的同步 `predict_time_series()`
- `time_series_forecast_agent_flow.py` 里注释掉的 `io_records` 旧逻辑
- `utils.py` 里未使用 import / legacy env 支持
- `reward.py` 里重复但不再必要的半死逻辑

---

## 10. 我给你的最简洁的下一步执行建议

如果你只做一轮“最值得的清理”，我建议按这 8 件事做：

1. 删除 `arft/agent_flow/tool_agent_flow.py`
2. 删除 repo 里的所有 cache / debug / artifact 文件
3. 给仓库补 `requirements.txt` 或 `environment.yml`
4. 统一所有 builder / README 的默认路径，彻底清掉 `same2` 残留
5. 修 `build_etth1_sft_dataset.py` 默认输入类型不一致问题
6. 让 `_predict_with_runtime_tools()` 支持 synthetic timestamp anchor
7. 把 `reward.py` 对 `verl` 的依赖改成 optional import
8. 删掉 `utils.py` 里未使用的同步预测旧接口

做完这 8 件事后，这个仓库会比现在**简洁很多、可信很多、也更适合继续调效果**。
