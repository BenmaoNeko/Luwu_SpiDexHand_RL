# rsl_rl env 接入说明

## 目的

这份文档说明 `rsl_rl` 环境层需要提供什么接口、返回什么数据，以及怎么让 runner 正常启动。

如果你要把自己的仿真环境接进 `rsl_rl`，先满足这里列出的最小契约，再处理训练配置和模型配置。

## 最小接口

`rsl_rl` 的 runner 依赖 `VecEnv` 抽象接口。代码入口在 `rsl_rl/env/vec_env.py`。

环境至少要提供这些属性：

- `num_envs`：并行环境个数
- `num_actions`：动作维度
- `max_episode_length`：最大 episode 长度。可以是标量，也可以是每个 env 一项的张量
- `episode_length_buf`：当前 episode 长度缓冲区
- `device`：环境所在设备
- `cfg`：环境配置对象或配置字典

环境至少要实现这两个方法：

### `get_observations()`

返回当前观测，类型是 `TensorDict`。

runner 在构造阶段就会先调用一次这个方法，用它来创建算法、actor 和 critic。也就是说，环境实例一创建好，就得能返回一份结构正确的观测。

### `step(actions)`

输入：

- `actions`，形状是 `(num_envs, num_actions)`

输出：

- `observations`：`TensorDict`
- `rewards`：形状 `(num_envs,)`
- `dones`：形状 `(num_envs,)`
- `extras`：额外信息字典

## 观测需要包含什么

`rsl_rl` 不直接约定“只有一个 obs tensor”。它约定的是“按组组织的 `TensorDict`”。

常见做法是让 `get_observations()` 和 `step()` 返回同一种结构。每个 key 表示一个 observation group。

例如：

- `policy`
- `critic`
- `actor`
- `student`
- `teacher`
- `rnd_state`

runner 和算法真正关心的是训练配置里的 `obs_groups`。这个配置把“模型需要的观测集合”映射到“环境实际返回的 group 列表”。

当前 `rsl_rl` 用到的集合名包括：

- `actor`
- `critic`
- `student`
- `teacher`
- `rnd_state`

如果 `obs_groups` 没配全，`rsl_rl/utils/utils.py` 里的 `resolve_obs_groups()` 会补默认值：

- 先找同名 group
- 找不到时，再退回到 `policy`
- 两者都没有时，直接报错

所以最稳的做法不是依赖默认补全，而是显式提供 `obs_groups`，并保证环境返回的 group 名和配置一致。

## extras 里建议和必须包含什么

`extras` 是 runner 训练循环里透传给算法和 logger 的补充信息。

### 建议包含

#### `time_outs`

类型通常是 `torch.Tensor`，形状和 `dones` 对齐。

这个字段表示 episode 结束是不是因为时间上限。PPO 在 `process_env_step()` 里会用它做 timeout bootstrapping。如果你的环境有固定 episode 长度，最好提供这个字段。

#### `log`

类型是 `dict[str, float | torch.Tensor]`。

这个字段给 logger 用。key 建议用带 `/` 前缀的字符串做命名空间，value 可以是标量，也可以是张量。若 value 是张量，logger 会取均值。

可以往这里放：

- episode reward
- episode length
- success rate
- task-specific metrics

### 不是硬性要求，但要注意

- `PPO.process_env_step()` 只显式消费 `time_outs`
- `OnPolicyRunner.learn()` 会把整个 `extras` 交给 `alg.process_env_step()` 和 `logger.process_env_step()`
- 如果你完全不提供 `extras`，至少也要返回空字典，不能缺位

## runner 正常启动前要满足什么

`OnPolicyRunner` 的启动路径很直接：先拿环境观测，再构建算法，再开始 rollout。代码在 `rsl_rl/runners/on_policy_runner.py`。

要让它正常启动，至少要满足下面这些条件。

### 一，环境对象先可用

在 `OnPolicyRunner.__init__()` 里，runner 会立刻做两件事：

- 读取 `self.env.get_observations()`
- 读取 `self.env.cfg`、`self.env.num_envs`

所以在你创建 runner 之前，环境必须已经：

- 完成底层 simulator 初始化
- 分配好 observation buffer
- 分配好 `episode_length_buf`
- 确定 `device`
- 能在当前状态下直接返回一份合法 `TensorDict`

### 二，训练配置能解析出算法类

`train_cfg["algorithm"]["class_name"]` 会交给 `resolve_callable()` 解析。

这意味着你需要提供：

- 一个可导入的类名
- 或一个完整的模块路径

如果这里写错，runner 在构造阶段就会失败。

### 三，观测结构和算法配置一致

算法构造时会拿环境返回的 `obs` 去解析 actor、critic 等输入。

如果有这些问题，通常会在构造早期报错：

- `obs_groups` 里的 group 名在环境观测里不存在
- actor/critic 需要的 observation set 没法补全
- RND 开启了，但没有 `rnd_state`
- distillation 开启了，但没有 `student` 或 `teacher`

### 四，step 返回的张量形状稳定

在训练循环里，runner 默认假设：

- `actions` 送进环境时是 `(num_envs, num_actions)`
- `rewards`、`dones` 能按 env 维对齐
- `observations` 能直接 `.to(device)`

如果维度不稳定，或者 batch 维和 `num_envs` 对不上，通常会在 rollout 的前几步直接炸掉。

### 五，环境和 runner 的 device 一致

runner 会把动作发到 `env.device`，然后再把 `obs`、`rewards`、`dones` 挪到训练 device。

这意味着至少要保证：

- 环境知道自己的 `device`
- `step()` 能接收发到这个 device 上的 action tensor
- 返回的 `TensorDict` 和张量都支持 `.to(device)`

如果你启用多卡训练，`device` 还要和 `LOCAL_RANK` 对应的 `cuda:{rank}` 一致，否则 `_configure_multi_gpu()` 会直接报错。

## learn 阶段还会额外依赖什么

runner 真正进入 `learn()` 之后，还会继续依赖环境提供稳定的数据流。

### rollout 阶段

每一步都会执行：

1. `alg.act(obs)` 采样动作
2. `env.step(actions)` 推进一步
3. `alg.process_env_step(obs, rewards, dones, extras)` 记录 transition
4. `logger.process_env_step(...)` 记录统计量

所以环境要保证：

- 每一步都返回同构的 observation layout
- `dones` 为真的 env 可以在下一步自动 reset，或者在环境内部处理 episode 切换
- `episode_length_buf` 随环境推进而更新

### return 计算阶段

PPO 会在 rollout 结束后再调用一次 critic 估值，所以最后一帧 `obs` 也必须合法。

如果最后一步返回了半初始化状态、缺字段状态，`compute_returns()` 会失败。

## 推荐的 env 设计

如果你想少踩坑，可以按这个方式设计环境层。

### 统一 observation schema

让 `reset` 后、`get_observations()`、`step()` 返回完全一致的 `TensorDict` 结构。不要第一步一个 key，后面又换另一个 key。

### 显式维护 episode 状态

把这些状态做成标准成员：

- `episode_length_buf`
- reset 标记
- timeout 标记
- 每个 env 的累积回报

这样更容易把 `dones`、`time_outs` 和 `extras["log"]` 对齐。

### 把日志指标集中到 extras["log"]

不要把日志指标散落在 observation 里。观测只放模型输入，统计量放 `extras["log"]`。

### 先跑通最小版本

最开始先只提供：

- 一个 `policy` group
- 基本的 `rewards`
- 基本的 `dones`
- 空的 `extras` 或仅 `time_outs`

等 PPO 跑通，再加 `critic` 独立观测、RND、distillation 或 symmetry。

## 最小检查清单

在你实例化 `OnPolicyRunner` 之前，先检查这些点：

- `env.num_envs` 已设置
- `env.num_actions` 已设置
- `env.max_episode_length` 已设置
- `env.episode_length_buf` 已分配
- `env.device` 已设置
- `env.cfg` 已设置
- `env.get_observations()` 能直接返回 `TensorDict`
- `env.step(actions)` 返回四元组
- observation groups 和 `obs_groups` 配置一致
- `rewards` 和 `dones` 的 batch 维等于 `num_envs`
- 如果有 time limit，`extras["time_outs"]` 已提供
- 如果开 RND，`rnd_state` 已提供
- 如果开 distillation，`student` 和 `teacher` 已提供

## 关键代码位置

- `rsl_rl/env/vec_env.py`：环境抽象接口
- `rsl_rl/runners/on_policy_runner.py`：runner 启动和训练循环
- `rsl_rl/algorithms/ppo.py`：env step 数据在 PPO 里的消费方式
- `rsl_rl/utils/utils.py`：`resolve_obs_groups()` 和 callable 解析逻辑

## 一句话理解

想让 `rsl_rl` 跑起来，环境至少要像一个“能批量 step 的 VecEnv”，稳定返回 `TensorDict` 观测、`rewards`、`dones` 和 `extras`，并让这些内容和训练配置里的 `obs_groups`、算法开关、device 设置完全对齐。
