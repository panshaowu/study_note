# Ray 在 verl 中的用法总结

本文档整理 verl 框架中使用 Ray 的主要模式和用法，便于快速理解 verl 的分布式架构。

---

## 1. 核心概念速查

| 概念 | Ray API | verl 中的用途 |
|------|---------|--------------|
| **Actor** | `@ray.remote` | Worker 进程，封装模型推理/训练逻辑 |
| **ObjectRef** | `ray.put()` / `ray.get()` | 跨进程传递大数据（TensorDict） |
| **PlacementGroup** | `placement_group()` | 控制 Worker 的节点分配 |
| **Scheduling Strategy** | `PlacementGroupSchedulingStrategy` | 指定 Worker 调度到特定 PG |
| **WorkerGroup** | `RayWorkerGroup` | verl 封装的 Worker 管理类 |

---

## 2. Ray Actor (@ray.remote)

### 2.1 基本模式

```python
@ray.remote
class MyWorker:
    def __init__(self, config):
        self.config = config

    def forward(self, data):
        return self.compute(data)
```

**verl 中的使用**: 所有 Worker 类都使用 `@ray.remote` 装饰，使其成为分布式执行单元。

### 2.2 Actor 资源指定

```python
@ray.remote(num_gpus=1)
class GPUWorker:
    pass

@ray.remote(resources={"NPU": 1})
class NPUWorker:
    pass
```

---

## 3. Ray Placement Group

### 3.1 创建 Placement Group

```python
from ray.util.placement_group import placement_group

# 定义 bundle（每个 bundle 包含 CPU + GPU）
bundle = {"CPU": 4, "GPU": 1}
bundles = [bundle.copy() for _ in range(num_workers)]

# 创建 PG
pg = placement_group(bundles=bundles, strategy="STRICT_PACK")
ray.get(pg.ready())  # 等待 PG 就绪
```

### 3.2 调度策略

```python
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

options = {
    "scheduling_strategy": PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0
    )
}

# 创建 Actor 到指定 bundle
worker = MyWorker.options(**options).remote(config)
```

### 3.3 verl 中的 ResourcePool

```python
class RayResourcePool:
    def get_placement_groups(self, strategy="STRICT_PACK", device_name="cuda"):
        bundle = {"CPU": self.max_colocate_count}
        if self.use_gpu:
            bundle[device_name] = 1
        # ...
```

---

## 4. 跨进程数据传输

### 4.1 ray.put / ray.get

```python
# 发送数据到 object store
data_ref = ray.put(tensor_dict)

# 获取数据
result = ray.get(data_ref)
```

### 4.2 parallel_put 并行写入

```python
from verl.utils.ray_utils import parallel_put

# 并行将数据列表写入 object store
data_refs = parallel_put(data_list, max_workers=16)
```

### 4.3 verl DataProto

`DataProto` 是 verl 的标准数据协议，内部使用 Ray ObjectRef 传输：

```python
data = DataProto(batch=tensor_dict, non_tensor_batch=dict)
# 传输时自动序列化为 Ray 对象
```

---

## 5. RayWorkerGroup 管理机制

### 5.1 初始化流程

```
ResourcePoolManager.create_resource_pool()
    ↓ 创建 RayResourcePool (含 PlacementGroups)
    ↓
RayWorkerGroup(resource_pool=pool, ray_cls_with_init=cls)
    ↓ _init_with_resource_pool()
    ↓ 遍历每个 PG，每个 bundle 创建一个 Worker
    ↓
WorkerGroup.execute_all_async(method_name, *args, **kwargs)
```

### 5.2 Worker 创建

```python
def _create_worker(self, rank, pg, local_rank, ...):
    env_vars = {
        "WORLD_SIZE": str(world_size),
        "RANK": str(rank),
        "MASTER_ADDR": self._master_addr,
        "MASTER_PORT": self._master_port,
    }

    # 设置 actor 选项
    ray_cls_with_init.update_options({
        "runtime_env": {"env_vars": env_vars},
        "name": f"{prefix}{name}_{pg_idx}:{local_rank}"
    })

    # 创建 actor
    worker = ray_cls_with_init(
        placement_group=pg,
        placement_group_bundle_idx=local_rank,
        use_gpu=True,
        num_gpus=1/resource_pool.max_colocate_count,
    )
```

### 5.3 方法调用

```python
# 异步调用所有 worker
refs = worker_group.execute_all_async("forward", data)

# 同步等待结果
results = ray.get(refs)

# 只在 rank 0 执行
result = worker_group.execute_rank_zero("sync_weights")
```

---

## 6. FusedWorker (角色融合)

将多个角色（Actor/Critic/Rollout）融合到单个 Ray Actor：

```python
from verl.single_controller.ray.base import create_colocated_worker_cls_fused

# 融合多个 Worker 类
fused_cls = create_colocated_worker_cls_fused({
    "actor": ActorWorkerCls,
    "critic": CriticWorkerCls,
    "ref": RefPolicyWorkerCls,
})

# 创建融合 WorkerGroup
wg = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=fused_cls)

# 分裂出各角色的视图
actor_wg = wg.spawn(prefix_set=["actor"])["actor"]
critic_wg = wg.spawn(prefix_set=["critic"])["critic"]
```

---

## 7. 常用 Ray 工具函数

### 7.1 获取节点 IP

```python
@ray.remote
def get_master_addr_port():
    addr = ray.util.get_node_ip_address()
    # 动态分配端口
    return addr, str(free_port)
```

### 7.2 检查 Actor 存活

```python
from ray.experimental.state.api import get_actor

def _is_worker_alive(worker):
    state_dict = get_actor(worker._actor_id.hex())
    return state_dict.get("state") == "ALIVE"
```

### 7.3 获取加速器 ID

```python
device_ids = ray.get_runtime_context().get_accelerator_ids("GPU")
```

---

## 8. 环境变量传递

verl 通过 `runtime_env` 传递分布式环境变量：

```python
env_vars = {
    "WORLD_SIZE": str(world_size),
    "RANK": str(rank),
    "LOCAL_RANK": str(local_rank),
    "LOCAL_WORLD_SIZE": str(local_world_size),
    "MASTER_ADDR": master_addr,
    "MASTER_PORT": master_port,
}
```

Worker 初始化时从环境变量读取配置：

```python
class Worker:
    def __init__(self):
        self._rank = int(os.environ["RANK"])
        self._world_size = int(os.environ["WORLD_SIZE"])
```

---

## 9. 典型使用场景

### 9.1 创建 WorkerGroup

```python
from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs

# 定义 worker 类
@ray.remote
class MyWorker(Worker):
    def train_step(self, batch):
        return self.model(batch)

# 创建
cls_with_init = RayClassWithInitArgs(cls=MyWorker, config=my_config)
wg = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=cls_with_init)

# 调用
wg.execute_all_async("train_step", batch)
```

### 9.2 管理 LLM 服务器

```python
@ray.remote
class LLMServer:
    def generate(self, prompts):
        return self.engine.generate(prompts)

# 创建多个 replica
servers = {f"server_{i}": LLMServer.remote() for i in range(num_replicas)}

# 负载均衡
load_balancer = GlobalRequestLoadBalancer.remote(servers)
```

---

## 10. 关键文件索引

| 文件 | 描述 |
|------|------|
| `verl/single_controller/ray/base.py` | RayWorkerGroup、RayResourcePool 实现 |
| `verl/single_controller/base/worker.py` | Worker 基类 |
| `verl/protocol.py` | DataProto 数据协议 |
| `verl/utils/ray_utils.py` | Ray 工具函数 |
| `verl/workers/rollout/llm_server.py` | LLM 服务器管理 |

---

## 11. 小结

verl 使用 Ray 作为单控制器架构的基础，通过以下方式实现分布式训练：

1. **Actor 化**: 所有 Worker 都是 Ray Actor，可跨节点调度
2. **Placement Group**: 控制 Worker 的资源分配和节点位置
3. **ObjectRef**: 高效传输 TensorDict 等大数据
4. **RayWorkerGroup 封装**: 提供简洁的 Worker 管理接口
5. **FusedWorker**: 支持将多个角色融合到单个进程减少通信