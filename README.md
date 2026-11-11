# Hangar Ingest 维修测量数据处理

将设备观测和维修工单归入可重建的日分区，记录时钟校准、隔离复核、水位和数据发布。接收位置和提交版本分别保存；校准更正会影响分区有效性。

## 防泄漏训练样本

`hangar_ingest.training` 按「飞机、预测时刻、回看窗、预测跨度、标签条件」生成训练样本，并严格区分两类时间：

- **事件时间**（`event_time`、工单 `opened_at/closed_at`）：事情在飞机上实际发生的时刻。
- **提交时间**（观测 `created_at`、审计 `at`、校准 `committed_at`）：系统何时知道这件事。

特征只使用「预测点之前已发生」且「知识截止时已提交」的数据；标签使用单独冻结的后续观察截止。预测跨度未走完的样本标记为右删失（`censored`），绝不会算作负例。事后更正（工单修订、校准重校）保存在只追加的版本表里，不能穿越知识时点；缺少历史证据时样本标记为不可构建（`unbuildable`）并给出机器可读原因，而不是静默读取当前值。

每个设备特征块输出窗口均值、分位数、覆盖率、IQR 异常比例和距最近已知维修时间，并在 `lineage` 中记录全部观测 ID、提交时间、校准修订号与冻结参数、工单修订，可回溯到原始版本。数据划分按飞机分组并按时间前推（展开窗口），清除共享飞机与重叠标签区间，输出带摘要的可审计划分清单。

相关操作以 `training.` 为前缀，可用 `python3 -m hangar_ingest commands` 查看：`training.preview`、`training.save`、`training.get`、`training.samples`、`training.split`、`training.split.verify` 等；多飞机合并运行使用 `training.preview.multi` 与 `training.save.multi`。

## 运行环境

使用 Python 3.11，核心功能依赖标准库和 SQLite。演示和测试只使用本地合成数据，不需要账户、设备或外部运行服务。请从项目根目录执行命令。

## 检查与运行

构建检查验证语法，测试使用临时数据目录，不会连接远端服务。

```sh
python3 -m compileall -q hangar_ingest
python3 -m unittest discover -s tests -v
python3 -m hangar_ingest --db :memory: demo --fixture fixtures/demo.json
```

演示会展示产品的实际数据处理和输出，运行结束自行退出。查看现有接口：

```sh
python3 -m hangar_ingest commands
```

## 目录与数据

hangar_ingest/ 保存领域逻辑和请求入口，tests/ 包含正常行为、边界与恢复测试。例子中的标识和记录为合成资料，不包含真实人员或生产测量。业务目录和本地数据库由调用者指定，应与源码分开保存。

## 干净环境复现

容器定义固定运行时版本。构建时准备环境，运行时无需网络；每次启动会复制只读源码到临时工作目录，测试产生的文件不会写回仓库。

```sh
docker build -t local-product-check .
docker run --rm --network none local-product-check
```
