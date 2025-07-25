# MLflow 使用说明

## 安装依赖
```bash
pip install -r requirements.txt
```

## 启动 MLflow UI
在运行训练脚本之前或之后，可以启动 MLflow UI 来查看实验记录：

```bash
mlflow ui
```

然后在浏览器中访问 `http://localhost:5000` 来查看实验结果。

## 主要功能替换

### SwanLab → MLflow 功能对应

1. **实验跟踪**：
   - SwanLab: `SwanLabCallback` 
   - MLflow: `report_to="mlflow"` 在 TrainingArguments 中

2. **参数记录**：
   - SwanLab: `config` 参数
   - MLflow: `mlflow.log_params()`

3. **模型保存**：
   - SwanLab: 自动保存
   - MLflow: `mlflow.pytorch.log_model()`

4. **测试结果记录**：
   - SwanLab: `swanlab.Text()` 和 `swanlab.log()`
   - MLflow: `mlflow.log_table()` 和 `mlflow.log_text()`

## 查看结果

训练完成后，您可以在 MLflow UI 中看到：
- 训练参数
- 训练指标（loss, learning_rate 等）
- 保存的模型
- 测试预测结果表格
- 样例预测文本文件

## 优势

- MLflow 提供更好的模型版本管理
- 支持模型部署功能
- 与主流机器学习工具有更好的集成
- 社区支持更广泛
