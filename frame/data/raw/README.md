# 数据目录说明

当前正式数据集为 HEEW 的区域级汇总数据，不需要复制到本目录。文件位于项目根目录：

```text
dataset/HEEW/cleaned_data/Total_energy.csv
dataset/HEEW/cleaned_data/Total_weather.csv
```

运行审计和基线时分别传入：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\audit_data.py `
  --energy-file dataset\HEEW\cleaned_data\Total_energy.csv `
  --weather-file dataset\HEEW\cleaned_data\Total_weather.csv `
  --output-dir frame\reports\phase1
```

HEEW 负荷文件和气象文件通过 `Year`、`Month`、`Day`、`Hour` 合并。三个预测目标为：

- `Electricity` → `electricity`
- `Cooling` → `cooling`
- `Heat` → `heating`

默认历史外生变量包括温度、露点、湿度、风速、阵风、气压、降水以及日历特征。原始数据不得直接覆盖或修改；审计和清洗结果写入 `frame/reports/phase1`。
