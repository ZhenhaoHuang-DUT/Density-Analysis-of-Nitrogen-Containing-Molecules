# SMILES2MOL

<p align="center">
  <strong>科研导向的分子构建与分析工具库</strong>
</p>

<p align="center">
  <a href="#特性">特性</a> •
  <a href="#安装">安装</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#核心模块">核心模块</a> •
  <a href="#使用示例">使用示例</a> •
  <a href="#高级功能">高级功能</a> •
  <a href="#常见问题">常见问题</a>
</p>

## 📋 概述

**SMILES2MOL** 是一个针对计算化学和药物发现研究设计的 Python 工具库，专注于提供严格、可配置的 SMILES 到 RDKit Mol 对象的转换。该库强调**科研语义清晰性**和**决策透明性**，确保每个分子构建步骤都有明确的科研理由。

## ✨ 特性

- ✅ **严格的分子诊断**：全面的预检查机制，包括金属检测、有机组分分析、电荷验证等
- ✅ **可配置的构建策略**：通过 `MolBuildConfig` 类精细控制分子构建的每个环节
- ✅ **科研语义明确**：所有配置参数都有明确的科研意义，避免隐式规则
- ✅ **决策透明**：返回详细的报告，解释每个接受或拒绝决策的原因
- ✅ **异常处理完善**：优雅的错误处理机制，便于调试和自动化流程
- ✅ **多种预设配置**：针对不同科研场景的预设配置

## 📦 安装

### 依赖要求
- Python 3.7+
- RDKit (2020.03+)

### 从源代码安装
```bash
git clone https://github.com/yourusername/smiles2mol.git
cd smiles2mol
pip install -e .
```

### 通过 pip 安装
```bash
pip install smiles2mol
```

## 🚀 快速开始

```python
from smiles2mol import build_mol_from_smiles, DEFAULT_ORGANIC_CONFIG

# 构建一个有机分子
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # 阿司匹林
mol, report = build_mol_from_smiles(smiles, DEFAULT_ORGANIC_CONFIG)

if mol is not None:
    print(f"成功构建分子，原子数: {mol.GetNumAtoms()}")
    print(f"构象数: {mol.GetNumConformers()}")
else:
    print(f"构建失败，原因: {report['reasons']}")
```

## 🏗️ 核心模块

### 1. 配置系统 (`config.py`)

#### `MolBuildConfig` 类

控制 SMILES → RDKit Mol 构象生成的完整配置，所有字段都有明确的科研语义：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `require_parse_success` | bool | True | 是否要求 SMILES 解析成功 |
| `require_no_warnings` | bool | True | 是否不允许任何警告 |
| `skip_diagnosis_if_provided` | bool | True | 如果提供预检查结果，是否跳过诊断 |
| `require_total_charge_zero` | bool | True | 是否要求总形式电荷为 0 |
| `allow_radicals` | bool | False | 是否允许自由基（未配对电子） |
| `allow_isotopes` | bool | True | 是否允许同位素标记 |
| `require_organic_component` | bool | True | 是否要求含有有机组分（含碳） |
| `allow_metal` | bool | False | 是否允许金属元素 |
| `require_metal` | bool | False | 是否必须含有金属元素 |
| `allow_metal_organic_complex` | bool | False | 是否允许金属-有机配合物 |
| `allow_unusual_valence` | bool | False | 是否允许异常价态 |
| `embed_3d` | bool | True | 是否生成 3D 构象 |
| `optimize_geometry` | bool | True | 是否进行几何优化 |
| `max_embed_attempts` | int | 10 | 最大构象嵌入尝试次数 |
| `raise_on_reject` | bool | False | 拒绝时是否抛出异常 |

#### 预设配置

| 配置名称 | 适用场景 | 关键特性 |
|----------|----------|----------|
| `DEFAULT_ORGANIC_CONFIG` | 标准有机分子 | 要求无金属、总电荷为零、含碳 |
| `METAL_COMPLEX_CONFIG` | 金属配合物 | 允许金属、非零电荷、异常价态 |
| `CHARGED_SPECIES_CONFIG` | 带电物种 | 允许非零电荷、金属、无有机组分 |
| `DEBUG_LENIENT_CONFIG` | 调试宽松模式 | 允许几乎所有特征，仅用于调试 |

### 2. 预检查系统 (`utils.py`)

#### `precheck_smiles()` 函数

执行严格的 SMILES 预检查，返回结构化结果：

```python
from smiles2mol.utils import precheck_smiles

result = precheck_smiles("CC(=O)Oc1ccccc1C(=O)O", allow_sanitize_fail=False)

print(f"解析成功: {result.success}")
print(f"总电荷: {result.total_formal_charge}")
print(f"包含金属: {result.contains_metal}")
print(f"包含有机组分: {result.contains_organic_component}")
print(f"警告: {result.warnings}")
```

**返回的 `PrecheckResult` 对象包含**：
- `smiles`：输入的 SMILES 字符串
- `mol`：解析后的 RDKit Mol 对象（若成功）
- `success`：解析是否成功
- `errors`/`warnings`：错误和警告信息
- `fragment_counter`：片段计数
- `irreducible_counter`：不可约片段计数
- `total_formal_charge`：总形式电荷
- `contains_metal`：是否包含金属元素
- `contains_organic_component`：是否包含有机组分（基于碳）
- `has_radical`：是否有自由基
- `has_isotopes`：是否有同位素标记
- `has_unusual_valence`：是否有异常价态
- `action_recommendation`：行动建议

#### 元素分类定义

- **金属元素**：包括碱金属、碱土金属、过渡金属、镧系/锕系元素等
- **有机相关元素**：H, C, N, O, F, P, S, Cl, Br, I

### 3. 核心构建函数 (`core.py`)

#### `build_mol_from_smiles()` 函数

主入口函数，协调整个分子构建流程：

```python
def build_mol_from_smiles(
    smiles: str,
    config: MolBuildConfig = DEFAULT_ORGANIC_CONFIG,
    diagnosis: Optional[PrecheckResult] = None,
    diagnostic_only: bool = False,
) -> Tuple[Optional[Chem.Mol], Dict[str, Any]]:
```

**参数说明**：
- `smiles`：输入的 SMILES 字符串
- `config`：分子构建配置（默认为 `DEFAULT_ORGANIC_CONFIG`）
- `diagnosis`：预计算的诊断结果（可选，用于复用诊断）
- `diagnostic_only`：是否仅进行诊断，不构建分子

**返回值**：
- `mol`：成功构建的 RDKit Mol 对象，失败时为 `None`
- `report`：包含完整构建报告的字典

**构建报告结构**：
```python
report = {
    "precheck": PrecheckResult,     # 预检查结果对象
    "decision": str,               # 决策结果: 'accepted'/'rejected'/'error'/'diagnostic_only'
    "reasons": List[str],          # 决策原因列表
    "opt_results": List[Dict],     # 优化结果（如果执行了优化）
    "exception": Dict[str, Any],   # 异常信息（如果有）
}
```

**决策结果说明**：
- `accepted`: 分子被接受并成功构建
- `rejected`: 分子被拒绝（不符合配置要求）
- `error`: 构建过程中发生异常
- `diagnostic_only`: 仅进行诊断，未构建分子

## 📝 使用示例

### 示例 1：基本有机分子构建

```python
from smiles2mol import build_mol_from_smiles, DEFAULT_ORGANIC_CONFIG

smiles = "CC(=O)Oc1ccccc1C(=O)O"  # 阿司匹林
mol, report = build_mol_from_smiles(smiles, DEFAULT_ORGANIC_CONFIG)

if report["decision"] == "accepted":
    print(f"成功构建: {mol.GetNumAtoms()} 原子")
    print(f"构象能量: {report['opt_results'][0]['energy']:.2f} kcal/mol")
else:
    print(f"构建失败: {report['reasons']}")
```

### 示例 2：金属配合物分析

```python
from smiles2mol import build_mol_from_smiles, METAL_COMPLEX_CONFIG
from smiles2mol.utils import precheck_smiles

# 铁配合物
smiles = "[Fe+2]C(C)C"

# 首先进行预检查
pre = precheck_smiles(smiles, allow_sanitize_fail=True)
print(f"包含金属: {pre.contains_metal}")
print(f"金属元素: {pre.metal_element_counts}")
print(f"有机组分: {pre.contains_organic_component}")
print(f"建议: {pre.action_recommendation}")

# 使用金属配合物配置构建
mol, report = build_mol_from_smiles(smiles, METAL_COMPLEX_CONFIG)
if mol:
    print(f"成功构建金属配合物，总电荷: {pre.total_formal_charge}")
```

### 示例 3：带电物种处理

```python
from smiles2mol import build_mol_from_smiles, CHARGED_SPECIES_CONFIG

# 氯化铵（带电荷盐）
smiles = "[NH4+].[Cl-]"

# 使用默认配置会被拒绝
mol1, report1 = build_mol_from_smiles(smiles)
print(f"默认配置: {report1['decision']} - {report1['reasons']}")

# 使用带电物种配置会被接受
mol2, report2 = build_mol_from_smiles(smiles, CHARGED_SPECIES_CONFIG)
if mol2:
    print(f"带电配置: 成功构建，总原子数 {mol2.GetNumAtoms()}")
```

### 示例 4：诊断模式与自定义配置

```python
from smiles2mol import build_mol_from_smiles, MolBuildConfig

# 自定义配置：只进行诊断，不生成3D构象
custom_config = MolBuildConfig(
    require_parse_success=True,
    require_no_warnings=False,  # 允许警告
    require_total_charge_zero=True,
    allow_radicals=False,
    embed_3d=False,  # 不生成3D构象
    optimize_geometry=False,
    require_organic_component=False,  # 允许无机分子
)

# 诊断模式
smiles = "[Na+].[Cl-]"
mol, report = build_mol_from_smiles(smiles, custom_config, diagnostic_only=True)

print(f"预检查成功: {report['precheck'].success}")
print(f"总电荷: {report['precheck'].total_formal_charge}")
print(f"片段: {report['precheck'].fragment_counter}")
print(f"动作建议: {report['precheck'].action_recommendation}")
```

### 示例 5：批量处理与错误处理

```python
from smiles2mol import build_mol_from_smiles, DEFAULT_ORGANIC_CONFIG

smiles_list = [
    "CC(=O)Oc1ccccc1C(=O)O",  # 阿司匹林
    "[Fe+2]C(C)C",            # 铁配合物
    "invalid_smiles",         # 无效SMILES
    "[NH4+].[Cl-]",           # 氯化铵
]

results = []
for smiles in smiles_list:
    mol, report = build_mol_from_smiles(smiles, DEFAULT_ORGANIC_CONFIG)
    
    result = {
        "smiles": smiles,
        "decision": report["decision"],
        "success": mol is not None,
    }
    
    if report["decision"] == "rejected":
        result["reasons"] = report["reasons"]
    elif report["decision"] == "error":
        result["error"] = report["exception"]["message"]
    
    results.append(result)

# 统计结果
accepted = sum(1 for r in results if r["success"])
print(f"接受: {accepted}/{len(results)}")
print(f"拒绝: {len(results)-accepted}/{len(results)}")
```

### 示例 6：高级用法 - 复用诊断结果

```python
from smiles2mol import build_mol_from_smiles, DEFAULT_ORGANIC_CONFIG
from smiles2mol.utils import precheck_smiles

# 复杂分子，先进行预检查
smiles = "C1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)O"
pre = precheck_smiles(smiles, allow_sanitize_fail=True)

print("=== 预检查结果 ===")
print(f"警告数量: {len(pre.warnings)}")
print(f"包含有机组分: {pre.contains_organic_component}")
print(f"总电荷: {pre.total_formal_charge}")

# 使用预检查结果构建分子（避免重复解析）
mol1, report1 = build_mol_from_smiles(smiles, diagnosis=pre)
print(f"构建结果: {report1['decision']}")

# 尝试不同配置
from smiles2mol import DEBUG_LENIENT_CONFIG
mol2, report2 = build_mol_from_smiles(smiles, DEBUG_LENIENT_CONFIG, diagnosis=pre)
print(f"宽松配置构建结果: {report2['decision']}")
```

## 🔧 高级功能

### 1. 片段分析与不可约组成

```python
from smiles2mol.utils import precheck_smiles

smiles = "[Na+].[Cl-].CC(=O)O"
pre = precheck_smiles(smiles)

print(f"原始片段: {pre.fragment_counter}")
print(f"不可约组成: {pre.irreducible_counter}")
print(f"各片段电荷: {pre.fragment_charges}")
```

### 2. 价态异常检测

```python
from smiles2mol.utils import precheck_smiles

# 丙二烯（可能触发价态警告）
smiles = "C=C=[CH2]"
pre = precheck_smiles(smiles)

if pre.has_unusual_valence:
    print(f"检测到异常价态: {pre.warnings}")
```

### 3. 金属-有机配位分析

```python
from smiles2mol.utils import precheck_smiles

smiles = "[Fe+2]C1=CC=CC=C1"  # 铁苯配合物
pre = precheck_smiles(smiles)

print(f"包含金属: {pre.contains_metal}")
print(f"包含有机组分: {pre.contains_organic_component}")
print(f"动作建议: {pre.action_recommendation}")
```

## ❓ 常见问题与解决方案

### Q1: 为什么我的金属配合物被拒绝？
**A**: 默认配置 (`DEFAULT_ORGANIC_CONFIG`) 不允许金属。请使用 `METAL_COMPLEX_CONFIG` 或创建自定义配置，设置 `allow_metal=True`。

### Q2: 如何允许带电荷的分子？
**A**: 默认配置要求总电荷为零。使用 `CHARGED_SPECIES_CONFIG` 或设置 `require_total_charge_zero=False`。

### Q3: 如何跳过3D构象生成？
**A**: 设置 `embed_3d=False`。这对于纯盐或只需要2D信息的场景很有用。

### Q4: 如何处理RDKit版本差异？
**A**: 库已处理 ETKDGv3 参数兼容性。如果遇到问题，会自动回退到 ETKDGv2。

### Q5: 如何获取详细的失败原因？
**A**: 检查返回的 `report['reasons']` 列表，其中包含所有拒绝原因。

### Q6: 如何处理无效的 SMILES？
**A**: 库会自动检测并拒绝无效的 SMILES，错误信息会包含在 `report['reasons']` 中。

### Q7: 什么是不可约组成？
**A**: 通过计算所有片段数量的最大公约数（GCD）约化得到的组成，用于识别分子最基本的化学计量比。

### Q8: 何时使用诊断模式？
**A**: 当您只想了解分子的性质而不需要3D构象时，使用 `diagnostic_only=True`。

### Q9: 如何复用预检查结果？
**A**: 使用 `precheck_smiles()` 获取诊断结果，然后传递给 `build_mol_from_smiles()` 的 `diagnosis` 参数，避免重复解析。

### Q10: 异常价态检测可靠吗？
**A**: 检测主要针对主族/有机元素。过渡金属的价态变化较大，默认不纳入检测以避免误报。可通过 `check_only_main_group=False` 调整。

## 📊 内部流程说明

### 构建流程的三个阶段：

1. **预检查阶段**：
   - 执行 `precheck_smiles()` 解析 SMILES
   - 收集分子结构、电荷、元素组成等信息
   - 检测金属、自由基、同位素、异常价态等

2. **决策阶段**：
   - 根据配置评估预检查结果
   - 使用 `_evaluate_precheck()` 函数判断是否通过
   - 生成拒绝原因或继续下一步

3. **构象生成阶段**：
   - 如果需要，添加氢原子
   - 使用 ETKDGv3 算法生成 3D 构象
   - 可选：使用 MMFF/UFF 进行几何优化

### 构象生成细节：
- 使用 ETKDGv3 算法，随机种子固定为 42 以保证可重复性
- 支持手性保持 (`enforceChirality=True`)
- 优化失败时自动回退到 UFF 力场
- 可配置最大尝试次数 (`max_embed_attempts`)

## 🛠️ 自定义配置指南

### 创建自定义配置：

```python
from smiles2mol import MolBuildConfig

# 用于药物筛选的配置
DRUG_SCREENING_CONFIG = MolBuildConfig(
    require_parse_success=True,
    require_no_warnings=True,
    require_total_charge_zero=True,  # 药物分子通常为中性
    allow_radicals=False,           # 药物通常不是自由基
    allow_isotopes=True,            # 允许同位素标记
    require_organic_component=True, # 必须有碳
    allow_metal=False,              # 排除金属配合物
    embed_3d=True,
    optimize_geometry=True,
    max_embed_attempts=20,          # 更多尝试次数
)

# 用于无机化学的配置
INORGANIC_CONFIG = MolBuildConfig(
    require_parse_success=True,
    require_no_warnings=False,      # 无机配合物常有警告
    require_total_charge_zero=False, # 允许带电
    allow_radicals=True,            # 允许自由基
    allow_isotopes=True,
    require_organic_component=False, # 允许纯无机物
    allow_metal=True,               # 允许金属
    allow_metal_organic_complex=True,
    allow_unusual_valence=True,     # 无机化学常有异常价态
    embed_3d=True,
    optimize_geometry=True,
)
```

## 📈 性能建议

1. **批量处理**：对于大量分子，考虑先收集所有诊断结果，再根据需求构建
2. **配置复用**：为常见场景创建预设配置，避免重复配置
3. **诊断复用**：如果需要对同一分子尝试不同配置，复用预检查结果
4. **适当降级**：对于简单分子，可设置 `embed_3d=False` 和 `optimize_geometry=False` 加速

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request。主要贡献方向：

### 新功能建议：
- 新的预检查功能（如芳香性检测、官能团识别）
- 更多的预设配置（针对特定研究领域）
- 性能优化（并行处理、缓存机制）
- 扩展支持（其他分子格式、数据库集成）

### 代码规范：
- 遵循现有代码结构
- 添加充分的注释
- 包含单元测试
- 更新文档

### 报告问题：
- 提供重现步骤
- 包含输入 SMILES 和配置
- 说明期望行为与实际行为

## 📄 许可证

MIT License

```text
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📞 联系方式

- **GitHub Issues**: [项目 Issues 页面](https://github.com/yourusername/smiles2mol/issues)
- **Email**: your.email@example.com
- **文档**: [项目 Wiki](https://github.com/yourusername/smiles2mol/wiki)

## 📚 参考文献与相关项目

### 相关工具：
- **RDKit**: 化学信息学基础工具库
- **OpenBabel**: 化学文件格式转换
- **MoleculeNet**: 分子机器学习基准数据集
- **DeepChem**: 深度学习化学工具包

### 推荐阅读：
- RDKit 文档: https://www.rdkit.org/docs/
- SMILES 语法: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
- 分子构象生成: https://pubs.acs.org/doi/10.1021/acs.jcim.5b00654

## 🎯 应用场景

### 药物发现：
- 化合物库预处理
- 虚拟筛选前处理
- ADMET 性质预测数据准备

### 计算化学：
- 量子化学计算输入准备
- 分子动力学模拟初始结构
- 构象搜索基准测试

### 材料科学：
- 金属有机框架 (MOF) 构建
- 配合物结构分析
- 晶体结构预处理

### 教育研究：
- 化学信息学教学
- 分子可视化
- 自动化的分子分析

---

<p align="center">
  <em>✨ SMILES2MOL - 让分子构建更严谨、更透明 ✨</em>
</p>

<p align="center">
  <strong>重要提示</strong>: 本工具库设计用于科研目的，建议在使用前充分理解分子化学原理。<br>
  对于重要的科研决策，建议结合人工验证和实验确认。
</p>

<p align="center">
  <a href="#smiles2mol">返回顶部</a>
</p>

---

**版本信息**: v0.1.0 | **最后更新**: 2024年1月 | **作者**: Your Name