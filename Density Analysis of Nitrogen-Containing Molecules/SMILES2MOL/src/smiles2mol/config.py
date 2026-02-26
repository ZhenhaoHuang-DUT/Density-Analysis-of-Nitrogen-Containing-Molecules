# src/smiles2mol/config.py
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MolBuildConfig:
    """
    控制 SMILES → RDKit Mol 构象生成策略的配置对象

    所有字段都有明确的“科研语义”，避免隐式规则
    """

    # ========== 解析与诊断 ==========
    require_parse_success: bool = True
    require_no_warnings: bool = True
    skip_diagnosis_if_provided: bool = True

    # ========== 电荷与电子 ==========
    require_total_charge_zero: bool = True
    allow_radicals: bool = False
    allow_isotopes: bool = True

    # ========== 元素组成 ==========
    require_organic_component: bool = True
    allow_metal: bool = False
    require_metal: bool = False   # 与 allow_metal 互斥，由 core 检查

    # ========== 金属配位相关 ==========
    allow_metal_organic_complex: bool = False
    allow_unusual_valence: bool = False

    # ========== 构象生成 ==========
    embed_3d: bool = True
    optimize_geometry: bool = True
    max_embed_attempts: int = 10

    # ========== 失败处理 ==========
    raise_on_reject: bool = False


# ---------- 常用预设（科研中非常重要） ----------

# ---------- 常用预设（科研中非常重要） ----------

DEFAULT_ORGANIC_CONFIG = MolBuildConfig()
# 默认配置：要求总电荷为0，允许同位素，不允许金属

METAL_COMPLEX_CONFIG = MolBuildConfig(
    allow_metal=True,
    allow_metal_organic_complex=True,
    require_no_warnings=False,
    allow_unusual_valence=True,
    require_total_charge_zero=False,  # 金属配合物常有电荷
    require_organic_component=False,  # 可能有无机金属配合物
)

CHARGED_SPECIES_CONFIG = MolBuildConfig(
    require_total_charge_zero=False,
    require_no_warnings=False,
    require_organic_component=False,
    allow_metal=True,  # 添加这一行，允许金属
)

DEBUG_LENIENT_CONFIG = MolBuildConfig(
    require_no_warnings=False,
    allow_metal=True,
    allow_radicals=True,
    allow_unusual_valence=True,
    require_total_charge_zero=False,
    require_organic_component=False,
)