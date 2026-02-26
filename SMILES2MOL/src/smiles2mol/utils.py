"""
my_rdkit_lib.utils (最终版)

功能：
- 严格但仅做一次 RDKit Mol 解析（precheck_smiles）。
- 基于 Mol 执行片段分析、不可约组成（GCD 约化）、电荷统计、金属/有机判定、同位素/自由基/异常价态检测、拓扑配位指示等。
- 返回结构化 PrecheckResult（包含 errors, warnings, action_recommendation），便于 core 决策。
- 禁止在本模块中执行 AddHs/Embed/优化 等 3D 相关操作。

使用：
    from my_rdkit_lib.utils import precheck_smiles, PrecheckResult
    res = precheck_smiles("CC(=O)Oc1ccccc1C(=O)O")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter
import math

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import RDLogger

# -------------------------
# 抑制 RDKit 控制台日志（避免 Parse Error 散落 stderr）
# 如需调试 RDKit 输出，请注释掉下一行
RDLogger.DisableLog('rdApp.*')

# -------------------------
# 自定义异常类型
# -------------------------
class UtilsError(Exception):
    """基类错误"""
    pass

class InvalidSmilesError(UtilsError):
    """SMILES 无法解析为 Mol"""
    pass

class SanitizeMolError(UtilsError):
    """RDKit SanitizeMol 失败"""
    pass

# -------------------------
# 元素类别（基于原子序数）
# -------------------------
METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51,
    55, 56,
    *range(57, 72),  # lanthanides
    *range(72, 81),
    *range(81, 84),
    *range(87, 104),
}

# 认定为“有机相关元素”的原子序号（H, C, N, O, F, P, S, Cl, Br, I）
ORGANIC_ATOMIC_NUMBERS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}


# -------------------------
# dataclass 返回结构
# -------------------------
@dataclass
class FragmentInfo:
    canonical_smiles: str
    count: int
    formal_charge: int
    atom_counts: Dict[str, int] = field(default_factory=dict)
    is_single_atom: bool = False

@dataclass
class PrecheckResult:
    smiles: str
    mol: Optional[Chem.Mol]  # 成功解析时为 RDKit Mol，否则为 None
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # 片段/组成信息（基于 Mol 严格计算）
    fragment_counter: Dict[str, int] = field(default_factory=dict)
    irreducible_counter: Dict[str, int] = field(default_factory=dict)
    fragments: List[FragmentInfo] = field(default_factory=list)

    # 电荷/基态信息
    total_formal_charge: Optional[int] = None
    fragment_charges: List[int] = field(default_factory=list)

    # 标志位
    contains_metal: Optional[bool] = None
    metal_element_counts: Dict[str, int] = field(default_factory=dict)
    contains_organic_component: Optional[bool] = None
    organic_element_counts: Dict[str, int] = field(default_factory=dict)
    has_radical: Optional[bool] = None
    has_isotopes: Optional[bool] = None
    has_unusual_valence: Optional[bool] = None

    # 建议（用于 pipeline 决策）
    action_recommendation: Optional[str] = None


# -------------------------
# 内部辅助函数（接受 RDKit Mol）
# -------------------------
def _atom_is_metal(atom: rdchem.Atom) -> bool:
    return atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS

def _atom_is_organic(atom: rdchem.Atom) -> bool:
    return atom.GetAtomicNum() in ORGANIC_ATOMIC_NUMBERS

def total_formal_charge(mol: Chem.Mol) -> int:
    return sum(int(a.GetFormalCharge()) for a in mol.GetAtoms())

def fragment_mols(mol: Chem.Mol) -> List[Chem.Mol]:
    return list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))

def canonical_smiles_for_mol(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def fragment_composition(mol: Chem.Mol) -> Tuple[Dict[str,int], List[FragmentInfo]]:
    """返回 canonical smiles -> count 的计数，以及 FragmentInfo 列表（atom_counts/formal_charge/is_single）"""
    frags = fragment_mols(mol)
    canonical_list = []
    frag_infos: List[FragmentInfo] = []
    for f in frags:
        cs = canonical_smiles_for_mol(f)
        canonical_list.append(cs)
        fc = sum(int(a.GetFormalCharge()) for a in f.GetAtoms())
        atom_counts = Counter([a.GetSymbol() for a in f.GetAtoms()])
        is_single = (len(f.GetAtoms()) == 1)
        fi = FragmentInfo(canonical_smiles=cs, count=1, formal_charge=fc,
                          atom_counts=dict(atom_counts), is_single_atom=is_single)
        frag_infos.append(fi)
    counter = Counter(canonical_list)
    # 合并 frag_infos 的 count 字段按 counter
    merged_infos: Dict[str, FragmentInfo] = {}
    for fi in frag_infos:
        key = fi.canonical_smiles
        if key not in merged_infos:
            merged_infos[key] = FragmentInfo(
                canonical_smiles=fi.canonical_smiles,
                count=counter[key],
                formal_charge=fi.formal_charge,
                atom_counts=fi.atom_counts,
                is_single_atom=fi.is_single_atom
            )
    return dict(counter), list(merged_infos.values())

def gcd_reduce_counter(counter: Dict[str,int]) -> Dict[str,int]:
    if not counter:
        return {}
    vals = list(counter.values())
    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(g, v)
    if g > 1:
        return {k: v // g for k, v in counter.items()}
    return dict(counter)

def metal_element_counts(mol: Chem.Mol) -> Dict[str,int]:
    counts = Counter()
    for a in mol.GetAtoms():
        if _atom_is_metal(a):
            counts[a.GetSymbol()] += 1
    return dict(counts)

def organic_element_counts(mol: Chem.Mol) -> Dict[str,int]:
    counts = Counter()
    for a in mol.GetAtoms():
        if a.GetAtomicNum() in ORGANIC_ATOMIC_NUMBERS:
            counts[a.GetSymbol()] += 1
    return dict(counts)

def contains_organic_component_by_carbon(mol: Chem.Mol) -> Tuple[bool, Dict[str,int]]:
    """
    更保守的“是否含有机组分”判定：
      - 若 mol 中存在 carbon (C) 则认为含有机组分
    返回 (has_carbon, organic_element_counts)
    """
    counts = organic_element_counts(mol)
    return (counts.get('C', 0) > 0), counts

def detect_radicals_detailed(mol: Chem.Mol) -> Dict[str, object]:
    """
    返回字典：
      {
        'total_radical_electrons': int,
        'radical_on_metals': bool,
        'radical_on_nonmetals': bool,
        'radical_atom_summary': [(symbol, idx, n_rad_elec), ...]
      }
    """
    total = 0
    radical_meta = False
    radical_nonmeta = False
    summary = []
    for a in mol.GetAtoms():
        n = int(a.GetNumRadicalElectrons())
        if n > 0:
            total += n
            summary.append((a.GetSymbol(), a.GetIdx(), n))
            if _atom_is_metal(a):
                radical_meta = True
            else:
                radical_nonmeta = True
    return {
        'total_radical_electrons': total,
        'radical_on_metals': radical_meta,
        'radical_on_nonmetals': radical_nonmeta,
        'radical_atom_summary': summary
    }

def detect_isotopes(mol: Chem.Mol) -> bool:
    for a in mol.GetAtoms():
        if a.GetIsotope() and a.GetIsotope() != 0:
            return True
    return False

def detect_unusual_valence(
    mol: Chem.Mol,
    tolerance: int = 2,
    check_only_main_group: bool = True
) -> Tuple[bool, List[str]]:
    """
    更稳健的价态异常检测：
      - 只在能获取到正向 default_val ( > 0 ) 时进行比较
      - 可选择只检查主族/有机相关元素（避免对过渡金属误报）
    参数:
      - tolerance: 若 tot_val > default_val + tolerance 则认为异常（保守判断）
      - check_only_main_group: 若 True，仅对非金属 / 有机相关元素进行检测（推荐 True）
    返回:
      - (has_unusual_valence, messages)
    说明:
      - 许多过渡金属与配位化学会超出默认价态，因此一般不把它们纳入此类警告中
    """
    messages: List[str] = []
    pt = Chem.GetPeriodicTable()
    unusual = False

    # 定义要检查的原子（默认：有机相关/主族元素）
    if check_only_main_group:
        # 推荐检查的原子序数集合（H C N O P S F Cl Br I 等）
        check_atomic_nums = ORGANIC_ATOMIC_NUMBERS.copy()
    else:
        # 若 user 想更广泛地检查，可以设置为 None 或空集合以表示检查所有能取到 default_val 的元素
        check_atomic_nums = None

    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        # 若限定检查范围并且该原子不在集合内，则跳过
        if check_atomic_nums is not None and an not in check_atomic_nums:
            continue

        try:
            totval = atom.GetTotalValence()
        except Exception:
            # 若无法获取则跳过
            continue

        # 获取 default valence
        try:
            default_val = pt.GetDefaultValence(atom.GetSymbol())
        except Exception:
            default_val = None

        # 只有 default_val 为正整数时才比较；否则跳过（防止 -1 或 0 导致误判）
        if default_val is None or (isinstance(default_val, int) and default_val <= 0):
            # 无定义的 default_val，跳过比较
            continue

        # 进行保守判断
        if totval > default_val + tolerance:
            unusual = True
            messages.append(
                f"原子 {atom.GetSymbol()} (idx={atom.GetIdx()}) 的总价态 {totval} 明显大于常见值 {default_val}（容忍度 {tolerance}）"
            )

    return unusual, messages

def metal_organic_topology_check(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """
    基于拓扑判断是否存在 metal--organic 配位（更强的证据）：
      - 找到金属原子
      - 检查金属是否直接与任何 '有机元素' 原子相连（C/N/O/P/S 等）
    返回 (has_coordination_like_bonds, messages)
    """
    msgs = []
    has_coord = False
    for a in mol.GetAtoms():
        if _atom_is_metal(a):
            neighs = a.GetNeighbors()
            for n in neighs:
                if _atom_is_organic(n):
                    has_coord = True
                    msgs.append(f"金属 {a.GetSymbol()} (idx={a.GetIdx()}) 与有机原子 {n.GetSymbol()} (idx={n.GetIdx()}) 直接相连")
    return has_coord, msgs


# -------------------------
# 更稳健的 Mol 解析：MolFromSmiles(sanitize=False) + SanitizeMol 捕获异常
# -------------------------
def _mol_from_smiles_strict(smiles: str, res: PrecheckResult) -> Optional[Chem.Mol]:
    """
    尝试解析 SMILES（不立刻 sanitize），再手动 sanitize 捕获错误信息并写入 res.errors。
    返回 mol 或 None（并在 res.errors 中附加信息）。
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    except Exception as e:
        res.errors.append(f"RDKit MolFromSmiles 抛出异常: {e}")
        return None

    if mol is None:
        res.errors.append("RDKit 无法解析 SMILES（MolFromSmiles 返回 None）")
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        res.errors.append(f"SanitizeMol 失败: {e}")
        return None

    return mol


# -------------------------
# 主入口：一次解析 + 全面检测
# -------------------------
def precheck_smiles(smiles: str, allow_sanitize_fail: bool = False) -> PrecheckResult:
    """
    严格的 precheck：只调用一次 MolFromSmiles(sanitize=True)（或在 sanitize 失败时返回错误）
    返回 PrecheckResult（包含 mol, fragments, flags, warnings, errors, recommendation）
    参数:
      - smiles: 输入 SMILES 字符串
      - allow_sanitize_fail: 若 True，则在 sanitize 失败时仍返回 mol=None 与错误信息（不抛异常）
    """
    res = PrecheckResult(smiles=smiles, mol=None, success=False)
    if not smiles or not isinstance(smiles, str):
        res.errors.append("输入 SMILES 为空或类型不正确")
        res.action_recommendation = "reject: empty_or_invalid_input"
        return res

    # 解析并 sanitize（strict）
    mol = _mol_from_smiles_strict(smiles, res)
    if mol is None:
        if allow_sanitize_fail:
            res.action_recommendation = "reject: sanitize_or_parse_failed"
            return res
        else:
            # 抛出异常并带上具体错误
            raise SanitizeMolError("; ".join(res.errors))

    # 成功解析
    res.mol = mol
    res.success = True

    # fragments / composition
    try:
        counter, fragment_infos = fragment_composition(mol)
        res.fragment_counter = counter
        res.fragments = fragment_infos
        res.irreducible_counter = gcd_reduce_counter(counter)

        # ===== 新增：不可约 SMILES + 不可约 mol =====
        if res.irreducible_counter:
            irreducible_smiles_list = []
            for smi in sorted(res.irreducible_counter):
                irreducible_smiles_list.extend(
                    [smi] * res.irreducible_counter[smi]
                )
            irreducible_smiles = ".".join(irreducible_smiles_list)

            # 覆盖 smiles（符号层）
            res.smiles = irreducible_smiles

            # 覆盖 mol（结构层）
            try:
                irreducible_mol = Chem.MolFromSmiles(irreducible_smiles, sanitize=True)
                if irreducible_mol is None:
                    res.warnings.append(
                        f"不可约 SMILES 无法重新生成 mol: {irreducible_smiles}"
                    )
                else:
                    res.mol = irreducible_mol
            except Exception as e:
                res.warnings.append(
                    f"不可约 mol 构建异常: {e}"
                )
    except Exception as e:
        res.warnings.append(f"片段分析异常: {e}")

    # charge
    try:
        res.total_formal_charge = total_formal_charge(mol)
        frag_mols = fragment_mols(mol)
        res.fragment_charges = [sum(int(a.GetFormalCharge()) for a in f.GetAtoms()) for f in frag_mols]
    except Exception as e:
        res.warnings.append(f"电荷统计异常: {e}")

    # metal/organic counts (organic 判定以碳为主)
    try:
        me_counts = metal_element_counts(mol)
        org_counts = organic_element_counts(mol)  # full counts
        has_org_by_c, _ = contains_organic_component_by_carbon(mol)
        res.metal_element_counts = me_counts
        res.organic_element_counts = org_counts
        res.contains_metal = len(me_counts) > 0
        res.contains_organic_component = has_org_by_c
    except Exception as e:
        res.warnings.append(f"元素计数异常: {e}")

    # radicals detailed
    try:
        rad_info = detect_radicals_detailed(mol)
        res.has_radical = rad_info['total_radical_electrons'] > 0
        if rad_info['total_radical_electrons'] > 0:
            res.warnings.append(
                f"检测到未配对电子: total={rad_info['total_radical_electrons']}, "
                f"on_metals={rad_info['radical_on_metals']}, on_nonmetals={rad_info['radical_on_nonmetals']}. "
                "请人工确认是否为真实自由基或因金属离子表示导致的假阳性。"
            )
    except Exception as e:
        res.warnings.append(f"自由基检测异常: {e}")

    # isotopes / unusual valence
    try:
        res.has_isotopes = detect_isotopes(mol)
    except Exception as e:
        res.warnings.append(f"同位素检测异常: {e}")

    try:
        unusual, msgs = detect_unusual_valence(mol)
        res.has_unusual_valence = unusual
        if msgs:
            res.warnings.extend(msgs)
    except Exception as e:
        res.warnings.append(f"价态异常检测异常: {e}")

    # topology: metal-organic adjacency
    try:
        coord_like, coord_msgs = metal_organic_topology_check(mol)
        if coord_msgs:
            res.warnings.extend(coord_msgs)
        if res.contains_metal:
            if coord_like:
                res.action_recommendation = "proceed_with_caution: likely_metal_organic_complex"
            else:
                res.action_recommendation = "inspect: contains_metal_but_no_direct_coordination"
        else:
            if res.has_unusual_valence:
                res.action_recommendation = "inspect: organic_with_unusual_valence"
            else:
                res.action_recommendation = "proceed: likely_organic_molecule"
    except Exception as e:
        res.warnings.append(f"配位拓扑检测异常: {e}")
        if res.action_recommendation is None:
            res.action_recommendation = "inspect: topology_check_failed"

    # pure single-atom salt skip suggestion
    try:
        if res.fragments:
            all_single = all(fi.is_single_atom for fi in res.fragments)
            if all_single:
                res.warnings.append("所有片段均为单原子离子（纯盐）。通常无需 3D 构象生成。")
                if res.total_formal_charge == 0:
                    res.action_recommendation = "skip_3D_generation: pure_ionic_salt"
    except Exception as e:
        res.warnings.append(f"单原子片段判断异常: {e}")

    return res


# -------------------------
# 快速自测（仅在作为脚本执行时运行）
# -------------------------
if __name__ == "__main__":
    import rdkit
    print("RDKit version:", rdkit.__version__)
    tests = [
        "[2H-]1[Fe+2]2([2H-][Fe+2]31[N-](C(=C(C(=[N]3c1c(cccc1C)C)C)C)C)c1c(C)cccc1C)[N](=C(C)C(=C(C)[N-]2c1c(C)cccc1C)C)c1c(cccc1C)C.c1(ccccc1)C",   # aspirin (organic)
        "[2H+]",             # non-bracketed (likely parse fail)
        "[2H]I.[2H]I",             # ionic
        "[Fe+2]C(C)C",             # Fe2+ + small organic
        "[Cu+2].[NH4+].[Cl-].[Cl-]", # mixed ionic
        "[B-](c1c2c(cccc2ccc1)C[N+](C)(C)C)(c1c(cc(cc1C)C)C)(C#N)c1c(C)cc(C)cc1C",             # benzene
        "[Ag+]123([CH2]=[CH2]3)[N]3=C(C(F)(F)F)N=C(C(F)(F)F)N3[B@H-](N3C(=NC(=[N]13)C(F)(F)F)C(F)(F)F)N1[N]2=C(N=C1C(F)(F)F)C(F)(F)F.[Ag+]123([CH2]=[CH2]3)[N]3=C(N=C(N3[B@@H-](N3C(=NC(=[N]13)C(F)(F)F)C(F)(F)F)N1[N]2=C(N=C1C(F)(F)F)C(F)(F)F)C(F)(F)F)C(F)(F)F",                 # likely parse fail in some RDKit versions
        "[B-]([C-]12[CH]3=[CH]4[CH]5=[CH]1[Fe+2]16782345[CH-]2[CH]1=[CH]6[CH]7=[CH]82)(c1c(cccc1)C)([F][K+]12345[O]6CC[O]2CC[O]3CC[O]4CC[O]5CC[O]1CC6)c1c(C)cccc1.ClC(Cl)Cl",                   # invalid smiles
        "",                        # empty
    ]
    for s in tests:
        print("="*80)
        print("SMILES:", s)
        try:
            r = precheck_smiles(s, allow_sanitize_fail=True)
        except Exception as e:
            print("抛出异常:", type(e).__name__, str(e))
            continue
        print("success:", r.success)
        print("errors:", r.errors)
        print("warnings:", r.warnings)
        print("fragments:", r.fragment_counter)
        print("irreducible:", r.irreducible_counter)
        print("total_charge:", r.total_formal_charge)
        print("contains_metal:", r.contains_metal, r.metal_element_counts)
        print("contains_organic (by carbon):", r.contains_organic_component, r.organic_element_counts)
        print("has_radical:", r.has_radical, "has_isotopes:", r.has_isotopes)
        print("has_unusual_valence:", r.has_unusual_valence)
        print("action_recommendation:", r.action_recommendation)
