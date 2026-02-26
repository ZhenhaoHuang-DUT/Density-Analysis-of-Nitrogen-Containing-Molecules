from typing import Optional, Dict, Any, Tuple, List
import traceback

from rdkit import Chem
from rdkit.Chem import AllChem

from .utils import precheck_smiles, PrecheckResult
from .config import MolBuildConfig, DEFAULT_ORGANIC_CONFIG
def _evaluate_precheck(
    pre: PrecheckResult,
    cfg: MolBuildConfig
) -> Tuple[bool, List[str]]:
    """
    根据 MolBuildConfig 判断 precheck 是否通过
    返回 (passed, reasons)
    """
    reasons: List[str] = []

    if cfg.require_parse_success and not pre.success:
        reasons.append("SMILES parse/sanitize failed")
        return False, reasons

    if pre.errors:
        reasons.append(f"precheck errors: {pre.errors}")
        return False, reasons

    if cfg.require_no_warnings and pre.warnings:
        reasons.append(f"warnings not allowed: {pre.warnings}")
        return False, reasons

    # -------- 电荷 --------
    if cfg.require_total_charge_zero:
        if pre.total_formal_charge != 0:
            reasons.append(f"total formal charge != 0 ({pre.total_formal_charge})")
            return False, reasons

    # -------- 元素组成 --------
    if cfg.require_organic_component and not pre.contains_organic_component:
        reasons.append("no organic component (no carbon)")
        return False, reasons

    if cfg.require_metal and not pre.contains_metal:
        reasons.append("metal required but not found")
        return False, reasons

    if not cfg.allow_metal and pre.contains_metal:
        reasons.append(f"contains metal but not allowed: {pre.metal_element_counts}")
        return False, reasons

    # -------- 电子结构 --------
    if not cfg.allow_radicals and pre.has_radical:
        reasons.append("contains unpaired electrons (radical)")
        return False, reasons

    if not cfg.allow_isotopes and pre.has_isotopes:
        reasons.append("explicit isotopes not allowed")
        return False, reasons

    # -------- 价态 / 配位 --------
    if pre.has_unusual_valence and not cfg.allow_unusual_valence:
        reasons.append("unusual valence not allowed")
        return False, reasons

    if (
        pre.contains_metal
        and pre.contains_organic_component
        and not cfg.allow_metal_organic_complex
    ):
        reasons.append("metal–organic complex not allowed")
        return False, reasons

    return True, []


def _embed_and_optimize(
        mol: Chem.Mol,
        cfg: MolBuildConfig
) -> Tuple[Chem.Mol, List[Dict[str, Any]]]:
    if not cfg.embed_3d:
        # 移除所有可能的构象
        if mol.GetNumConformers() > 0:
            mol.RemoveAllConformers()
        return mol, []

        # 如果进行3D嵌入，添加氢原子
    mol = Chem.AddHs(mol)
    try:
        # 尝试使用 ETKDGv3
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        # 检查是否有 maxAttempts 属性（不同 RDKit 版本可能不同）
        if hasattr(params, 'maxAttempts'):
            params.maxAttempts = cfg.max_embed_attempts
        elif hasattr(params, 'maxIterations'):
            params.maxIterations = cfg.max_embed_attempts
        params.enforceChirality = True

        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
    except Exception as e:
        # 如果 ETKDGv3 失败，回退到 ETKDGv2
        print(f"警告: ETKDGv3 失败，使用 ETKDGv2: {e}")
        params = AllChem.ETKDG()
        params.randomSeed = 42
        params.maxIterations = cfg.max_embed_attempts
        params.enforceChirality = True
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)

    if not conf_ids:
        raise RuntimeError("ETKDG embedding failed")

    opt_results = []

    if cfg.optimize_geometry:
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                res = AllChem.MMFFOptimizeMoleculeConfs(mol)
            else:
                res = AllChem.UFFOptimizeMoleculeConfs(mol)
        except Exception:
            res = AllChem.UFFOptimizeMoleculeConfs(mol)

        for cid, (status, energy) in zip(conf_ids, res):
            opt_results.append(
                {"conf_id": int(cid), "status": int(status), "energy": float(energy)}
            )

    return mol, opt_results
def build_mol_from_smiles(
    smiles: str,
    config: MolBuildConfig = DEFAULT_ORGANIC_CONFIG,
    diagnosis: Optional[PrecheckResult] = None,
    diagnostic_only: bool = False,
) -> Tuple[Optional[Chem.Mol], Dict[str, Any]]:
    """
    主入口：
      - 返回 (mol or None, report)

    report 包含：
      - precheck
      - decision
      - reasons
      - opt_results
      - exception（若有）
    """
    report: Dict[str, Any] = {
        "precheck": None,
        "decision": None,
        "reasons": [],
        "opt_results": [],
        "exception": None,
    }

    try:
        # ---------- 1. precheck ----------
        pre = diagnosis or precheck_smiles(smiles, allow_sanitize_fail=True)
        report["precheck"] = pre

        if diagnostic_only:
            report["decision"] = "diagnostic_only"
            return None, report

        # ---------- 2. 决策 ----------
        passed, reasons = _evaluate_precheck(pre, config)
        if not passed:
            report["decision"] = "rejected"
            report["reasons"] = reasons
            return None, report

        if pre.mol is None:
            report["decision"] = "rejected"
            report["reasons"].append("no mol object after precheck")
            return None, report

        # ---------- 3. 构象 ----------
        mol = Chem.Mol(pre.mol)  # copy
        mol, opt_results = _embed_and_optimize(mol, config)

        report["decision"] = "accepted"
        report["opt_results"] = opt_results
        return mol, report

    except Exception as e:
        report["decision"] = "error"
        report["exception"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        return None, report
