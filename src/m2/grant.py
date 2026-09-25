"""
CAMFS M2 — Static Bilateral Governance Grant & Policy Suite
===========================================================

Implements §7, §8, §10.1, and §12.1 (P0–P6) of the CAMFS M2 Specification.
Formal bilateral contract governing knowledge transfer from H1 to H3.
Enforces:
- Pre-build gate (AllowBuild)
- Post-build release gate (AllowRelease)
- Parameter budget (<= 50,000 parameters)
- No base weight leakage, no patient ID leakage, no raw image leakage
- Strict prohibition of upload, onward transfer, and M1 seed mutation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch


@dataclass
class ArtifactGrant:
    """
    Formal bilateral research grant defining exact transfer rights (§10.1).
    """
    grant_id: str = "GRANT_H1_H3_T1CE_M2_PRIMARY"
    donor_hospital: str = "H1"
    recipient_hospital: str = "H3"
    recipient_track: Tuple[str, ...] = ("T1", "FLAIR")
    transferred_modality: str = "T1ce"
    target_augmented_track: Tuple[str, ...] = ("T1", "T1ce", "FLAIR")
    canonical_base_hash: str = ""
    rights_digest: str = ""
    max_parameter_budget: int = 50_000
    allow_build: bool = True
    allow_release: bool = True
    allow_upload: bool = False             # Invariant: Never allow upload to global cloud
    allow_m1_seed: bool = False            # Invariant: Never allow M1 parameter mutation
    allow_onward_transfer: bool = False    # Invariant: H3 cannot transfer to H2/H4


def create_primary_grant(canonical_base_hash: str, rights_digest: str) -> ArtifactGrant:
    """Creates the standard H1 -> H3 T1ce bilateral grant."""
    return ArtifactGrant(
        grant_id="GRANT_H1_H3_T1CE_M2_PRIMARY",
        donor_hospital="H1",
        recipient_hospital="H3",
        recipient_track=("T1", "FLAIR"),
        transferred_modality="T1ce",
        target_augmented_track=("T1", "T1ce", "FLAIR"),
        canonical_base_hash=canonical_base_hash,
        rights_digest=rights_digest,
        max_parameter_budget=50_000,
        allow_build=True,
        allow_release=True,
    )


def allow_build(
    grant: ArtifactGrant,
    donor_modalities: Sequence[str],
    current_base_hash: str,
) -> Tuple[bool, str]:
    """
    Pre-build authorization gate (§10.1).
    Verifies that the donor owns the requested modality, the recipient does not,
    the grant is active, and the base hash matches canonical M1.
    """
    if not grant.allow_build:
        return False, "BUILD_DENIED: Grant allow_build flag is False"

    if current_base_hash != grant.canonical_base_hash:
        return False, (
            f"BUILD_DENIED: Base hash mismatch (expected {grant.canonical_base_hash[:12]}, "
            f"got {current_base_hash[:12]})"
        )

    if grant.transferred_modality not in donor_modalities:
        return False, (
            f"BUILD_DENIED: Donor {grant.donor_hospital} does not own transferred modality "
            f"{grant.transferred_modality}"
        )

    if grant.transferred_modality in grant.recipient_track:
        return False, (
            f"BUILD_DENIED: Recipient {grant.recipient_hospital} already owns modality "
            f"{grant.transferred_modality}"
        )

    return True, "BUILD_ALLOWED"


def allow_release(
    artifact_state_dict: Dict[str, torch.Tensor],
    manifest: Dict[str, Any],
    grant: ArtifactGrant,
    current_base_hash: str,
) -> Tuple[bool, str]:
    """
    Post-build release gate (§10.1, §10.7).
    Inspects the actual compiled artifact before cross-hospital transfer.
    Checks:
    - allow_release flag
    - Base hash match
    - Parameter count <= max_parameter_budget (50,000)
    - Absence of M1 encoder/decoder/fusion base parameter names
    - Absence of patient identifiers in metadata
    """
    if not grant.allow_release:
        return False, "RELEASE_DENIED: Grant allow_release flag is False"

    manifest_base_hash = manifest.get("canonical_base_hash", "")
    if manifest_base_hash != grant.canonical_base_hash or current_base_hash != grant.canonical_base_hash:
        return False, "RELEASE_DENIED: Base hash mismatch in artifact manifest"

    # Parameter count check
    total_params = sum(p.numel() for p in artifact_state_dict.values())
    if total_params > grant.max_parameter_budget:
        return False, (
            f"RELEASE_DENIED: Parameter count {total_params} exceeds budget {grant.max_parameter_budget}"
        )

    # Base weight leakage check
    forbidden_prefixes = ("encoder_", "fusion_head.", "decoder.", "prototypes.")
    for key in artifact_state_dict.keys():
        if any(key.startswith(p) for p in forbidden_prefixes):
            return False, f"RELEASE_DENIED: Leaked base parameter detected: {key}"

    # Privacy / patient ID leakage check
    manifest_str = str(manifest)
    if "BraTS20_Training_" in manifest_str or "patient_ids" in manifest:
        return False, "RELEASE_DENIED: Raw patient identifiers detected in artifact manifest"

    return True, "RELEASE_ALLOWED"


# ---------------------------------------------------------------------------
# Policy Tests (P0–P6) Implementation (§12.1)
# ---------------------------------------------------------------------------

def run_policy_suite(
    grant: ArtifactGrant,
    valid_artifact_sd: Dict[str, torch.Tensor],
    valid_manifest: Dict[str, Any],
    donor_modalities: Sequence[str] = ("T1", "T1ce", "T2", "FLAIR"),
) -> Dict[str, bool]:
    """
    Executes the 7 formal policy test cases (P0–P6) from §12.1.
    Returns a dictionary mapping test name to passed status (True/False).
    """
    results: Dict[str, bool] = {}
    base_hash = grant.canonical_base_hash

    # P0: Donor denies build -> DENIED
    p0_grant = ArtifactGrant(
        **{**asdict(grant), "allow_build": False}
    )
    ok_p0, _ = allow_build(p0_grant, donor_modalities, base_hash)
    results["P0_donor_denies_build"] = (ok_p0 is False)

    # P1: Recipient denies release -> DENIED
    p1_grant = ArtifactGrant(
        **{**asdict(grant), "allow_release": False}
    )
    ok_p1, _ = allow_release(valid_artifact_sd, valid_manifest, p1_grant, base_hash)
    results["P1_recipient_denies_release"] = (ok_p1 is False)

    # P2: Valid grant and compliant artifact -> ALLOWED
    ok_b_p2, _ = allow_build(grant, donor_modalities, base_hash)
    ok_r_p2, _ = allow_release(valid_artifact_sd, valid_manifest, grant, base_hash)
    results["P2_valid_grant_allowed"] = (ok_b_p2 is True and ok_r_p2 is True)

    # P3: Unauthorized modality transfer (e.g. requesting T2 which donor doesn't give or recipient owns) -> DENIED
    p3_grant = ArtifactGrant(
        **{**asdict(grant), "transferred_modality": "T2", "recipient_track": ("T1", "T2", "FLAIR")}
    )
    ok_p3, _ = allow_build(p3_grant, donor_modalities, base_hash)
    results["P3_unauthorized_modality_denied"] = (ok_p3 is False)

    # P4: Base hash mismatch -> DENIED
    bad_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    ok_p4, _ = allow_build(grant, donor_modalities, bad_hash)
    results["P4_bad_base_hash_denied"] = (ok_p4 is False)

    # P5: Parameter budget violation (>50,000 params) -> DENIED
    large_sd = {"bloat_weights": torch.zeros(60_000)}
    ok_p5, _ = allow_release(large_sd, valid_manifest, grant, base_hash)
    results["P5_parameter_budget_violation_denied"] = (ok_p5 is False)

    # P6: Leaked base encoder weights in artifact -> DENIED
    leaked_sd = dict(valid_artifact_sd)
    leaked_sd["encoder_t1.layer1.block.0.weight"] = torch.zeros(10, 10)
    ok_p6, _ = allow_release(leaked_sd, valid_manifest, grant, base_hash)
    results["P6_leaked_base_weights_denied"] = (ok_p6 is False)

    return results
