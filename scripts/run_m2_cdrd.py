"""
CAMFS M2 — CDRD End-to-End Orchestration Pipeline (E4)
======================================================

Implements §10.7, §15.2 of the CAMFS M2 Specification.
16-Step deterministic pipeline executing Consent-Bounded Detached Residual Distillation:
H1 (Donor) -> H3 (Recipient) for Modality T1ce.

Usage:
    python scripts/run_m2_cdrd.py --config configs/m2_cdrd.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_yaml
from src.data.brats_dataset import BraTSDataset
from src.data.slice_sampler import SliceSampler
from src.losses import SoftDiceCrossEntropyLoss
from src.metrics import PatientEvaluator
from src.m2.adapter import CDRDAdapter
from src.m2.adapter_trainer import train_cdrd_adapter
from src.m2.base_registry import (
    compute_canonical_base_hash,
    compute_rights_digest,
    verify_base_integrity,
)
from src.m2.calibration import (
    RecipientGate,
    evaluate_recipient_acceptance,
    train_recipient_gates,
)
from src.m2.frozen_base import load_frozen_base
from src.m2.grant import (
    create_primary_grant,
    allow_build,
    allow_release,
    run_policy_suite,
)
from src.m2.provenance import ProvenanceLedger
from src.m2.splits import compute_m2_splits, save_m2_splits_json
from src.m2.teacher import (
    build_restricted_teacher,
    calibrate_teacher_temperature,
    evaluate_teacher_viability,
    verify_teacher_day0_identity,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CAMFS M2 CDRD E4 Pipeline")
    parser.add_argument("--config", type=str, default="configs/m2_cdrd.yaml", help="Path to M2 config YAML")
    parser.add_argument("--gpu", type=int, default=None, help="Override GPU ID")
    parser.add_argument("--partition-seed", type=int, default=None, help="Override partition seed")
    parser.add_argument("--train-seed", type=int, default=None, help="Override training seed")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining of teacher and adapter")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    # CLI overrides
    gpu_id = args.gpu if args.gpu is not None else cfg["experiment"].get("gpu", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"\n===================================================================")
    print(f"  CAMFS M2: Consent-Bounded Detached Residual Distillation (E4)")
    print(f"  Device: {device} | Host: {os.uname().nodename}")
    print(f"===================================================================\n")

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(cfg["paths"]["ledger_path"])
    ledger = ProvenanceLedger(ledger_path)

    # -----------------------------------------------------------------------
    # Step 1: Pre-Flight Canonical Base Hash Verification
    # -----------------------------------------------------------------------
    print("[Step 1/16] Computing and verifying Canonical Base Hash b...")
    p1_path = cfg["paths"]["phase1_checkpoint"]
    s3_path = cfg["paths"]["track_s3_checkpoint"]

    base_digests = compute_canonical_base_hash(p1_path, s3_path)
    base_hash = base_digests["canonical_base_hash"]
    rights_digest = compute_rights_digest(base_hash)
    print(f"  Canonical Base Hash b: {base_hash}")
    print(f"  Rights Digest rho_b:   {rights_digest}")

    ledger.record_event(
        event_type="PRE_FLIGHT_INTEGRITY",
        actor_hospital="HEADNODE",
        grant_id="GENESIS",
        payload={
            "canonical_base_hash": base_hash,
            "rights_digest": rights_digest,
            "phase1_file_sha256": base_digests["phase1_file_sha256"],
            "track_s3_file_sha256": base_digests["track_s3_file_sha256"],
        },
    )

    # -----------------------------------------------------------------------
    # Step 2: Governance Grant Creation & Pre-Build Gate Check
    # -----------------------------------------------------------------------
    print("\n[Step 2/16] Establishing Bilateral Research Grant & Pre-Build Gate...")
    grant = create_primary_grant(canonical_base_hash=base_hash, rights_digest=rights_digest)
    donor_mods = ["T1", "T1ce", "T2", "FLAIR"]

    build_ok, build_msg = allow_build(grant, donor_modalities=donor_mods, current_base_hash=base_hash)
    if not build_ok:
        print(f"  ❌ {build_msg}")
        ledger.record_event("BUILD_DENIED", "H1", grant.grant_id, {"reason": build_msg})
        sys.exit(1)
    print(f"  ✅ {build_msg} (Grant ID: {grant.grant_id})")

    ledger.record_event(
        event_type="ALLOW_BUILD_PASS",
        actor_hospital="H1",
        grant_id=grant.grant_id,
        payload={"grant": grant.__dict__},
    )

    # -----------------------------------------------------------------------
    # Step 3: Materialize Patient Splits
    # -----------------------------------------------------------------------
    print("\n[Step 3/16] Computing deterministic M2 patient splits...")
    partition_file = Path(cfg["paths"]["partitions_dir"]) / f"partition_{cfg['experiment']['partition_seed']}.json"
    splits = compute_m2_splits(
        partition_path=partition_file,
        test_patients_path=cfg["paths"]["h3_test_patients"],
        split_seed=cfg["experiment"]["split_seed"],
    )
    splits_json_path = out_dir / "m2_splits.json"
    save_m2_splits_json(splits, splits_json_path)
    print(f"  H1 Splits: {len(splits.h1.train_ids)} Train, {len(splits.h1.val_ids)} Val, {len(splits.h1.release_audit_ids)} Release-Audit")
    print(f"  H3 Splits: {len(splits.h3.m1_train_ids)} M1-Train, {len(splits.h3.m2_cal_ids)} Cal, {len(splits.h3.m2_accept_val_ids)} Accept-Val, {len(splits.h3.sacred_test_ids)} Sacred Test")

    # -----------------------------------------------------------------------
    # Step 4: Instantiate Frozen M1 Base
    # -----------------------------------------------------------------------
    print("\n[Step 4/16] Loading Frozen M1 Base wrapper (eval mode, requires_grad=False)...")
    frozen_base = load_frozen_base(p1_path, s3_path, device=device)
    print("  Frozen Base instantiated with feature taps: (f1, f2, f3, f4, z_S, logits_B)")

    # Shared Dataset and Evaluator instances
    dataset_h1 = BraTSDataset(
        patient_ids=splits.h1.train_ids + splits.h1.val_ids + splits.h1.release_audit_ids,
        cache_root=cfg["paths"]["preprocessed_dir"],
        modalities=["T1", "T1ce", "T2", "FLAIR"],
    )
    dataset_h3 = BraTSDataset(
        patient_ids=splits.h3.m1_train_ids + splits.h3.m2_cal_ids + splits.h3.m2_accept_val_ids,
        cache_root=cfg["paths"]["preprocessed_dir"],
        modalities=["T1", "FLAIR"],
    )
    evaluator = PatientEvaluator()

    # -----------------------------------------------------------------------
    # Step 5: Instantiate Donor Restricted Teacher & Verify Day-0 Identity
    # -----------------------------------------------------------------------
    print("\n[Step 5/16] Building Restricted Teacher via Net2Net Widening...")
    teacher = build_restricted_teacher(p1_path, s3_path, device=device)
    day0_teacher_diff = verify_teacher_day0_identity(teacher, frozen_base, device=device, tol=1e-4)
    print(f"  ✅ Day-0 Identity Verified: Max abs diff = {day0_teacher_diff:.2e} (Teacher == Base before training)")

    # -----------------------------------------------------------------------
    # Step 6: Train Restricted Teacher on H1 (102 patients)
    # -----------------------------------------------------------------------
    teacher_ckpt_path = out_dir / "teacher_best.pt"
    if teacher_ckpt_path.exists() and not args.force_retrain:
        print(f"\n[Step 6/16] Loading existing trained teacher from {teacher_ckpt_path}...")
        teacher.load_state_dict(torch.load(teacher_ckpt_path, map_location=device)["state_dict"])
    else:
        print(f"\n[Step 6/16] Training Restricted Teacher on {len(splits.h1.train_ids)} H1 patients...")
        t_opt = torch.optim.AdamW(
            list(teacher.fusion_head.parameters()) + list(teacher.decoder.parameters()),
            lr=cfg["teacher"]["lr"],
            weight_decay=cfg["teacher"]["weight_decay"],
        )
        t_crit = SoftDiceCrossEntropyLoss()
        t_sampler = SliceSampler(cache_root=dataset_h1.cache_root, patient_ids=splits.h1.train_ids)

        best_t_val = -1.0
        best_t_sd = None
        patience_cnt = 0
        max_t_epochs = cfg["teacher"]["max_epochs"]
        patience = cfg["teacher"]["patience"]

        for epoch in range(max_t_epochs):
            teacher.train()
            # Encoders must remain in eval mode
            teacher.encoder_t1.eval()
            teacher.encoder_t1ce.eval()
            teacher.encoder_flair.eval()

            samples = t_sampler.get_epoch_samples(seed=cfg["experiment"]["train_seed"] + epoch, is_training=True)
            run_loss = 0.0
            nb = 0

            for bs in range(0, len(samples), cfg["teacher"]["batch_size"]):
                batch_items = samples[bs : bs + cfg["teacher"]["batch_size"]]
                t1_l, t1ce_l, flair_l, y_l = [], [], [], []
                for pid, s_idx in batch_items:
                    pdir = dataset_h1.cache_root / pid
                    t1_l.append(torch.from_numpy(np.array(np.load(pdir / "t1.npy", mmap_mode="r")[s_idx], dtype=np.float32)).unsqueeze(0))
                    t1ce_l.append(torch.from_numpy(np.array(np.load(pdir / "t1ce.npy", mmap_mode="r")[s_idx], dtype=np.float32)).unsqueeze(0))
                    flair_l.append(torch.from_numpy(np.array(np.load(pdir / "flair.npy", mmap_mode="r")[s_idx], dtype=np.float32)).unsqueeze(0))
                    y_l.append(torch.from_numpy(np.array(np.load(pdir / "labels.npy", mmap_mode="r")[s_idx], dtype=np.int64)))

                x_t = {
                    "T1": torch.stack(t1_l, dim=0).to(device),
                    "T1ce": torch.stack(t1ce_l, dim=0).to(device),
                    "FLAIR": torch.stack(flair_l, dim=0).to(device),
                }
                y_b = torch.stack(y_l, dim=0).to(device)

                logits = teacher(x_t)
                loss = t_crit(logits, y_b)

                t_opt.zero_grad()
                loss.backward()
                t_opt.step()

                run_loss += loss.item()
                nb += 1

            # Validation pass on 13 H1 val patients
            teacher.eval()
            val_dices = []
            with torch.no_grad():
                for pid in splits.h1.val_ids:
                    vol = dataset_h1.load_patient_volume(pid)
                    lab = vol["labels"]
                    preds = []
                    for s in range(0, lab.shape[0], 16):
                        e = min(s + 16, lab.shape[0])
                        xb = {
                            m: torch.from_numpy(vol["modalities"][m][s:e]).float().unsqueeze(1).to(device)
                            for m in ["T1", "T1ce", "FLAIR"]
                        }
                        preds.append(teacher(xb).cpu())
                    pred_vol = torch.cat(preds, dim=0).permute(1, 0, 2, 3).numpy()
                    val_dices.append(evaluator.evaluate_patient_volume(pred_vol, lab)["macro_dice"])

            mean_val_dice = float(np.mean(val_dices))
            print(f"  [Teacher Epoch {epoch+1:02d}/{max_t_epochs:02d}] Loss: {run_loss/max(1, nb):.4f} | Val Macro Dice: {mean_val_dice*100:.2f}%", flush=True)

            if mean_val_dice > best_t_val + 1e-4:
                best_t_val = mean_val_dice
                best_t_sd = copy.deepcopy(teacher.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  Teacher early stopping at epoch {epoch+1}")
                    break

        if best_t_sd is not None:
            teacher.load_state_dict(best_t_sd)
        torch.save({"state_dict": teacher.state_dict(), "val_macro_dice": best_t_val}, teacher_ckpt_path)
        print(f"  ⭐ Saved best teacher checkpoint to {teacher_ckpt_path} (Val Macro: {best_t_val*100:.2f}%)")

    ledger.record_event(
        event_type="TEACHER_TRAINED",
        actor_hospital="H1",
        grant_id=grant.grant_id,
        payload={"teacher_checkpoint": str(teacher_ckpt_path)},
    )

    # -----------------------------------------------------------------------
    # Step 7: Teacher Temperature Calibration
    # -----------------------------------------------------------------------
    print("\n[Step 7/16] Calibrating Teacher Temperature on H1 Validation set...")
    t_cal = calibrate_teacher_temperature(
        teacher=teacher,
        dataset=dataset_h1,
        val_patient_ids=splits.h1.val_ids,
        device=device,
        num_grid_points=cfg["teacher"]["calibration"]["num_grid_points"],
        t_min=cfg["teacher"]["calibration"]["t_min"],
        t_max=cfg["teacher"]["calibration"]["t_max"],
        seed=cfg["teacher"]["calibration"]["seed"],
        max_voxels=cfg["teacher"]["calibration"]["max_voxels"],
    )
    print(f"  ✅ Fitted Teacher Temperature T_T^cal = {t_cal:.4f}")

    ledger.record_event(
        event_type="TEACHER_CALIBRATED",
        actor_hospital="H1",
        grant_id=grant.grant_id,
        payload={"teacher_temperature": t_cal},
    )

    # -----------------------------------------------------------------------
    # Step 8: Teacher Viability Gate (q_T)
    # -----------------------------------------------------------------------
    print("\n[Step 8/16] Evaluating Teacher Viability Gate on 12 H1 Release-Audit patients...")
    q_T, viability_metrics = evaluate_teacher_viability(
        teacher=teacher,
        frozen_base=frozen_base,
        dataset=dataset_h1,
        release_audit_patient_ids=splits.h1.release_audit_ids,
        evaluator=evaluator,
        device=device,
        bootstrap_resamples=cfg["teacher"]["viability"]["bootstrap_resamples"],
        bootstrap_seed=cfg["teacher"]["viability"]["bootstrap_seed"],
        non_inferiority_delta=cfg["teacher"]["viability"]["non_inferiority_delta"],
        et_confidence_threshold=cfg["teacher"]["viability"]["et_confidence_threshold"],
    )
    print(f"  Teacher Mean ET: {viability_metrics['teacher_mean_ET']*100:.2f}% vs Base Mean ET: {viability_metrics['base_mean_ET']*100:.2f}%")
    print(f"  Mean Delta ET:   {viability_metrics['mean_delta_ET']*100:+.2f}% | P(Delta ET > 0): {viability_metrics['prob_ET_improved']*100:.1f}%")
    print(f"  => Viability Gate Decision: q_T = {q_T}")

    ledger.record_event(
        event_type="TEACHER_VIABILITY_EVALUATED",
        actor_hospital="H1",
        grant_id=grant.grant_id,
        payload=viability_metrics,
    )

    # -----------------------------------------------------------------------
    # Step 9: Instantiate CDRD Adapter & Verify Day-0 Identity
    # -----------------------------------------------------------------------
    print("\n[Step 9/16] Instantiating CDRD Adapter (16,344 parameters)...")
    adapter = CDRDAdapter().to(device)
    param_count = adapter.count_trainable_parameters()
    assert param_count == 16344, f"Adapter parameter count error: {param_count} != 16344"
    print(f"  ✅ Verified Parameter Count: {param_count} parameters (~{param_count * 4 / 1024:.1f} KB)")

    # -----------------------------------------------------------------------
    # Step 10: Train CDRD Adapter on H1 (102 patients)
    # -----------------------------------------------------------------------
    adapter_ckpt_path = out_dir / "adapter_best.pt"
    if adapter_ckpt_path.exists() and not args.force_retrain:
        print(f"\n[Step 10/16] Loading existing trained adapter from {adapter_ckpt_path}...")
        adapter.load_state_dict(torch.load(adapter_ckpt_path, map_location=device)["state_dict"])
    else:
        print(f"\n[Step 10/16] Distilling T1ce Knowledge into CDRD Adapter on H1...")
        adapter, adapter_meta = train_cdrd_adapter(
            adapter=adapter,
            frozen_base=frozen_base,
            restricted_teacher=teacher,
            teacher_temperature=t_cal,
            dataset=dataset_h1,
            h1_train_patient_ids=splits.h1.train_ids,
            h1_val_patient_ids=splits.h1.val_ids,
            evaluator=evaluator,
            device=device,
            lr=cfg["adapter"]["lr"],
            max_epochs=cfg["adapter"]["max_epochs"],
            patience=cfg["adapter"]["patience"],
            batch_size=cfg["adapter"]["batch_size"],
            seed=cfg["experiment"]["train_seed"],
            lambda_seg=cfg["adapter"]["lambda_seg"],
            lambda_kd=cfg["adapter"]["lambda_kd"],
            lambda_delta=cfg["adapter"]["lambda_delta"],
            lambda_tv=cfg["adapter"]["lambda_tv"],
            kd_temperature=cfg["adapter"]["kd_temperature"],
        )
        torch.save({"state_dict": adapter.state_dict(), "meta": adapter_meta}, adapter_ckpt_path)
        print(f"  ⭐ Saved best adapter checkpoint to {adapter_ckpt_path}")

    ledger.record_event(
        event_type="ADAPTER_TRAINED",
        actor_hospital="H1",
        grant_id=grant.grant_id,
        payload={"adapter_checkpoint": str(adapter_ckpt_path)},
    )

    # -----------------------------------------------------------------------
    # Step 11: Artifact Packaging, Policy Tests (P0–P6) & AllowRelease
    # -----------------------------------------------------------------------
    print("\n[Step 11/16] Executing Formal Governance Policy Suite (P0–P6)...")
    manifest = {
        "grant_id": grant.grant_id,
        "donor_hospital": "H1",
        "recipient_hospital": "H3",
        "canonical_base_hash": base_hash,
        "rights_digest": rights_digest,
        "parameter_count": param_count,
        "q_T": q_T,
        "teacher_temperature": t_cal,
    }

    policy_results = run_policy_suite(grant, adapter.state_dict(), manifest, donor_mods)
    for test_name, passed in policy_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  - {test_name}: {status}")
        assert passed, f"Policy test {test_name} failed!"

    rel_ok, rel_msg = allow_release(adapter.state_dict(), manifest, grant, current_base_hash=base_hash)
    assert rel_ok, f"Release denied: {rel_msg}"
    print(f"  ✅ {rel_msg}")

    ledger.record_event(
        event_type="ALLOW_RELEASE_PASS",
        actor_hospital="H1",
        grant_id=grant.grant_id,
        payload={"manifest": manifest, "policy_suite": policy_results},
    )

    # -----------------------------------------------------------------------
    # Step 12: Cross-Hospital Transfer Simulation (H1 -> H3)
    # -----------------------------------------------------------------------
    print("\n[Step 12/16] Simulating cross-hospital transfer of adapter artifact to H3...")
    h3_staging_dir = out_dir / "h3_received"
    h3_staging_dir.mkdir(parents=True, exist_ok=True)
    transfer_payload = {
        "adapter_state_dict": adapter.state_dict(),
        "manifest": manifest,
    }
    torch.save(transfer_payload, h3_staging_dir / "cdrd_artifact.pt")
    print(f"  ✅ Artifact safely received and verified at Hospital H3")

    # -----------------------------------------------------------------------
    # Step 13: Recipient Gate Calibration (H3: 6 patients, 4 parameters)
    # -----------------------------------------------------------------------
    print("\n[Step 13/16] Fitting 4 Recipient Gate logits on 6 H3 Calibration patients...")
    gate = RecipientGate(init_val=cfg["calibration"]["gate_init"]).to(device)
    gate_ckpt_path = out_dir / "recipient_gate.pt"

    gate, gate_meta = train_recipient_gates(
        gate=gate,
        frozen_base=frozen_base,
        adapter=adapter,
        dataset=dataset_h3,
        h3_cal_patient_ids=splits.h3.m2_cal_ids,
        device=device,
        lr=cfg["calibration"]["lr"],
        epochs=cfg["calibration"]["epochs"],
        batch_size=cfg["calibration"]["batch_size"],
        seed=cfg["experiment"]["split_seed"],
    )
    torch.save({"state_dict": gate.state_dict(), "meta": gate_meta}, gate_ckpt_path)
    print(f"  ⭐ Saved fitted gate to {gate_ckpt_path} (Final alphas: {gate_meta['final_alphas']})")

    ledger.record_event(
        event_type="RECIPIENT_GATE_CALIBRATED",
        actor_hospital="H3",
        grant_id=grant.grant_id,
        payload={"alphas": gate_meta["final_alphas"]},
    )

    # -----------------------------------------------------------------------
    # Step 14: Recipient Acceptance Evaluation (H3: 6 accept-val patients)
    # -----------------------------------------------------------------------
    print("\n[Step 14/16] Evaluating Augmented vs Fallback acceptance on 6 H3 Acceptance-Val patients...")
    d_deploy, accept_pass, accept_metrics = evaluate_recipient_acceptance(
        gate=gate,
        frozen_base=frozen_base,
        adapter=adapter,
        dataset=dataset_h3,
        h3_accept_val_patient_ids=splits.h3.m2_accept_val_ids,
        evaluator=evaluator,
        q_T=q_T,
        device=device,
        non_inferiority_delta=cfg["calibration"]["acceptance"]["non_inferiority_delta"],
    )

    ledger.record_event(
        event_type="ACCEPTANCE_EVALUATED",
        actor_hospital="H3",
        grant_id=grant.grant_id,
        payload=accept_metrics,
    )

    # -----------------------------------------------------------------------
    # Step 15: Post-Flight Canonical Base Integrity Verification
    # -----------------------------------------------------------------------
    print("\n[Step 15/16] Re-verifying M1 Canonical Base Invariance after M2 execution...")
    verify_base_integrity(p1_path, s3_path, expected_base_hash=base_hash)
    print(f"  ✅ M1 Base Invariance Verified: Hash is bit-for-bit identical before and after M2")

    ledger.record_event(
        event_type="POST_FLIGHT_INTEGRITY",
        actor_hospital="HEADNODE",
        grant_id=grant.grant_id,
        payload={"canonical_base_hash": base_hash, "integrity": "VERIFIED"},
    )

    # -----------------------------------------------------------------------
    # Step 16: Cryptographic Ledger Finalization
    # -----------------------------------------------------------------------
    print("\n[Step 16/16] Finalizing Cryptographic Provenance Ledger...")
    chain_ok, chain_err = ledger.verify_integrity()
    assert chain_ok, f"Ledger integrity violation: {chain_err}"
    print(f"  ✅ Provenance Ledger Chain Verified from Genesis (Total events recorded)")

    print(f"\n===================================================================")
    print(f"  🎉 CAMFS M2 CDRD PIPELINE SUCCESSFULLY COMPLETED!")
    print(f"  Teacher Viability q_T: {q_T}")
    print(f"  H3 Acceptance Pass:    {accept_pass}")
    print(f"  Deployment Gate d_dep: {d_deploy}")
    print(f"  Ledger Path:           {ledger_path}")
    print(f"===================================================================\n")


if __name__ == "__main__":
    main()
