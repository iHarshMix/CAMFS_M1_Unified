"""
CAMFS M1 — Shared Experiment Utilities
======================================

Reusable execution helpers for primary runs and ablation studies:
  - Post-Phase 2 H2 private local head fine-tuning via Tier-1 Net2Net
  - Standardized pure 50-patient test set volume evaluation
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.data.brats_dataset import BraTSDataset
from src.federation.client import FederatedClient
from src.federation.server import FederatedServer
from src.governance.ledger import ProvenanceLedger
from src.logging import CheckpointManager, EvaluationCSVLogger, Phase2CSVLogger
from src.metrics import PatientEvaluator
from src.models.decoder import UNetDecoder
from src.models.encoder import UnimodalEncoder
from src.models.fusion import SubsetFusionHead, net2net_widen_fusion_head


def train_h2_private_local_head(
    h2_client: FederatedClient,
    server: FederatedServer,
    s2_checkpoint_path: Path,
    output_chkpt_dir: Path,
    p2_logger: Phase2CSVLogger,
    ledger: ProvenanceLedger,
    device: torch.device | str,
    max_epochs: int = 100,
    patience: int = 10,
    min_epochs: int = 20,
    max_patients: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Train H2's private local head for {T1, T1ce, T2} via Tier-1 Net2Net channel-widening (§9.5 Option b).

    Loads converged Track S2' {T1, T2} checkpoint, widens input slots to {T1, T1ce, T2}
    with zero-initialized T1ce slot, and fine-tunes locally with early stopping.
    Never transmitted across the network.
    """
    local_track_mods = ["T1", "T1ce", "T2"]

    if not s2_checkpoint_path.exists():
        print(f"WARNING: Track S2 checkpoint not found at {s2_checkpoint_path}. Cannot train H2 local head.", flush=True)
        return 0.0, 0

    s2_chkpt = CheckpointManager.load_checkpoint(s2_checkpoint_path, device=device, restore_rng=False)
    s2_fusion_state = s2_chkpt["state_dict"]["fusion"]
    s2_decoder_state = s2_chkpt["state_dict"]["decoder"]
    s2_source_round = s2_chkpt.get("round", 0)
    s2_source_dice = s2_chkpt.get("val_macro_dice", 0.0)

    # 1. Create base S2 fusion head and load converged weights
    s2_base_head = SubsetFusionHead(modality_subset=["T1", "T2"]).to(device)
    s2_base_head.load_state_dict(s2_fusion_state)

    # 2. Tier-1 Net2Net channel-widening: [T1, T2] -> [T1, T1ce, T2]
    local_fusion = net2net_widen_fusion_head(
        source_head=s2_base_head,
        source_modalities=["T1", "T2"],
        target_modalities=local_track_mods,
    ).to(device)

    # 3. Direct copy of decoder weights (shapes are identical)
    local_decoder = UNetDecoder(num_classes=4).to(device)
    local_decoder.load_state_dict(s2_decoder_state)

    # 4. Extract fused prototypes from S2 checkpoint
    local_fused_protos = s2_chkpt.get("prototypes", {}).get("fused", torch.zeros(4, 256, device=device))
    if isinstance(local_fused_protos, dict):
        local_fused_protos = local_fused_protos.get("S2", torch.zeros(4, 256, device=device))
    if not isinstance(local_fused_protos, torch.Tensor):
        local_fused_protos = torch.zeros(4, 256, device=device)
    local_fused_protos = local_fused_protos.to(device)

    ledger.record_event("H2_LOCAL_HEAD_SEED", {
        "track": "H2_local",
        "modalities": local_track_mods,
        "seed_mechanism": "Tier1_Net2Net_from_S2",
        "source_track": "S2",
        "source_checkpoint": str(s2_checkpoint_path.name),
        "source_round": s2_source_round,
        "source_val_dice": s2_source_dice,
    })

    # 5. Local fine-tuning loop with early stopping
    best_val_dice = 0.0
    best_epoch = 0
    no_improvement_count = 0

    print(f"\n=== Training H2 Private Local Head (Tier-1 Net2Net) for up to {max_epochs} epochs ===", flush=True)

    for local_epoch in range(1, max_epochs + 1):
        up = h2_client.train_phase2_round(
            track_id="H2_local",
            track_modalities=local_track_mods,
            frozen_encoders=server.encoders,
            global_fusion_head=local_fusion,
            global_decoder=local_decoder,
            global_fused_prototypes=local_fused_protos,
            local_epochs=1,
            max_patients=max_patients,
        )

        local_fusion.load_state_dict(up["fusion_state_dict"])
        local_decoder.load_state_dict(up["decoder_state_dict"])
        local_fused_protos = up["fused_prototypes"]

        v_dice, v_hd95 = h2_client.evaluate_phase2_validation(
            track_id="H2_local",
            track_modalities=local_track_mods,
            frozen_encoders=server.encoders,
            global_fusion_head=local_fusion,
            global_decoder=local_decoder,
            max_patients=max_patients,
        )

        val_macro_dice = v_dice["macro"]

        p2_logger.log_round(
            round_num=local_epoch,
            track_id="H2_local",
            hospital_id="H2",
            loss_dice_ce=up.get("loss_dice_ce", 0.0),
            loss_fused_align=up.get("loss_fused_align", 0.0),
            total_loss=up.get("total_loss", 0.0),
            val_dice_dict=v_dice,
            val_hd95_dict=v_hd95,
        )

        if local_epoch % 5 == 0 or local_epoch == 1 or local_epoch == max_epochs:
            print(f"  [H2 Local] Epoch {local_epoch}/{max_epochs} - "
                  f"Val Macro Dice: {val_macro_dice*100:.2f}% - "
                  f"Loss: {up.get('total_loss', 0.0):.4f}", flush=True)

        if val_macro_dice > best_val_dice + 1e-4:
            best_val_dice = val_macro_dice
            best_epoch = local_epoch
            no_improvement_count = 0

            CheckpointManager.save_checkpoint(
                filepath=output_chkpt_dir / "best_H2_local_head.pt",
                state_dict={
                    "fusion": local_fusion.state_dict(),
                    "decoder": local_decoder.state_dict(),
                },
                prototypes={"fused": local_fused_protos},
                current_round=local_epoch,
                val_macro_dice=val_macro_dice,
                extra_metadata={
                    "track": "H2_local",
                    "modalities": local_track_mods,
                    "seed_mechanism": "Tier1_Net2Net_from_S2",
                    "source_round": s2_source_round,
                },
            )
        else:
            no_improvement_count += 1

        if local_epoch >= min_epochs and no_improvement_count >= patience:
            print(f"  [H2 Local] Early stopping at epoch {local_epoch} "
                  f"(best: {best_val_dice*100:.2f}% at epoch {best_epoch})", flush=True)
            break

    ledger.record_event("H2_LOCAL_HEAD_TRAINED", {
        "track": "H2_local",
        "modalities": local_track_mods,
        "best_epoch": best_epoch,
        "best_val_dice": float(best_val_dice),
        "total_epochs": local_epoch,
        "network_transmission": "NEVER",
    })

    print(f"=== H2 Private Local Head Completed: Best Val Dice {best_val_dice*100:.2f}% at Epoch {best_epoch} ===", flush=True)
    return best_val_dice, best_epoch


def evaluate_all_tracks_on_pure_50(
    chkpt_dir: Path,
    results_dir: Path,
    partitions_dir: Path,
    partition_seed: int,
    preprocessed_dir: Path,
    device: torch.device | str,
    h2_custom_chkpt: Optional[str] = None,
    h2_custom_modalities: Optional[List[str]] = None,
    max_test_patients: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Standardized evaluation of all hospital models on the universal 50-patient held-out test set.
    Logs per-patient metrics to pure_50_test_patient_metrics.csv and returns hospital summaries.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    partition_file = partitions_dir / f"partition_{partition_seed}.json"

    with open(partition_file, "r") as f:
        partition_data = json.load(f)

    h3_info = partition_data["hospitals"]["H3"]
    pure_50_test_pids = h3_info.get("fixed_test_patient_ids", [])
    if not pure_50_test_pids:
        h3_test_file = partitions_dir / "h3_test_patients.json"
        if h3_test_file.exists():
            with open(h3_test_file, "r") as f:
                pure_50_test_pids = json.load(f)["patient_ids"]

    if max_test_patients is not None:
        pure_50_test_pids = pure_50_test_pids[:max_test_patients]

    # Load frozen Phase 1 encoders
    phase1_frozen_chkpt = chkpt_dir / "phase1_frozen.pt"
    if not phase1_frozen_chkpt.exists():
        raise FileNotFoundError(f"Frozen Phase 1 checkpoint not found: {phase1_frozen_chkpt}")

    p1_data = CheckpointManager.load_checkpoint(phase1_frozen_chkpt, device=device)
    encoders: Dict[str, UnimodalEncoder] = {}
    for m, enc_state in p1_data["state_dict"].items():
        enc = UnimodalEncoder(in_channels=1).to(device)
        enc.load_state_dict(enc_state)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        encoders[m] = enc

    # Default evaluation specifications
    h2_chkpt = h2_custom_chkpt or ("best_H2_local_head.pt" if (chkpt_dir / "best_H2_local_head.pt").exists() else "best_track_S2.pt")
    h2_mods = h2_custom_modalities or (["T1", "T1ce", "T2"] if h2_chkpt == "best_H2_local_head.pt" or h2_chkpt == "best_track_S4.pt" else ["T1", "T2"])

    eval_specs = {
        "H1": {"track_id": "S1", "modalities": ["T1", "T1ce", "T2", "FLAIR"], "chkpt_name": "best_track_S1.pt"},
        "H2": {"track_id": "H2_local" if "local" in h2_chkpt else "S2", "modalities": h2_mods, "chkpt_name": h2_chkpt},
        "H3": {"track_id": "S3", "modalities": ["T1", "FLAIR"], "chkpt_name": "best_track_S3.pt"},
        "H4": {"track_id": "S4", "modalities": ["T1", "T1ce", "T2"], "chkpt_name": "best_track_S4.pt"},
    }

    eval_csv_path = results_dir / "pure_50_test_patient_metrics.csv"
    eval_logger = EvaluationCSVLogger(eval_csv_path)
    evaluator = PatientEvaluator()
    hospital_summary = {}

    print(f"\n=== Evaluating on Universal 50 Pure Held-Out Test Patients (N={len(pure_50_test_pids)}) ===", flush=True)

    for hid, h_spec in eval_specs.items():
        chkpt_path = chkpt_dir / h_spec["chkpt_name"]
        if not chkpt_path.exists():
            print(f"WARNING: Checkpoint {chkpt_path.name} not found for {hid}. Skipping.", flush=True)
            continue

        chkpt_data = CheckpointManager.load_checkpoint(chkpt_path, device=device, restore_rng=False)
        state = chkpt_data["state_dict"]

        fusion_head = SubsetFusionHead(modality_subset=h_spec["modalities"]).to(device)
        fusion_head.load_state_dict(state["fusion"])
        fusion_head.eval()

        decoder = UNetDecoder(num_classes=4).to(device)
        decoder.load_state_dict(state["decoder"])
        decoder.eval()

        dataset = BraTSDataset(
            patient_ids=pure_50_test_pids,
            cache_root=preprocessed_dir,
            modalities=h_spec["modalities"],
        )

        h_patient_metrics = []

        with torch.no_grad():
            for pid in pure_50_test_pids:
                vol = dataset.load_patient_volume(pid)
                lab_vol = vol["labels"]
                num_slices = lab_vol.shape[0]
                batch_size = 16

                pred_logits_list = []
                for start in range(0, num_slices, batch_size):
                    end = min(start + batch_size, num_slices)

                    mod_feats_b = {}
                    for m in h_spec["modalities"]:
                        if m in vol["modalities"]:
                            x_m = torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(device)
                            h1, h2, h3, h4, z = encoders[m](x_m)
                            mod_feats_b[m] = (h1, h2, h3, h4, z)

                    f1, f2, f3, f4, z_S_b = fusion_head(mod_feats_b)
                    logits_b = decoder(z_S=z_S_b, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)
                    pred_logits_list.append(logits_b.cpu())

                pred_logits_3d = torch.cat(pred_logits_list, dim=0).permute(1, 0, 2, 3).numpy()
                p_metrics = evaluator.evaluate_patient_volume(pred_logits_3d, lab_vol)
                h_patient_metrics.append(p_metrics)

                dice_dict = {
                    "ET": p_metrics["dice_ET"],
                    "TC": p_metrics["dice_TC"],
                    "WT": p_metrics["dice_WT"],
                    "macro": p_metrics["macro_dice"],
                }
                hd95_dict = {
                    "ET": p_metrics["hd95_ET"],
                    "TC": p_metrics["hd95_TC"],
                    "WT": p_metrics["hd95_WT"],
                    "macro": p_metrics["macro_hd95"],
                }
                eval_logger.log_patient_metrics(
                    patient_id=pid,
                    track_id=h_spec["track_id"],
                    hospital_id=hid,
                    dice_dict=dice_dict,
                    hd95_dict=hd95_dict,
                )

        avg_macro = float(np.mean([m["macro_dice"] for m in h_patient_metrics])) * 100
        avg_WT = float(np.mean([m["dice_WT"] for m in h_patient_metrics])) * 100
        avg_TC = float(np.mean([m["dice_TC"] for m in h_patient_metrics])) * 100
        avg_ET = float(np.mean([m["dice_ET"] for m in h_patient_metrics])) * 100
        med_hd95 = float(np.median([m["macro_hd95"] for m in h_patient_metrics]))

        hospital_summary[hid] = {
            "track": h_spec["track_id"],
            "modalities": "+".join(h_spec["modalities"]),
            "WT": avg_WT,
            "TC": avg_TC,
            "ET": avg_ET,
            "Macro": avg_macro,
            "HD95_Median": med_hd95,
        }

        print(f"  [{hid}] Macro Dice: {avg_macro:.2f}% | WT: {avg_WT:.2f}% | TC: {avg_TC:.2f}% | ET: {avg_ET:.2f}% | HD95: {med_hd95:.2f} mm", flush=True)

    if hospital_summary:
        fed_avg_macro = float(np.mean([h["Macro"] for h in hospital_summary.values()]))
        fed_med_hd = float(np.median([h["HD95_Median"] for h in hospital_summary.values()]))
        print(f"  --> Federated Average Macro Dice: {fed_avg_macro:.2f}% (Median HD95: {fed_med_hd:.2f} mm)", flush=True)

    return hospital_summary
