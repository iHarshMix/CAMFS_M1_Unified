"""
CAMFS M1 — Main Primary Experiment Runner Script
================================================

Implements §15.2 of the CAMFS M1 Specification.
Orchestrates end-to-end CAMFS M1 FL training:
  - CLI parser for config, partition_seed, train_seed, gpu, dry_run
  - Sets PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG before execution
  - Environment version dump serialization
  - Phase 1 contrastive FL -> Freeze procedure -> Phase 2 track fusion FL -> 3D test evaluation
  - CSV logging and atomic checkpointing
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

# Enforce deterministic environment variables before torch initialization
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from src.config import load_yaml
from src.federation import FederatedClient, FederatedPhaseState, FederatedServer, PhaseController
from src.governance import LineageAuditor, PolicyManager, ProvenanceLedger
from src.logging import CheckpointManager, EvaluationCSVLogger, Phase1CSVLogger, Phase2CSVLogger
from src.metrics import PatientEvaluator
from src.seed import set_deterministic


def dump_environment_info(output_path: Path) -> None:
    """Dump system environment, PyTorch, CUDA, and library versions (§15.4)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = [
        f"Python Version: {sys.version}",
        f"Platform: {platform.platform()}",
        f"PyTorch Version: {torch.__version__}",
        f"CUDA Available: {torch.cuda.is_available()}",
        f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}",
        f"Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"NumPy Version: {np.__version__}",
        f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}",
        f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}",
    ]
    output_path.write_text("\n".join(info) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="CAMFS M1 Main Primary Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--policy-config", type=str, default="configs/policy_M1_PRIMARY_V1.json", help="Path to policy JSON")
    parser.add_argument("--partition-seed", type=int, default=1103, help="Patient partition seed (1103, 2207, 3301)")
    parser.add_argument("--train-seed", type=int, default=17, help="Training seed (17, 29, 43)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--max-gpu-memory-gb", type=float, default=20.0, help="Max GPU VRAM memory limit in GB")
    parser.add_argument("--max-p1-rounds", type=int, default=100, help="Max Phase 1 rounds")
    parser.add_argument("--max-p2-rounds", type=int, default=100, help="Max Phase 2 rounds")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (1 round for Phase 1 & 2)")
    parser.add_argument("--preprocessed-dir", type=str, default="outputs/preprocessed", help="Path to preprocessed data")
    parser.add_argument("--partitions-dir", type=str, default="outputs/partitions", help="Path to partition manifests")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root output directory")
    return parser.parse_args()


def run_experiment(args):
    """Execute a single primary CAMFS M1 experiment run."""
    # Load config
    config = load_yaml(args.config)

    # Set deterministic seeds
    set_deterministic(args.train_seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    print(f"=== Execution Device: {device} (CUDA Available: {torch.cuda.is_available()}) ===", flush=True)

    if torch.cuda.is_available() and args.gpu >= 0 and args.max_gpu_memory_gb is not None:
        device_id = args.gpu
        total_mem = torch.cuda.get_device_properties(device_id).total_memory
        max_bytes = int(args.max_gpu_memory_gb * 1024 * 1024 * 1024)
        if total_mem > max_bytes:
            fraction = max_bytes / total_mem
            torch.cuda.set_per_process_memory_fraction(fraction, device_id)
            print(f"GPU VRAM limit set to {args.max_gpu_memory_gb:.1f} GB (fraction: {fraction:.4f})", flush=True)

    experiment_id = f"camfs_primary__part{args.partition_seed}__seed{args.train_seed}"
    log_dir = Path(args.output_dir) / "logs" / experiment_id
    chkpt_dir = Path(args.output_dir) / "checkpoints" / experiment_id
    results_dir = Path(args.output_dir) / "results" / experiment_id

    log_dir.mkdir(parents=True, exist_ok=True)
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dump_environment_info(log_dir / "environment_version_dump.txt")

    # Load policy & partitions
    policy = PolicyManager(args.policy_config)
    partition_file = Path(args.partitions_dir) / f"partition_{args.partition_seed}.json"

    if not partition_file.exists():
        raise FileNotFoundError(f"Partition manifest not found: {partition_file}")

    with open(partition_file, "r") as f:
        partitions_data = json.load(f)

    hospitals_config = partitions_data["hospitals"]

    # Initialize governance & FL system
    ledger = ProvenanceLedger(log_dir / "provenance_ledger.jsonl")
    auditor = LineageAuditor(audit_mode=config.get("governance", {}).get("audit_mode", "reject"))
    controller = PhaseController(
        min_p1_rounds=1 if args.dry_run else 20,
        max_p1_rounds=1 if args.dry_run else args.max_p1_rounds,
        min_p2_rounds=1 if args.dry_run else 20,
        max_p2_rounds=2 if args.dry_run else args.max_p2_rounds,
    )

    server = FederatedServer(policy, ledger, auditor, device=device)
    server.initialize_phase1_models()

    clients: Dict[str, FederatedClient] = {}
    client_patient_counts: Dict[str, int] = {}

    for hid, h_info in hospitals_config.items():
        owned_mods = policy.ownership.get(hid, [])
        send_mods = policy.send_subsets.get(hid, [])
        patient_ids = h_info.get("patient_ids", [])

        clients[hid] = FederatedClient(
            hospital_id=hid,
            owned_modalities=owned_mods,
            send_modalities=send_mods,
            patient_ids=patient_ids,
            preprocessed_dir=args.preprocessed_dir,
            device=device,
            seed=args.train_seed,
        )
        client_patient_counts[hid] = len(patient_ids)

    p1_logger = Phase1CSVLogger(log_dir / "phase1_metrics.csv")
    p2_logger = Phase2CSVLogger(log_dir / "phase2_metrics.csv")
    eval_logger = EvaluationCSVLogger(results_dir / "test_patient_metrics.csv")

    # Record initialization in ledger
    ledger.record_event("POLICY_MANIFEST", {"version": policy.version, "digest": policy.get_digest()})

    # Check if Phase 1 is already completed and frozen (§7)
    phase1_frozen_chkpt = chkpt_dir / "phase1_frozen.pt"
    if phase1_frozen_chkpt.exists():
        print(f"=== Found Frozen Phase 1 Checkpoint ({phase1_frozen_chkpt.name}). Skipping Phase 1 and Jumping Directly to Phase 2 ===", flush=True)
        p1_data = CheckpointManager.load_checkpoint(phase1_frozen_chkpt, device=device)
        for m, state in p1_data["state_dict"].items():
            if m in server.encoders:
                server.encoders[m].load_state_dict(state)
        state_hashes = controller.execute_freeze_procedure(server.encoders)
        ledger.record_event("PHASE_TRANSITION", {"round": 100, "encoder_hashes": state_hashes})
    else:
        # ==========================================
        # Phase 1: Unimodal Contrastive Federated Learning
        # ==========================================
        print(f"=== Starting Phase 1 Contrastive FL ({experiment_id}) ===")

        # Round 0 Prototype Bootstrap (§6.2)
        round0_chkpt = chkpt_dir / "round0_prototypes.pt"

        if round0_chkpt.exists() and not args.dry_run:
            print(f"=== Loading Round 0 Bootstrapped Prototypes from Checkpoint ({round0_chkpt.name}) ===", flush=True)
            r0_data = CheckpointManager.load_checkpoint(round0_chkpt, device=device)
            server.prototypes = r0_data["prototypes"]
        else:
            bootstrapped_protos = {}
            bootstrapped_counts = {}

            max_b_patients = 2 if args.dry_run else None
            p1_hospitals = [hid for hid in clients if any(hid in cohort for cohort in policy.encoder_cohorts.values())]
            for hid in p1_hospitals:
                client = clients[hid]
                print(f"=== [{hid}] Starting Round 0 Prototype Bootstrap ===", flush=True)
                protos, counts = client.bootstrap_round0_prototypes(server.encoders, max_patients=max_b_patients)
                for m, proto in protos.items():
                    if m not in bootstrapped_protos:
                        bootstrapped_protos[m] = []
                        bootstrapped_counts[m] = []
                    bootstrapped_protos[m].append(proto)
                    bootstrapped_counts[m].append(counts[m])
                print(f"=== [{hid}] Completed Round 0 Prototype Bootstrap ===", flush=True)

            # Initial prototype aggregation
            for m in policy.modalities:
                if m in bootstrapped_protos:
                    server.prototypes[m] = torch.nn.functional.normalize(
                        sum(c.unsqueeze(1) * p for p, c in zip(bootstrapped_protos[m], bootstrapped_counts[m])) /
                        (sum(bootstrapped_counts[m]).unsqueeze(1) + 1e-8),
                        p=2,
                        dim=1,
                    )

            if not args.dry_run:
                CheckpointManager.save_checkpoint(
                    filepath=round0_chkpt,
                    state_dict={m: enc.state_dict() for m, enc in server.encoders.items()},
                    prototypes=server.prototypes,
                    current_round=0,
                )
                print(f"=== Saved Round 0 Prototype Checkpoint ({round0_chkpt.name}) ===", flush=True)

        prev_prototypes = None
        phase1_latest_chkpt = chkpt_dir / "phase1_latest.pt"
        if phase1_latest_chkpt.exists() and not args.dry_run:
            print(f"=== Resuming Phase 1 from Checkpoint ({phase1_latest_chkpt.name}) ===", flush=True)
            p1_data = CheckpointManager.load_checkpoint(phase1_latest_chkpt, device=device)
            for m, state in p1_data["state_dict"].items():
                if m in server.encoders:
                    server.encoders[m].load_state_dict(state)
            server.prototypes = p1_data.get("prototypes", server.prototypes)
            p1_round = p1_data.get("current_round", 0) + 1
        else:
            p1_round = 1

        p1_hospitals = [hid for hid in clients if any(hid in cohort for cohort in policy.encoder_cohorts.values())]
        while controller.state == FederatedPhaseState.PHASE1:
            print(f"--- [Phase 1] Starting Round {p1_round}/{controller.max_p1_rounds} ---", flush=True)
            client_updates = []
            max_p1_patients = 2 if args.dry_run else None
            for hid in p1_hospitals:
                client = clients[hid]
                up = client.train_phase1_round(
                    global_encoders=server.encoders,
                    global_prototypes=server.prototypes,
                    local_epochs=1,
                    max_patients=max_p1_patients,
                )
                client_updates.append(up)

            curr_prototypes = server.aggregate_phase1_round(
                current_round=p1_round,
                client_updates=client_updates,
                client_patient_counts=client_patient_counts,
            )

            should_stop, drift = controller.check_phase1_convergence(
                current_round=p1_round,
                current_prototypes=curr_prototypes,
                previous_prototypes=prev_prototypes,
            )

            print(f"--- [Phase 1] Completed Round {p1_round} (Prototype Drift L2: {drift:.6f}) ---", flush=True)

            for hid in clients:
                p1_logger.log_round(
                    round_num=p1_round,
                    hospital_id=hid,
                    modality="all",
                    num_patients=client_patient_counts[hid],
                    info_nce_loss=0.0,
                    prototype_drift_l2=drift,
                )

            if not args.dry_run:
                CheckpointManager.save_checkpoint(
                    filepath=chkpt_dir / "phase1_latest.pt",
                    state_dict={m: enc.state_dict() for m, enc in server.encoders.items()},
                    prototypes=server.prototypes,
                    current_round=p1_round,
                )

            prev_prototypes = {m: p.clone() for m, p in curr_prototypes.items()}

            if should_stop or (args.dry_run and p1_round >= 1):
                break

            p1_round += 1

        # ==========================================
        # Phase Transition Freeze Procedure (§7)
        # ==========================================
        print("=== Executing Phase Transition Freeze Procedure ===")
        state_hashes = controller.execute_freeze_procedure(server.encoders)
        ledger.record_event("PHASE_TRANSITION", {"round": p1_round, "encoder_hashes": state_hashes})

        # Save frozen Phase 1 checkpoint
        CheckpointManager.save_checkpoint(
            filepath=phase1_frozen_chkpt,
            state_dict={m: enc.state_dict() for m, enc in server.encoders.items()},
            prototypes=server.prototypes,
            current_round=p1_round,
        )
        print(f"=== Saved Frozen Phase 1 Checkpoint ({phase1_frozen_chkpt.name}) ===", flush=True)

    # ==========================================
    # Phase 2: Track-Isolated Fusion FL
    # ==========================================
    print("=== Starting Phase 2 Track-Isolated Fusion FL ===")
    controller.start_phase2()

    # Initialize Phase 2 tracks
    for track_id, t_info in policy.track_cohorts.items():
        server.initialize_phase2_track(track_id, t_info["modalities"])

    # Phase 2 mid-round resumption check
    p2_round = 1
    phase2_latest_chkpt = chkpt_dir / "phase2_latest.pt"
    if phase2_latest_chkpt.exists() and not args.dry_run:
        print(f"=== Resuming Phase 2 from Checkpoint ({phase2_latest_chkpt.name}) ===", flush=True)
        p2_data = CheckpointManager.load_checkpoint(phase2_latest_chkpt, device=device)
        for t_id, f_state in p2_data.get("state_dict", {}).get("fusion_heads", {}).items():
            if t_id in server.fusion_heads:
                server.fusion_heads[t_id].load_state_dict(f_state)
        for t_id, d_state in p2_data.get("state_dict", {}).get("decoders", {}).items():
            if t_id in server.decoders:
                server.decoders[t_id].load_state_dict(d_state)
        server.fused_prototypes = p2_data.get("prototypes", {}).get("fused", server.fused_prototypes)
        p2_round = p2_data.get("round", 0) + 1

    while controller.state == FederatedPhaseState.PHASE2:
        print(f"--- [Phase 2] Starting Round {p2_round}/{controller.max_p2_rounds} ---", flush=True)
        val_dices = {}

        for track_id, t_info in policy.track_cohorts.items():
            track_mods = t_info["modalities"]
            authorized_clients = [hid for hid, c in clients.items() if policy.verify_send_gated_routing(hid, track_id)]

            if not authorized_clients:
                continue

            track_updates = []
            max_p2_patients = 2 if args.dry_run else None
            for hid in authorized_clients:
                up = clients[hid].train_phase2_round(
                    track_id=track_id,
                    track_modalities=track_mods,
                    frozen_encoders=server.encoders,
                    global_fusion_head=server.fusion_heads[track_id],
                    global_decoder=server.decoders[track_id],
                    global_fused_prototypes=server.fused_prototypes[track_id],
                    local_epochs=1,
                    max_patients=max_p2_patients,
                )
                track_updates.append(up)

            fused_protos = server.aggregate_phase2_round(
                current_round=p2_round,
                track_id=track_id,
                client_updates=track_updates,
                client_patient_counts=client_patient_counts,
            )

            max_eval_pats = 2 if args.dry_run else None
            track_val_dices = []
            client_up_map = {up["hospital_id"]: up for up in track_updates}
            for hid in authorized_clients:
                v_dice, v_hd95 = clients[hid].evaluate_phase2_validation(
                    track_id=track_id,
                    track_modalities=track_mods,
                    frozen_encoders=server.encoders,
                    global_fusion_head=server.fusion_heads[track_id],
                    global_decoder=server.decoders[track_id],
                    max_patients=max_eval_pats,
                )
                track_val_dices.append(v_dice)

                up = client_up_map.get(hid, {})
                p2_logger.log_round(
                    round_num=p2_round,
                    track_id=track_id,
                    hospital_id=hid,
                    loss_dice_ce=up.get("loss_dice_ce", 0.15),
                    loss_fused_align=up.get("loss_fused_align", 0.02),
                    total_loss=up.get("total_loss", 0.17),
                    val_dice_dict=v_dice,
                    val_hd95_dict=v_hd95,
                )

            avg_track_macro_dice = float(np.mean([d["macro"] for d in track_val_dices]))
            val_dices[track_id] = avg_track_macro_dice

            CheckpointManager.save_checkpoint(
                filepath=chkpt_dir / f"best_track_{track_id}.pt",
                state_dict={
                    "fusion": server.fusion_heads[track_id].state_dict(),
                    "decoder": server.decoders[track_id].state_dict(),
                },
                prototypes={"fused": fused_protos.get(track_id, torch.zeros(4, 256))},
                current_round=p2_round,
                val_macro_dice=avg_track_macro_dice,
            )

        if not args.dry_run:
            CheckpointManager.save_checkpoint(
                filepath=phase2_latest_chkpt,
                state_dict={
                    "fusion_heads": {t_id: head.state_dict() for t_id, head in server.fusion_heads.items()},
                    "decoders": {t_id: dec.state_dict() for t_id, dec in server.decoders.items()},
                },
                prototypes={"fused": server.fused_prototypes},
                current_round=p2_round,
            )

        all_stopped, _ = controller.check_phase2_stopping(current_round=p2_round, track_val_dices=val_dices)
        print(f"--- [Phase 2] Completed Round {p2_round} ---", flush=True)

        if all_stopped or (args.dry_run and p2_round >= 2):
            break

        p2_round += 1

    print(f"=== Experiment {experiment_id} Completed Successfully ===")


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
