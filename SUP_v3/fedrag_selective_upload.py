from __future__ import annotations

import copy
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping

import torch
from transformers import BertTokenizer

try:
    from flgo.algorithm.fedbase import BasicClient, BasicServer
except ImportError:
    from FedE.flgo.algorithm.fedbase import BasicClient, BasicServer

try:
    from .hypernetwork import (
        BlockImportanceHyperNetwork,
        block_delta_stats,
        block_parameter_counts,
        build_block_map,
        clone_state_dict,
        estimate_payload_ratio,
        filter_delta_by_blocks,
        stats_to_features,
        subtract_state_dict,
        supervised_hn_update,
    )
    from .metrics import append_jsonl, ensure_dir, estimate_encrypted_bytes, write_csv, write_json
    from .upload_selectors import UploadSelector
except ImportError:
    from hypernetwork import (
        BlockImportanceHyperNetwork,
        block_delta_stats,
        block_parameter_counts,
        build_block_map,
        clone_state_dict,
        estimate_payload_ratio,
        filter_delta_by_blocks,
        stats_to_features,
        subtract_state_dict,
        supervised_hn_update,
    )
    from metrics import append_jsonl, ensure_dir, estimate_encrypted_bytes, write_csv, write_json
    from upload_selectors import UploadSelector


class Server(BasicServer):
    """FedRAG server with configurable selective-upload strategies."""

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.selection_strategy = self.option.get("selection_strategy", "hypernet")
        self.block_strategy = self.option.get("selective_block_strategy", "bert")
        self.topk_blocks = int(self.option.get("selective_topk_blocks", 3))
        self.warmup_rounds = int(self.option.get("selective_warmup_rounds", 2))
        self.always_upload = list(self.option.get("selective_always_upload", []))
        self.hn_lr = float(self.option.get("hn_lr", 5e-3))
        self.hn_embedding_dim = int(self.option.get("hn_embedding_dim", 64))
        self.hn_hidden_dim = int(self.option.get("hn_hidden_dim", 128))
        self.estimate_encryption = bool(self.option.get("estimate_encryption", False))
        self.encryption_expansion = float(self.option.get("encryption_expansion", 8.0))
        self.output_dir = ensure_dir(self.option.get("output_dir", "SUP_v2/outputs/manual_run"))

        self.block_map = build_block_map(self.model.state_dict(), self.block_strategy)
        self.block_names = list(self.block_map.keys())
        self.block_param_counts = block_parameter_counts(self.block_map, self.model.state_dict())
        self.total_trainable_params = sum(self.block_param_counts.values())
        self.client_last_stats: Dict[int, Dict[str, Dict[str, float]]] = {
            cid: {} for cid in range(self.num_clients)
        }
        self.selector = UploadSelector(
            strategy=self.selection_strategy,
            block_names=self.block_names,
            topk=self.topk_blocks,
            always_upload=self.always_upload,
            seed=int(self.option.get("seed", 0)),
        )

        self.hypernet = None
        self.hn_optimizer = None
        if self.selection_strategy == "hypernet":
            self.hypernet = BlockImportanceHyperNetwork(
                client_num=self.num_clients,
                block_names=self.block_names,
                embedding_dim=self.hn_embedding_dim,
                hidden_dim=self.hn_hidden_dim,
            ).to(self.device)
            self.hn_optimizer = torch.optim.Adam(self.hypernet.parameters(), lr=self.hn_lr)

        self.round_records = []
        self._write_run_metadata()

    def iterate(self):
        self.selected_clients = self.sample()
        client_updates = self.communicate(self.selected_clients)
        self.model = self.aggregate(self.model, client_updates)
        return len(client_updates.get("delta", [])) > 0

    def pack(self, client_id, mtype=0, *args, **kwargs):
        selection = self._select_upload_blocks(client_id)
        return {
            "model": copy.deepcopy(self.model),
            "upload_blocks": selection.upload_blocks,
            "selection_scores": selection.scores,
            "selection_strategy": selection.strategy,
            "block_map": self.block_map,
            "client_id": client_id,
            "__mtype__": mtype,
        }

    def unpack(self, packages_received_from_clients):
        res = {
            "delta": [],
            "stats": [],
            "client_id": [],
            "num_samples": [],
            "upload_blocks": [],
            "payload_ratio": [],
            "uploaded_params": [],
            "full_params": [],
            "encrypted_bytes": [],
            "selection_scores": [],
            "hn_loss": [],
        }
        for pkg in packages_received_from_clients:
            if pkg is None:
                continue
            client_id = int(pkg.get("client_id", pkg.get("__cid", -1)))
            stats = pkg.get("block_stats", pkg.get("stats", {}))
            # FLGo's simulator decorators may call `unpack` twice:
            # first on raw client replies, then again on clock-repacked packages.
            # Reuse the previously computed loss on the second pass.
            hn_loss = float(pkg.get("hn_loss", 0.0))
            if "block_stats" in pkg and self.hypernet is not None and self.hn_optimizer is not None:
                hn_loss = supervised_hn_update(
                    self.hypernet,
                    self.hn_optimizer,
                    client_id,
                    self.block_names,
                    stats,
                )
            self.client_last_stats[client_id] = stats
            for key in [
                "delta",
                "client_id",
                "num_samples",
                "upload_blocks",
                "payload_ratio",
                "uploaded_params",
                "full_params",
                "encrypted_bytes",
                "selection_scores",
            ]:
                if key == "client_id":
                    res[key].append(client_id)
                else:
                    res[key].append(pkg.get(key))
            res["stats"].append(stats)
            res["hn_loss"].append(hn_loss)
        return res

    def aggregate(self, model_old, client_updates: Mapping[str, List], *args, **kwargs):
        deltas = client_updates.get("delta", [])
        if not deltas:
            return model_old

        current_state = clone_state_dict(model_old.state_dict(), device="cpu")
        total_by_param = {
            name: torch.zeros_like(value, device="cpu")
            for name, value in current_state.items()
            if torch.is_tensor(value) and torch.is_floating_point(value)
        }
        weight_by_param = {name: 0.0 for name in total_by_param}

        for delta, sample_count in zip(deltas, client_updates.get("num_samples", [])):
            weight = float(sample_count)
            for name, value in delta.items():
                if name not in total_by_param:
                    continue
                total_by_param[name] += value.cpu() * weight
                weight_by_param[name] += weight

        updated_state = OrderedDict()
        for name, value in current_state.items():
            if name in total_by_param and weight_by_param[name] > 0:
                updated_state[name] = value + total_by_param[name] / weight_by_param[name]
            else:
                updated_state[name] = value
        model_old.load_state_dict(updated_state, strict=False)
        self._log_round(client_updates)
        return model_old

    def run(self):
        self.gv.logger.time_start("Total Time Cost")
        if not self._load_checkpoint() and self.eval_interval > 0:
            self.gv.logger.info("--------------Initial Evaluation--------------")
        while True:
            if self._if_exit():
                break
            self.gv.clock.step()
            updated = self.iterate()
            if updated is True or updated is None:
                self.gv.logger.info("--------------Round {}--------------".format(self.current_round))
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self._save_checkpoint()
                if self.gv.logger.early_stop():
                    break
                self.current_round += 1
                self.global_lr_scheduler(self.current_round)

        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end("Total Time Cost")
        self.gv.logger.save_output_as_json()
        self._save_artifacts()

    def _select_upload_blocks(self, client_id: int):
        if self.current_round <= self.warmup_rounds:
            return self.selector.select(client_id, self.current_round) if self.selection_strategy == "full" else self._full_selection()
        hypernet_scores = None
        if self.hypernet is not None:
            features = stats_to_features(self.block_names, self.client_last_stats.get(client_id, {})).to(self.device)
            hypernet_scores = self.hypernet.score_blocks(client_id, self.block_names, features)
        return self.selector.select(
            client_id=client_id,
            current_round=self.current_round,
            last_stats=self.client_last_stats.get(client_id, {}),
            hypernet_scores=hypernet_scores,
        )

    def _full_selection(self):
        class _FullSelection:
            strategy = "warmup_full"
            upload_blocks = ["__ALL__"]
            scores = {}

        return _FullSelection()

    def _log_round(self, client_updates: Mapping[str, List]) -> None:
        ratios = client_updates.get("payload_ratio", [])
        hn_losses = client_updates.get("hn_loss", [])
        uploaded_params = client_updates.get("uploaded_params", [])
        encrypted_bytes = client_updates.get("encrypted_bytes", [])
        record = {
            "round": int(self.current_round),
            "strategy": self.selection_strategy,
            "topk_blocks": self.topk_blocks,
            "warmup_rounds": self.warmup_rounds,
            "selected_clients": list(map(int, client_updates.get("client_id", []))),
            "avg_payload_ratio": sum(ratios) / max(len(ratios), 1),
            "total_uploaded_params": int(sum(uploaded_params)),
            "avg_uploaded_params": sum(uploaded_params) / max(len(uploaded_params), 1),
            "total_full_params": int(self.total_trainable_params * max(len(uploaded_params), 1)),
            "avg_hn_loss": sum(hn_losses) / max(len(hn_losses), 1),
            "total_encrypted_bytes_est": float(sum(encrypted_bytes)),
            "upload_blocks": client_updates.get("upload_blocks", []),
        }
        self.round_records.append(record)
        append_jsonl(Path(self.output_dir) / "round_logs.jsonl", record)
        self.gv.logger.info(
            "SUP_v2 round {} strategy={} avg_payload={:.4f} uploaded_params={}".format(
                record["round"],
                record["strategy"],
                record["avg_payload_ratio"],
                record["total_uploaded_params"],
            )
        )

    def _write_run_metadata(self) -> None:
        metadata = {
            "selection_strategy": self.selection_strategy,
            "topk_blocks": self.topk_blocks,
            "warmup_rounds": self.warmup_rounds,
            "block_strategy": self.block_strategy,
            "block_names": self.block_names,
            "block_param_counts": self.block_param_counts,
            "total_trainable_params": self.total_trainable_params,
            "estimate_encryption": self.estimate_encryption,
            "encryption_expansion": self.encryption_expansion,
            "option": {key: str(value) for key, value in self.option.items()},
        }
        write_json(Path(self.output_dir) / "run_metadata.json", metadata)

    def _save_artifacts(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = Path(self.output_dir) / f"retriever_state_{timestamp}.bin"
        torch.save(self.model.state_dict(), model_path)
        hf_model_dir = Path(self.output_dir) / f"retriever_hf_{timestamp}"
        hf_model_dir_saved = False
        if hasattr(self.model, "model") and hasattr(self.model.model, "save_pretrained"):
            hf_model_dir.mkdir(parents=True, exist_ok=True)
            self.model.model.save_pretrained(hf_model_dir)
            embedding_model = os.environ.get("FEDE_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            try:
                BertTokenizer.from_pretrained(embedding_model).save_pretrained(hf_model_dir)
            except Exception as exc:
                write_json(hf_model_dir / "tokenizer_save_warning.json", {"warning": str(exc)})
            hf_model_dir_saved = True
        write_json(Path(self.output_dir) / "round_logs.json", self.round_records)
        write_csv(Path(self.output_dir) / "round_logs.csv", self.round_records)
        if self.hypernet is not None:
            torch.save(
                {
                    "hypernet": self.hypernet.state_dict(),
                    "block_names": self.block_names,
                    "round_records": self.round_records,
                },
                Path(self.output_dir) / f"hypernet_{timestamp}.pt",
            )
        write_json(
            Path(self.output_dir) / "final_artifacts.json",
            {
                "model_state_path": str(model_path),
                "hf_model_dir": str(hf_model_dir) if hf_model_dir_saved else "",
            },
        )


class Client(BasicClient):
    def train(self, teacher_model, local_model):
        teacher_model.eval()
        local_model.train()
        optimizer = self.calculator.get_optimizer(
            local_model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )
        teacher_model.to(self.device)
        local_model.to(self.device)
        for _ in range(self.num_steps):
            batch_data = self.get_batch_data()
            local_model.zero_grad()
            server_logits = self.calculator.compute_server_loss(teacher_model, batch_data)
            loss, _, _ = self.calculator.compute_client_loss(server_logits, local_model, batch_data)
            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(parameters=local_model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        return local_model

    def unpack(self, received_pkg):
        return (
            received_pkg["model"],
            received_pkg["upload_blocks"],
            received_pkg["block_map"],
            int(received_pkg["client_id"]),
            received_pkg.get("selection_scores", {}),
            received_pkg.get("selection_strategy", ""),
        )

    def reply(self, svr_pkg):
        teacher_model, upload_blocks, block_map, client_id, selection_scores, selection_strategy = self.unpack(svr_pkg)
        self.client_id = client_id
        local_model = copy.deepcopy(teacher_model)
        initial_state = clone_state_dict(local_model.state_dict(), device="cpu")
        self.train(teacher_model, local_model)
        final_state = clone_state_dict(local_model.state_dict(), device="cpu")
        delta = subtract_state_dict(final_state, initial_state, device="cpu")
        stats = block_delta_stats(delta, block_map)
        sparse_delta = filter_delta_by_blocks(delta, block_map, upload_blocks)
        payload_ratio = estimate_payload_ratio(block_map, final_state, upload_blocks)
        full_params = sum(block_parameter_counts(block_map, final_state).values())
        uploaded_params = int(round(full_params * payload_ratio))
        encrypted_bytes = 0.0
        if bool(self.option.get("estimate_encryption", False)):
            encrypted_bytes = estimate_encrypted_bytes(
                uploaded_params,
                encryption_expansion=float(self.option.get("encryption_expansion", 8.0)),
            )
        return {
            "client_id": client_id,
            "delta": sparse_delta,
            "block_stats": stats,
            "upload_blocks": upload_blocks,
            "payload_ratio": payload_ratio,
            "uploaded_params": uploaded_params,
            "full_params": full_params,
            "encrypted_bytes": encrypted_bytes,
            "selection_scores": selection_scores,
            "selection_strategy": selection_strategy,
            "num_samples": getattr(self, "datavol", 1),
        }
