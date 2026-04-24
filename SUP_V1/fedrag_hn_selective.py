from __future__ import annotations

import copy
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Iterable, List, Mapping

import torch

try:
    from FedE.flgo.algorithm.fedbase import BasicClient, BasicServer
except ImportError:
    try:
        from flgo.algorithm.fedbase import BasicClient, BasicServer
    except ImportError:
        from .fedbase import BasicClient, BasicServer

try:
    from .hypernetwork import (
        BlockImportanceHyperNetwork,
        block_delta_stats,
        build_block_map,
        clone_state_dict,
        estimate_payload_ratio,
        filter_delta_by_blocks,
        select_topk_blocks,
        stats_to_features,
        subtract_state_dict,
        supervised_hn_update,
    )
except ImportError:
    from hypernetwork import (
        BlockImportanceHyperNetwork,
        block_delta_stats,
        build_block_map,
        clone_state_dict,
        estimate_payload_ratio,
        filter_delta_by_blocks,
        select_topk_blocks,
        stats_to_features,
        subtract_state_dict,
        supervised_hn_update,
    )


class Server(BasicServer):
    """FedRAG with hypernetwork-guided selective delta upload.

    This V1 server keeps the original FedRAG local objective but changes the
    communication contract: clients upload only selected block deltas plus tiny
    block-level statistics for hypernetwork training.
    """

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.block_strategy = self.option.get("selective_block_strategy", "bert")
        self.selective_topk_blocks = int(self.option.get("selective_topk_blocks", 3))
        self.selective_warmup_rounds = int(self.option.get("selective_warmup_rounds", 1))
        self.selective_min_score = float(self.option.get("selective_min_score", 0.0))
        self.selective_always_upload = list(
            self.option.get("selective_always_upload", [])
        )
        self.hn_lr = float(self.option.get("hn_lr", 5e-3))
        self.hn_embedding_dim = int(self.option.get("hn_embedding_dim", 64))
        self.hn_hidden_dim = int(self.option.get("hn_hidden_dim", 128))

        self.block_map = build_block_map(self.model.state_dict(), self.block_strategy)
        self.block_names = list(self.block_map.keys())
        self.client_last_stats: Dict[int, Dict[str, Dict[str, float]]] = {
            cid: {} for cid in range(self.num_clients)
        }
        self.client_last_upload_blocks: Dict[int, List[str]] = {
            cid: ["__ALL__"] for cid in range(self.num_clients)
        }
        self.hypernet = BlockImportanceHyperNetwork(
            client_num=self.num_clients,
            block_names=self.block_names,
            embedding_dim=self.hn_embedding_dim,
            hidden_dim=self.hn_hidden_dim,
        ).to(self.device)
        self.hn_optimizer = torch.optim.Adam(self.hypernet.parameters(), lr=self.hn_lr)
        self.communication_log = []

    def pack(self, client_id, mtype=0, *args, **kwargs):
        upload_blocks = self._choose_upload_blocks(client_id)
        self.client_last_upload_blocks[client_id] = upload_blocks
        return {
            "model": copy.deepcopy(self.model),
            "upload_blocks": upload_blocks,
            "block_map": self.block_map,
            "client_id": client_id,
            "__mtype__": mtype,
        }

    def iterate(self):
        self.selected_clients = self.sample()
        client_updates = self.communicate(self.selected_clients)
        self.model = self.aggregate(self.model, client_updates)
        return len(client_updates.get("delta", [])) > 0

    def unpack(self, packages_received_from_clients):
        if len(packages_received_from_clients) == 0:
            return {
                "delta": [],
                "stats": [],
                "client_id": [],
                "num_samples": [],
                "upload_blocks": [],
                "payload_ratio": [],
                "hn_loss": [],
            }
        res = {
            "delta": [],
            "stats": [],
            "client_id": [],
            "num_samples": [],
            "upload_blocks": [],
            "payload_ratio": [],
            "hn_loss": [],
        }
        for pkg in packages_received_from_clients:
            client_id = int(pkg["client_id"])
            stats = pkg["block_stats"]
            hn_loss = supervised_hn_update(
                self.hypernet,
                self.hn_optimizer,
                client_id,
                self.block_names,
                stats,
            )
            self.client_last_stats[client_id] = stats
            for key in ["delta", "client_id", "num_samples", "upload_blocks", "payload_ratio"]:
                res[key].append(pkg[key])
            res["stats"].append(stats)
            res["hn_loss"].append(hn_loss)
        return res

    def aggregate(self, model_old, client_updates: Mapping[str, List], *args, **kwargs):
        deltas = client_updates.get("delta", [])
        client_ids = client_updates.get("client_id", [])
        num_samples = client_updates.get("num_samples", [])
        if len(deltas) == 0:
            return model_old

        current_state = clone_state_dict(model_old.state_dict(), device="cpu")
        total_by_param = {
            name: torch.zeros_like(value, device="cpu")
            for name, value in current_state.items()
            if torch.is_tensor(value) and torch.is_floating_point(value)
        }
        weight_by_param = {name: 0.0 for name in total_by_param}

        for delta, weight in zip(deltas, num_samples):
            sample_weight = float(weight)
            for name, tensor_delta in delta.items():
                if name not in total_by_param:
                    continue
                total_by_param[name] += tensor_delta.cpu() * sample_weight
                weight_by_param[name] += sample_weight

        updated_state = OrderedDict()
        for name, value in current_state.items():
            if name in total_by_param and weight_by_param[name] > 0:
                updated_state[name] = value + total_by_param[name] / weight_by_param[name]
            else:
                updated_state[name] = value

        model_old.load_state_dict(updated_state, strict=False)
        avg_ratio = sum(client_updates.get("payload_ratio", [1.0])) / max(len(deltas), 1)
        avg_hn_loss = sum(client_updates.get("hn_loss", [0.0])) / max(len(deltas), 1)
        self.communication_log.append(
            {
                "round": self.current_round,
                "clients": client_ids,
                "avg_payload_ratio": avg_ratio,
                "avg_hn_loss": avg_hn_loss,
                "upload_blocks": client_updates.get("upload_blocks", []),
            }
        )
        self.gv.logger.info(
            "SUP_V1 round {} avg payload ratio {:.4f}, hn loss {:.6f}".format(
                self.current_round,
                avg_ratio,
                avg_hn_loss,
            )
        )
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
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(self.model.state_dict(), f"sup_v1_model_{timestamp}.bin")
        torch.save(
            {
                "hypernet": self.hypernet.state_dict(),
                "block_names": self.block_names,
                "communication_log": self.communication_log,
            },
            f"sup_v1_hypernet_{timestamp}.pt",
        )

    def _choose_upload_blocks(self, client_id: int) -> List[str]:
        if self.current_round <= self.selective_warmup_rounds:
            return ["__ALL__"]
        features = stats_to_features(
            self.block_names,
            self.client_last_stats.get(client_id, {}),
        ).to(self.device)
        scores = self.hypernet.score_blocks(client_id, self.block_names, features)
        return select_topk_blocks(
            scores,
            topk=self.selective_topk_blocks,
            min_score=self.selective_min_score,
            always_upload=self.selective_always_upload,
        )


class Client(BasicClient):
    def train(self, teacher_model, local_model):
        local_model.train()
        teacher_model.eval()
        optimizer = self.calculator.get_optimizer(
            local_model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )
        teacher_model.to(self.device)
        local_model.to(self.device)

        for iter_id in range(self.num_steps):
            batch_data = self.get_batch_data()
            local_model.zero_grad()
            server_loss = self.calculator.compute_server_loss(teacher_model, batch_data)
            client_loss, client_only, server_only = self.calculator.compute_client_loss(
                server_loss,
                local_model,
                batch_data,
            )
            client_loss.backward()
            optimizer.step()
            if iter_id == self.num_steps - 1:
                print(
                    "SUP_V1 client {}, loss {}, contrastive {}, distill {}".format(
                        self.client_id,
                        client_loss,
                        client_only,
                        server_only,
                    )
                )
        return local_model

    def unpack(self, received_pkg):
        return (
            received_pkg["model"],
            received_pkg["upload_blocks"],
            received_pkg["block_map"],
            received_pkg["client_id"],
        )

    def reply(self, svr_pkg):
        teacher_model, upload_blocks, block_map, client_id = self.unpack(svr_pkg)
        self.client_id = client_id

        local_model = copy.deepcopy(teacher_model)
        initial_state = clone_state_dict(local_model.state_dict(), device="cpu")
        self.train(teacher_model, local_model)
        final_state = clone_state_dict(local_model.state_dict(), device="cpu")
        delta = subtract_state_dict(final_state, initial_state, device="cpu")
        stats = block_delta_stats(delta, block_map)
        sparse_delta = filter_delta_by_blocks(delta, block_map, upload_blocks)
        payload_ratio = estimate_payload_ratio(block_map, final_state, upload_blocks)
        return {
            "client_id": client_id,
            "delta": sparse_delta,
            "block_stats": stats,
            "upload_blocks": upload_blocks,
            "payload_ratio": payload_ratio,
            "num_samples": getattr(self, "datavol", 1),
        }
