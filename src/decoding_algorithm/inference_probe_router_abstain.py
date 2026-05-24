import torch

from .inference import Inference as BaseInference


class ProbeDecisionAbstainLayer(torch.nn.Module):
    def __init__(self, probe_payload, shared_state, router_stats, lower_threshold=0.45, upper_threshold=0.55):
        super().__init__()
        self.weight = probe_payload["weight"].float()
        self.bias = probe_payload["bias"].float()
        self.mean = probe_payload["mean"].float()
        self.std = probe_payload["std"].float()
        self.pooling = probe_payload.get("config", {}).get("pooling", "mean_tokens")
        self.shared_state = shared_state
        self.router_stats = router_stats
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def _pool(self, x):
        if self.pooling == "last_token":
            return x[:, -1, :]
        return x.mean(dim=1)

    def forward(self, x):
        input_dtype = x.dtype
        feat = self._pool(x).float()
        mean = self.mean.to(device=x.device, dtype=torch.float32)
        std = self.std.to(device=x.device, dtype=torch.float32)
        weight = self.weight.to(device=x.device, dtype=torch.float32)
        bias = self.bias.to(device=x.device, dtype=torch.float32)

        feat = (feat - mean.unsqueeze(0)) / std.unsqueeze(0)
        logits = torch.matmul(feat, weight.unsqueeze(1)).squeeze(-1) + bias
        probs = torch.sigmoid(logits)

        use_gevd = probs > self.upper_threshold
        use_svd = probs < self.lower_threshold
        use_bypass = (~use_gevd) & (~use_svd)

        self.shared_state["use_gevd"] = use_gevd.detach()
        self.shared_state["use_svd"] = use_svd.detach()
        self.shared_state["use_bypass"] = use_bypass.detach()
        self.shared_state["probs"] = probs.detach()

        probs_cpu = probs.detach().float().cpu()
        self.router_stats["num_decisions"] += int(probs_cpu.numel())
        self.router_stats["prob_values"].extend(probs_cpu.tolist())
        self.router_stats["gevd"] += int(use_gevd.sum().item())
        self.router_stats["svd"] += int(use_svd.sum().item())
        self.router_stats["bypass"] += int(use_bypass.sum().item())

        return x.to(input_dtype)


class ProbeAbstainAdapterLayer(torch.nn.Module):
    def __init__(
        self,
        svd_positive_projection,
        svd_negative_projection,
        gevd_positive_projection,
        gevd_negative_projection,
        combine_sea_embeddings,
        feature_function,
        shared_state,
    ):
        super().__init__()
        self.svd_positive_projection = svd_positive_projection
        self.svd_negative_projection = svd_negative_projection
        self.gevd_positive_projection = gevd_positive_projection
        self.gevd_negative_projection = gevd_negative_projection
        self.combine_sea_embeddings = combine_sea_embeddings
        self.feature_function = feature_function
        self.shared_state = shared_state

    def non_linear_feature_func(self, X, func="squared-exponential"):
        if func == "squared-exponential":
            length_scale = 1
            return torch.exp(-1 * X**2 / (2 * length_scale**2))
        if func == "tanh":
            return torch.tanh(X)
        if func == "elu":
            positive_X = X * (X >= 0)
            negative_X = (torch.exp(X) - 1) * (X < 0)
            return positive_X + negative_X

    def inv_non_linear_feature_func(self, X, func="squared-exponential"):
        eps = (torch.ones(X.shape) * 1e-4).to(X.device)
        if func == "squared-exponential":
            length_scale = 1
            return -torch.log(torch.max(X, eps)) * 2 * length_scale**2
        if func == "tanh":
            X = torch.min(X, 1 - eps)
            X = torch.max(X, -1 + eps)
            return torch.atanh(X)
        if func == "elu":
            positive_X = X * (X >= 0)
            negative_X = (torch.log(torch.max(X, -1 + eps) + 1)) * (X < 0)
            return positive_X + negative_X

    def _samplewise_project(self, projection, x_per_sample):
        return torch.matmul(projection.unsqueeze(0), x_per_sample)

    def forward(self, x):
        input_dtype = x.dtype
        use_gevd = self.shared_state.get("use_gevd")
        use_svd = self.shared_state.get("use_svd")
        use_bypass = self.shared_state.get("use_bypass")
        if use_gevd is None or use_svd is None or use_bypass is None:
            return x

        svd_positive_projection = self.svd_positive_projection.to(device=x.device, dtype=input_dtype)
        svd_negative_projection = self.svd_negative_projection.to(device=x.device, dtype=input_dtype)
        gevd_positive_projection = self.gevd_positive_projection.to(device=x.device, dtype=input_dtype)
        gevd_negative_projection = self.gevd_negative_projection.to(device=x.device, dtype=input_dtype)

        bs, _, _ = x.size()
        x_per_sample = x.permute(0, 2, 1)
        original_norm = torch.linalg.vector_norm(x_per_sample.float(), dim=1, keepdim=True)

        if self.feature_function:
            x_per_sample = self.non_linear_feature_func(x_per_sample, self.feature_function)

        pos_svd = self._samplewise_project(svd_positive_projection, x_per_sample)
        neg_svd = self._samplewise_project(svd_negative_projection, x_per_sample)
        pos_gevd = self._samplewise_project(gevd_positive_projection, x_per_sample)
        neg_gevd = self._samplewise_project(gevd_negative_projection, x_per_sample)

        pos_x = torch.where(use_gevd.view(bs, 1, 1), pos_gevd, pos_svd)
        neg_x = torch.where(use_gevd.view(bs, 1, 1), neg_gevd, neg_svd)

        if self.feature_function:
            pos_x = self.inv_non_linear_feature_func(pos_x, self.feature_function)
            neg_x = self.inv_non_linear_feature_func(neg_x, self.feature_function)

        if self.combine_sea_embeddings == "l2_norm":
            combined = pos_x + neg_x
            combined_norm = torch.linalg.vector_norm(combined.float(), dim=1, keepdim=True)
            combined = combined * original_norm / (combined_norm + 1e-8)
        elif self.combine_sea_embeddings == "average":
            combined = (pos_x + neg_x) / 2
        else:
            raise ValueError(f"Unsupported combine_sea_embeddings: {self.combine_sea_embeddings}")

        combined = torch.where(use_bypass.view(bs, 1, 1), x_per_sample, combined)
        return combined.permute(0, 2, 1).to(input_dtype)


class model_with_probe_abstain_router(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(
        self,
        svd_projections,
        gevd_projections,
        probe_payload,
        apply_sea_layers=None,
        L=None,
        combine_sea_embeddings="l2_norm",
        feature_function=None,
        shared_state=None,
        router_stats=None,
        lower_threshold=0.45,
        upper_threshold=0.55,
    ):
        svd_positive_proj, svd_negative_proj = svd_projections
        gevd_positive_proj, gevd_negative_proj = gevd_projections

        total_layers = len(self.model.model.layers)
        if apply_sea_layers == "all":
            target_layers = list(range(total_layers))
        elif apply_sea_layers == "first-L":
            target_layers = list(range(0, int(L)))
        elif apply_sea_layers == "last-L":
            target_layers = list(range(total_layers - int(L), total_layers))
        elif apply_sea_layers == "last":
            target_layers = [total_layers - 1]
        elif apply_sea_layers == "specific":
            target_layers = [int(i) for i in L.split(",")]
        else:
            target_layers = []

        if not target_layers:
            print("No target layers selected for probe abstain router.")
            return self.model

        svd_positive_projection = torch.load(svd_positive_proj, map_location="cpu")
        svd_negative_projection = torch.load(svd_negative_proj, map_location="cpu")
        gevd_positive_projection = torch.load(gevd_positive_proj, map_location="cpu")
        gevd_negative_projection = torch.load(gevd_negative_proj, map_location="cpu")

        probe_layer_index = int(probe_payload.get("config", {}).get("layer_index", target_layers[0]))
        edit_layers = list(target_layers)

        print("Probe abstain router assignment:")
        print("  probe layer:", probe_layer_index)
        print("  edit layers:", edit_layers)
        print("  abstain thresholds:", lower_threshold, upper_threshold)

        self.model.model.layers[probe_layer_index].mlp = torch.nn.Sequential(
            self.model.model.layers[probe_layer_index].mlp,
            ProbeDecisionAbstainLayer(
                probe_payload,
                shared_state,
                router_stats,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            ),
        )

        # 将编辑 adapter 挂到每一个选定的目标层。
        # 如果 probe 层本身也在编辑层集合中，实际执行顺序为：
        #   原始 MLP -> ProbeDecisionAbstainLayer -> ProbeAbstainAdapterLayer
        # 因此路由决策读取的仍然是未编辑的 MLP 输出，编辑会在同一层内紧接着应用。
        for edit_layer_index in edit_layers:
            self.model.model.layers[edit_layer_index].mlp = torch.nn.Sequential(
                self.model.model.layers[edit_layer_index].mlp,
                ProbeAbstainAdapterLayer(
                    svd_positive_projection[edit_layer_index].cuda(),
                    svd_negative_projection[edit_layer_index].cuda(),
                    gevd_positive_projection[edit_layer_index].cuda(),
                    gevd_negative_projection[edit_layer_index].cuda(),
                    combine_sea_embeddings,
                    feature_function,
                    shared_state,
                ),
            )
        return self.model


class InferenceProbeRouterAbstain(BaseInference):
    def __init__(
        self,
        model_name,
        lora_name,
        dataset_name,
        device="cuda",
        max_gpu_memory=39,
        amateur_model_name=None,
        num_gpus=-1,
        amateur_model_nums_gpus=-1,
        sea_probe_router=False,
        svd_positive_proj=None,
        svd_negative_proj=None,
        gevd_positive_proj=None,
        gevd_negative_proj=None,
        probe_path=None,
        apply_sea_layers=None,
        L=None,
        combine_sea_embeddings="l2_norm",
        feature_function=None,
        lower_threshold=0.45,
        upper_threshold=0.55,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.amateur_model_name = amateur_model_name
        self.lora_name = lora_name
        self.device = device
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.shared_state = {}
        self.router_stats = {"svd": 0, "gevd": 0, "bypass": 0, "num_decisions": 0, "prob_values": []}

        self.model, self.tokenizer = self.load_model(model_name, num_gpus)

        if amateur_model_name is not None:
            self.amateur_model, self.amateur_model_tokenizer = self.load_model(
                amateur_model_name, amateur_model_nums_gpus, num_gpus
            )

        if sea_probe_router:
            probe_payload = torch.load(probe_path, map_location="cpu")
            updated_wrapper = model_with_probe_abstain_router(self.model)
            _ = updated_wrapper.get_model(
                (svd_positive_proj, svd_negative_proj),
                (gevd_positive_proj, gevd_negative_proj),
                probe_payload,
                apply_sea_layers=apply_sea_layers,
                L=L,
                combine_sea_embeddings=combine_sea_embeddings,
                feature_function=feature_function,
                shared_state=self.shared_state,
                router_stats=self.router_stats,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
            print("Probe abstain router has been added!\n")

        self.all_gpu_nums = num_gpus + amateur_model_nums_gpus
        assert self.all_gpu_nums <= 8

    def summarize_router_stats(self):
        summary = {
            "svd": int(self.router_stats["svd"]),
            "gevd": int(self.router_stats["gevd"]),
            "bypass": int(self.router_stats["bypass"]),
        }
        prob_values = self.router_stats.get("prob_values", [])
        if prob_values:
            tensor = torch.tensor(prob_values, dtype=torch.float32)
            summary["prob_mean"] = float(tensor.mean().item())
            summary["prob_min"] = float(tensor.min().item())
            summary["prob_max"] = float(tensor.max().item())
            summary["prob_p50"] = float(torch.quantile(tensor, 0.50).item())
            summary["prob_p90"] = float(torch.quantile(tensor, 0.90).item())
            summary["prob_p95"] = float(torch.quantile(tensor, 0.95).item())
            summary["prob_p99"] = float(torch.quantile(tensor, 0.99).item())
        else:
            summary["prob_mean"] = None
            summary["prob_min"] = None
            summary["prob_max"] = None
            summary["prob_p50"] = None
            summary["prob_p90"] = None
            summary["prob_p95"] = None
            summary["prob_p99"] = None
        return summary
