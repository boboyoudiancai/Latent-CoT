import os
import tempfile
from pathlib import Path

os.environ["STARVLA_DISABLE_DEEPSPEED"] = "1"

import torch
from torch import nn
from omegaconf import OmegaConf
from types import SimpleNamespace
from unittest.mock import patch
from starVLA.training.train_starvla import (
    _load_thinking_stage1_modules,
    _validate_thinking_accumulation,
    _validate_thinking_stage1_checkpoint,
    setup_optimizer_and_scheduler,
    VLATrainer,
)


def test_stage2_rejects_pre_thinking_checkpoint():
    with tempfile.TemporaryDirectory(prefix="qwenpi-checkpoint-test-") as directory:
        original = Path(directory) / "old.pt"
        stage1 = Path(directory) / "stage1.pt"
        torch.save({"qwen_vl_interface.weight": torch.ones(1)}, original)
        torch.save({"latent_decoder.model.lm_head.weight": torch.ones(1)}, stage1)
        try:
            _validate_thinking_stage1_checkpoint(original)
        except ValueError:
            pass
        else:
            raise AssertionError("Stage 2 accepted a pre-thinking checkpoint")
        _validate_thinking_stage1_checkpoint(stage1)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = nn.Linear(2, 2)
        self.spatial_forcing_projector = nn.Linear(2, 2)
        self.action_model = nn.Linear(2, 2)
        self.project_layers = nn.Linear(2, 2)


def test_stage2_loads_only_strict_modules_and_preserves_action():
    with tempfile.TemporaryDirectory(prefix="qwenpi-transfer-test-") as directory:
        source, target = _TinyModel(), _TinyModel()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(3.0)
        checkpoint = dict(source.state_dict())
        checkpoint["latent_decoder.model.weight"] = torch.ones(1)
        path = Path(directory) / "stage1.pt"
        torch.save(checkpoint, path)
        action = {k: v.clone() for k, v in target.action_model.state_dict().items()}
        projection = {k: v.clone() for k, v in target.project_layers.state_dict().items()}
        _load_thinking_stage1_modules(target, path)
        for name in ("qwen_vl_interface", "spatial_forcing_projector"):
            for value in getattr(target, name).state_dict().values():
                assert torch.all(value == 3)
        assert all(torch.equal(v, action[k]) for k, v in target.action_model.state_dict().items())
        assert all(torch.equal(v, projection[k]) for k, v in target.project_layers.state_dict().items())

        for bad_checkpoint in (
            {k: v for k, v in checkpoint.items() if not k.startswith("spatial_forcing_projector.")},
            {**checkpoint, "spatial_forcing_projector.weight": torch.ones(1)},
        ):
            torch.save(bad_checkpoint, path)
            fresh = _TinyModel()
            before = {k: v.clone() for k, v in fresh.state_dict().items()}
            try:
                _load_thinking_stage1_modules(fresh, path)
            except ValueError:
                pass
            else:
                raise AssertionError("Incompatible stage 1 checkpoint was accepted")
            assert all(torch.equal(value, before[key]) for key, value in fresh.state_dict().items())


def test_stage1_rejects_training_checkpoint():
    with tempfile.TemporaryDirectory(prefix="qwenpi-stage1-test-") as directory:
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.config = OmegaConf.create({
            "output_dir": directory,
            "framework": {"latent_reasoning": {"enable": True, "require_base_initialization": True}},
            "trainer": {"pretrained_checkpoint": "old_qwenpi.pt"},
        })
        try:
            trainer._init_checkpointing()
        except ValueError as error:
            assert "base pretrained VLM" in str(error)
        else:
            raise AssertionError("Stage 1 accepted an old training checkpoint")


def test_accumulation_guard_and_scheduler_scaling():
    cfg = OmegaConf.create({
        "framework": {"latent_reasoning": {"enable": True}},
        "trainer": {
            "gradient_accumulation_steps": 2, "scale_scheduler_by_world_size": True,
            "learning_rate": {"base": 1e-5},
            "optimizer": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0, "fused": False},
            "scheduler_specific_kwargs": {"min_lr": 5e-7}, "lr_scheduler_type": "cosine_with_min_lr",
            "num_warmup_steps": 300, "max_train_steps": 10000,
        },
    })
    _validate_thinking_accumulation(cfg, SimpleNamespace(gradient_accumulation_steps=2))
    try:
        _validate_thinking_accumulation(cfg, SimpleNamespace(gradient_accumulation_steps=1))
    except ValueError:
        pass
    else:
        raise AssertionError("Accumulation mismatch was accepted")
    model = nn.Linear(2, 2)
    params = [{"params": list(model.parameters()), "name": "base"}]
    with patch("starVLA.training.train_starvla.accelerator", SimpleNamespace(num_processes=8)), \
         patch("starVLA.training.train_starvla.build_param_lr_groups", return_value=params), \
         patch("starVLA.training.train_starvla.get_scheduler") as get_scheduler:
        setup_optimizer_and_scheduler(model, cfg)
    assert get_scheduler.call_args.kwargs["num_warmup_steps"] == 2400
    assert get_scheduler.call_args.kwargs["num_training_steps"] == 80000


def test_lara_uniform_step_sampling():
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
    dataset = SimpleNamespace(all_steps=[(5, 0), (5, 1), (8, 0)])
    mixture = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
    mixture.mode = "val"
    mixture.datasets = [dataset]
    mixture._dataset_sampling_weights = [1.0]
    mixture.sampling_mode = "lara_uniform_steps"
    assert {mixture.sample_step(index)[1:] for index in range(100)} == set(dataset.all_steps)


def test_thinking_stage_configs():
    root = Path("examples/simBenchmarks/LIBERO/train_files")
    stage1 = OmegaConf.load(root / "starvla_qwenpi_thinking_stage1.yaml")
    stage2 = OmegaConf.load(root / "starvla_qwenpi_thinking_stage2.yaml")
    assert stage1.trainer.pretrained_checkpoint is None
    assert stage1.framework.latent_reasoning.require_base_initialization
    assert (stage1.trainer.max_train_steps, stage1.datasets.vla_data.per_device_batch_size,
            stage1.trainer.gradient_accumulation_steps) == (10000, 7, 2)
    assert list(stage1.trainer.optimizer.betas) == [0.9, 0.999]
    assert stage1.trainer.optimizer.weight_decay == 0.0
    assert (stage2.trainer.max_train_steps, stage2.datasets.vla_data.per_device_batch_size,
            stage2.trainer.gradient_accumulation_steps) == (40000, 14, 1)
    assert stage2.trainer.num_warmup_steps == 5000
    assert stage2.trainer.scale_scheduler_by_world_size
    assert stage2.trainer.reload_modules == "qwen_vl_interface,spatial_forcing_projector"
    for cfg in (stage1, stage2):
        assert cfg.framework.future.enable and cfg.framework.spatial_forcing_aux.enable
        assert cfg.datasets.vla_data.sampling_mode == "lara_uniform_steps"


def test_accelerated_scheduler_reference_timing():
    from accelerate.scheduler import AcceleratedScheduler
    from transformers import get_scheduler
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_scheduler("cosine_with_min_lr", optimizer, num_warmup_steps=2400,
                              num_training_steps=80000, scheduler_specific_kwargs={"min_lr": 5e-7})
    wrapped = AcceleratedScheduler(scheduler, SimpleNamespace(step_was_skipped=False), split_batches=False)
    with patch("accelerate.scheduler.AcceleratorState", return_value=SimpleNamespace(num_processes=8)):
        for _ in range(300):
            optimizer.step()
            wrapped.step()
    assert scheduler.last_epoch == 2400
    assert abs(optimizer.param_groups[0]["lr"] - 1e-4) < 1e-10


def test_stage2_scheduler_has_5000_real_warmup_steps():
    from accelerate.scheduler import AcceleratedScheduler
    from transformers import get_scheduler
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_scheduler("cosine_with_min_lr", optimizer, num_warmup_steps=5000 * 8,
                              num_training_steps=40000 * 8, scheduler_specific_kwargs={"min_lr": 5e-7})
    wrapped = AcceleratedScheduler(scheduler, SimpleNamespace(step_was_skipped=False), split_batches=False)
    with patch("accelerate.scheduler.AcceleratorState", return_value=SimpleNamespace(num_processes=8)):
        for _ in range(625):
            optimizer.step()
            wrapped.step()
        assert abs(optimizer.param_groups[0]["lr"] - 1.25e-5) < 1e-10
        for _ in range(5000 - 625):
            optimizer.step()
            wrapped.step()
        assert scheduler.last_epoch == 5000 * 8
        assert abs(optimizer.param_groups[0]["lr"] - 1e-4) < 1e-10
        for _ in range(40000 - 5000):
            optimizer.step()
            wrapped.step()
    assert scheduler.last_epoch == 40000 * 8
    assert abs(optimizer.param_groups[0]["lr"] - 5e-7) < 1e-10
