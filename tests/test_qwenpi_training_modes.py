"""CPU regression tests for ZeRO-2 parameter groups and training/eval modes."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ["STARVLA_DISABLE_DEEPSPEED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
from starVLA.model.modules.vlm.latent_text_decoder import FullQwenLatentTextDecoder
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.train_starvla import VLATrainer


def make_interface():
    interface = _QWen3_VL_Interface.__new__(_QWen3_VL_Interface)
    nn.Module.__init__(interface)
    interface.model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.2)).eval()
    interface.visual_ema = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.2)).eval()
    interface.visual_ema.requires_grad_(False)
    interface.future_teacher_update_mode = "ema"
    return interface


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = make_interface()
        self.action_model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.2))
        self.project_layers = nn.Linear(2, 2)
        self.latent_decoder = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.2)).eval()
        self.other = nn.Linear(2, 2)
        self.frozen_base = nn.Parameter(torch.ones(2), requires_grad=False)
        self.raise_in_prediction = False
        self.modes_seen = None

    def predict_action(self, examples, **kwargs):
        self.modes_seen = (self.training, self.action_model.training, self.action_model[1].training,
                           self.qwen_vl_interface.visual_ema.training)
        if self.raise_in_prediction:
            raise RuntimeError("prediction failed")
        return {"normalized_actions": np.array([example["action"] for example in examples])}


class TrainingModeTests(unittest.TestCase):
    @staticmethod
    def prepare_trainer(model):
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.model = model
        trainer.optimizer = object()
        trainer.vla_train_dataloader = object()
        trainer.lr_scheduler = object()
        trainer.accelerator = SimpleNamespace(gradient_accumulation_steps=1)
        trainer.config = OmegaConf.create({"seed": 42, "trainer": {"freeze_modules": ""},
                                           "framework": {"latent_reasoning": {"enable": True}}})
        trainer.completed_steps = 0
        trainer._resume_state_dir = None
        trainer.resume_from_checkpoint = None
        trainer._save_initial_configs = lambda: None
        trainer._init_checkpointing = lambda: None
        trainer._init_wandb = lambda: None
        trainer.print_trainable_parameters = lambda model: None
        trainer.setup_distributed_training = lambda accelerator, *components: components
        trainer.prepare_training()
        return trainer

    def test_optimizer_ignores_already_frozen_and_config_frozen_parameters(self):
        model = TinyModel()
        model.action_model.requires_grad_(False)
        model.project_layers.requires_grad_(False)
        model.latent_decoder[0].bias.requires_grad_(False)
        config = OmegaConf.create({"trainer": {
            "freeze_modules": "other",
            "learning_rate": {"base": 1e-5, "qwen_vl_interface": 1e-5,
                              "action_model": 1e-4, "project_layers": 1e-4,
                              "latent_decoder": 1e-4},
        }})
        groups = build_param_lr_groups(model, config)
        names = {group["name"] for group in groups}
        self.assertNotIn("action_model", names)
        self.assertNotIn("project_layers", names)
        self.assertTrue(all(group["params"] for group in groups))
        actual = [parameter for group in groups for parameter in group["params"]]
        self.assertTrue(all(parameter.requires_grad for parameter in actual))
        self.assertEqual(len(actual), len({id(parameter) for parameter in actual}))
        expected = {id(parameter) for name, parameter in model.named_parameters()
                    if parameter.requires_grad and not name.startswith("other.")}
        self.assertEqual({id(parameter) for parameter in actual}, expected)
        torch.optim.AdamW(groups, lr=1e-5)

    def test_prepare_training_enables_loaded_models_and_preserves_teacher_eval(self):
        model = TinyModel()
        self.assertFalse(model.qwen_vl_interface.model.training)
        self.assertFalse(model.latent_decoder.training)
        self.prepare_trainer(model)
        self.assertTrue(model.qwen_vl_interface.model.training)
        self.assertTrue(model.latent_decoder.training)
        self.assertTrue(model.latent_decoder[1].training)
        self.assertFalse(model.qwen_vl_interface.visual_ema.training)
        self.assertFalse(model.qwen_vl_interface.visual_ema[1].training)

    def test_real_qwen_decoder_uses_internal_gradient_checkpointing(self):
        from transformers import Qwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

        config = Qwen3VLConfig(
            text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 2, "num_attention_heads": 4,
                         "num_key_value_heads": 2, "head_dim": 8,
                         "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
            vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                           "num_heads": 4, "out_hidden_size": 32,
                           "num_position_embeddings": 16, "deepstack_visual_indexes": []},
        )
        qwen = Qwen3VLForConditionalGeneration(config).eval()
        qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        layers = qwen.model.language_model.layers
        for layer in layers:
            layer._gradient_checkpointing_func = Mock(wraps=layer._gradient_checkpointing_func)
        model = TinyModel()
        model.latent_decoder = FullQwenLatentTextDecoder(
            qwen_model=qwen, hidden_size=32, vocab_size=64, use_forward_checkpoint=False,
        )
        self.prepare_trainer(model)
        latent = torch.randn(2, 1, 32, requires_grad=True)
        tokens = torch.randint(0, 64, (2, 4))
        loss = model.latent_decoder(latent, tokens, tokens, torch.ones_like(tokens))["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(layer.training and layer.gradient_checkpointing for layer in layers))
        self.assertTrue(all(layer._gradient_checkpointing_func.call_count > 0 for layer in layers))
        self.assertIsNotNone(latent.grad)
        self.assertGreater(latent.grad.abs().sum().item(), 0)
        self.assertIsNotNone(qwen.lm_head.weight.grad)

    def test_evaluation_disables_dropout_and_restores_modes_even_on_failure(self):
        model = TinyModel().train()
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.model = model
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer._get_next_batch = lambda: [{"action": np.zeros((1, 2))}]
        metrics = trainer.eval_action_model({})
        self.assertEqual(model.modes_seen, (False, False, False, False))
        self.assertEqual(metrics["mse_score"], 0.0)
        self.assertTrue(model.training)
        self.assertTrue(model.action_model[1].training)
        self.assertFalse(model.qwen_vl_interface.visual_ema.training)

        model.raise_in_prediction = False
        model.eval()
        trainer.eval_action_model({})
        self.assertFalse(model.training)
        self.assertFalse(model.action_model[1].training)
        model.raise_in_prediction = True
        model.train()
        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            trainer.eval_action_model({})
        self.assertTrue(model.training)
        self.assertTrue(model.action_model[1].training)
        self.assertFalse(model.qwen_vl_interface.visual_ema.training)

    def test_teacher_stays_eval_for_ema_and_frozen_modes(self):
        for update_mode in ("ema", "frozen"):
            interface = make_interface()
            interface.future_teacher_update_mode = update_mode
            interface.eval()
            interface.train()
            self.assertTrue(interface.model.training)
            self.assertFalse(interface.visual_ema.training)


if __name__ == "__main__":
    unittest.main()
