"""Run in a fresh process with STARVLA_DISABLE_DEEPSPEED=1 and accumulation=2."""
import os
os.environ["STARVLA_DISABLE_DEEPSPEED"] = "1"
os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from starVLA.training.train_starvla import accelerator


def test_two_microbatches_make_one_optimizer_step():
    assert accelerator.gradient_accumulation_steps == 2
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    boundaries = []
    initial_scheduler_step = scheduler.scheduler.last_epoch
    for _ in range(2):
        with accelerator.accumulate(model):
            loss = (model(torch.ones(1, 1)) - 1).square().mean()
            accelerator.backward(loss)
            boundaries.append(accelerator.sync_gradients)
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad()
    assert boundaries == [False, True]
    assert torch.allclose(model.weight.detach().cpu(), torch.tensor([[0.2]]))
    assert scheduler.scheduler.last_epoch == initial_scheduler_step + 1
