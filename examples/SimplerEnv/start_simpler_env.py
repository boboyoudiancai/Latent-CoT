import os

# from IPython import embed; embed()
from examples.SimplerEnv.custom_argparse import get_args
from examples.SimplerEnv.model2simpler_interface import M1Inference

from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

import numpy as np



# if os.environ.get("DEBUG", None):
#     import debugpy
#     debugpy.listen(("0.0.0.0", 10092))
#     print("🔍 Rank 0 waiting for debugger attach on port 10092...")
#     debugpy.wait_for_client()
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # import debugpy 
    # if os.environ.get("DEBUG", None):

    #     debugpy.listen(("0.0.0.0", 10092))  # listen port 
    #     print("Waiting for debugger to attach...")
    #     debugpy.wait_for_client()  # wait for VS Code attach

    model = M1Inference(
        policy_ckpt_path=args.ckpt_path, # to get unnormalization stats
        policy_setup=args.policy_setup,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5,                 # cfg from 1.5 to 7 also performs well
        enable_latent_reasoning=args.enable_latent_reasoning,
        thinking_token_count=args.thinking_token_count,
        img_next_count=args.img_next_count,
        cot_mode=getattr(args, "cot_mode", "implicit"),
        think_max_len=getattr(args, "think_max_len", 64),
        think_temp=getattr(args, "think_temp", 0.1),
        think_topp=getattr(args, "think_topp", 0.9),
        # action_ensemble=False,         # 禁用 action ensemble
    )

    # policy model creation; update this if you are using a new policy model
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
