import os
import tqdm
import json
import copy
import math

import torch
import logging
import argparse

import numpy as np
from rouge import Rouge

import dataclasses
from xopen import xopen

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.llama import H2OLlamaForCausalLM
from utils.lmeval_warp import LMEvalLM
from lm_eval import evaluator

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="")

    parser.add_argument("--enable_h2o_generation", action='store_true')
    parser.add_argument("--num_heavy_hitter_tokens", type=int, default=-1)
    parser.add_argument("--num_window_length", type=int, default=256)

    parser.add_argument("--enable_position_rolling", action='store_true')

    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    set_seed(args)

    args.model_name = "meta-llama/Meta-Llama-3-8B"
    args.enable_h2o_generation = False

    print(args)

    model_name = args.model_name

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if args.num_heavy_hitter_tokens == -1:
        print('not assign number of heavy hitter tokens, use half of the cache size: {}'.format(args.num_window_length // 2))
        args.num_heavy_hitter_tokens = args.num_window_length // 2

    model_warp = LMEvalLM(model_name, config=config, args=args)

    results = evaluator.simple_evaluate(
        model=model_warp,
        tasks=["coqa"],
        log_samples=True,
        # limit=8
    )

    print(results["results"])


