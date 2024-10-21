import logging

import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from utils.llama import H2OLlamaForCausalLM

logger = logging.getLogger(__name__)


class LMEvalLM(HFLM):
    def __init__(
            self,
            model_name: str,
            config,
            args,
            device: str = "cuda:0",
            dtype: str = "float16",
            batch_size: int = 1,
            load_compressed_weight: bool = False,
    ) -> None:
        super().__init__(
            pretrained=model_name,
            backend="default",
            revision="main",
            subfolder=None,
            tokenizer=None,
            truncation=False,
            max_length=None,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_batch_size=64,
            trust_remote_code=False,
            use_fast_tokenizer=True,
            add_bos_token=False,
            prefix_token_id=None,
            parallelize=False,
            # device_map_option="auto",
            peft=None,
            delta=None,
            autogptq=False,
        )

        self._model = None
        # call GC
        import gc
        gc.collect()

        if args.enable_h2o_generation:
            config.num_heavy_hitter_tokens = args.num_heavy_hitter_tokens
            config.num_window_length = args.num_window_length
            config.enable_position_rolling = args.enable_position_rolling
            model = H2OLlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                config=config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
        self._model = model

        self._model = self._model.half()
        self._model.to(self.device)
