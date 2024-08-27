import gc
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback

from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from .rloo_config import RLOOConfig


INVALID_LOGPROB = 1.0


class RLOOTrainer(Trainer):
    def __init__(
        self,
        config: RLOOConfig,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.policy = policy
        # TODO WHY
        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, args.rloo_k, "`local_batch_size` must be a multiple of rloo_k"
        )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, reward_model]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        # self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        self.optimizer, self.dataloader = accelerator.prepare(self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)
            # For mixed precision
            # self.reward_model = self.accelerator.prepare(self.reward_model)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        # self.model = self.accelerator.prepare(self.model)
        model = self.accelerator.prepare(self.model)
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        self.reward_model = self.accelerator.prepare(self.reward_model)
        reward_model = self.reward_model
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            # min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            # eos_token_id=self.tokenizer.eos_token_id,
            eos_token_id=args.stop_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with (torch.no_grad()):
                queries = data["input_ids"].to(device)
                queries = queries.repeat(args.rloo_k, 1)
                context_length = queries.shape[1]
                query_responses = []
                responses = []
                response_lens = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                scores_armo = []
                scores_fsfairx = []
                reward_dist_entropy = []
                rewards_adjusted_all = []
                rewards_adjusted_armo_all = []
                gating_output_armo_all = []
                gating_output_all = []
                sequence_lengths = []
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    query_responses, logprobs = batch_generation(
                        unwrapped_model, #TODO unwrapped model is the same as the ref policy if using a single model with lora
                        # model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                        context_length,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    # logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    # all_logprob = F.log_softmax(logits, dim=-1)
                    # logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    # del logits, all_logprob
                    torch.cuda.empty_cache()

                    # ref_output = forward(ref_policy, query_response, tokenizer.pad_token_id)
                    # ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits = forward(ref_policy, query_response, tokenizer.pad_token_id).logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    # ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logits = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_logits, 2, response.unsqueeze(-1)).squeeze(-1)
                    del ref_logits
                    # del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response, response_len = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )

                    # TODO mask out bos_token ?! its removed for training the reward model
                    # TODO also check sequenth length, seems to be incorrect
                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    #TODO for my armo style reward model, remove bos token as this is how the model was trained
                    postprocessed_query_response = torch.where(
                        postprocessed_query_response == tokenizer.bos_token_id, tokenizer.pad_token_id, postprocessed_query_response)
                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    (score, qt_estimates, entropy, gating_output, rewards_adjusted, score_armo, rewards_adjusted_armo,
                    gating_output_armo, score_fsfairx) = get_reward(
                        reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                    )

                    responses.append(response)
                    response_lens.append(response_len)
                    postprocessed_responses.append(postprocessed_response)
                    # logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    scores_armo.append(score_armo)
                    scores_fsfairx.append(score_fsfairx)
                    reward_dist_entropy.append(entropy)
                    gating_output_all.append(gating_output)
                    gating_output_armo_all.append(gating_output_armo)
                    rewards_adjusted_all.append(rewards_adjusted)
                    rewards_adjusted_armo_all.append(rewards_adjusted_armo)
                responses = torch.cat(responses, 0)
                response_lens = torch.cat(response_lens, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                # logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                scores_armo = torch.cat(scores_armo, 0)
                scores_fsfairx = torch.cat(scores_fsfairx, 0)
                reward_dist_entropy = torch.cat(reward_dist_entropy, 0)
                rewards_adjusted_all = torch.cat(rewards_adjusted_all, 0)
                rewards_adjusted_armo_all = torch.cat(rewards_adjusted_armo_all, 0)
                gating_output_all = torch.cat(gating_output_all, 0)
                gating_output_armo_all = torch.cat(gating_output_armo_all, 0)
                # del (logprob, ref_logprob, score)
                del (ref_logprob, score)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                # contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
                # TODO for my armo style reward model, remove bos token as this is how the model was trained
                scores = args.reward_bias + args.reward_scale * scores
                contain_eos_token = torch.any(postprocessed_responses == args.stop_token_id, dim=-1)
                if args.non_eos_penalty:
                    scores_eos = torch.where(contain_eos_token, scores, args.penalty_reward_value)
                    scores_armo_eos = torch.where(contain_eos_token, scores_armo, args.penalty_reward_value)
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = (-args.kl_coef * kl).sum(1)
                entropy_reward = -args.entropy_coef * reward_dist_entropy.squeeze(1)
                rlhf_reward = scores_eos + non_score_reward
                # rlhf_reward = scores_eos + non_score_reward + entropy_reward

                # vectorized RLOO advantages implementation
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                advantages = rlhf_reward - baseline
                advantages = advantages.flatten()
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            #TODO why not use truncated responses?
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]

                            output = forward(model, mb_query_responses, tokenizer.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            new_logprobs = new_logprobs.sum(1)
                            mb_logprobs = mb_logprobs.sum(1)
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()
                            loss = pg_loss
                            accelerator.backward(loss)
                            _grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[
                                    ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = pg_clipfrac
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    self.state.global_step += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_all_logprobs, new_logprobs,
                        logprobs_diff, ratio, pg_losses, pg_losses2,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                entropy_reward = entropy_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                # metrics["loss/grad_norm"] = self.accelerator.gather(_grad_norm.detach().item() if isinstance(_grad_norm, torch.Tensor) else _grad_norm).mean().item()
                metrics["loss/grad_norm"] = self.accelerator.gather(_grad_norm.detach()).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/entropy_reward"] = self.accelerator.gather(entropy_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores_with_eos"] = self.accelerator.gather(scores_eos.mean()).mean().item()
                metrics["objective/scores_original"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["objective/scores_armo"] = self.accelerator.gather(scores_armo.mean()).mean().item()
                metrics["objective/scores_armo_with_eos"] = self.accelerator.gather(scores_armo_eos.mean()).mean().item()
                metrics["objective/scores_fsfairx"] = self.accelerator.gather(scores_fsfairx.mean()).mean().item()
                metrics["objective/reward_dist_entropy"] = self.accelerator.gather(reward_dist_entropy.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/contain_eos_token"] = self.accelerator.gather(contain_eos_token).float().mean().item()
                metrics["val/response_len"] = self.accelerator.gather(response_lens).float().mean().item()
                # metrics["val/num_eos_tokens"] = (responses == args.stop_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode

                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

                gating_output = self.accelerator.gather(gating_output_all).mean(0)
                rewards_adjusted = self.accelerator.gather(rewards_adjusted_all).mean(0)
                gating_output_armo_all = self.accelerator.gather(gating_output_armo_all).mean(0)
                rewards_adjusted_armo_all = self.accelerator.gather(rewards_adjusted_armo_all).mean(0)
                if "wandb" in args.report_to and self.accelerator.process_index == 0 and self.state.global_step > 5:
                    import wandb
                    costum_logs = {}
                    for i, a in enumerate(self.accelerator.unwrap_model(self.reward_model).attributes):
                        costum_logs[f"gating_output/{a}"] = gating_output[i].item()
                        costum_logs[f"rewards_adjusted/{a}"] = rewards_adjusted[i].item()
                        costum_logs[f"gating_output_armo/{a}"] = gating_output_armo_all[i].item()
                        costum_logs[f"rewards_adjusted_armo/{a}"] = rewards_adjusted_armo_all[i].item()

                    wandb.log({**costum_logs, "train/global_step": self.state.global_step})



            del kl, mean_kl, mean_entropy, scores

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                # self._save_checkpoint(model, trial=None, metrics=metrics)
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        tokenizer = self.tokenizer
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model,
                        query,
                        query.shape[0],
                        tokenizer.pad_token_id,
                        generation_config,
                        context_length,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response, response_len = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )
                    table["query"].extend(gather_object(tokenizer.batch_decode(query, skip_special_tokens=True)))
                    table["model response"].extend(gather_object(tokenizer.batch_decode(postprocessed_response)))

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    (score, qt_estimates, entropy, gating_output, rewards_adjusted, score_armo, rewards_adjusted_armo,
                    gating_output_armo, score_fsfairx) = get_reward(
                        self.reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                    )
                    score = args.reward_bias + args.reward_scale * score
                    score_list = self.accelerator.gather(score).float().cpu().numpy()
                    table["score"].extend(score_list)
                    # table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())


                    qt_estimates_list = self.accelerator.gather(qt_estimates).float().cpu().numpy()
                    qt_estimates_list = args.reward_bias + args.reward_scale * qt_estimates_list
                    entropy_list = self.accelerator.gather(entropy.squeeze(1)).float().cpu().numpy()
                    gating_output = self.accelerator.gather(gating_output).float().cpu().numpy()
                    rewards_adjusted = self.accelerator.gather(rewards_adjusted).float().cpu().numpy()
                    table["reward dist entropy"].extend(entropy_list)
                    if "wandb" in args.report_to:
                        import wandb
                        import time
                        st = time.time()
                        quantiles = self.accelerator.unwrap_model(self.reward_model).quantiles.cpu().numpy()
                        print("Getting quantiles with unwrap took seconds: ", time.time() - st)
                    #
                        if self.accelerator.process_index == 0:
                            print("Create plots")
                            for qt, entopy, s in zip(qt_estimates_list, entropy_list, score_list):
                                plot_obj = plot_quantile_histogram(quantiles, qt, s)
                                table["reward_distribution"].extend([wandb.Image(plot_obj)])
                                plt.close()
                            for gat, rew in zip(gating_output, rewards_adjusted):
                                plt.bar(range(len(gat)), gat)
                                table["gating_output"].extend([wandb.Image(plt)])
                                plt.close()
                                plt.bar(range(len(rew)), rew)
                                table["rewards_adjusted"].extend([wandb.Image(plt)])
                                plt.close()

                if sampling:
                    break
        table["model response"] = [resp.replace("[PAD]", "") for resp in table["model response"]]
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df.iloc[0 : 0 + 5])
        if "wandb" in args.report_to:
            import wandb

            if wandb.run is not None:
                wandb.log({f"completions step {self.state.global_step}": wandb.Table(dataframe=df)})

def plot_quantile_histogram(quantiles, values, prediction):
    # quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    # values =    [1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    # Create an interpolation function
    interp_func = interp1d(quantiles, values, kind='linear', fill_value="extrapolate")

    # Generate a larger sample of quantiles
    sample_quantiles = np.linspace(0, 1, 1000)
    sample_values = interp_func(sample_quantiles)
    plt.figure(figsize=(10, 6))
    plt.hist(sample_values, bins=30, density=True, alpha=0.6, color='b', edgecolor='black')

    # Plot the interpolated PDF as a line plot for reference
    plt.plot(sample_values, np.zeros_like(sample_values), 'o', label='Interpolated Data Points', markersize=2)
    plt.axvline(prediction, color='r', linestyle='--', linewidth=2, label='Expectation (point estimate)')

    plt.title('Histogram Approximating the PDF from Quantiles')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    return plt