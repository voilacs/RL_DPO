**DPO: Direct Preference Optimization**

This repository provides a reference implementation of the DPO algorithm for training language models from preference data, as described in Direct Preference Optimization: Your Language Model is Secretly a Reward Model. It supports both DPO and its variants like Conservative DPO and IPO.

Running with docker on IIITD cuda gpus:

We had with us 2 GPUs (40GB each):

Batch Size Calculation: 

batch size per gpu had to remain constant for that we did this:

batch size per GPU = batch_size / (gradient_accumulation_steps * num_gpus)

For example, if batch_size=64, gradient_accumulation_steps=2, and num_gpus=2, then:

batch size per GPU = 64 / (2 * 2) = 16

python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=32 eval_batch_size=16 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16


