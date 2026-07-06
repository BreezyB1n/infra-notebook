from typing import Optional, Union

from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification
)

class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template

        self.final_prompts = []

        for prompt in prompts:
            if apply_chat_template:
                messages = [{"role": "user", "content": prompt}]
                # 对prompt格式化处理
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize = False,
                    add_generation_prompt = True, # prompt末尾提示模型generate
                )
            else:
                # bos_token 只是提醒模型begin of sequence
                if tokenizer.bos_token is not None:
                    prompt = tokenizer.bos_token + prompt
            
            self.final_prompts.append(prompt)

    def __len__(self):
        return len(self.final_prompts)
        
    def __getitem__(self, index):
        return self.final_prompts[index]

@dataclass
class PPOConfig:
    # actor, critic, ref model架构通常一样，共用一个path
    actor_model_path: str = ""
    reward_model_path: str = ""

    actor_lr: float = 5e-5
    critic_lr: float = 5e-5

    # 从数据集中一次取多少条Prompt
    rollout_batch_size: int = 8
    # 实际生成samples时喂给模型的prompt数量，8 / 2 = 4次推理
    micro_rollout_batch_size: int = 2
    # 训练时用多少条experience做反向传播
    micro_train_batch_size: int = 2

    # 每个prompt数据用来生成多少条回答
    n_samples_per_prompt: int = 2
    max_new_tokens: int = 50
    # 控制整条序列长度
    max_length: int = 256

    # 一共迭代多少次，生成一批经验 -> 用这批经验训练若干轮
    episodes: int = 3
    # 同一批 rollout 经验重复训练多少，PPO 的一个特点是：生成经验很贵，所以会复用同一批经验训练几次
    max_epochs: int = 5

# 冻结模型参数的工具
def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

class Critic(nn.Module):
    def __init__(self, base_model) -> None:
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, num_actions):
        # attention_mask负责保证batch纬度统一
        # 得到每一个token的hidden_state
        outputs = self.base_model(
            input_ids,
            attention_mask,
        )
        # hidden_states: (B, S, hidden_dim)
        hidden_states = outputs.last_hidden_state

        # values: (B, S)
        values = self.value_head(hidden_states).squeeze(-1) # 去掉最后大小为1的维度方便后续计算 -> (B, S)

        # 让 value 和 action log_prob 在时间步上对齐
        values = values[:, :-1]
        # 只保留生成回答部分的 value -> (B, num_actions)
        values = values[:, -num_actions:]

        return values # 结果会用来计算advantages，returns，value_loss
    
# 初始化模型
def init_models(config: PPOConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # actor, critic, ref model架构通常一样，共用一个tokenizer
    actor_tokenizer = AutoTokenizer.from_pretrained(config.actor_model_path)
    reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_path)

    actor_tokenizer.padding_token_side = "left"

    if actor_tokenizer.pad_token is None:
        actor_tokenizer.pad_token = actor_tokenizer.eos_token
    
    actor_model = AutoModelForCausalLM.from_pretrained(
        config.actor_model_path
    ).to(device)

    ref_model = AutoModelForCausalLM.from_pretrained(
        config.actor_model_path
    ).to(device)

    reward_model = AutoModelForCausalLM.from_pretrained(
        config.reward_model_path
    ).to(device)

    critic_base_model = AutoModelForCausalLM.from_pretrained(
        config.actor_model_path
    ).to(device)

    critic_model = Critic(critic_base_model).to(device)

    # 冻结模型参数
    freeze_model(ref_model)
    freeze_model(reward_model)

    return {
        "device": device,
        "actor_tokenizer": actor_tokenizer,
        "reward_tokenizer": reward_tokenizer,
        "actor_model": actor_model,
        "ref_model": ref_model,
        "reward_model": reward_model,
        "critic_model": critic_model,
    }

# 初始化优化器
def init_optimizers(models, config: PPOConfig):
    optimizer_actor = torch.optim.AdamW(
        models["actor_model"].parameters(),
        lr = config.actor_lr,
    )

    optimizer_critic = torch.optim.AdamW(
        models["critic_model"].parameters(),
        lr = config.critic_lr
    )

    return optimizer_actor, optimizer_critic

#  ------------------------------ Model、Data准备完成，进入Rollout代码 -----------------------------------

@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor # 哪些位置是actor生成的token
    num_actions: int
    response_length: torch.Tensor # 每条回答实际生成了多少 token
    total_length: torch.Tensor

def generate_samples(
    prompts,
    actor_model,
    actor_tokenizer,
    device,
    max_length,
    max_new_tokens,
    n_samples_per_prompt,
    micro_rollout_batch_size,      
):
    # 进入推理模式
    actor_model.eval()

    all_prompts = []
    for prompt in prompts:
        # 复制prompt，每个prompt做n_samples_per_prompt次采样
        for _ in range(n_samples_per_prompt):
            all_prompts.append(prompt)

    # 保存生成的samples
    samples_list = []

    for start in range(0, len(all_prompts), micro_rollout_batch_size):
        batch_prompts = all_prompts[start: start + micro_rollout_batch_size]

        inputs = actor_tokenizer(
            batch_prompts,
            padding='max_length',
            max_length=max_length,
            truncation=True, # 超长阶段
            return_tensors="pt", # 返回PyTorch张量
        ).to(device)

        '''
            tokenizer输出:
            {
                "input_ids": tensor(...),
                "attention_mask": tensor(...),
            }
        '''
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            # (B, P + R)
            # generate() 做了 采样 + 解码 ，把 logits 变成了 token id
            seqs = actor_model.generate(
                **inputs, # 相当于字典解包
                max_new_tokens = max_new_tokens,
                eos_token_id = actor_tokenizer.eos_token_id,
                pad_token_id = actor_tokenizer.pad_token_id,
            )
        
            prompt_length = input_ids.size(1)

            # 从完整序列中切出 response 部分 (B, R)
            response_ids = seqs[:, prompt_length:]

            # seqs.ne(pad_token_id): 当前位置不是 pad token -> True, long转为1/0
            attention_mask = seqs.ne(actor_tokenizer.pad_token_id).long()
            action_mask = response_ids.ne(actor_tokenizer.pad_token_id).long()

            samples = Samples(
                seqs=seqs,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.sum(dim=-1),
                total_length=attention_mask.sum(dim=-1),
            )

            samples_list.append(samples)
    
    return samples_list
    
    
def compute_action_log_probs

