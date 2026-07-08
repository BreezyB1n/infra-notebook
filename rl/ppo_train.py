from typing import Optional, Union

from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # last_hidden_states确保基于之前的回答打分
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
    

def compute_action_log_probs(model, seqs, attention_mask, num_actions):
    '''
    给定完整序列 seqs = prompt + response
    计算 actor/ref 对 response 每个 token 的 log probability

    actor vs old_actor: 用于 PPO ratio, 决定怎么更新 actor
    actor vs ref_model: 用于 KL penalty, 限制 actor 不要跑偏
    '''
    outputs = model(
        input_ids=seqs,
        attention_mask = attention_mask,
    )

    # (B, S, V)
    logits = outputs.logits

    # (B, S - 1, vocab_size)
    # 对 每个位置logits求log_softmax
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1) # dim=-1代表在最后一维上做操作

    # (B, S - 1)
    labels = seqs[:, 1:]

    # (B, S - 1)
    token_log_probs = log_probs.gather(
        dim=-1, # 在词表维度（最后一维）上，按 index 给出的索引取值
        index=labels.unsqueeze(-1), # labels.unsqueeze(-1) 把 [[2, 3]] 变成 [[[2], [3]]]
    ).squeeze(-1) # 假如得到[[[-3.08], [-3.13]]] -> squeeze(-1) -> [[-3.08, -3.13]]
    '''
    [log P(r1|p1,p2,p3),
    log P(r2|p1,p2,p3,r1),
    log P(r3|p1,p2,p3,r1,r2)]
    '''
    action_log_prbs = token_log_probs[:, -num_actions:]

    return action_log_prbs

# ------------------------------- reward model -----------------------

def compute_reward_scores(
    reward_model,
    reward_tokenizer,
    actor_tokenizer,
    seqs,
    device,
):
    # 将token还原为text
    texts = actor_tokenizer.decode(
        seqs,
        skip_special_tokens=True,
    )

    inputs = reward_tokenizer(
        texts,
        padding = True,
        truncation = True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        scores = reward_model(
            **inputs
        ).logits
    
    return scores

# KL计算
def compute_approx_kl(action_log_probs, ref_log_probs, action_mask=None):
    """
    计算每一个token的kl值
    """
    kl = action_log_probs - ref_log_probs

    if action_mask is not None:
        kl = kl * action_mask
    
    return kl

# token-level rewards
def compute_reward(
    kl,
    reward_scores,
    action_mask,
    kl_coef=0.1,
    clip_reward_value=0.2,
):
    rewards = -kl_coef * kl

    reward_scores = torch.clamp(
        reward_scores,
        min=-clip_reward_value,
        max=clip_reward_value,
    )

    batch_size = rewards.size(0)
    response_length = action_mask.sum(dim=1)

    for i in range(batch_size):
        last_action_index = response_length[i] - 1
        rewards[i, last_action_index] += reward_scores[i, 0]

    rewards = rewards * action_mask

    return rewards

# 计算 advantages, returns 来训练 actor, returns
def compute_advantages_and_returns(
    values,
    rewards,
    action_mask,
    gamma=0.99,
    lam=0.95      
):
    """
    values: (B, num_actions)
    rewards: (B, num_actions)
    """

    values = values * action_mask
    rewards = rewards * action_mask

    batch_size, num_actions = rewards.shape

    advantages = torch.zeros_like(rewards) # (B, num_actions)
    # 上一轮循环的A_t + 1
    last_GAE = torch.zeros(batch_size, device=rewards.device)

    # 从后向前算
    for t in reversed(range(num_actions)):
        if t == num_actions - 1:
            # V(s_t+1) = 0, pad token = 0
            next_value = torch.zeros(batch_size, device=rewards.device)
            next_valid = torch.zeros(batch_size, device=rewards.device)
        else:
            # 得到V(s_t+1), 如果是pad token后续计算会设置为0
            next_value = values[:, t + 1]
            next_valid = action_mask[:, t + 1] # s_t+1是否是pad token

        # 计算delta
        delta = rewards[:, t] + gamma * next_value * next_valid - values[:, t]

        last_GAE = delta + gamma * lam * last_GAE * next_valid

        advantages[:, t] = last_GAE
    
    advantages = advantages * action_mask
    returns = advantages + values # critic 训练目标

    return advantages.detach(), returns.detach() # advantages 和 returns 是训练目标，不是要被继续优化的中间变量。

# Experience 保存 rollout 数据快照
@dataclass
class Experience:
    # 完整序列 prompts + response
    seqs: torch.Tensor
    # 有效 token mask
    attention_mask: torch.Tensor
    # 生成 token mask
    action_mask: torch.Tensor

    # rollout 时 actor 对 action_token的 log probability
    old_action_log_probs: torch.Tensor
    # rollout 时 critic 对 action_token的value预测
    old_values: torch.Tensor

    advantages: torch.Tensor

    returns: torch.Tensor

    reward_scores: Optional[torch.Tensor] = None
    kl: Optional[torch.Tensor] = None

class ExperienceBuffer(Dataset):
    def __init__(self):
        self.items = []

    def append(self, experiences):
        self.items.extend(experiences)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
@dataclass
class ExperienceBatch:
    seqs: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    old_action_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

# 有 Experience 是"一条一条"的，训练的时候需要"一批一批"的。ExperienceBatch + collate_fn 就是把多条经验拼成一个 batch，方便模型批量计算
def collate_experiences(batch):
    return ExperienceBatch(
        seqs = torch.cat([x.reqs for x in batch], dim=0),
        attention_mask=torch.cat([x.attention_mask for x in batch], dim=0),
        action_mask=torch.cat([x.action_mask for x in batch], dim=0),
        old_action_log_probs=torch.cat([x.old_action_log_probs for x in batch], dim=0),
        old_values=torch.cat([x.old_values for x in batch], dim=0),
        advantages=torch.cat([x.advantages for x in batch], dim=0),
        returns=torch.cat([x.returns for x in batch], dim=0),
    )

# 将结果封装成experience
def make_experience(
    sample,
    old_action_log_probs,
    old_values,
    advantages,
    returns,
    reward_scores=None,
    kl=None,
):
    return Experience(
        seqs=sample.seqs.detach(),
        attention_mask=sample.attention_mask.detach(),
        action_mask=sample.action_mask.detach(),
        old_action_log_probs=old_action_log_probs.detach(),
        old_values=old_values.detach(),
        advantages=advantages.detach(),
        returns=returns.detach(),
        reward_scores=None if reward_scores is None else reward_scores.detach(),
        kl=None if kl is None else kl.detach(),
    )

def compute_policy_loss(
        new_action_log_probs,
        old_action_log_probs,
        advantages,
        action_mask,
        clip_eps=0.2,
):
    ratio = torch.exp(new_action_log_probs - old_action_log_probs)

    unclipped_objective = ratio * advantages

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps
    )

    clipped_objective = clipped_ratio * advantages

    objective = torch.minimum(
        unclipped_objective,
        clipped_objective,
    )

    loss = -objective
    
    loss = loss * action_mask

    # 对每条回答内部取平均
    """
    loss.sum(dim=-1) = [
    0.5+0.3+0.7 = 1.5,   # 回答A的总loss
    0.2+0.4+0.1+0.6+0.3+0.5+0.2+0.4 = 2.7,  # 回答B的总loss
    ]

    action_mask.sum(dim=-1) = [3, 8]  # 真实token数量

    loss.sum(dim=-1) / action_mask.sum(dim=-1) = [
        1.5 / 3 = 0.5,    # 回答A: 每个词平均 loss = 0.5
        2.7 / 8 = 0.3375, # 回答B: 每个词平均 loss = 0.3375
    ]
    """
    loss = loss.sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1)

    return loss.mean()

def compute_value_loss(
    new_values,
    returns,
    action_mask      
):
    # shape: (B, S)
    loss = (new_values - returns) ** 2

    loss = loss * action_mask

    loss = loss.sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1)

    return loss.mean()

def train_step(
    batch,
    models,
    optimizer_actor,
    optimizer_critic,
    clip_eps=0.2,
):
    actor_model = models["actor_model"]
    critic_model = models["critic_model"]

    actor_model.train()
    critic_model.train()
    
    # 1. 更新 actor
    optimizer_actor.zero_grad()

    new_action_log_probs = compute_action_log_probs(
        model=actor_model,
        seqs=batch.seqs,
        attention_mask=batch.attention_mask,
        num_actions=batch.action_mask.size(1),
    )

    policy_loss = compute_policy_loss(
        new_action_log_probs=new_action_log_probs,
        old_action_log_probs=batch.old_action_log_probs,
        advantages=batch.advantages,
        action_mask=batch.action_mask,
        clip_eps=clip_eps,
    )

    # 计算每个参数梯度
    policy_loss.backward()
    # 根据梯度更新模型参数
    optimizer_actor.step()

    # 2. 更新critic
    optimizer_critic.zero_grad()

    new_values = critic_model(
        input_ids=batch.seqs,
        attention_mask=batch.attention_mask,
        num_actions=batch.num_actions,
    )

    value_loss = compute_value_loss(
        new_values=new_values,
        returns=batch.returns,
        action_mask=batch.action_mask,
    )

    value_loss.backward()
    optimizer_critic.step()

    return {
        "policy_loss": policy_loss.item(), # item将单元素tensor转成python数值
        "value_loss": value_loss.item(),
    }

def generate_experiences(
    samples_list,
    models,
    config
):
    actor_model = models["actor_model"]
    critic_model = models["critic_model"]
    ref_model = models["ref_model"]
    reward_model = models["reward_model"]

    actor_tokenizer = models["actor_tokenizer"]
    reward_tokenizer = models["reward_tokenizer"]
    device = models["device"]

    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []

    for sample in samples_list:
        with torch.no_grad():
            # 计算后续 PPO ration
            old_action_log_probs = compute_action_log_probs(
                model=actor_model,
                seqs=sample.seqs,
                attention_mask=sample.attention_mask,
                num_actions=sample.action_mask
            )

            # 计算 KL 惩罚
            ref_action_log_probs = compute_action_log_probs(
                model=ref_model,
                seqs=sample.seqs,
                attention_mask=sample.attention_mask,
                num_actions=sample.action_mask
            )

            # 计算GAE 和 PPO clip
            old_values = critic_model(
                input_ids=sample.seqs,
                attention_mask=sample.attention_mask,
                action_mask=sample.action_mask,
            )

            # r_t
            reward_scores = compute_reward_scores(
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                actor_tokenizer=actor_tokenizer,
                seqs=sample.seqs,
                device=sample.device
            )

            # KL
            kl = compute_approx_kl(
                old_action_log_probs, # 每次更新后 rollout的log_probs
                ref_action_log_probs,
                sample.attention_mask,
            )

            rewards = compute_reward(
                kl=kl,
                reward_scores=reward_scores,
                action_mask=sample.action_mask,
                kl_coef=0.1,
                clip_reward_value=0.2,
            )

            advantages, returns = compute_advantages_and_returns(
                values=old_values,
                rewards=rewards,
                action_mask=sample.action_mask,
                gamma=0.99,
                lam=0.95,
            )

        experience = make_experience(
            sample=sample,
            old_action_log_probs=old_action_log_probs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            reward_scores=reward_scores,
            kl=kl,
        )

        experiences.append(experience)
    
    return experiences


def train(
    config,
    models,
    optimizer_actor,
    optimizer_critic,
    prompts_dataloader,
    writer=None,
):
    buffer = ExperienceBuffer()
    global_step = 0

    for episode in range(config.episodes):
        print(f"episode {episode + 1}/{config.episodes}")

        for prompt_batch in prompts_dataloader:
            # 1. rollout：actor 生成回答
            samples_list = generate_samples(
                prompts=prompt_batch,
                actor_model=models["actor_model"],
                actor_tokenizer=models["actor_tokenizer"],
                device=models["device"],
                max_length=config.max_length,
                max_new_tokens=config.max_new_tokens,
                n_samples_per_prompt=config.n_samples_per_prompt,
                micro_rollout_batch_size=config.micro_rollout_batch_size,
            )

            # 2. 把 samples 转成 PPO experiences
            experiences = generate_experiences(
                samples_list=samples_list,
                models=models,
                config=config,
            )

            buffer.clear()
            buffer.append(experiences)

            train_dataloader = DataLoader(
                buffer,
                batch_size=config.micro_train_batch_size,
                shuffle=True,
                collate_fn=collate_experiences,
            )

            # 3. 同一批 rollout 数据训练多轮
            for epoch in range(config.max_epochs):
                for batch in train_dataloader:
                    metrics = train_step(
                        batch=batch,
                        models=models,
                        optimizer_actor=optimizer_actor,
                        optimizer_critic=optimizer_critic,
                        clip_eps=0.2,
                    )

                    if writer is not None:
                        writer.add_scalar(
                            "loss/policy",
                            metrics["policy_loss"],
                            global_step,
                        )
                        writer.add_scalar(
                            "loss/value",
                            metrics["value_loss"],
                            global_step,
                        )

                    print(
                        f"step={global_step} "
                        f"epoch={epoch} "
                        f"policy_loss={metrics['policy_loss']:.4f} "
                        f"value_loss={metrics['value_loss']:.4f}"
                    )

                    global_step += 1

            # 4. 当前 rollout 数据已经用完，清空
            buffer.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()