# Multi-Drafter Speculative Decoding with Alignment Feedback

Paper: Taehyeon Kim, Hojung Jung, Se-Young Yun, "Multi-Drafter Speculative Decoding with Alignment Feedback", arXiv:2604.05417v1, 2026.

## High-Level Summary

This paper proposes **MetaSD**, a speculative decoding framework that uses multiple drafter models instead of relying on a single small model. Standard speculative decoding accelerates LLM inference by letting a smaller drafter propose future tokens, then asking the target LLM to verify them in parallel. This preserves the target model's output distribution, but the speedup depends heavily on how well the drafter matches the target model on the current prompt and generation context.

The central problem is that a single drafter is usually specialized or biased toward some task distribution. A code-specialized drafter may work well for code but poorly for translation; a summarization drafter may work early in a long answer but become less suitable when the generation shifts into reasoning. MetaSD addresses this by treating drafter selection as an online decision problem: at each speculative decoding round, choose one drafter from a pool, observe how well it aligns with the target model, and update future choices.

The key contribution is to use **alignment feedback** as a reward signal for a **multi-armed bandit (MAB)** algorithm. Each drafter is an arm. Each speculative decoding cycle is one bandit round. The algorithm balances exploration of different drafters with exploitation of the drafter that currently appears most aligned with the target model.

## Motivation

Speculative decoding is useful because LLM inference is often memory-bandwidth bound. Instead of generating one target-model token per forward pass, a smaller drafter proposes a block of tokens, and the target model verifies that block in parallel. Accepted tokens reduce the number of expensive target-model calls.

However, the performance of speculative decoding depends on the acceptance rate. If the drafter's token distribution is close to the target model's distribution, more proposed tokens are accepted and inference is faster. If the drafter is misaligned, few tokens are accepted and speculative decoding provides little benefit.

The paper argues that modern deployments often face diverse, changing workloads, so a single drafter is brittle. Static multi-drafter ensembles are also unattractive because drafting and verifying with all drafters increases compute. MetaSD instead keeps the standard speculative decoding compute pattern: only one drafter is active at each round, but the active drafter is chosen adaptively.

## Method

MetaSD maintains a pool of `K` drafters. At each decoding round:

1. A bandit policy selects one drafter.
2. That drafter proposes up to `Nmax` future tokens.
3. The target model verifies the proposed tokens.
4. The accepted tokens are appended to the generation.
5. The algorithm computes an alignment reward from the drafter and target distributions.
6. The bandit policy updates its estimate of that drafter's quality.

The paper instantiates this framework mainly with **UCB** (Upper Confidence Bound), called **MetaSD-UCB**. UCB selects the drafter with the best empirical reward plus an exploration bonus. This naturally tries under-sampled drafters while converging toward the best-performing one.

The framework supports both:

- **Black-box speculative decoding**, where drafters are independent models that only use normal token-level inputs and outputs.
- **White-box speculative decoding**, where drafters can use target-model internal representations, as in EAGLE-style draft models.

## Reward Design: Block Divergence

The paper's main reward-design idea is **Block Divergence (BD)**. The reward is based on the total variation distance between the target model's next-token distribution and the drafter's next-token distribution over the speculative block.

Intuitively:

- If the drafter distribution is close to the target distribution, BD reward is high.
- If the drafter distribution is far from the target distribution, BD reward is low.

The paper compares BD against a more direct reward, **Block Efficiency (BE)**, which is the number of accepted tokens divided by `Nmax`. BE is simple and directly tied to speedup, but it can be sparse: many rounds may yield zero or low acceptance, producing noisy feedback. BD is denser because it measures distributional alignment even when few tokens are accepted.

The paper proves that maximizing expected BD is equivalent to maximizing expected accepted tokens under its assumptions, while BD gives a stronger feedback signal than BE in many cases. Empirically, BD has lower variance and better separation between good and bad drafters, which helps the bandit converge faster.

## Regret Objective

Standard bandit regret assumes a fixed number of rounds. MetaSD does not fit this directly because the number of rounds is itself affected by the policy: better drafters accept more tokens per round, so generation finishes in fewer rounds.

The paper defines **stopping time regret**:

`REG(pi, B) = E[tau(pi, B)] - E[tau(pi*, B)]`

where `B` is the target number of generated tokens and `tau(pi, B)` is the number of decoding rounds needed by policy `pi` to generate `B` tokens. Minimizing this regret is equivalent to maximizing expected accepted tokens and therefore aligns with speculative decoding's actual objective.

For MetaSD-UCB with BD reward, the paper proves a logarithmic regret upper bound in the target sequence length `B`. The bound improves with larger `Nmax` because each speculative block provides more distributional observations, reducing reward variance. The analysis also argues that BD yields tighter constants than BE.

## Experimental Setup

The paper evaluates MetaSD on both black-box and white-box speculative decoding.

Target model:

- Vicuna 7B.

Black-box drafters:

- Vicuna 68M drafters specialized for different task domains.

White-box drafters:

- EAGLE-style drafters fine-tuned on different task-specific datasets.

Tasks:

- Code generation from MT-Bench.
- Translation from WMT16.
- Summarization from CNN/DailyMail.
- Question answering from Natural Questions.
- Math reasoning from GSM8K.
- Multilingual translation tasks.
- Out-of-domain settings such as finance and RAG-style inputs.

Baselines include single specialized drafters, one-size-fits-all drafters, PLD, Lookahead, BPD, Medusa, EAGLE, EXP3, Sequential Halving, random selection, drafting with all drafters, and classifier/MoE-style routing.

## Main Results

MetaSD-UCB generally outperforms static single-drafter selection because it can adaptively choose the drafter that is currently most aligned with the target model.

For black-box speculative decoding, specialized drafters perform best only on their own domains and degrade on unrelated tasks. MetaSpS-UCB remains competitive across tasks and often approaches or surpasses the best static drafter without knowing the task label in advance.

For white-box speculative decoding, MetaEagle-UCB similarly improves robustness across tasks. It gets strong speedups while preserving the lossless decoding property of speculative decoding.

The paper also shows that naively using all drafters is inefficient. Drafting with all drafters may include a strong candidate, but the verification cost grows and speedup can collapse. MetaSD avoids that by using only one drafter per round.

Compared with MoE-style classifier routing, MetaSD is more robust to perturbed prompts and out-of-domain inputs. Classifier routing makes a static input-level decision, while MetaSD updates decisions token-by-token using target-model alignment feedback.

## RL and Online Learning Perspective

MetaSD is best understood as an **online reinforcement learning / bandit control problem embedded inside inference**. It does not train the target LLM or drafter models with policy gradients, but it uses RL-style sequential decision-making to choose actions during generation.

### State, Action, Reward, Transition

The implicit environment is the ongoing LLM generation process.

- **State:** the current prompt plus generated prefix, target length progress, and the current reward estimates for each drafter.
- **Action:** choose one drafter from the drafter pool.
- **Reward:** alignment feedback from the selected drafter, mainly BD reward.
- **Transition:** execute one speculative decoding round, accept some number of tokens, update the generated prefix, and update bandit statistics.
- **Episode termination:** stop once `B` target tokens have been generated.

This is not a full Markov decision process in the paper's main formulation because the policy does not learn a rich context-conditioned value function. It is closer to a stochastic MAB where each arm corresponds to a drafter. Still, because the generation prefix changes over time and the best drafter may change with context, the problem has clear connections to contextual and non-stationary RL.

### Why Bandits Instead of Full RL?

The paper chooses MAB algorithms because they are lightweight, training-free, and suitable for inference-time adaptation. A full RL policy would require collecting trajectories, defining a larger state representation, training a router, and maintaining generalization across prompts. That would add complexity and possibly extra compute.

Bandits are a pragmatic fit because the immediate action has a measurable local reward: how well the selected drafter aligns with the target model during the current speculative block. The algorithm can improve online using only this local feedback.

### Exploration vs. Exploitation

The core RL tradeoff is exploration versus exploitation:

- Exploration: try drafters whose quality is uncertain.
- Exploitation: use the drafter with the best observed alignment.

UCB handles this by adding an uncertainty bonus to each drafter's empirical reward. Drafters with few observations receive larger bonuses, so they are sampled early. As evidence accumulates, the policy concentrates on the drafter with the strongest BD reward.

This matters because the best drafter is not known beforehand and may differ by task. Without exploration, the system could overcommit to a poor drafter. Without exploitation, it would waste rounds on inferior drafters and lose speedup.

### Reward Shaping

BD reward is a form of reward shaping. The obvious reward is the number of accepted tokens, but that reward can be noisy and sparse. BD gives a denser signal by measuring distributional closeness between the drafter and target model across the block.

From an RL perspective, the paper's reward design is important because it improves credit assignment. The bandit gets useful information even when acceptance outcomes are not very informative. The theoretical and empirical results both suggest that better-shaped rewards lead to faster identification of the best arm.

### Regret as the RL Objective

The paper's stopping time regret is the right objective for an inference-control problem. Standard cumulative reward regret does not directly capture latency because the number of decision rounds is not fixed. A better policy shortens the episode by accepting more tokens per round.

Thus, minimizing stopping time regret is equivalent to minimizing decoding rounds, which corresponds to maximizing inference speedup. This is an important adaptation of bandit theory to a variable-horizon generation process.

### Non-Stationarity

The paper explicitly discusses non-stationary settings. In generation, the best drafter may change within the same output. For example, a prompt may begin with summarization and later require reasoning. A summarization drafter could be best early, while a reasoning drafter becomes better later.

The appendix discusses extensions such as:

- **Discounted UCB**, which downweights old observations.
- **Sliding-window UCB**, which estimates rewards using only recent rounds.
- **EXP3**, which can handle adversarial reward changes.

The paper notes that standard non-stationary bandit definitions are awkward here because distribution shifts occur at the token level, while the bandit acts at the speculative-block level. A single decoding round may span multiple context shifts. This leaves room for future theoretical work.

Empirically, the paper includes a non-stationary translation task where each query requires translating two languages. MetaSD-UCB achieves a speedup ratio of 1.722, exceeding the best static drafter/upper-bound router speedup of 1.581 in that setup. The reason is that a dynamic bandit can switch behavior during generation, while a static router is limited by a single drafter assignment.

### Relation to Contextual Bandits and Routing

The paper's current method is mostly non-contextual within a query: it uses observed rewards rather than a learned state encoder to predict the best drafter from the prompt. This is simpler and more robust than classifier routing, but it may spend early rounds exploring.

A natural extension is a contextual bandit that uses prompt embeddings, hidden states, or task features as context while still updating from BD reward. This could reduce cold-start exploration, especially when the prompt clearly indicates a domain. The paper lists contextual bandits as a limitation/future direction.

### RL Takeaways

The main RL lesson is that inference-time systems can benefit from lightweight online decision-making without training a large policy. MetaSD converts drafter routing into a bandit problem with a carefully shaped reward and a task-specific regret objective. This gives a practical middle ground between static routing and expensive learned controllers.

For an RL project, the most relevant ideas are:

- Designing rewards that are dense, low-variance, and aligned with the true system objective.
- Choosing a regret objective that matches the actual episode cost, not just generic cumulative reward.
- Handling non-stationarity when the environment changes during an episode.
- Comparing static supervised routing against online adaptive control.
- Extending non-contextual bandits into contextual or non-stationary bandit policies.

## Limitations and Future Directions

MetaSD still has several limitations.

First, switching drafters can incur KV-cache costs. The paper discusses Sequential Halving as one way to reduce switching overhead, but real serving systems would need careful cache management.

Second, UCB assumes relatively stable reward distributions. The paper discusses non-stationary extensions, but the theoretical guarantees for token-level distribution shifts inside speculative blocks remain open.

Third, the method may spend early decoding rounds exploring suboptimal drafters. Contextual bandits or learned priors could reduce this cost.

Fourth, maintaining multiple drafters requires additional memory. The paper argues that only one drafter is active at a time, so memory bandwidth remains comparable to single-drafter speculative decoding, but total DRAM/VRAM capacity requirements still matter in deployment.

Finally, the paper's experiments center on Vicuna 7B and specific drafter families. More evidence would be needed for larger target models, production batching, and modern serving stacks.

## Key Takeaways

MetaSD reframes speculative decoding as an online drafter-selection problem. Instead of picking one drafter globally, it adapts during inference using alignment feedback from the target model.

The paper's strongest idea is the combination of:

- Multiple heterogeneous drafters.
- Bandit-based online selection.
- Block Divergence as a dense alignment reward.
- Stopping time regret as the objective matched to inference speed.

The result is a training-free, lossless speculative decoding framework that improves robustness across tasks and can adapt to changing generation contexts.
