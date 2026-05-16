# When RL Meets Adaptive Speculative Training: A Unified Training-Serving System

Paper: Junxiong Wang, Fengxiang Bie, Jisen Li, Zhongzhu Zhou, Zelei Shao, Qingyang Wu, Yinghui Liu, Yubo Wang, Avner May, Sri Yanamandra, Tri Dao, Percy Liang, Ce Zhang, Ben Athiwaratkun, Shuaiwen Leon Song, Chenfeng Xu, Xiaoxia Wu, "When RL Meets Adaptive Speculative Training: A Unified Training-Serving System", arXiv:2602.06932v3, 2026.

Website: https://aurora-spec-ai.github.io/

## High-Level Summary

This paper introduces **Aurora**, a unified training-serving system for speculative decoding. Its main claim is that speculative decoding should not be treated as a static "train the drafter offline, then deploy it" pipeline. Instead, the drafter/speculator should be continuously trained from live inference traces and periodically hot-swapped back into the serving system.

Speculative decoding uses a lightweight draft model to propose multiple future tokens, then asks a stronger target model to verify them. If the draft model aligns well with the target model, many tokens are accepted and inference is faster. In practice, however, a drafter trained offline can become stale because production traffic shifts, prompt patterns change, serving kernels and batch sizes affect actual speedup, and target models may be updated.

Aurora closes this loop. The inference server serves requests with speculative decoding while streaming accepted and rejected draft outcomes into a buffer. A separate training server consumes that data, updates the speculator asynchronously, and pushes new speculator weights back into the inference server without interrupting active service.

The paper frames this as an **asynchronous RL-style system**: the speculator is the policy, the target model plus verifier is the environment, accepted tokens are positive feedback, and rejected draft branches provide negative or counterfactual feedback.

## Motivation

The paper identifies three main problems with conventional speculative decoding deployment.

First, **day-0 support is hard**. New frontier models are released frequently, but a useful speculator usually requires offline data generation, training, calibration, and deployment. This delays acceleration for new models.

Second, **offline metrics do not reliably predict production speedup**. Acceptance rate matters, but real speedup also depends on the serving stack: kernels, precision, batching, scheduling, hardware, draft overhead, and target-model architecture. A drafter with better offline acceptance can still produce worse end-to-end throughput.

Third, **domain drift degrades static drafters**. Production traffic is local and time-varying. A drafter trained on a broad static corpus may be suboptimal for the current request stream, and a previously good drafter may become stale after the target model or traffic mix changes.

Aurora's answer is to move speculative training into the serving loop. Instead of optimizing a drafter globally over a fixed dataset, it optimizes for the current deployed distribution and measures utility directly from live inference.

## Background: Speculative Decoding

In speculative decoding, a draft model proposes `gamma` future tokens and a target model verifies them. If the average acceptance probability is `alpha`, the expected number of produced tokens per verifier step is approximately:

`E[L] = (1 - alpha^(gamma + 1)) / (1 - alpha)`

If `c` is the draft-to-target cost ratio, the wall-clock speedup is approximately:

`Speedup ~= E[L] / (1 + gamma c)`

This equation highlights a systems tradeoff. A deeper speculator may improve acceptance but cost more per draft step. A smaller speculator may be cheap but poorly aligned. Aurora avoids relying only on offline estimates by evaluating the speculator in the actual serving loop.

## System Architecture

Aurora has two decoupled components.

The **Inference Server** runs an SGLang speculative decoding engine with a fixed target model and a draft/speculator model. For each request, it generates draft tokens, verifies them with the target, returns the response, and streams serving traces into a distributed data buffer. The traces include accepted prefixes, rejected branches, target logits, hidden states needed for EAGLE-style training, and rejection-trajectory metadata.

The **Training Server** runs asynchronously on separate GPU resources. It reads batches from the buffer, trains a copy of the draft model, and periodically sends updated speculator weights back to the inference server. The inference server hot-swaps those weights without downtime.

The design is intentionally lazy and non-disruptive. Unlike many asynchronous RL systems, Aurora does not aggressively push every new policy update to actors. Frequent synchronization can invalidate caches, create latency jitter, or cause transient regressions. Aurora therefore treats synchronization frequency as a serving-system control knob.

## Implementation Details

Aurora is built around GPU-aware RPC between inference and training processes. The implementation uses batched RPC transfers with TensorPipe and expandable CUDA memory segments to reduce memory fragmentation and enable efficient GPU-to-GPU communication.

One important systems idea is the **zero-copy target model design**. The training server does not load a duplicate target model. Instead, the inference server transmits precomputed target information, such as selected hidden states, logits, input/output tokens, and rejection metadata. This avoids loading an 8B-70B target model on the training side and lets the training server load only the smaller draft model.

The paper also introduces an efficient **Tree Attention** training mechanism. Speculative decoding creates a tree-like structure of accepted and rejected branches. Training each branch separately would be expensive, so Aurora builds a custom attention mask that respects the speculative tree's causal structure and processes accepted and rejected branches in one batched forward/backward pass.

The appendix also describes compatibility with disaggregated serving systems. Aurora's training server can be treated as a third role next to prefill and decode servers, consuming hidden states and logits over the same communication fabric.

## Training Objective

Aurora trains the draft model using both accepted and rejected speculative outcomes.

The accepted-token objective is imitation: train the draft model to match the target model on verifier-approved continuations.

The rejected-branch objective is counterfactual feedback: rejected draft branches show what the speculator should avoid. The paper calls this **Discard Sampling**. It applies a KL-based objective to discarded/rejected sequences, optionally with top-k filtering to focus on high-probability disagreements and reduce gradient noise.

The loss is:

`L = E_accept[KL(p_target || p_draft)] + lambda_discard E_discard[KL(p_target || p_draft)]`

The paper experiments with variants such as forward KL, reverse KL, next-token prediction auxiliary loss, and discard-token training. A key empirical finding is that most gains come from closing the loop with online updates; discard-token training gives extra benefit mainly when the lookahead is larger and rejected branches contain more useful signal.

## Experiments

Aurora treats a prompt corpus as a live request stream rather than a supervised static dataset. It does not use ground-truth assistant responses. The serving system processes requests sequentially, verifies speculative drafts, and uses only the resulting inference traces for online training.

The main online-training experiments use a 40k-44k prompt stream spanning:

- Mathematical reasoning from GSM8K.
- Text-to-SQL from Spider.
- Code from CodeSearchNet.
- Finance from Finance-Alpaca.
- General conversational instructions.

The paper evaluates two traffic patterns:

- **Mixed streams**, where prompts are shuffled to approximate stationary traffic.
- **Ordered streams**, where prompts are grouped by domain to induce abrupt distribution shifts.

The core metrics are:

- **Speculative acceptance length:** average number of draft tokens accepted per verification step.
- **Per-request throughput:** `(T_input + T_output) / t_request`, measuring user-perceived tokens per second.

## Main Results

In day-0 experiments with Qwen3-8B and EAGLE-3, Aurora can start from a randomly initialized speculator and improve during serving.

In the mixed-stream setting, the acceptance length reaches 3.08, surpassing both the static pretrained baseline at 2.63 and a pretrained-then-finetuned baseline at 2.99. Throughput stabilizes at 302.3 tokens/s.

In the ordered domain-shift setting, the untrained speculator starts with near-zero acceptance and converges to 2.46 accepted tokens within about 10,000 requests, close to the pretrained baseline's 2.57. Throughput stabilizes at 295.6 tokens/s, competitive with the static speculator's 288.8 tokens/s.

These results support the paper's central claim: online adaptation can make a cold-start speculator useful during live serving, reducing the need for a long offline pretraining cycle.

## Synchronization Policy

The synchronization study shows a tradeoff between adaptation speed and serving stability.

More frequent updates reduce policy staleness and improve post-shift acceptance recovery, but they can hurt throughput because weight refreshes disrupt serving. Very lazy updates are stable but adapt slowly. The paper finds that moderately lazy synchronization gives a good Pareto point, preserving throughput while retaining most of the adaptation benefit.

This is one of the paper's most important systems lessons. In ordinary asynchronous RL, frequent synchronization is often desirable. In production speculative decoding, synchronization itself has user-visible cost, so the optimal policy is not simply "update as fast as possible."

## Speculative Algorithm Findings

The paper compares static drafters against several online adaptation variants:

- Frozen pretrained draft model.
- Aurora with forward KL on accepted tokens.
- Aurora with reverse KL on accepted tokens.
- Aurora with reverse KL plus next-token prediction.
- Aurora with discard/rejected-token training.

The main result is that online fine-tuning consistently improves acceptance length and throughput compared with the frozen baseline. Most of the benefit comes from using on-policy serving data and continuously adapting to the current traffic. More complex objectives provide smaller incremental improvements.

Discarded tokens become more useful when the lookahead is larger. With a short lookahead, a strong pretrained speculator may already accept many tokens, leaving limited rejected-branch signal. With lookahead 10, rejected branches provide more useful supervision and improve performance more clearly.

The paper also studies batch size. Aurora's relative throughput gains are larger at smaller batch sizes because baseline decoding is less amortized and each skipped target step matters more. At larger batch sizes, the target model is already better utilized, and speculative overhead becomes a larger fraction of total cost.

## Scaling to Frontier Open Models

Aurora is evaluated on two large recent open models:

- **MiniMax M2.1**, a 229B-parameter MoE model served on 4 H200 GPUs with tensor parallelism in FP8.
- **Qwen3-Coder-Next**, an 80B-total-parameter MoE/hybrid model served on 4 H200 GPUs with tensor and expert parallelism in FP8.

For MiniMax M2.1, Aurora raises mean accepted draft length to about 2.8 and achieves about 1.45x throughput improvement in the main trace. Table results show speedups from 1.57x at batch size 1 to 1.25x at batch size 32.

For Qwen3-Coder-Next, Aurora raises accepted draft length above 3 and reports around 1.23x throughput improvement in the main deployment. Table results show up to 1.51x at batch size 1, 1.23x at batch size 8, and diminishing returns at larger batch sizes. At batch size 32, speculative overhead can dominate and make decoding slightly slower than the baseline.

## RL and Online Learning Perspective

Aurora is explicitly presented as an **asynchronous reinforcement-learning-style system**, although it is not a conventional RL algorithm like PPO or Q-learning. Its contribution is to import the actor-learner systems pattern into speculative decoding and adapt it to production serving constraints.

### State, Action, Reward, Transition

The online speculative decoding process can be mapped into RL terms:

- **Policy:** the draft/speculator model.
- **Environment:** the target model plus verifier.
- **State:** the current prompt, generated prefix, target-model hidden states, and speculative decoding context.
- **Action:** propose a token sequence or speculative tree.
- **Reward:** accepted tokens provide positive reward; rejected branches provide zero, negative, or counterfactual feedback.
- **Transition:** the verifier accepts a prefix, rejects the remaining proposals, updates the generated text, and emits traces for training.
- **Objective:** maximize expected accepted length and end-to-end serving throughput.

This is a short-horizon online learning loop. Each speculative step gives immediate structured feedback, and the policy is updated continuously from recent production traces.

### Why It Is RL-Style Instead of Standard Supervised Training

Offline drafter training resembles supervised distillation: collect target-model logits or activations on a static corpus, then train the draft model to imitate the target. Aurora instead trains from the interaction between the deployed policy and the verifier under live traffic.

The distinction matters because the data distribution is policy- and traffic-dependent. A speculator's mistakes determine which branches are rejected, and the serving workload determines which contexts appear. This makes the training signal closer to on-policy or off-policy RL experience than to a fixed supervised dataset.

That said, Aurora's optimizer is still based on KL/imitation losses rather than explicit policy-gradient return optimization. The RL framing is mainly about the system loop, feedback structure, asynchronous actor-learner architecture, and serving-aligned objective.

### Positive and Negative Feedback

A key RL idea in the paper is using both success and failure signals.

Accepted tokens are similar to demonstrations: they show the speculator which proposals matched the verifier. Rejected branches are counterfactual negative feedback: they show what the current policy proposed but the verifier refused. Discard Sampling turns these rejected branches into a training signal.

This is important because speculative decoding naturally generates failure data during normal serving. Aurora does not need a separate exploration stage to collect mistakes; mistakes occur as part of the draft-verify process.

### Asynchrony and Staleness

Aurora follows an asynchronous actor-learner pattern:

- Serving replicas act as actors and produce experience.
- A distributed buffer stores experience.
- A learner trains the speculator on separate GPUs.
- Updated weights are periodically pushed back to serving replicas.

The paper highlights a difference from standard asynchronous RL. In many RL systems, the main concern is learner throughput and sample efficiency, so stale policies are undesirable. In Aurora, service stability is equally important. Frequent policy refresh can hurt latency, invalidate caches, and reduce throughput. Therefore, Aurora deliberately accepts some policy staleness in exchange for predictable serving performance.

### Reward Alignment With Serving Utility

Aurora's reward is not just "predict the target model well." The true objective is serving efficiency: accepted length, latency, tokens per second, and cost per output token.

This is why the paper argues that offline acceptance metrics are insufficient. A model with higher acceptance may still be slower if the draft overhead is too high or if synchronization affects serving. Aurora closes the loop by measuring and improving performance in the actual serving environment.

From an RL perspective, this is a reward-specification lesson: optimize the feedback that reflects deployed utility, not a proxy that is only loosely correlated with it.

### Non-Stationarity

Aurora is designed for non-stationary traffic. The request stream can shift between math, SQL, code, finance, and conversational domains. It can also shift because users change behavior, prompts are modified, or the target model is updated.

The ordered-stream experiments intentionally create abrupt domain shifts. The frozen speculator degrades after shifts, while online variants recover by training on new serving traces. This makes Aurora a test-time adaptation system as much as a speculative decoding system.

### Exploration and Safety

Aurora does not rely on explicit RL exploration in the usual sense. Speculative decoding is lossless with respect to the target model because the verifier rejects bad draft tokens. This creates a useful safety property: an imperfect or randomly initialized speculator may hurt latency initially, but it should not change output quality as long as the verifier is correct.

That property makes day-0 deployment possible. The system can deploy a weak speculator, collect feedback, and improve online while the target model preserves correctness. The remaining risk is serving performance, not generation quality.

### RL Takeaways

For an RL project, Aurora is useful because it shows how RL systems ideas can be applied outside classic reward-maximizing agents.

The main lessons are:

- Treat inference as an online interaction loop, not just static model execution.
- Use production feedback to train the component that directly affects serving efficiency.
- Separate actors and learners to keep serving responsive.
- Control policy synchronization because staleness and update overhead are deployment-level tradeoffs.
- Use both positive and negative feedback from natural verifier outcomes.
- Evaluate the true deployed objective, not only offline proxy metrics.

## Limitations and Open Questions

Aurora's strongest results depend on careful systems engineering. The training loop requires extra GPU resources, RPC infrastructure, buffering, and safe hot-swapping. This is more complex than loading a static drafter.

A cold-start speculator can initially reduce throughput before it adapts. The paper shows recovery, but production systems would still need guardrails, rollback logic, or traffic shaping to avoid hurting users during early adaptation.

The paper frames training as asynchronous RL, but the actual optimization remains KL-based imitation/fine-tuning. More direct RL objectives could be explored, especially objectives that optimize latency or throughput rather than accepted-token length.

Synchronization policy is treated empirically. A more formal controller could adapt refresh frequency based on observed throughput, acceptance length, cache state, and drift.

Finally, speculative decoding gains diminish at larger batch sizes and can become negative when verification overhead dominates. Aurora improves the speculator, but it does not eliminate the fundamental systems tradeoff between draft cost, verification cost, and batch-level hardware utilization.

## Key Takeaways

Aurora reframes speculative decoding as a live adaptive system. Instead of training a drafter offline and hoping it remains useful, Aurora continuously trains the speculator from production traces and pushes updates back into serving.

The paper's core ideas are:

- Close the training-serving loop for speculative decoding.
- Treat accepted and rejected drafts as RL-style feedback.
- Use asynchronous actor-learner infrastructure to train without blocking inference.
- Hot-swap speculator updates with lazy synchronization.
- Optimize for actual serving throughput under the current traffic distribution.

The result is a system that supports day-0 speculator deployment, adapts to domain shift, and scales to large open models while preserving the lossless verification property of speculative decoding.
