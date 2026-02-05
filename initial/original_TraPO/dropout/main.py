import torch
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

M = 2
# 模拟数据：假设有 N 个问题，每个问题对应熵 H_x 和正确率 A_x
N = 512
H = torch.rand(N) * M  # 熵范围 [0, 2]
# A = torch.rand(N)        # 正确率范围 [0, 1]
possible_values = torch.tensor([0.01, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.99])
indices = torch.randint(0, len(possible_values), (N,))
A = possible_values[indices]

# 变分参数：每个问题的 dropout 率 phi_x ~ Beta(alpha_x, beta_x)
alpha = torch.ones(N, requires_grad=True)  # 初始 alpha=1
beta = torch.ones(N, requires_grad=True)   # 初始 beta=1

# 优化器
optimizer = torch.optim.Adam([alpha, beta], lr=0.01)

# 先验参数：phi_x 的先验为 Beta(alpha_0, beta_0)
alpha_0 = 1.0
beta_0 = 1.0

# 变分推断
for step in range(5000):
    # 变分分布 q(phi_x | alpha, beta)
    q_phi = dist.Beta(alpha, beta)
    
    # 采样 phi_x
    phi_x = q_phi.rsample()
    
    # 定义似然 p(H_x | phi_x) 和 p(A_x | phi_x)
    # 先验知识：
    #   1. H_x 低 → phi_x 高：假设 p(H_x | phi_x) = Exponential(rate=1 + 10*phi_x)
    #   2. A_x 高 → phi_x 高：假设 p(A_x | phi_x) = Beta(concentration1=10*phi_x, concentration0=10*(1-phi_x))
    likelihood_H = dist.Exponential(1.0 + 10.0 * phi_x)
    likelihood_A = dist.Beta(10.0 * phi_x, 10.0 * (1.0 - phi_x))
    
    # 计算对数似然 log p(H_x | phi_x) + log p(A_x | phi_x)
    log_likelihood = likelihood_H.log_prob(H) + likelihood_A.log_prob(A)
    
    # 计算 KL 散度 KL(q(phi_x) || p(phi_x))
    prior = dist.Beta(alpha_0, beta_0)
    kl_divergence = dist.kl_divergence(q_phi, prior)
    
    # ELBO = 似然期望 - KL 散度
    elbo = log_likelihood.mean() - kl_divergence.mean()
    
    # 损失 = -ELBO
    loss = -elbo
    
    # 优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, ELBO: {elbo.item():.2f}, Mean phi_x: {phi_x.mean().item():.3f}")

# 推断结果：每个问题的 dropout 率 phi_x = alpha / (alpha + beta)
inferred_phi = alpha.detach() / (alpha.detach() + beta.detach())

# 打印前 5 个问题的熵、正确率和推断的 dropout 率
print("\nInferred dropout rates (phi_x):")
for i in range(20):
    # print(alpha.detach()[:10], beta.detach()[0:10])
    print(f"Question {i}: H_x={H[i]:.3f}, A_x={A[i]:.3f} → phi_x={inferred_phi[i]:.3f}")



# 设置绘图风格
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))

# 1. 绘制推断的 phi_x 的分布
plt.subplot(2, 2, 1)
sns.histplot(inferred_phi.numpy(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of Inferred Dropout Rates (phi_x)')
plt.xlabel('phi_x')
plt.ylabel('Frequency')

# 2. 绘制 phi_x 与 H 的关系
plt.subplot(2, 2, 2)
plt.scatter(H.numpy(), inferred_phi.numpy(), alpha=0.6, color='salmon')
plt.title('Dropout Rate vs Entropy')
plt.xlabel('Entropy (H)')
plt.ylabel('Inferred phi_x')
plt.plot([0, 10], [inferred_phi.mean(), inferred_phi.mean()], 'k--', lw=1)  # 平均线

# 3. 绘制 phi_x 与 A 的关系
plt.subplot(2, 2, 3)
plt.scatter(A.numpy(), inferred_phi.numpy(), alpha=0.6, color='lightgreen')
plt.title('Dropout Rate vs Accuracy')
plt.xlabel('Accuracy (A)')
plt.ylabel('Inferred phi_x')
plt.plot([0, 1], [inferred_phi.mean(), inferred_phi.mean()], 'k--', lw=1)  # 平均线

# 4. 绘制 alpha 和 beta 的分布
plt.subplot(2, 2, 4)
plt.scatter(alpha.detach().numpy(), beta.detach().numpy(), alpha=0.6, color='purple')
plt.title('Distribution of Alpha and Beta Parameters')
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.plot([0, max(alpha.detach().max(), beta.detach().max())], [0, max(alpha.detach().max(), beta.detach().max())], 'k--', lw=1)  # 对角线

plt.tight_layout()
plt.savefig("/ossfs/workspace/aml0/484999/code/LUFFY/dropout/fig/alpha-beta-end.png", dpi=300)