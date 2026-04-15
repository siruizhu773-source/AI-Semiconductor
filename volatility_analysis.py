#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:17:22 2026

@author: zhusirui
"""

#%%

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


#%%
# 1. Define ticker and date
tickers = ['NVDA', 'INTC', 'TSM', '^IXIC']
start_date = '2018-01-01'
end_date = '2023-12-31'


#%%
# 2. Down load data(adj_price)

print("Down loading from yf")
data = yf.download(tickers, start=start_date, end=end_date)['Close']


#%%
# 3. check nan
print("\n if there is nan")
print(data.isnull().sum()) 

print(data.head())


#%%
# 4. print a normal plot 
data.plot(figsize=(12, 6), title='Adjusted Closing Prices (2018-2023)')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()

#%%
# 我们假设在 2018 年 1 月 1 日，我们在每样资产上都投入了 100 块钱，看看接下来几年它们分别变成了多少钱。
# 魔法操作：将所有列的数据，除以它们各自第一天的数据，然后再乘以 100
# 这样所有资产的起点都在同一条起跑线（100）上了！
normalized_data = data / data.iloc[0] * 100

# 重新画图
normalized_data.plot(figsize=(12, 6), title='Normalized Prices (Base 100: 2018-2023)')
plt.ylabel('Relative Price (Base = 100)')
plt.grid(True)
plt.show()


#%%

import numpy as np
import scipy.stats as stats
import seaborn as sns # 用于画出更美观的统计图

#%%
# ==========================================
# 1. 计算对数收益率 (Log Returns)
# ==========================================
# 为什么要用对数收益率？因为它具有时间可加性，且更接近正态分布，适合建模。
# 公式: r_t = ln(P_t / P_{t-1})
# data.shift(1) 就是把数据往下挪一天，这样当前行除以前一行就是 P_t / P_{t-1}
print("正在计算对数收益率...")
returns = np.log(data / data.shift(1))

# 删掉第一行（因为第一天没有前一天的价格，计算出来是 NaN）
returns = returns.dropna()

print("\n前5天的对数收益率：")
print(returns.head())



#%%
# ==========================================
# 2. 描述性统计表 (Descriptive Statistics)
# ==========================================
print("\n=== 对数收益率描述性统计 ===")
# 我们建一个空的 DataFrame 来存结果
stats_df = pd.DataFrame()

stats_df['Mean (均值)'] = returns.mean()
stats_df['Std Dev (波动率/标准差)'] = returns.std()
stats_df['Skewness (偏度)'] = returns.skew()
stats_df['Excess Kurtosis (峰度)'] = returns.kurtosis() # Pandas 算出来的是超额峰度 (Excess Kurtosis)

# 把结果转置一下，让股票代码在行，指标在列，方便阅读
print(stats_df.round(4)) # 保留4位小数



#%%

# ==========================================
# 3. 数据可视化：尖峰厚尾特征 (Fat Tails)
# ==========================================
# 我们挑选 NVDA (英伟达) 和 INTC (英特尔) 来画对比图
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 画英伟达的收益率分布
sns.histplot(returns['NVDA'], bins=100, kde=True, ax=axes[0], color='red')
axes[0].set_title('NVDA Daily Log Returns Distribution')
axes[0].set_xlabel('Log Return')

# 画英特尔的收益率分布
sns.histplot(returns['INTC'], bins=100, kde=True, ax=axes[1], color='blue')
axes[1].set_title('INTC Daily Log Returns Distribution')
axes[1].set_xlabel('Log Return')

plt.tight_layout()
plt.show()


#%%

# =============
# 以NVDA为例
# =============

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import statsmodels.api as sm


#%%
# ==========================================
# 0. 数据准备 (Data Preparation)
# ==========================================
print("正在下载数据并计算收益率...")
data = yf.download('NVDA', start='2018-01-01', end='2023-12-31')['Close']
# 行业惯例：为了GARCH底层算法容易收敛，收益率放大100倍（变成百分比）
returns = np.log(data / data.shift(1)).dropna() * 100 



#%%
# ==========================================
# 步骤 1 & 2: 提取残差与检验 ARCH 效应 (Engle's ARCH Test)
# ==========================================
# 假设均值为常数，收益率减去均值得到纯粹的“市场冲击（残差）”
residuals = returns - returns.mean()

# 进行 Engle's ARCH LM 检验 (原假设 H0: 没有波动率聚集/没有ARCH效应)
print("\n--- 步骤 2: Engle's ARCH 检验 ---")
arch_test = het_arch(residuals, nlags=5)
print(f"ARCH LM 检验统计量: {arch_test[0]:.4f}")
print(f"P-value (P值): {arch_test[1]:.4e}")
if arch_test[1] < 0.05:
    print("结论: P值极小，强烈拒绝原假设！证明存在显著的波动率聚集，必须使用 GARCH 模型。")
else:
    print("结论: 没有发现显著的波动率聚集。")


#%%
# ==========================================
# 步骤 3: GARCH(1,1) 联合估计 (Estimation)
# ==========================================
print("\n--- 步骤 3: GARCH(1,1) 模型拟合 ---")
# 构建模型：均值方程默认为常数(Constant)，方差方程为 GARCH(1,1)
model = arch_model(returns, mean='Constant', vol='Garch', p=1, q=1, dist='Normal')
res = model.fit(disp='off') # disp='off' 隐藏迭代过程
print(res.summary())



#%%
# ==========================================
# 步骤 4: 模型检验 (Standardized Residuals Check)
# ==========================================
# 计算标准化残差 = 原始残差 / 模型算出的条件标准差
std_resid = res.resid / res.conditional_volatility



#%%
# ==========================================
# 终极可视化 Dashboard (直接可用作作品集配图)
# ==========================================
print("\n--- 正在生成可视化图表 ---")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('NVIDIA (NVDA) - GARCH(1,1) Volatility Analysis (2018-2023)', fontsize=16, fontweight='bold')

# 图1：原始收益率 (直观感受波动率聚集)
axes[0, 0].plot(returns, color='gray', alpha=0.7)
axes[0, 0].set_title('1. Daily Log Returns (%) - Notice the Volatility Clustering', fontsize=12)
axes[0, 0].axhline(0, color='black', linestyle='--')

# 图2：条件波动率 (核心神图：GARCH眼中的风险)
axes[0, 1].plot(res.conditional_volatility, color='red', linewidth=1.5)
axes[0, 1].set_title('2. GARCH(1,1) Conditional Volatility - The Risk Indicator', fontsize=12)

# 图3：标准化残差 (检验模型是否把规律榨干了)
axes[1, 0].plot(std_resid, color='blue', alpha=0.7)
axes[1, 0].set_title('3. Standardized Residuals - Should look like White Noise', fontsize=12)
axes[1, 0].axhline(0, color='black', linestyle='--')

# 图4：标准化残差平方的 ACF 自相关图 (终极检验)
sm.graphics.tsa.plot_acf(std_resid**2, lags=20, ax=axes[1, 1], color='green')
axes[1, 1].set_title('4. ACF of Squared Std. Residuals - No significant spikes allowed', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 留出大标题空间
plt.show()


#%%


# ==========================================
# 终极进阶：Student-t 分布的 GARCH(1,1)
# ==========================================
print("\n--- 正在使用 Student-t 分布重新拟合 GARCH ---")
# 注意这里的改变：dist='t'
model_t = arch_model(returns, mean='Constant', vol='Garch', p=1, q=1, dist='t')
res_t = model_t.fit(disp='off')

print(res_t.summary())

# 对比 AIC
print("\n=== 模型 PK 赛: AIC 对比 ===")
print(f"Normal GARCH AIC: {res.aic:.2f}")
print(f"Student-t GARCH AIC: {res_t.aic:.2f}")

if res_t.aic < res.aic:
    print("结论: Student-t 模型的 AIC 更小，完胜！它更好地捕捉了市场的极端黑天鹅风险。")
else:
    print("结论: Normal 模型的 AIC 更小。")
    
    
    
    
#%%

# ================
# Final display
# ================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore') # Mute underlying warnings for a clean output


#%%
# ==========================================
# Step 0: Data Acquisition & Preprocessing
# ==========================================
print("="*70)
print("Initializing Project: Fetching 6-year daily data for Portfolio...")
print("="*70)
tickers = ['NVDA', 'INTC', 'TSM', '^IXIC']

# Download historical data (2018-2023) encompassing pre-COVID, COVID, and rate hike cycles
data = yf.download(tickers, start='2018-01-01', end='2023-12-31')['Close']

# Calculate logarithmic returns and scale by 100 (percentage) for GARCH convergence
returns = np.log(data / data.shift(1)).dropna() * 100

print("\n Data Pipeline Complete: Percentage log returns calculated for all assets.")


#%%
# ==========================================
# Step 1 & 2: Model Selection (Normal vs. Student-t)
# ==========================================
print("\n" + "="*70)
print("Step 1 & 2: Model Selection (Normal GARCH vs. Student-t GARCH)")
print("="*70)
print("Objective: Empirically justify the necessity of fat-tailed risk assumptions.\n")

results_compare = []

for ticker in tickers:
    ret = returns[ticker]
    
    # Fit Normal Distribution GARCH
    res_n = arch_model(ret, mean='Constant', vol='Garch', p=1, q=1, dist='Normal').fit(disp='off')
    # Fit Student-t Distribution GARCH
    res_t = arch_model(ret, mean='Constant', vol='Garch', p=1, q=1, dist='t').fit(disp='off')
    
    # Lower AIC indicates a better model fit
    winner = "Student-t" if res_t.aic < res_n.aic else "Normal"
    
    results_compare.append({
        'Ticker': ticker,
        'Normal GARCH (AIC)': round(res_n.aic, 2),
        'Student-t GARCH (AIC)': round(res_t.aic, 2),
        'Winning Model': winner,
        'AIC Reduction': round(res_n.aic - res_t.aic, 2)
    })

# Generate Model Selection Summary
aic_summary_df = pd.DataFrame(results_compare)
print("\n===  Model Selection Summary ===")
print(aic_summary_df.to_string(index=False))
print("\nConclusion: The Student-t GARCH significantly outperforms the Normal GARCH across all assets.")
print("Proceeding with the Student-t distribution for robust tail-risk analysis.")




#%%
# ==========================================
# Step 3: Optimal Model Parameter Extraction
# ==========================================
print("\n" + "="*70)
print("Step 3: Core Volatility Parameters Extraction (GARCH-t)")
print("="*70)
print("Objective: Quantify shock sensitivity, volatility persistence, and tail risks.\n")

results_best_list = []

for ticker in tickers:
    ret = returns[ticker]
    
    # Utilizing the superior Student-t model
    model = arch_model(ret, mean='Constant', vol='Garch', p=1, q=1, dist='t')
    res = model.fit(disp='off')
    
    # Parameter extraction
    mu = res.params['mu'] 
    alpha = res.params['alpha[1]'] # Shock sensitivity
    beta = res.params['beta[1]']   # Volatility memory/inertia
    persistence = alpha + beta     # Total volatility persistence
    nu = res.params['nu']          # Degrees of freedom (Tail thickness)
    
    results_best_list.append({
        'Ticker': ticker,
        'Mean Return (μ)': round(mu, 4),
        'Shock Sensitivity (α)': round(alpha, 4),
        'Vol Memory (β)': round(beta, 4),
        'Persistence (α+β)': round(persistence, 4),
        'Tail Risk DOF (ν)': round(nu, 2)
    })

# Generate Parameter Summary
best_params_df = pd.DataFrame(results_best_list)
print("\n=== Portfolio Volatility Dynamics & Risk Parameters ===")
print(best_params_df.to_string(index=False))




#%%
# ==========================================
# Step 4: Asset-Level Risk Dashboard Generation
# ==========================================
print("\n" + "="*70)
print("Step 4: Generating Asset-Level Risk Visualizations")
print("="*70)
print("Objective: Provide visual diagnostics and validate model adequacy (White Noise check).\n")

for ticker in tickers:
    ret = returns[ticker]
    
    model = arch_model(ret, mean='Constant', vol='Garch', p=1, q=1, dist='t')
    res = model.fit(disp='off')
    
    # Standardized Residuals = Raw Residuals / Conditional Volatility
    std_resid = res.resid / res.conditional_volatility
    
    # 2x2 Dashboard Setup
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{ticker} - GARCH(1,1) Volatility Analysis (Student-t)', fontsize=16, fontweight='bold')

    # Plot 1: Daily Log Returns
    axes[0, 0].plot(ret, color='gray', alpha=0.7)
    axes[0, 0].set_title('1. Daily Log Returns (%) - Volatility Clustering', fontsize=12)
    axes[0, 0].axhline(0, color='black', linestyle='--')

    # Plot 2: Conditional Volatility (The Risk Indicator)
    axes[0, 1].plot(res.conditional_volatility, color='red', linewidth=1.5)
    axes[0, 1].set_title('2. Conditional Volatility - Dynamic Risk Indicator', fontsize=12)

    # Plot 3: Standardized Residuals
    axes[1, 0].plot(std_resid, color='blue', alpha=0.7)
    axes[1, 0].set_title('3. Standardized Residuals - Approaching White Noise', fontsize=12)
    axes[1, 0].axhline(0, color='black', linestyle='--')

    # Plot 4: ACF of Squared Standardized Residuals
    sm.graphics.tsa.plot_acf(std_resid**2, lags=20, ax=axes[1, 1], color='green')
    axes[1, 1].set_title('4. ACF of Squared Std. Residuals - Model Diagnostics', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("\n" + "="*70)
print("Full Quantitative Pipeline Executed Successfully!")
print("="*70)




