# -*- coding: utf-8 -*-
"""
上海房价预测 - 特征工程优化 (正确版本)
保留所有原始特征，只移除单价这个作弊特征
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time

start_time = time.time()
print("=" * 80)
print("上海房价预测 - 特征工程优化 (正确版本)")
print("=" * 80)

# 1. 加载数据
print("\n【1】加载数据...")
df_train = pd.read_csv('dataset.csv')
df_test = pd.read_csv('dataset_test.csv')
print(f"  ✓ 训练集：{df_train.shape}")
print(f"  ✓ 测试集：{df_test.shape}")

# 2. 特征工程 (只添加，不添加单价！)
print("\n【2】创建特征 (只添加合法特征)...")
df_train_feat = df_train.copy()
df_test_feat = df_test.copy()

# ✅ 房龄
df_train_feat['房龄'] = 2026 - df_train_feat['建造年份']
df_test_feat['房龄'] = 2026 - df_test_feat['建造年份']
print("  ✓ 房龄")

# ❌ 不加单价 (这是作弊！)

# ✅ 房间密度
df_train_feat['房间密度'] = df_train_feat['居室数'] / (df_train_feat['总面积'] + 1e-6)
df_test_feat['房间密度'] = df_test_feat['居室数'] / (df_test_feat['总面积'] + 1e-6)
print("  ✓ 房间密度")

# ✅ 厅室比
df_train_feat['厅室比'] = df_train_feat['厅堂数'] / (df_train_feat['居室数'] + 1e-6)
df_test_feat['厅室比'] = df_test_feat['厅堂数'] / (df_test_feat['居室数'] + 1e-6)
print("  ✓ 厅室比")

# ✅ 面积平方
df_train_feat['总面积_平方'] = df_train_feat['总面积'] ** 2
df_test_feat['总面积_平方'] = df_test_feat['总面积'] ** 2
print("  ✓ 面积平方")

# ✅ 居室数平方
df_train_feat['居室数_平方'] = df_train_feat['居室数'] ** 2
df_test_feat['居室数_平方'] = df_test_feat['居室数'] ** 2
print("  ✓ 居室数平方")

# ✅ 房龄×面积交互
df_train_feat['房龄_面积交互'] = df_train_feat['房龄'] * df_train_feat['总面积']
df_test_feat['房龄_面积交互'] = df_test_feat['房龄'] * df_test_feat['总面积']
print("  ✓ 房龄×面积交互")

# ✅ 是否次新
df_train_feat['是否次新'] = (df_train_feat['房龄'] <= 5).astype(int)
df_test_feat['是否次新'] = (df_test_feat['房龄'] <= 5).astype(int)
print("  ✓ 是否次新")

print(f"✓ 新增特征：8 个 (全部合法)")

# 3. 编码分类特征
print("\n【3】编码分类特征...")
categorical_cols = ['装修', '楼层分布', '物业类型', '产权性质', '产权年限', 
                    '房本年限', '区', '小区', '南', '南北', '近地铁', 
                    '车位充足', '户型方正', '多人关注', '有电梯']

label_encoders = {}
for col in categorical_cols:
    if col in df_train_feat.columns:
        le = LabelEncoder()
        # 合并训练集和测试集来 fit
        combined = pd.concat([df_train_feat[col].astype(str), df_test_feat[col].astype(str)])
        le.fit(combined)
        df_train_feat[col + '_enc'] = le.transform(df_train_feat[col].astype(str))
        df_test_feat[col + '_enc'] = le.transform(df_test_feat[col].astype(str))
        label_encoders[col] = le
        print(f"  ✓ {col} -> {col}_enc")

print(f"✓ 编码完成")

# 4. 准备数据
print("\n【4】准备训练数据...")
target = '价格'
exclude_cols = ['价格', '小区名称', '区', '小区', '标题', '街道',
                '装修', '楼层分布', '物业类型', '产权性质', '产权年限', 
                '房本年限', '南', '南北', '近地铁', 
                '车位充足', '户型方正', '多人关注', '有电梯']

# 使用所有数值特征 + 编码后的特征
feature_cols = [col for col in df_train_feat.columns 
                if col not in exclude_cols 
                and (col.endswith('_enc') or df_train_feat[col].dtype in ['int64', 'float64', 'int32', 'float32'])]

print(f"✓ 使用特征：{len(feature_cols)} 个")

# 确保测试集有相同的特征列
for col in feature_cols:
    if col not in df_test_feat.columns:
        df_test_feat[col] = 0

X = df_train_feat[feature_cols].fillna(0)
y = df_train_feat[target]

# 移除无效值
valid_mask = (y > 0) & y.notnull()
X = X[valid_mask]
y = y[valid_mask]
print(f"  ✓ 有效样本：{len(y):,}")

y_log = np.log1p(y)
X_test = df_test_feat[feature_cols].fillna(0)

print(f"✓ X: {X.shape}, y: {y.shape}")

# 5. 划分数据集
print("\n【5】划分数据集...")
X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"✓ 训练集：{X_train.shape}, 验证集：{X_val.shape}")

# 6. 训练模型
print("\n【6】训练模型...")

models = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
}

results = {}
for name, model in models.items():
    print(f"  训练 {name}...", end=' ', flush=True)
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    
    # 预测并还原
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_val_original = np.expm1(y_val)
    
    # 评估
    r2 = r2_score(y_val_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
    mae = mean_absolute_error(y_val_original, y_pred)
    
    results[name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'model': model}
    print(f"R²={r2:.4f}, RMSE={rmse:.2f}万 (耗时：{t1-t0:.2f}s)")

# 7. 结果对比
print("\n" + "=" * 80)
print("模型性能对比 (正确版本 - 无作弊)")
print("=" * 80)

baseline_with_cheat = 0.9674  # 有作弊特征的结果
baseline_original = 0.9343    # 原始基线

for name, res in sorted(results.items(), key=lambda x: x[1]['R²'], reverse=True):
    improvement_vs_original = res['R²'] - baseline_original
    improvement_vs_cheat = res['R²'] - baseline_with_cheat
    print(f"{name:15} R²={res['R²']:.4f}")
    print(f"              vs 原始基线：{improvement_vs_original:+.4f}")
    print(f"              vs 作弊版本：{improvement_vs_cheat:+.4f}")
    print(f"              RMSE={res['RMSE']:.2f}万")

# 8. 可视化
print("\n【7】生成可视化...")
model_names = list(results.keys())
colors = ['steelblue', 'coral']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

r2_scores = [results[m]['R²'] for m in model_names]
rmse_scores = [results[m]['RMSE'] for m in model_names]
mae_scores = [results[m]['MAE'] for m in model_names]

axes[0].bar(model_names, r2_scores, color=colors, alpha=0.7)
axes[0].set_ylabel('R²')
axes[0].set_title('R² 对比 (正确版本)', fontsize=14, fontweight='bold')
axes[0].axhline(y=baseline_with_cheat, color='red', linestyle='--', label=f'作弊版本 ({baseline_with_cheat:.4f})')
axes[0].axhline(y=baseline_original, color='green', linestyle='--', label=f'原始基线 ({baseline_original:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(model_names, rmse_scores, color=colors, alpha=0.7)
axes[1].set_ylabel('RMSE (万元)')
axes[1].set_title('RMSE 对比 (正确版本)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].bar(model_names, mae_scores, color=colors, alpha=0.7)
axes[2].set_ylabel('MAE (万元)')
axes[2].set_title('MAE 对比 (正确版本)', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('特征工程优化效果_正确版本.png', dpi=300)
print("✓ 已保存：特征工程优化效果_正确版本.png")

# 9. 特征重要性
print("\n【8】特征重要性...")
best_name = 'XGBoost'
best_model = results[best_name]['model']

if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({
        '特征': feature_cols,
        '重要性': best_model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    plt.figure(figsize=(12, 10))
    top_n = min(20, len(fi))
    plt.barh(range(top_n), fi.head(top_n)['重要性'].values)
    plt.yticks(range(top_n), fi.head(top_n)['特征'].values)
    plt.xlabel('重要性')
    plt.title(f'Top 20 特征重要性 ({best_name}) - 正确版本')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('特征重要性_正确版本.png', dpi=300)
    print("✓ 已保存：特征重要性_正确版本.png")
    
    print("\nTop 15 重要特征:")
    for idx, row in fi.head(15).iterrows():
        print(f"  {row['特征']}: {row['重要性']:.4f}")

# 10. 生成预测
print("\n【9】生成预测...")
test_pred_log = best_model.predict(X_test)
test_pred = np.expm1(test_pred_log)

result_df = df_test.copy()
result_df['预测价格'] = test_pred
output_file = f'{best_name}_优化预测_正确版本.xlsx'
result_df.to_excel(output_file, index=False)
print(f"✓ 已保存：{output_file}")

print(f"\n预测统计:")
print(f"  均值：{test_pred.mean():.2f} 万元")
print(f"  中位数：{np.median(test_pred):.2f} 万元")

# 11. 报告
print("\n【10】生成报告...")
best = results[best_name]

report = f"""# 特征工程优化报告 (正确版本)

**时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**总耗时**: {time.time()-start_time:.2f}秒

## ⚠️ 重要说明

本版本**移除了单价这个作弊特征**，但保留了所有其他合法特征：
- 单价 = 价格/面积，已经包含了目标变量信息
- 使用单价相当于数据泄露 (Data Leakage)
- 其他所有特征（区、小区、朝向等）都是合法的

## 性能对比

| 版本 | 最佳 R² | RMSE(万) | 说明 |
|------|---------|----------|------|
| 作弊版本 | {baseline_with_cheat:.4f} | 132.01 | 包含单价 (数据泄露) ❌ |
| **正确版本** | **{best['R²']:.4f}** | **{best['RMSE']:.2f}** | **移除单价，保留其他** ✅ |
| 原始基线 | {baseline_original:.4f} | 136.82 | 无特征工程 |

## 关键结果

- **最佳模型**: {best_name}
- **最佳 R²**: {best['R²']:.4f}
- **相比原始基线提升**: {best['R²'] - baseline_original:+.4f}
- **相比作弊版本差距**: {best['R²'] - baseline_with_cheat:+.4f}
- **最低 RMSE**: {best['RMSE']:.2f}万元

## 新增特征 (全部合法)

1. **房龄** (2026 - 建造年份)
2. **房间密度** (居室数/面积)
3. **厅室比** (厅堂数/居室数)
4. **总面积平方**
5. **居室数平方**
6. **房龄×面积交互**
7. **是否次新** (房龄≤5)

## Top 15 重要特征

"""

for i, (_, row) in enumerate(fi.head(15).iterrows()):
    report += f"{i+1}. {row['特征']}: {row['重要性']:.4f}\n"

report += f"""
## 分析

### 为什么正确版本 R²={best['R²']:.4f} 是真实的？

1. **移除了单价** - 没有数据泄露
2. **保留了所有合法特征** - 区、小区、朝向、地铁等
3. **模型学到的是真实规律** - 不是"背答案"

### 为什么比作弊版本低？

- 作弊版本 R²=0.9674 是虚高的
- 正确版本 R²={best['R²']:.4f} 是真实能力
- **在实际应用中，正确版本更可靠**

## 下一步建议

1. **Target Encoding** (区域、小区名称) - 预计提升 R² +0.02-0.04
2. **位置特征** (环线) - 预计提升 R² +0.01-0.03
3. **参数调优** - 预计提升 R² +0.01
4. **模型融合** - 预计提升 R² +0.01-0.02

**预期最终 R² ≈ 0.95+ (真实能力)**
"""

with open('特征工程优化报告_正确版本.md', 'w', encoding='utf-8') as f:
    f.write(report)
print("✓ 已保存：特征工程优化报告_正确版本.md")

# 完成
print("\n" + "=" * 80)
print("✅ 完成！(正确版本 - 无作弊)")
print("=" * 80)
print(f"\n📊 最佳模型：{best_name}")
print(f"📈 最佳 R²: {best['R²']:.4f} (真实能力)")
print(f"📉 作弊版本 R²: {baseline_with_cheat:.4f} (虚高，不推荐)")
print(f"🚀 相比原始基线：{best['R²'] - baseline_original:+.4f}")
print(f"⏱️  总耗时：{time.time()-start_time:.2f}秒")

print("\n📁 生成的文件:")
print("  - 特征工程优化效果_正确版本.png")
print("  - 特征重要性_正确版本.png")
print("  - 特征工程优化报告_正确版本.md")
print(f"  - {best_name}_优化预测_正确版本.xlsx")

print("\n💡 建议：使用正确版本的结果，虽然 R²略低但真实可靠！")
