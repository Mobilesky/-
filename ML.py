import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import shap
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# ====================== 配置区 ======================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

# 创建输出文件夹，避免文件混乱
output_dir = "result_tables"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -----------------------------------------------------------------------------
# 多目标综合性能筛选：TOPSIS+熵权法核心函数
# -----------------------------------------------------------------------------
def topsis_entropy_weight(pareto_df, target_cols):
    data = pareto_df[target_cols].values
    n, m = data.shape

    # 正向指标标准化
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    norm_data = (data - min_vals) / (max_vals - min_vals + 1e-8)

    # 熵权法计算权重
    p = norm_data / np.sum(norm_data, axis=0, keepdims=True)
    e = -np.sum(p * np.log(p + 1e-8), axis=0) / np.log(n)
    weight = (1 - e) / np.sum(1 - e)

    # 加权标准化矩阵
    weighted_norm = norm_data * weight

    # 正负理想解
    ideal_best = np.max(weighted_norm, axis=0)
    ideal_worst = np.min(weighted_norm, axis=0)

    # 欧氏距离
    d_best = np.sqrt(np.sum((weighted_norm - ideal_best) ** 2, axis=1))
    d_worst = np.sqrt(np.sum((weighted_norm - ideal_worst) ** 2, axis=1))

    # 综合贴近度
    score = d_worst / (d_best + d_worst + 1e-8)

    # 结果整合
    result_df = pareto_df.copy()
    result_df['综合性能评分'] = score
    result_df['性能排名'] = result_df['综合性能评分'].rank(ascending=False, method='min').astype(int)
    result_df = result_df.sort_values(by='综合性能评分', ascending=False).reset_index(drop=True)
    best_solution = result_df.iloc[0]

    print("\n📊 熵权法计算的性能权重:")
    for col, w in zip(target_cols, weight):
        print(f"  - {col}: {w:.4f}")

    return result_df, best_solution

# -----------------------------------------------------------------------------
# 1. 数据加载与预处理
# -----------------------------------------------------------------------------
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # 记录原始输入数据量
    initial_size = len(df)
    
    # 修复列名：替换特殊字符
    df.columns = df.columns.str.replace('℃', 'degC', regex=False)
    df.columns = df.columns.str.replace('⁻', '-', regex=False)
    
    df = df.dropna()
    
    # 手动指定目标列顺序
    target_cols = ['YS/MPa', 'UTS/MPa', 'EL/%']
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    # 3σ准则剔除异常值
    for target in target_cols:
        mean, std = df[target].mean(), df[target].std()
        df = df[(df[target] >= mean-3*std) & (df[target] <= mean+3*std)]
    
    # 记录最终数据量
    final_size = len(df)
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # 【规范划分】先8:2划分，测试集全程锁死，不参与交叉验证
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            scaler, feature_cols, target_cols, df, initial_size, final_size)

# -----------------------------------------------------------------------------
# 2. 【核心修改】多模型训练与5折交叉验证选优
# -----------------------------------------------------------------------------
def train_and_evaluate_models(X_train, X_test, y_train, y_test, target_names, cv_folds=5):
    """
    5折交叉验证选最优模型：
    1. 在训练集内做cv_folds折交叉验证，计算平均R²和标准差
    2. 以交叉验证平均R²为核心标准，选出泛化能力最优的模型
    3. 最优模型用全量训练集重训练，独立测试集做最终性能评估
    """
    # 模型池
    models_pool = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Extra Trees': ExtraTreesRegressor(random_state=42)
    }
    
    best_models, performance_results = {}, {}
    all_perf_data = []  # 保存所有模型的交叉验证+测试集结果

    for i, target in enumerate(target_names):
        print(f"\n===== Predicting: {target} =====")
        print(f"--- 开展{cv_folds}折交叉验证，筛选最优模型 ---")
        y_train_t, y_test_t = y_train[:,i], y_test[:,i]
        
        model_cv_results = {}  # 保存每个模型的交叉验证结果
        # 1. 遍历所有模型，做5折交叉验证
        for name, model in models_pool.items():
            # 5折交叉验证，在训练集内计算R²
            cv_r2_scores = cross_val_score(model, X_train, y_train_t, cv=cv_folds, scoring='r2', n_jobs=-1)
            cv_mean_r2 = np.mean(cv_r2_scores)
            cv_std_r2 = np.std(cv_r2_scores)
            model_cv_results[name] = {'cv_mean_r2': cv_mean_r2, 'cv_std_r2': cv_std_r2}
            print(f"{name:16s} | {cv_folds}折交叉验证平均R²={cv_mean_r2:.4f} | 标准差={cv_std_r2:.4f}")
        
        # 2. 【核心选优】以交叉验证平均R²为标准，选出最优模型
        best_model_name = max(model_cv_results, key=lambda x: model_cv_results[x]['cv_mean_r2'])
        best_cv_mean_r2 = model_cv_results[best_model_name]['cv_mean_r2']
        print(f"\n✅ {target} 最优模型: {best_model_name} (交叉验证平均R²={best_cv_mean_r2:.4f})")
        
        # 3. 最优模型用全量训练集重训练，独立测试集做最终性能评估
        best_model = copy.deepcopy(models_pool[best_model_name])
        best_model.fit(X_train, y_train_t)
        y_pred_test = best_model.predict(X_test)
        
        # 计算最终测试集性能指标
        r2_test = r2_score(y_test_t, y_pred_test)
        mae_test = mean_absolute_error(y_test_t, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test_t, y_pred_test))
        
        # 计算训练集拟合指标，用于过拟合验证
        y_pred_train = best_model.predict(X_train)
        r2_train = r2_score(y_train_t, y_pred_train)
        
        # 保存最优模型
        best_models[target] = best_model
        performance_results[target] = {
            best_model_name: {'R2': r2_test, 'MAE': mae_test, 'RMSE': rmse_test}
        }
        
        # 打印最终测试集结果
        print(f"--- 最优模型独立测试集最终性能 ---")
        print(f"{best_model_name:16s} | 测试集R²={r2_test:.4f} | MAE={mae_test:.2f} | RMSE={rmse_test:.2f}")
        
        # 收集所有模型的完整数据，用于保存CSV
        for name in models_pool.keys():
            all_perf_data.append({
                'Target': target,
                'Model': name,
                'CV_Mean_R2': model_cv_results[name]['cv_mean_r2'],
                'CV_Std_R2': model_cv_results[name]['cv_std_r2'],
                'Test_R2': r2_test if name == best_model_name else np.nan,
                'Test_MAE': mae_test if name == best_model_name else np.nan,
                'Test_RMSE': rmse_test if name == best_model_name else np.nan,
                'Train_R2': r2_train if name == best_model_name else np.nan,
                'Is_Best_Model': '是' if name == best_model_name else '否'
            })
    
    # 保存所有交叉验证+性能结果到CSV
    perf_df = pd.DataFrame(all_perf_data)
    perf_df.to_csv(f'{output_dir}/model_cv_performance_all.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 模型5折交叉验证+性能表已保存至: {output_dir}/model_cv_performance_all.csv")
    
    return best_models, performance_results

# -----------------------------------------------------------------------------
# 3. 保存预测值 vs 真实值数据
# -----------------------------------------------------------------------------
def save_predictions(best_models, X_test, y_test, target_cols):
    for i, target in enumerate(target_cols):
        y_true = y_test[:, i]
        y_pred = best_models[target].predict(X_test)
        
        pred_df = pd.DataFrame({
            'Index': range(len(y_true)),
            'Experimental_Value': y_true,
            'Predicted_Value': y_pred,
            'Absolute_Error': np.abs(y_true - y_pred)
        })
        
        safe_name = target.replace('/', '_')
        pred_df.to_csv(f'{output_dir}/prediction_{safe_name}.csv', index=False)
        print(f"✅ 预测数据已保存至: {output_dir}/prediction_{safe_name}.csv")

# -----------------------------------------------------------------------------
# 4. 绘图1：模型性能对比柱状图
# -----------------------------------------------------------------------------
def plot_model_performance(performance_results, target_cols):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, target in enumerate(target_cols):
        perf = performance_results[target]
        names = list(perf.keys())
        r2_scores = [perf[m]['R2'] for m in names]
        
        axes[i].bar(names, r2_scores, color=colors[0], width=0.6)
        axes[i].set_title(f'{target.replace("/","_")} (Best Model)', fontsize=12)
        axes[i].set_ylabel('$R^2$ Score', fontsize=11)
        axes[i].set_ylim(0, 1.0)
        axes[i].tick_params(axis='x', rotation=30)
        for j, v in enumerate(r2_scores):
            axes[i].text(j, v+0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('best_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# 5. 绘图2：预测值vs真实值散点图
# -----------------------------------------------------------------------------
def plot_prediction_scatter(best_models, X_test, y_test, target_cols):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, target in enumerate(target_cols):
        y_true = y_test[:, i]
        y_pred = best_models[target].predict(X_test)
        r2 = r2_score(y_true, y_pred)
        
        axes[i].scatter(y_true, y_pred, c=colors[0], s=20, alpha=0.7)
        min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[i].plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5)
        axes[i].set_title(f'{target.replace("/","_")}', fontsize=12)
        axes[i].set_xlabel('Experimental Value', fontsize=11)
        axes[i].set_ylabel('Predicted Value', fontsize=11)
        axes[i].text(0.05, 0.95, f'$R^2$={r2:.4f}', transform=axes[i].transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('prediction_vs_experimental.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# 6. SHAP分析 + 保存SHAP数据
# -----------------------------------------------------------------------------
def shap_analysis_jmst(best_models, X_train, feature_cols, target_cols):
    for target in target_cols:
        model = best_models[target]
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        except:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
            shap_values = explainer.shap_values(X_train)
        
        # 保存SHAP特征重要性
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature': feature_cols,
            'SHAP_Importance': shap_importance
        }).sort_values(by='SHAP_Importance', ascending=False)
        
        safe_name = target.replace('/', '_')
        shap_df.to_csv(f'{output_dir}/shap_importance_{safe_name}.csv', index=False)
        print(f"✅ SHAP数据已保存至: {output_dir}/shap_importance_{safe_name}.csv")
        
        # 绘图
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_train, feature_names=feature_cols, plot_type='bar', show=False)
        plt.title(f'Feature Importance (SHAP) | {safe_name}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'shap_feature_importance_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_train, feature_names=feature_cols, show=False)
        plt.title(f'SHAP Beeswarm | {safe_name}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'shap_beeswarm_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

# -----------------------------------------------------------------------------
# 7. Pearson相关系数热图 + 保存矩阵
# -----------------------------------------------------------------------------
def plot_pearson_heatmap(df, feature_cols, target_cols):
    corr_df = df[feature_cols + target_cols].corr()
    corr_df.to_csv(f'{output_dir}/pearson_correlation.csv')
    print(f"✅ 相关系数矩阵已保存至: {output_dir}/pearson_correlation.csv")
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_df, cmap='coolwarm', center=0, square=True, linewidths=0.5, annot=False)
    plt.title('Pearson Correlation Coefficient Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('pearson_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# 8. 多目标逆向设计
# -----------------------------------------------------------------------------
class SteelProblem(Problem):
    def __init__(self, scaler, best_models, feature_ranges, target_cols):
        self.scaler = scaler
        self.models = best_models
        self.target_cols = target_cols
        xl, xu = [r[0] for r in feature_ranges], [r[1] for r in feature_ranges]
        super().__init__(n_var=len(feature_ranges), n_obj=3, n_constr=0, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):
        x_scaled = self.scaler.transform(x)
        ys = self.models[self.target_cols[0]].predict(x_scaled)
        uts = self.models[self.target_cols[1]].predict(x_scaled)
        el = self.models[self.target_cols[2]].predict(x_scaled)
        out["F"] = np.column_stack([-ys, -uts, -el])

def inverse_design(scaler, best_models, df, feature_cols, target_cols):
    feature_ranges = [(df[c].min(), df[c].max()) for c in feature_cols]
    problem = SteelProblem(scaler, best_models, feature_ranges, target_cols)
    
    algorithm = NSGA2(pop_size=80, sampling=FloatRandomSampling(),
                      crossover=SBX(prob=0.9), mutation=PM(eta=20))
    res = minimize(problem, algorithm, get_termination("n_gen", 50), seed=42, verbose=True)
    
    pareto_X, pareto_F = res.X, -res.F
    res_df = pd.DataFrame(pareto_X, columns=feature_cols)
    pred_target_cols = [f'{col}_pred' for col in target_cols]
    res_df[pred_target_cols] = pareto_F
    
    return res_df, pred_target_cols

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    file_path = "data.csv" 
    
    print("="*50)
    print("STEP 1: 数据预处理")
    print("="*50)
    X_train, X_test, y_train, y_test, scaler, feat_cols, tgt_cols, df, initial_size, final_size = load_and_preprocess_data(file_path)
    print("确认目标列顺序:", tgt_cols)
    
    print(f"\n📊 数据量统计:")
    print(f"  - 原始输入数据量: {initial_size}")
    print(f"  - 去除缺失值/异常值后: {final_size}")
    print(f"  - 剔除样本数: {initial_size - final_size}")
    print(f"  - 训练集样本数: {len(X_train)} | 独立测试集样本数: {len(X_test)}")
    
    print("\n" + "="*50)
    print("STEP 2: 5折交叉验证选优 + 模型训练评估")
    print("="*50)
    best_models, perf = train_and_evaluate_models(X_train, X_test, y_train, y_test, tgt_cols, cv_folds=5)
    
    print("\n" + "="*50)
    print("STEP 3: 保存预测数据表格")
    print("="*50)
    save_predictions(best_models, X_test, y_test, tgt_cols)
    
    print("\n" + "="*50)
    print("STEP 4: 生成可视化图表 & 保存数据")
    print("="*50)
    plot_model_performance(perf, tgt_cols)
    plot_prediction_scatter(best_models, X_test, y_test, tgt_cols)
    plot_pearson_heatmap(df, feat_cols, tgt_cols)
    shap_analysis_jmst(best_models, X_train, feat_cols, tgt_cols)
    
    print("\n" + "="*50)
    print("STEP 5: 多目标逆向设计")
    print("="*50)
    pareto_res, pred_tgt_cols = inverse_design(scaler, best_models, df, feat_cols, tgt_cols)
    
    print("\n" + "="*50)
    print("STEP 6: 综合性能最优解筛选（TOPSIS+熵权法）")
    print("="*50)
    sorted_pareto, best_sol = topsis_entropy_weight(pareto_res, pred_tgt_cols)
    
    sorted_pareto.to_csv(f'{output_dir}/pareto_solutions_with_score.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 带综合评分的帕累托解已保存至: {output_dir}/pareto_solutions_with_score.csv")
    
    print("\n🏆 综合性能最优解详情:")
    print("--- 成分/工艺参数 ---")
    for col in feat_cols:
        print(f"  - {col}: {best_sol[col]:.4f}")
    print("\n--- 预测力学性能 ---")
    for col in pred_tgt_cols:
        print(f"  - {col}: {best_sol[col]:.2f}")
    print(f"\n  - 综合性能评分: {best_sol['综合性能评分']:.4f} (满分1.0)")
    print(f"  - 性能排名: 第{int(best_sol['性能排名'])}名 (共{len(sorted_pareto)}个方案)")
    
    print("\n" + "="*50)
    print(f"✅ 所有任务完成！数据表格已保存在 '{output_dir}' 文件夹中。")
    print("="*50)
