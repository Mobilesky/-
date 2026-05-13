高锰钢力学性能预测与多目标逆向智能设计系统
 
符合材料信息学顶刊学术规范的全流程机器学习解决方案
 
📋 项目简介
 
本项目是一套AI驱动的金属材料力学性能预测与工艺优化系统，针对高锰钢等金属材料，实现成分-轧制工艺-力学性能的智能化建模、精准预测与最优工艺自动筛选。系统严格遵循材料/机器学习顶刊学术规范，采用训练集内5折交叉验证选优+独立测试集盲测验证的严谨流程，彻底规避数据泄露风险，所有结果可直接用于学术论文发表与工业研发落地。
 
✨ 核心功能
 
1. 自动化数据预处理
- 自动完成原始数据清洗、缺失值剔除、3σ准则异常值过滤
- Min-Max标准化处理，消除不同量纲影响
- 自动统计并打印原始数据量、有效数据量、剔除样本数
2. 多模型训练与5折交叉验证智能选优
- 集成5种前沿机器学习模型：随机森林、XGBoost、CatBoost、梯度提升、极端随机树
- 训练集内5折交叉验证，以平均R²+标准差为标准筛选泛化能力最优模型
- 独立测试集全程隔离，仅用于最终一次性泛化性能评估
- 输出R²、MAE、RMSE三大核心评估指标
3. 多维度可解释性分析
- SHAP可解释性分析：自动计算特征重要性，输出特征影响规律
- Pearson相关系数分析：生成相关系数矩阵与热图
- 所有分析结果自动保存为CSV表格与期刊级图片
4. 多目标逆向工艺优化
- 基于NSGA-II算法生成帕累托最优工艺解集
- 融合TOPSIS+熵权法多属性决策模型，自动筛选力学综合性能最优方案
- 输出最优成分/工艺参数与对应的预测力学性能
5. 标准化成果输出
- 自动生成7类顶刊级标准化数据表格
- 自动绘制符合JMST等材料顶刊格式要求的可视化图表
- 所有结果分类保存，便于后续论文作图与数据追溯
 
🛠️ 环境依赖
 
-Python 3.9 及以上版本
-熊猫>=2. 0.0
-numpy >= 1.24.0
-matplotlib >= 3.7.0
-seaborn >= 0.12.0
-scikit-learn >= 1.3.0
-xgboost >= 2.0.0
-catboost >= 1.2.0
-shap >= 0.44.0
-pymoo >= 0.6.0
 
一键安装依赖
 
bash
  
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost shap pymoo
 
 
🚀 快速开始
 
1. 准备数据集
 
将你的实验数据保存为CSV格式，命名为 data.csv ，放在项目根目录下。
数据集必须包含以下列（列名可根据实际情况修改，需与代码中 target_cols 对应）：
 
- 输入特征：成分参数（C、Mn等）、轧制工艺参数（初始厚度、轧制温度、保温时间、压下率、轧制速度等）
- 输出目标： YS/MPa （屈服强度）、 UTS/MPa （抗拉强度）、 EL/% （伸长率）
 
2. 配置代码
 
打开主程序文件，根据你的数据集修改以下配置：
 
python
  
# 在load_and_preprocess_data函数中，修改为你的目标列名
target_cols = ['YS/MPa', 'UTS/MPa', 'EL/%']
 
 
3. 运行代码
 
bash
  
python main.py
 
 
4. 查看结果
 
所有结果将自动保存在 result_tables 文件夹中，生成的图片保存在项目根目录下。
 
📁 文件结构
 
plaintext
  
.
├── main.py              # 主程序代码
├── data.csv             # 你的实验数据集（需自行准备）
├── result_tables/       # 所有输出数据表格
│   ├── model_cv_performance_all.csv    # 5折交叉验证+模型性能表
│   ├── prediction_YS_MPa.csv           # 屈服强度预测值vs实验值
│   ├── prediction_UTS_MPa.csv          # 抗拉强度预测值vs实验值
│   ├── prediction_EL_%.csv             # 伸长率预测值vs实验值
│   ├── shap_importance_YS_MPa.csv      # 屈服强度SHAP特征重要性
│   ├── shap_importance_UTS_MPa.csv     # 抗拉强度SHAP特征重要性
│   ├── shap_importance_EL_%.csv        # 伸长率SHAP特征重要性
│   ├── pearson_correlation.csv         # Pearson相关系数矩阵
│   ├── pareto_optimal_solutions.csv    # 帕累托最优解集
│   └── pareto_solutions_with_score.csv # 带综合评分的帕累托解
├── best_model_performance.png          # 最优模型性能对比图
├── prediction_vs_experimental.png      # 预测值vs真实值散点图
├── shap_feature_importance_*.png       # SHAP特征重要性条形图
├── shap_beeswarm_*.png                 # SHAP蜂群图
├── pearson_correlation_heatmap.png     # Pearson相关系数热图
└── pareto_front.png                    # 3D帕累托前沿图
 
 
📊 结果说明
 
输出文件 内容说明 用途 
 model_cv_performance_all.csv  所有模型的5折交叉验证平均R²、标准差，以及最优模型的测试集性能 模型性能对比、论文结果撰写 
 prediction_*.csv  每个目标的实验值、模型预测值、绝对误差 绘制预测散点图、误差分析 
 shap_importance_*.csv  每个特征的SHAP重要性排序 特征影响机制分析、论文图表绘制 
 pearson_correlation.csv  所有变量之间的Pearson相关系数 线性相关性分析、补充SHAP分析 
 pareto_solutions_with_score.csv  带综合性能评分的帕累托最优解，按评分排序 筛选最优工艺方案、工程应用 
 
📚 学术规范说明
 
本代码严格遵循材料/机器学习顶刊学术规范：
 
1. 数据划分：一次性8:2划分为训练集与独立测试集，测试集全程不参与任何模型训练、选优、调参
2. 模型选优：仅在训练集内进行5折交叉验证，以交叉验证平均R²为标准筛选最优模型
3. 性能评估：最优模型在全量训练集上重训练后，仅用独立测试集做最终一次性性能评估
4. 可解释性分析：SHAP分析基于仅用训练集训练的最优模型完成，无数据泄露
5. 逆向优化：基于已验证泛化能力的最优模型进行，结果可靠
 
📝 更新日志
 
- v2.0：实现训练集内5折交叉验证选优，符合顶刊学术规范；新增TOPSIS+熵权法综合性能筛选
- v1.5：将MSE指标替换为RMSE；新增所有结果分类输出为CSV表格
- v1.4：修复模型引用覆盖的致命错误；替换SVR为Extra Trees，解决R²为负的问题
- v1.3：新增SHAP可解释性分析与Pearson相关系数分析
- v1.2：新增多目标逆向设计功能，基于NSGA-II算法生成帕累托最优解
- v1.0：初始版本，实现多模型训练与性能评估
 
⚠️ 免责声明
 
1. 本代码仅供学术研究使用，请勿用于商业用途
2. 模型预测结果仅供参考，实际工艺参数需通过实验验证
3. 使用本代码发表论文时，请注明代码来源
