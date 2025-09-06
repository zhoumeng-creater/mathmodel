"""
核心模型与优化算法模块
包含Y浓度预测、自适应分组和风险优化
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class YConcentrationPredictor:
    """Y浓度预测模型 - 集成学习"""
    
    def __init__(self):
        """初始化集成学习模型"""
        # CART决策树
        self.cart = DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=30,
            random_state=42
        )
        
        # 随机森林（轻量级）
        self.rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=20,
            random_state=42
        )
        
        # LightGBM（轻量级）
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.lgb_model = None
        
        # 集成权重（将基于验证集性能自动计算）
        self.weights = {'cart': 0.3, 'rf': 0.4, 'lgb': 0.3}
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练集成模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（用于计算权重）
            y_val: 验证标签
        """
        # 训练CART
        self.cart.fit(X_train, y_train)
        
        # 训练随机森林
        self.rf.fit(X_train, y_train)
        
        # 训练LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        self.lgb_model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=100
        )
        
        # 如果提供了验证集，自动计算最优权重
        if X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)
    
    def _optimize_weights(self, X_val, y_val):
        """
        基于验证集性能优化集成权重
        
        参数:
            X_val: 验证特征
            y_val: 验证标签
        """
        # 获取各模型预测
        pred_cart = self.cart.predict(X_val)
        pred_rf = self.rf.predict(X_val)
        pred_lgb = self.lgb_model.predict(X_val, num_iteration=self.lgb_model.best_iteration)
        
        # 计算各模型MSE
        mse_cart = mean_squared_error(y_val, pred_cart)
        mse_rf = mean_squared_error(y_val, pred_rf)
        mse_lgb = mean_squared_error(y_val, pred_lgb)
        
        # 基于MSE倒数计算权重
        inv_mse = [1/mse_cart, 1/mse_rf, 1/mse_lgb]
        total = sum(inv_mse)
        
        self.weights = {
            'cart': inv_mse[0] / total,
            'rf': inv_mse[1] / total,
            'lgb': inv_mse[2] / total
        }
    
    def predict(self, X):
        """
        集成预测
        
        参数:
            X: 特征矩阵
            
        返回:
            y_pred: 预测的Y浓度
        """
        pred_cart = self.cart.predict(X)
        pred_rf = self.rf.predict(X)
        pred_lgb = self.lgb_model.predict(X, num_iteration=self.lgb_model.best_iteration)
        
        # 加权集成
        y_pred = (self.weights['cart'] * pred_cart +
                  self.weights['rf'] * pred_rf +
                  self.weights['lgb'] * pred_lgb)
        
        return y_pred
    
    def predict_failure_prob(self, X, week_col_idx=0):
        """
        预测检测失败概率（Y浓度<4%）
        
        参数:
            X: 特征矩阵
            week_col_idx: Week特征的列索引
            
        返回:
            failure_prob: 失败概率
        """
        y_pred = self.predict(X)
        
        # 使用逻辑函数建模失败概率
        # P_fail = 1 / (1 + exp(theta_0 + theta_1 * y_pred))
        theta_0 = 2.0  # 基础参数
        theta_1 = 0.5  # Y浓度系数
        
        failure_prob = 1 / (1 + np.exp(theta_0 + theta_1 * y_pred))
        
        return failure_prob


class AdaptiveGrouping:
    """基于CART的自适应分组"""
    
    def __init__(self, max_depth=4, min_samples_leaf=50, min_groups=3, max_groups=7):
        """
        初始化自适应分组器
        
        参数:
            max_depth: 决策树最大深度（约束条件）
            min_samples_leaf: 最小叶节点样本数（约束条件）
            min_groups: 最小分组数
            max_groups: 最大分组数
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_groups = min_groups
        self.max_groups = max_groups
        self.tree = None
        self.groups = []
        
    def fit(self, X, y, sample_weights=None):
        """
        训练分组模型
        
        参数:
            X: 特征矩阵
            y: Y浓度值
            sample_weights: 样本权重
        """
        # 创建风险标签（用于分组）
        # 将Y浓度转换为风险等级
        risk_labels = self._create_risk_labels(X, y)
        
        # 训练CART分类树
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=0.01,  # 纯度阈值约束
            random_state=42
        )
        
        self.tree.fit(X, risk_labels, sample_weight=sample_weights)
        
        # 提取分组
        self._extract_groups(X, y)
    
    def _create_risk_labels(self, X, y):
        """
        创建风险标签用于分组
        
        参数:
            X: 特征矩阵
            y: Y浓度
            
        返回:
            risk_labels: 风险等级标签
        """
        # 基于Y浓度和特征创建风险等级
        risk_scores = np.zeros(len(y))
        
        # 考虑Y浓度达标情况
        risk_scores[y < 4] = 2  # 高风险
        risk_scores[(y >= 4) & (y < 6)] = 1  # 中风险
        risk_scores[y >= 6] = 0  # 低风险
        
        # 考虑BMI因素（假设BMI是第二个特征）
        if X.shape[1] > 1:
            bmi_values = X[:, 1]
            risk_scores[bmi_values > 35] += 1
        
        # 离散化为K个等级
        n_classes = min(self.max_groups, len(np.unique(risk_scores)))
        risk_labels = pd.qcut(risk_scores, n_classes, labels=False, duplicates='drop')
        
        return risk_labels
    
    def _extract_groups(self, X, y):
        """
        从决策树中提取分组
        
        参数:
            X: 特征矩阵
            y: Y浓度
        """
        # 获取叶节点分配
        leaf_ids = self.tree.apply(X)
        unique_leaves = np.unique(leaf_ids)
        
        # 确保分组数在约束范围内
        n_groups = len(unique_leaves)
        if n_groups < self.min_groups or n_groups > self.max_groups:
            # 调整分组数
            n_groups = np.clip(n_groups, self.min_groups, self.max_groups)
        
        # 构建分组信息
        self.groups = []
        for leaf_id in unique_leaves:
            mask = (leaf_ids == leaf_id)
            if np.sum(mask) >= self.min_samples_leaf:
                group_info = {
                    'leaf_id': leaf_id,
                    'indices': np.where(mask)[0],
                    'size': np.sum(mask),
                    'mean_bmi': np.mean(X[mask, 1]) if X.shape[1] > 1 else 0,
                    'mean_week': np.mean(X[mask, 0]) if X.shape[1] > 0 else 0,
                    'mean_y': np.mean(y[mask]),
                    'features': X[mask],
                    'y_values': y[mask]
                }
                self.groups.append(group_info)
        
        # 按平均BMI排序
        self.groups.sort(key=lambda x: x['mean_bmi'])
    
    def get_groups(self):
        """
        获取分组结果
        
        返回:
            groups: 分组列表
        """
        return self.groups
    
    def predict_group(self, X):
        """
        预测新样本的分组
        
        参数:
            X: 特征矩阵
            
        返回:
            group_ids: 分组ID
        """
        if self.tree is None:
            raise ValueError("模型未训练")
        
        leaf_ids = self.tree.apply(X)
        
        # 将叶节点ID映射到分组索引
        leaf_to_group = {}
        for i, group in enumerate(self.groups):
            leaf_to_group[group['leaf_id']] = i
        
        group_ids = [leaf_to_group.get(lid, -1) for lid in leaf_ids]
        return np.array(group_ids)


class RiskOptimizer:
    """风险优化器"""
    
    def __init__(self, weights=None):
        """
        初始化风险优化器
        
        参数:
            weights: 风险权重字典
        """
        # 默认风险权重
        self.weights = weights or {
            'alpha': 0.3,   # 时间风险权重
            'beta': 0.3,    # 失败风险权重
            'gamma': 0.2,   # 成本风险权重
            'delta': 0.2    # 异质性风险权重
        }
        
        # 时点范围约束
        self.time_range = (10, 25)
        
        # 达标概率阈值
        self.target_success_rate = 0.8
    
    def calculate_time_risk(self, t):
        """
        计算时间风险（离散化）
        
        参数:
            t: 检测时点（孕周）
            
        返回:
            risk: 时间风险值
        """
        if t <= 11:
            return 0.05
        elif t == 12:
            return 0.1
        elif 13 <= t <= 20:
            return 0.2 + 0.1 * (t - 12) / 8
        else:  # t > 20
            return 1.0
    
    def calculate_failure_risk(self, t, y_pred_func, features):
        """
        计算失败风险
        
        参数:
            t: 检测时点
            y_pred_func: Y浓度预测函数
            features: 组内特征
            
        返回:
            risk: 失败风险值
        """
        # 修改特征中的Week值为t
        features_at_t = features.copy()
        if features_at_t.shape[1] > 0:
            features_at_t[:, 0] = t  # 假设Week是第一个特征
        
        # 预测Y浓度
        y_pred = y_pred_func(features_at_t)
        
        # 计算失败概率（Y<4%）
        failure_prob = np.mean(y_pred < 4.0)
        
        # 失败风险 = 失败概率 * (1 + 复检成本)
        retest_cost = 0.5
        risk = failure_prob * (1 + retest_cost)
        
        return risk
    
    def calculate_cost_risk(self, t):
        """
        计算成本风险
        
        参数:
            t: 检测时点
            
        返回:
            risk: 成本风险值
        """
        if t < 12:
            return 0.3  # 可能需要复检
        elif 12 <= t <= 20:
            return 0.0  # 最优时段
        else:  # t > 20
            return 0.2  # 延迟成本
    
    def calculate_heterogeneity_risk(self, features):
        """
        计算异质性风险
        
        参数:
            features: 组内特征
            
        返回:
            risk: 异质性风险值
        """
        if len(features) == 0:
            return 0.0
        
        # 计算变异系数（标准差/均值）
        cv_values = []
        for col in range(features.shape[1]):
            col_data = features[:, col]
            if np.mean(col_data) != 0:
                cv = np.std(col_data) / np.abs(np.mean(col_data))
                cv_values.append(cv)
        
        # 平均变异系数作为异质性度量
        risk = np.mean(cv_values) if cv_values else 0.0
        
        return np.clip(risk, 0, 1)
    
    def calculate_total_risk(self, t, group_features, y_pred_func):
        """
        计算总风险
        
        参数:
            t: 检测时点
            group_features: 组内特征
            y_pred_func: Y浓度预测函数
            
        返回:
            total_risk: 总风险值
        """
        # 计算各风险组件
        r_time = self.calculate_time_risk(t)
        r_fail = self.calculate_failure_risk(t, y_pred_func, group_features)
        r_cost = self.calculate_cost_risk(t)
        r_hetero = self.calculate_heterogeneity_risk(group_features)
        
        # 加权求和
        total_risk = (self.weights['alpha'] * r_time +
                     self.weights['beta'] * r_fail +
                     self.weights['gamma'] * r_cost +
                     self.weights['delta'] * r_hetero)
        
        return total_risk
    
    def check_success_constraint(self, t, group_features, y_pred_func):
        """
        检查达标概率约束
        
        参数:
            t: 检测时点
            group_features: 组内特征
            y_pred_func: Y浓度预测函数
            
        返回:
            satisfied: 是否满足约束
        """
        # 修改特征中的Week值为t
        features_at_t = group_features.copy()
        if features_at_t.shape[1] > 0:
            features_at_t[:, 0] = t
        
        # 预测Y浓度
        y_pred = y_pred_func(features_at_t)
        
        # 计算达标比例
        success_rate = np.mean(y_pred >= 4.0)
        
        # 动态调整阈值（高风险组要求更高）
        mean_bmi = np.mean(group_features[:, 1]) if group_features.shape[1] > 1 else 25
        if mean_bmi > 35:
            adjusted_threshold = self.target_success_rate + 0.1
        else:
            adjusted_threshold = self.target_success_rate
        
        return success_rate >= adjusted_threshold
    
    def optimize_timepoints(self, groups, y_pred_func):
        """
        优化各组检测时点
        
        参数:
            groups: 分组列表
            y_pred_func: Y浓度预测函数
            
        返回:
            optimal_timepoints: 各组最优时点
            risks: 各组风险值
        """
        optimal_timepoints = []
        risks = []
        
        for i, group in enumerate(groups):
            # 网格搜索最优时点
            best_t = self.time_range[0]
            best_risk = float('inf')
            
            time_points = range(self.time_range[0], self.time_range[1] + 1)
            
            for t in time_points:
                # 检查约束条件
                if self.check_success_constraint(t, group['features'], y_pred_func):
                    # 计算风险
                    risk = self.calculate_total_risk(t, group['features'], y_pred_func)
                    
                    # 准单调性约束：BMI越高，时点应该越晚（允许小偏差）
                    if i > 0 and t < optimal_timepoints[i-1] - 2:
                        risk *= 1.5  # 惩罚违反单调性
                    
                    if risk < best_risk:
                        best_risk = risk
                        best_t = t
            
            optimal_timepoints.append(best_t)
            risks.append(best_risk)
        
        return optimal_timepoints, risks


class MonteCarloSimulator:
    """蒙特卡洛误差分析"""
    
    def __init__(self, n_simulations=1000, error_rate=0.05):
        """
        初始化模拟器
        
        参数:
            n_simulations: 模拟次数
            error_rate: 测量误差率（5%）
        """
        self.n_simulations = n_simulations
        self.error_rate = error_rate
    
    def simulate_measurement_error(self, y_true):
        """
        模拟测量误差
        
        参数:
            y_true: 真实Y浓度
            
        返回:
            y_observed: 带误差的观测值
        """
        # 相对误差和绝对误差
        relative_error = np.random.normal(0, self.error_rate, size=y_true.shape)
        absolute_error = np.random.normal(0, 0.1, size=y_true.shape)
        
        # Y_obs = Y_true * (1 + ε_m) + ε_a
        y_observed = y_true * (1 + relative_error) + absolute_error
        
        # 确保非负
        y_observed = np.maximum(y_observed, 0)
        
        return y_observed
    
    def analyze_robustness(self, optimal_timepoints, groups, y_pred_func, risk_optimizer):
        """
        分析方案鲁棒性
        
        参数:
            optimal_timepoints: 原始最优时点
            groups: 分组
            y_pred_func: 预测函数
            risk_optimizer: 风险优化器
            
        返回:
            robustness_metrics: 鲁棒性指标
        """
        timepoint_variations = []
        risk_variations = []
        success_rate_variations = []
        
        for _ in range(self.n_simulations):
            # 对每个组模拟误差影响
            simulated_timepoints = []
            simulated_risks = []
            
            for i, group in enumerate(groups):
                # 添加测量误差
                y_with_error = self.simulate_measurement_error(group['y_values'])
                
                # 重新计算最优时点（简化：只评估原始时点附近）
                t_original = optimal_timepoints[i]
                t_range = range(max(10, t_original-1), min(26, t_original+2))
                
                best_t = t_original
                best_risk = float('inf')
                
                for t in t_range:
                    if risk_optimizer.check_success_constraint(t, group['features'], y_pred_func):
                        risk = risk_optimizer.calculate_total_risk(t, group['features'], y_pred_func)
                        if risk < best_risk:
                            best_risk = risk
                            best_t = t
                
                simulated_timepoints.append(best_t)
                simulated_risks.append(best_risk)
            
            timepoint_variations.append(simulated_timepoints)
            risk_variations.append(np.mean(simulated_risks))
        
        # 计算鲁棒性指标
        timepoint_variations = np.array(timepoint_variations)
        
        robustness_metrics = {
            'timepoint_stability': 1 - np.mean(np.std(timepoint_variations, axis=0)) / 5,  # 归一化稳定性
            'risk_cv': np.std(risk_variations) / np.mean(risk_variations),  # 风险变异系数
            'mean_risk_change': np.mean(risk_variations) - np.mean([
                risk_optimizer.calculate_total_risk(t, g['features'], y_pred_func) 
                for t, g in zip(optimal_timepoints, groups)
            ]),
            'confidence_intervals': {
                f'group_{i}': (np.percentile(timepoint_variations[:, i], 5),
                              np.percentile(timepoint_variations[:, i], 95))
                for i in range(len(groups))
            }
        }
        
        return robustness_metrics