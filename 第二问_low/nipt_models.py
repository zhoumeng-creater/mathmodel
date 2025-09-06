"""
NIPT核心模型模块
包含Y染色体浓度预测、风险评估和BMI分组优化
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class YConcentrationPredictor:
    """Y染色体浓度预测器（CART决策树回归）"""
    
    def __init__(self, max_depth: int = 8, min_samples_split: int = 20, 
                 min_samples_leaf: int = 10):
        """
        初始化预测器
        
        参数:
            max_depth: 树的最大深度（避免过拟合）
            min_samples_split: 内部节点最小样本数
            min_samples_leaf: 叶节点最小样本数
        """
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        self.logistic_model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        特征工程：构建原始特征和衍生特征
        
        参数:
            data: 包含孕周、BMI、年龄、身高、体重的数据框
            
        返回:
            特征矩阵
        """
        features = []
        
        # 原始特征
        features.append(data['孕周'].values)
        features.append(data['BMI'].values)
        features.append(data['年龄'].values)
        features.append(data['身高'].values)
        features.append(data['体重'].values)
        
        # 衍生特征
        features.append(np.floor(data['BMI'].values / 2))  # BMI粗粒度分组
        features.append(data['孕周'].values ** 2)  # 孕周平方（非线性）
        features.append(data['BMI'].values * data['孕周'].values)  # BMI×孕周交互
        features.append(data['体重'].values / data['身高'].values)  # 体重身高比
        
        return np.column_stack(features)
    
    def train(self, data: pd.DataFrame):
        """
        训练CART模型和逻辑回归模型
        
        参数:
            data: 训练数据，必须包含Y染色体浓度
        """
        # 构建特征
        X = self.build_features(data)
        y = data['Y染色体浓度'].values
        
        # 训练CART回归模型（预测Y浓度）
        self.model.fit(X, y)
        
        # 训练逻辑回归模型（预测成功概率）
        X_scaled = self.scaler.fit_transform(X)
        y_binary = (y >= 4.0).astype(int)  # 二分类：是否达标
        self.logistic_model.fit(X_scaled, y_binary)
        
        self.is_trained = True
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        预测Y染色体浓度
        
        参数:
            data: 待预测数据
            
        返回:
            预测的Y浓度值
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        X = self.build_features(data)
        return self.model.predict(X)
    
    def get_success_probability(self, data: pd.DataFrame) -> np.ndarray:
        """
        计算检测成功概率（Y浓度≥4%的概率）
        
        参数:
            data: 输入数据
            
        返回:
            成功概率数组
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        X = self.build_features(data)
        X_scaled = self.scaler.transform(X)
        return self.logistic_model.predict_proba(X_scaled)[:, 1]


class RiskEvaluator:
    """风险评估器"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """
        初始化风险评估器
        
        参数:
            alpha: 延迟风险权重
            beta: 失败风险权重
            gamma: 成本风险权重
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cost_test = 1000  # 单次检测成本（元）
        self.cost_repeat = 1500  # 重复检测成本（元）
        
    def calculate_delay_risk(self, actual_week: float, optimal_week: float = 12) -> float:
        """
        计算延迟风险（分段函数）
        
        参数:
            actual_week: 实际检测孕周
            optimal_week: 最优检测孕周（默认12周）
            
        返回:
            延迟风险值
        """
        x = actual_week - optimal_week
        
        if x <= 0:
            return 0  # 提前检测无延迟风险
        elif 0 < x <= 2:
            return 0.1 * x
        elif 2 < x <= 4:
            return 0.2 + 0.3 * (x - 2)
        else:  # x > 4
            return 0.8 + 0.5 * (x - 4)
    
    def calculate_fail_risk(self, success_prob: float) -> float:
        """
        计算检测失败风险
        
        参数:
            success_prob: 检测成功概率
            
        返回:
            失败风险值
        """
        fail_prob = 1 - success_prob
        return fail_prob * self.cost_repeat
    
    def calculate_cost_risk(self, retest_prob: float) -> float:
        """
        计算成本风险
        
        参数:
            retest_prob: 需要重新检测的概率
            
        返回:
            成本风险值
        """
        return self.cost_test + retest_prob * self.cost_test
    
    def total_risk(self, actual_week: float, success_prob: float, 
                   retest_prob: float, group_weight: float = 1.0) -> float:
        """
        计算综合风险
        
        参数:
            actual_week: 实际检测孕周
            success_prob: 检测成功概率
            retest_prob: 重新检测概率
            group_weight: 组权重
            
        返回:
            综合风险值
        """
        r_delay = self.calculate_delay_risk(actual_week)
        r_fail = self.calculate_fail_risk(success_prob)
        r_cost = self.calculate_cost_risk(retest_prob)
        
        # 归一化处理
        r_fail_norm = r_fail / self.cost_repeat
        r_cost_norm = r_cost / (2 * self.cost_test)
        
        total = self.alpha * r_delay + self.beta * r_fail_norm + self.gamma * r_cost_norm
        return total * group_weight


class BMIGroupOptimizer:
    """BMI分组优化器"""
    
    def __init__(self, predictor: YConcentrationPredictor, 
                 evaluator: RiskEvaluator):
        """
        初始化优化器
        
        参数:
            predictor: Y浓度预测器
            evaluator: 风险评估器
        """
        self.predictor = predictor
        self.evaluator = evaluator
        
        # 约束参数
        self.min_group_width = 2.0  # 最小组宽（BMI单位）
        self.min_group_size = 30  # 最小组样本数
        self.min_success_prob = 0.85  # 最小成功概率
        self.week_range = (10, 25)  # 孕周范围
        
    def initialize_groups_kmeans(self, data: pd.DataFrame, K: int) -> List[float]:
        """
        使用K-means初始化BMI分组
        
        参数:
            data: 包含BMI和最早达标时间的数据
            K: 分组数
            
        返回:
            BMI分组边界点列表
        """
        # 计算每个样本的最早达标时间
        earliest_times = []
        for _, row in data.iterrows():
            # 简化计算：基于BMI估算最早达标时间
            if row['BMI'] < 28:
                earliest_time = 11
            elif row['BMI'] < 32:
                earliest_time = 13
            elif row['BMI'] < 36:
                earliest_time = 15
            else:
                earliest_time = 17
            earliest_times.append(earliest_time)
        
        # 构建聚类特征
        X = np.column_stack([data['BMI'].values, earliest_times])
        
        # K-means聚类
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # 根据聚类结果确定边界
        boundaries = [data['BMI'].min()]
        for k in range(K):
            cluster_bmi = data['BMI'].values[labels == k]
            if len(cluster_bmi) > 0:
                boundaries.append(cluster_bmi.max())
        
        # 去重排序
        boundaries = sorted(list(set(boundaries)))
        
        # 确保满足最小组宽约束
        validated_boundaries = [boundaries[0]]
        for b in boundaries[1:]:
            if b - validated_boundaries[-1] >= self.min_group_width:
                validated_boundaries.append(b)
        
        # 确保覆盖全范围
        if validated_boundaries[-1] < data['BMI'].max():
            validated_boundaries.append(data['BMI'].max() + 0.1)
        
        return validated_boundaries
    
    def optimize_detection_time(self, data: pd.DataFrame, 
                               bmi_range: Tuple[float, float]) -> Tuple[int, float]:
        """
        网格搜索最佳检测时点
        
        参数:
            data: 该BMI组的数据
            bmi_range: BMI范围
            
        返回:
            (最佳孕周, 最小风险值)
        """
        best_week = self.week_range[0]
        min_risk = float('inf')
        
        # 筛选该组数据
        group_data = data[(data['BMI'] >= bmi_range[0]) & 
                         (data['BMI'] < bmi_range[1])].copy()
        
        if len(group_data) < self.min_group_size:
            return best_week, min_risk
        
        # 网格搜索
        for week in range(self.week_range[0], self.week_range[1] + 1):
            # 构建预测数据
            pred_data = group_data.copy()
            pred_data['孕周'] = week
            
            # 预测成功概率
            success_probs = self.predictor.get_success_probability(pred_data)
            mean_success_prob = success_probs.mean()
            
            # 检查约束：达标概率≥0.85
            if mean_success_prob < self.min_success_prob:
                continue
            
            # 计算重测概率（简化：基于历史数据）
            retest_prob = 0.1 if week >= 13 else 0.2
            
            # 计算该周的总风险
            group_weight = len(group_data) / len(data)
            risk = self.evaluator.total_risk(week, mean_success_prob, 
                                            retest_prob, group_weight)
            
            if risk < min_risk:
                min_risk = risk
                best_week = week
        
        return best_week, min_risk
    
    def local_search(self, data: pd.DataFrame, boundaries: List[float], 
                    detection_times: List[int], max_iter: int = 10) -> Tuple[List[float], List[int]]:
        """
        局部搜索优化（调整边界和时点）
        
        参数:
            data: 全部数据
            boundaries: 当前BMI边界
            detection_times: 当前检测时点
            max_iter: 最大迭代次数
            
        返回:
            优化后的(边界, 时点)
        """
        best_boundaries = boundaries.copy()
        best_times = detection_times.copy()
        best_risk = self._calculate_total_risk(data, best_boundaries, best_times)
        
        for iteration in range(max_iter):
            improved = False
            
            # 尝试调整边界（除首尾外）
            for i in range(1, len(best_boundaries) - 1):
                for delta in [-1, 1]:
                    new_boundaries = best_boundaries.copy()
                    new_boundaries[i] += delta
                    
                    # 验证约束
                    if not self._validate_boundaries(new_boundaries):
                        continue
                    
                    # 重新优化时点
                    new_times = []
                    for j in range(len(new_boundaries) - 1):
                        bmi_range = (new_boundaries[j], new_boundaries[j + 1])
                        week, _ = self.optimize_detection_time(data, bmi_range)
                        new_times.append(week)
                    
                    # 计算新风险
                    new_risk = self._calculate_total_risk(data, new_boundaries, new_times)
                    
                    if new_risk < best_risk:
                        best_risk = new_risk
                        best_boundaries = new_boundaries
                        best_times = new_times
                        improved = True
            
            # 尝试调整时点
            for i in range(len(best_times)):
                for delta in [-1, 1]:
                    new_week = best_times[i] + delta
                    if new_week < self.week_range[0] or new_week > self.week_range[1]:
                        continue
                    
                    new_times = best_times.copy()
                    new_times[i] = new_week
                    
                    # 验证约束
                    if not self._validate_times(data, best_boundaries, new_times):
                        continue
                    
                    # 计算新风险
                    new_risk = self._calculate_total_risk(data, best_boundaries, new_times)
                    
                    if new_risk < best_risk:
                        best_risk = new_risk
                        best_times = new_times
                        improved = True
            
            if not improved:
                break
        
        return best_boundaries, best_times
    
    def _validate_boundaries(self, boundaries: List[float]) -> bool:
        """验证边界约束"""
        for i in range(len(boundaries) - 1):
            if boundaries[i + 1] - boundaries[i] < self.min_group_width:
                return False
            if boundaries[i + 1] <= boundaries[i]:
                return False
        return True
    
    def _validate_times(self, data: pd.DataFrame, boundaries: List[float], 
                       times: List[int]) -> bool:
        """验证时间约束和达标概率约束"""
        for i, week in enumerate(times):
            bmi_range = (boundaries[i], boundaries[i + 1])
            group_data = data[(data['BMI'] >= bmi_range[0]) & 
                            (data['BMI'] < bmi_range[1])].copy()
            
            if len(group_data) < self.min_group_size:
                return False
            
            group_data['孕周'] = week
            success_probs = self.predictor.get_success_probability(group_data)
            
            if success_probs.mean() < self.min_success_prob:
                return False
        
        return True
    
    def _calculate_total_risk(self, data: pd.DataFrame, boundaries: List[float], 
                             times: List[int]) -> float:
        """计算给定方案的总风险"""
        total_risk = 0
        
        for i in range(len(boundaries) - 1):
            bmi_range = (boundaries[i], boundaries[i + 1])
            group_data = data[(data['BMI'] >= bmi_range[0]) & 
                            (data['BMI'] < bmi_range[1])].copy()
            
            if len(group_data) == 0:
                continue
            
            group_data['孕周'] = times[i]
            success_probs = self.predictor.get_success_probability(group_data)
            mean_success_prob = success_probs.mean()
            
            retest_prob = 0.1 if times[i] >= 13 else 0.2
            group_weight = len(group_data) / len(data)
            
            risk = self.evaluator.total_risk(times[i], mean_success_prob, 
                                            retest_prob, group_weight)
            total_risk += risk
        
        return total_risk