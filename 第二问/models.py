"""
NIPT优化模型核心组件
严格实现第二问数学模型方案
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')


class YConcentrationPredictor:
    """Y染色体浓度预测模型（随机森林）"""
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        初始化预测器
        参数严格按照方案：使用随机森林进行Y浓度预测
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=20,  # 方案中的最小叶节点样本数
            random_state=random_state
        )
        self.is_fitted = False
        
    def prepare_features(self, data):
        """
        特征工程 - 严格按照方案构建特征向量
        X = [Week, BMI, Week², BMI², Week×BMI, Age, Height, Weight]
        """
        features = pd.DataFrame()
        features['Week'] = data['Week']
        features['BMI'] = data['BMI']
        features['Week_squared'] = data['Week'] ** 2
        features['BMI_squared'] = data['BMI'] ** 2
        features['Week_BMI_interaction'] = data['Week'] * data['BMI']
        features['Age'] = data['Age']
        features['Height'] = data['Height']
        features['Weight'] = data['Weight']
        
        return features
    
    def fit(self, data):
        """训练模型"""
        # 准备特征
        X = self.prepare_features(data)
        y = data['Y_concentration']
        
        # 训练随机森林
        self.model.fit(X, y)
        self.is_fitted = True
        
        # 计算训练误差用于后续分析
        predictions = self.model.predict(X)
        self.train_rmse = np.sqrt(np.mean((predictions - y) ** 2))
        
        return self
    
    def predict(self, week, bmi, age, height, weight):
        """
        预测Y浓度
        输入可以是标量或数组
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 构建特征矩阵
        week = np.atleast_1d(week)
        bmi = np.atleast_1d(bmi)
        age = np.atleast_1d(age)
        height = np.atleast_1d(height)
        weight = np.atleast_1d(weight)
        
        features = np.column_stack([
            week, bmi, week**2, bmi**2, week*bmi,
            age, height, weight
        ])
        
        return self.model.predict(features)
    
    def get_success_probability(self, week, bmi_group_data, threshold=4.0):
        """
        计算某BMI组在特定孕周达标概率
        P(Y_conc >= 4% | week, BMI_group)
        """
        # 对组内所有样本预测
        predictions = []
        for _, row in bmi_group_data.iterrows():
            pred = self.predict(
                week, row['BMI'], row['Age'], 
                row['Height'], row['Weight']
            )[0]
            predictions.append(pred)
        
        # 计算达标比例
        predictions = np.array(predictions)
        success_rate = np.mean(predictions >= threshold)
        
        return success_rate


class RiskCalculator:
    """风险计算器 - 严格按照方案的风险函数"""
    
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2):
        """
        初始化风险权重
        α: 时间风险权重
        β: 失败风险权重  
        γ: 延迟风险权重
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 逻辑回归参数（用于失败概率预测）
        self.theta = np.array([-5.0, 0.3, 0.1, -0.01])  # [θ0, θ1, θ2, θ3]
        
    def timing_risk(self, t):
        """
        时间风险函数 - 严格按照方案定义
        R_timing(t)
        """
        t = np.atleast_1d(t)
        risk = np.zeros_like(t, dtype=float)
        
        # 早期（≤12周）：风险0.1
        mask_early = t <= 12
        risk[mask_early] = 0.1
        
        # 中期（12-27周）：风险线性增长
        mask_middle = (t > 12) & (t <= 27)
        risk[mask_middle] = 0.3 + 0.05 * (t[mask_middle] - 12)
        
        # 晚期（>27周）：风险1.0
        mask_late = t > 27
        risk[mask_late] = 1.0
        
        return risk if len(risk) > 1 else risk[0]
    
    def failure_risk(self, t, bmi):
        """
        检测失败风险 - 使用逻辑回归模型
        R_failure(t, BMI) = P_fail(t, BMI) × C_retest
        """
        # 逻辑回归：P_fail = 1 / (1 + exp(θ0 + θ1*t + θ2*BMI + θ3*t*BMI))
        linear_comb = (self.theta[0] + self.theta[1] * t + 
                      self.theta[2] * bmi + self.theta[3] * t * bmi)
        p_fail = 1 / (1 + np.exp(linear_comb))
        
        # 假设重测成本系数为2.0
        c_retest = 2.0
        
        return p_fail * c_retest
    
    def delay_risk(self, t, p_not_reach_threshold, t_max=25, t_min=10):
        """
        延迟诊断风险
        R_delay(t) = P(Y_conc < 4% | t) × (t_max - t) / (t_max - t_min)
        """
        normalized_delay = (t_max - t) / (t_max - t_min)
        return p_not_reach_threshold * normalized_delay
    
    def total_risk(self, t, bmi, p_not_reach_threshold):
        """
        总风险函数 - 加权组合三类风险
        Risk_total = α × R_timing + β × R_failure + γ × R_delay
        """
        r_timing = self.timing_risk(t)
        r_failure = self.failure_risk(t, bmi)
        r_delay = self.delay_risk(t, p_not_reach_threshold)
        
        total = (self.alpha * r_timing + 
                self.beta * r_failure + 
                self.gamma * r_delay)
        
        return total


class BMIGrouping:
    """BMI分组管理类"""
    
    def __init__(self, k, boundaries):
        """
        初始化分组
        k: 分组数量
        boundaries: 分组边界点 [b0, b1, ..., bk]
        """
        self.k = k
        self.boundaries = np.sort(boundaries)  # 确保边界有序
        
    def assign_groups(self, data):
        """
        将样本分配到各组
        返回带有group标签的数据
        """
        data = data.copy()
        data['group'] = pd.cut(
            data['BMI'], 
            bins=self.boundaries,
            labels=range(self.k),
            include_lowest=True,
            right=False  # 左闭右开区间 [bi-1, bi)
        )
        return data
    
    def get_group_stats(self, data):
        """获取各组统计信息"""
        grouped_data = self.assign_groups(data)
        stats = []
        
        for g in range(self.k):
            group_data = grouped_data[grouped_data['group'] == g]
            if len(group_data) > 0:
                stats.append({
                    'group': g,
                    'bmi_range': f"[{self.boundaries[g]:.1f}, {self.boundaries[g+1]:.1f})",
                    'count': len(group_data),
                    'mean_bmi': group_data['BMI'].mean(),
                    'mean_week': group_data['Week'].mean(),
                    'mean_y_conc': group_data['Y_concentration'].mean()
                })
        
        return pd.DataFrame(stats)
    
    def check_balance_constraint(self, data, n_min=20, n_max=500):
        """
        检查分组平衡约束
        n_min <= |G_k| <= n_max
        """
        grouped_data = self.assign_groups(data)
        for g in range(self.k):
            group_size = len(grouped_data[grouped_data['group'] == g])
            if group_size < n_min or group_size > n_max:
                return False
        return True
    
    def get_group_boundaries(self):
        """返回分组边界的字符串表示"""
        boundaries_str = []
        for i in range(self.k):
            boundaries_str.append(
                f"Group {i+1}: [{self.boundaries[i]:.2f}, {self.boundaries[i+1]:.2f})"
            )
        return boundaries_str