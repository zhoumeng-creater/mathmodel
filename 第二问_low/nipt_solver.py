"""
NIPT问题主求解器
整合所有组件，实现两阶段优化
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from nipt_models import YConcentrationPredictor, RiskEvaluator, BMIGroupOptimizer
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class NIPTSolver:
    """NIPT问题求解器（整合所有组件）"""
    
    def __init__(self, risk_weights: Optional[Dict[str, float]] = None):
        """
        初始化求解器
        
        参数:
            risk_weights: 风险权重字典 {'delay': α, 'fail': β, 'cost': γ}
        """
        # 设置风险权重（默认值或用户指定）
        if risk_weights is None:
            risk_weights = {'delay': 0.5, 'fail': 0.3, 'cost': 0.2}
        
        # 初始化组件
        self.predictor = YConcentrationPredictor()
        self.evaluator = RiskEvaluator(
            alpha=risk_weights['delay'],
            beta=risk_weights['fail'],
            gamma=risk_weights['cost']
        )
        self.optimizer = BMIGroupOptimizer(self.predictor, self.evaluator)
        
        # 存储结果
        self.optimal_solution = None
        self.sensitivity_results = None
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        参数:
            data: 原始数据
            
        返回:
            处理后的男胎数据
        """
        # 打印列名以便调试
        print("数据列名:", data.columns.tolist()[:10])  # 显示前10个列名
        
        # 筛选男胎数据（Y染色体浓度非空）
        # 注意：Y染色体浓度在Excel中可能是小数形式（0.045表示4.5%）
        male_data = data[data['Y染色体浓度'].notna()].copy()
        
        # 将Y染色体浓度转换为百分比（如果是小数形式）
        if male_data['Y染色体浓度'].max() < 1:  # 说明是小数形式
            male_data['Y染色体浓度'] = male_data['Y染色体浓度'] * 100
            print("Y染色体浓度已转换为百分比形式")
        
        # 提取孕周数值（从"周数+天数"格式中）
        def parse_week(week_str):
            if pd.isna(week_str):
                return np.nan
            try:
                week_str = str(week_str)
                if '+' in week_str:
                    parts = week_str.split('+')
                    weeks = float(parts[0])
                    days = float(parts[1]) if len(parts) > 1 else 0
                    return weeks + days / 7
                else:
                    # 可能是纯数字格式
                    return float(week_str)
            except:
                return np.nan
        
        male_data['孕周'] = male_data['孕妇本次检测时的孕周（周数+天数）'].apply(parse_week)
        
        # 创建简化列名映射
        male_data['年龄'] = male_data['孕妇年龄']
        male_data['身高'] = male_data['孕妇身高']
        male_data['体重'] = male_data['孕妇体重']
        male_data['BMI'] = male_data['孕妇BMI指标']
        
        # 删除缺失关键数据的行
        required_cols = ['孕周', 'BMI', '年龄', '身高', '体重', 'Y染色体浓度']
        male_data = male_data.dropna(subset=required_cols)
        
        # 打印数据范围以便调试
        print(f"Y染色体浓度范围: {male_data['Y染色体浓度'].min():.2f}% - {male_data['Y染色体浓度'].max():.2f}%")
        print(f"BMI范围: {male_data['BMI'].min():.1f} - {male_data['BMI'].max():.1f}")
        print(f"孕周范围: {male_data['孕周'].min():.1f} - {male_data['孕周'].max():.1f}")
        
        # 异常值处理（IQR方法）
        for col in ['BMI', 'Y染色体浓度']:
            Q1 = male_data[col].quantile(0.25)
            Q3 = male_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before_count = len(male_data)
            male_data = male_data[(male_data[col] >= lower) & (male_data[col] <= upper)]
            after_count = len(male_data)
            if before_count > after_count:
                print(f"移除{col}异常值: {before_count - after_count}条")
        
        # 确保孕周在合理范围
        male_data = male_data[(male_data['孕周'] >= 10) & (male_data['孕周'] <= 25)]
        
        print(f"预处理完成，保留{len(male_data)}条有效数据")
        
        return male_data
    
    def two_stage_optimization(self, data: pd.DataFrame) -> Dict:
        """
        两阶段优化主流程
        
        参数:
            data: 预处理后的数据
            
        返回:
            优化结果字典
        """
        # 训练预测模型
        print("训练预测模型...")
        self.predictor.train(data)
        
        # 第一阶段：确定最优分组数K和初始边界
        print("\n第一阶段：确定最优分组数K...")
        best_K = self._find_optimal_K(data)
        print(f"最优分组数K = {best_K}")
        
        # 初始化分组
        boundaries = self.optimizer.initialize_groups_kmeans(data, best_K)
        print(f"初始BMI边界: {[f'{b:.1f}' for b in boundaries]}")
        
        # 第二阶段：优化各组检测时点
        print("\n第二阶段：优化检测时点...")
        detection_times = []
        group_risks = []
        
        for i in range(len(boundaries) - 1):
            bmi_range = (boundaries[i], boundaries[i + 1])
            best_week, min_risk = self.optimizer.optimize_detection_time(data, bmi_range)
            detection_times.append(best_week)
            group_risks.append(min_risk)
            print(f"BMI组[{bmi_range[0]:.1f}, {bmi_range[1]:.1f}): 最佳检测时点 = {best_week}周, 风险 = {min_risk:.4f}")
        
        # 局部搜索优化
        print("\n局部搜索优化...")
        optimized_boundaries, optimized_times = self.optimizer.local_search(
            data, boundaries, detection_times, max_iter=10
        )
        
        # 计算最终风险
        final_risk = self.optimizer._calculate_total_risk(data, optimized_boundaries, optimized_times)
        
        # 构建结果
        self.optimal_solution = {
            'K': best_K,
            'boundaries': optimized_boundaries,
            'detection_times': optimized_times,
            'group_risks': group_risks,
            'total_risk': final_risk,
            'group_info': self._get_group_info(data, optimized_boundaries, optimized_times)
        }
        
        return self.optimal_solution
    
    def _find_optimal_K(self, data: pd.DataFrame) -> int:
        """
        找到最优的分组数K
        
        参数:
            data: 数据
            
        返回:
            最优K值
        """
        K_candidates = [3, 4, 5]  # 候选分组数
        best_K = 3
        best_score = -1
        
        for K in K_candidates:
            # 构建聚类特征
            earliest_times = []
            for _, row in data.iterrows():
                if row['BMI'] < 28:
                    earliest_time = 11
                elif row['BMI'] < 32:
                    earliest_time = 13
                elif row['BMI'] < 36:
                    earliest_time = 15
                else:
                    earliest_time = 17
                earliest_times.append(earliest_time)
            
            X = np.column_stack([data['BMI'].values, earliest_times])
            
            # 计算轮廓系数
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            if K > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_K = K
        
        return best_K
    
    def _get_group_info(self, data: pd.DataFrame, boundaries: List[float], 
                       times: List[int]) -> List[Dict]:
        """
        获取每组的详细信息
        
        参数:
            data: 数据
            boundaries: BMI边界
            times: 检测时点
            
        返回:
            组信息列表
        """
        group_info = []
        
        for i in range(len(boundaries) - 1):
            bmi_range = (boundaries[i], boundaries[i + 1])
            group_data = data[(data['BMI'] >= bmi_range[0]) & 
                            (data['BMI'] < bmi_range[1])].copy()
            
            if len(group_data) > 0:
                group_data['孕周'] = times[i]
                success_probs = self.predictor.get_success_probability(group_data)
                
                info = {
                    'group_id': i + 1,
                    'bmi_range': f"[{bmi_range[0]:.1f}, {bmi_range[1]:.1f})",
                    'detection_week': times[i],
                    'sample_size': len(group_data),
                    'mean_success_prob': success_probs.mean(),
                    'std_success_prob': success_probs.std(),
                    'min_success_prob': success_probs.min(),
                    'max_success_prob': success_probs.max()
                }
                group_info.append(info)
        
        return group_info
    
    def sensitivity_analysis(self, data: pd.DataFrame, 
                           error_range: float = 0.05) -> Dict:
        """
        误差敏感性分析
        
        参数:
            data: 数据
            error_range: 误差范围（默认±5%）
            
        返回:
            敏感性分析结果
        """
        if self.optimal_solution is None:
            raise ValueError("请先运行two_stage_optimization获取最优解")
        
        print("\n进行敏感性分析...")
        
        # 原始最优解
        original_boundaries = self.optimal_solution['boundaries']
        original_times = self.optimal_solution['detection_times']
        original_risk = self.optimal_solution['total_risk']
        
        # 测试不同误差情况
        error_levels = [-error_range, 0, error_range]
        results = []
        
        for error in error_levels:
            # 模拟测量误差
            noisy_data = data.copy()
            noisy_data['Y染色体浓度'] = noisy_data['Y染色体浓度'] * (1 + error)
            
            # 重新训练模型
            temp_predictor = YConcentrationPredictor()
            temp_predictor.train(noisy_data)
            
            # 使用原始方案计算风险
            temp_optimizer = BMIGroupOptimizer(temp_predictor, self.evaluator)
            risk = temp_optimizer._calculate_total_risk(noisy_data, 
                                                       original_boundaries, 
                                                       original_times)
            
            # 计算成功概率变化
            success_probs = []
            for i in range(len(original_boundaries) - 1):
                bmi_range = (original_boundaries[i], original_boundaries[i + 1])
                group_data = noisy_data[(noisy_data['BMI'] >= bmi_range[0]) & 
                                       (noisy_data['BMI'] < bmi_range[1])].copy()
                
                if len(group_data) > 0:
                    group_data['孕周'] = original_times[i]
                    probs = temp_predictor.get_success_probability(group_data)
                    success_probs.append(probs.mean())
            
            results.append({
                'error_level': error,
                'total_risk': risk,
                'risk_change': (risk - original_risk) / original_risk,
                'mean_success_prob': np.mean(success_probs),
                'robust': abs(risk - original_risk) / original_risk < 0.1
            })
        
        self.sensitivity_results = {
            'error_range': error_range,
            'results': results,
            'conclusion': '方案稳健' if all(r['robust'] for r in results) else '方案对误差敏感'
        }
        
        return self.sensitivity_results
    
    def get_optimal_solution(self) -> Optional[Dict]:
        """
        获取最优解
        
        返回:
            最优解字典，如未运行优化则返回None
        """
        return self.optimal_solution
    
    def get_recommendation(self, bmi: float) -> Dict:
        """
        根据BMI值给出检测建议
        
        参数:
            bmi: 孕妇的BMI值
            
        返回:
            检测建议字典
        """
        if self.optimal_solution is None:
            raise ValueError("请先运行优化获取解决方案")
        
        boundaries = self.optimal_solution['boundaries']
        times = self.optimal_solution['detection_times']
        
        # 找到对应的组
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= bmi < boundaries[i + 1]:
                return {
                    'bmi': bmi,
                    'group': i + 1,
                    'bmi_range': f"[{boundaries[i]:.1f}, {boundaries[i + 1]:.1f})",
                    'recommended_week': times[i],
                    'group_info': self.optimal_solution['group_info'][i]
                }
        
        # BMI超出范围
        if bmi < boundaries[0]:
            return {
                'bmi': bmi,
                'group': 1,
                'recommended_week': times[0],
                'note': 'BMI低于最小边界，使用第一组建议'
            }
        else:
            return {
                'bmi': bmi,
                'group': len(times),
                'recommended_week': times[-1],
                'note': 'BMI高于最大边界，使用最后一组建议'
            }