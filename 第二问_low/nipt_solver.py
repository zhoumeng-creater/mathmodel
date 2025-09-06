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
        数据预处理 - 完全借鉴第三问的成功方法
        
        参数:
            data: 原始数据
            
        返回:
            处理后的男胎数据
        """
        print(f"原始数据形状: {data.shape}")
        print(f"前5行前10列预览:")
        print(data.iloc[:5, :10])
        
        # 创建处理后的DataFrame
        processed_df = pd.DataFrame()
        
        # 按列索引读取（A=0, B=1, C=2, ...）
        # C列(索引2): 年龄
        if data.shape[1] > 2:
            processed_df['年龄'] = pd.to_numeric(data.iloc[:, 2], errors='coerce')
        
        # D列(索引3): 身高
        if data.shape[1] > 3:
            processed_df['身高'] = pd.to_numeric(data.iloc[:, 3], errors='coerce')
        
        # E列(索引4): 体重
        if data.shape[1] > 4:
            processed_df['体重'] = pd.to_numeric(data.iloc[:, 4], errors='coerce')
        
        # J列(索引9): 检测孕周 - 先读取原始值
        if data.shape[1] > 9:
            processed_df['Week_raw'] = data.iloc[:, 9]
            print(f"\n孕周原始数据(J列)前10个值:")
            print(data.iloc[:10, 9].tolist())
        
        # K列(索引10): BMI
        if data.shape[1] > 10:
            processed_df['BMI'] = pd.to_numeric(data.iloc[:, 10], errors='coerce')
        
        # V列(索引21): Y染色体浓度 - 关键列！
        if data.shape[1] > 21:
            print(f"\nV列(索引21)前10个值:")
            print(data.iloc[:10, 21].tolist())
            y_values = pd.to_numeric(data.iloc[:, 21], errors='coerce')
            
            # 统计非空值
            non_na_count = y_values.notna().sum()
            print(f"Y染色体浓度非空值数量: {non_na_count}")
            
            if non_na_count > 0:
                # 检查数值范围
                y_min = y_values.dropna().min()
                y_max = y_values.dropna().max()
                print(f"Y染色体浓度原始范围: {y_min:.6f} - {y_max:.6f}")
                
                # 判断是否需要转换（如果最大值小于1，说明是小数形式）
                if y_max < 1:
                    print(f"检测到小数形式，转换为百分比")
                    processed_df['Y染色体浓度'] = y_values * 100
                else:
                    processed_df['Y染色体浓度'] = y_values
            else:
                print(f"警告：Y染色体浓度列全为空！")
                processed_df['Y染色体浓度'] = y_values
        else:
            print(f"错误：数据只有{data.shape[1]}列，无法读取V列(需要至少22列)")
            return pd.DataFrame()
        
        print(f"\n筛选前样本数: {len(processed_df)}")
        print(f"Y染色体浓度列的统计:")
        print(f"  - 非空值: {processed_df['Y染色体浓度'].notna().sum()}")
        print(f"  - 空值: {processed_df['Y染色体浓度'].isna().sum()}")
        
        # 筛选男胎数据（Y染色体浓度不为空）
        male_mask = processed_df['Y染色体浓度'].notna()
        processed_df = processed_df[male_mask].copy()
        
        print(f"筛选男胎后样本数: {len(processed_df)}")
        
        if len(processed_df) == 0:
            print("警告：没有找到男胎数据！检查是否列索引偏移...")
            # 如果没有数据，返回空DataFrame
            return pd.DataFrame()
        
        # 处理孕周数据 - 关键步骤！
        if 'Week_raw' in processed_df.columns:
            def parse_week(week_str):
                """解析孕周字符串，转换为数值"""
                if pd.isna(week_str):
                    return np.nan
                    
                try:
                    if isinstance(week_str, (int, float)):
                        return float(week_str)
                    
                    week_str = str(week_str).strip()
                    
                    # 处理 "12w+3" 格式（w表示周）
                    if 'w' in week_str.lower():
                        # 去除w/W，然后按+分割
                        week_str = week_str.lower().replace('w', '')
                        if '+' in week_str:
                            parts = week_str.split('+')
                            weeks = float(parts[0])
                            days = float(parts[1]) if len(parts) > 1 else 0
                            return weeks + days / 7.0
                        else:
                            return float(week_str)
                    # 处理 "12+3" 格式
                    elif '+' in week_str:
                        parts = week_str.split('+')
                        weeks = float(parts[0])
                        days = float(parts[1]) if len(parts) > 1 else 0
                        return weeks + days / 7.0
                    # 处理 "12周3天" 格式
                    elif '周' in week_str:
                        import re
                        # 提取数字
                        numbers = re.findall(r'\d+', week_str)
                        if numbers:
                            weeks = float(numbers[0])
                            days = float(numbers[1]) if len(numbers) > 1 else 0
                            return weeks + days / 7.0
                        return np.nan
                    else:
                        # 尝试直接转换为数字
                        return float(week_str)
                except Exception as e:
                    print(f"  解析孕周失败: '{week_str}' - {e}")
                    return np.nan
            
            processed_df['孕周'] = processed_df['Week_raw'].apply(parse_week)
            processed_df = processed_df.drop('Week_raw', axis=1)
            
            # 打印解析结果
            print(f"\n孕周解析结果:")
            print(f"  成功解析: {processed_df['孕周'].notna().sum()}")
            print(f"  解析失败: {processed_df['孕周'].isna().sum()}")
            if processed_df['孕周'].notna().sum() > 0:
                print(f"  孕周范围: {processed_df['孕周'].min():.1f} - {processed_df['孕周'].max():.1f}")
        else:
            print("警告：没有找到孕周数据！")
            processed_df['孕周'] = 15  # 使用默认值
        
        # 处理缺失值 - 使用中位数填充
        print(f"\n处理缺失值前的统计:")
        for col in processed_df.columns:
            na_count = processed_df[col].isna().sum()
            if na_count > 0:
                print(f"  {col}: {na_count}个缺失值")
        
        numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in processed_df.columns:
                median_value = processed_df[col].median()
                if pd.notna(median_value):
                    processed_df[col].fillna(median_value, inplace=True)
                else:
                    # 使用默认值
                    default_vals = {'年龄': 30, '身高': 165, '体重': 60, 'BMI': 22, '孕周': 15}
                    processed_df[col].fillna(default_vals.get(col, 0), inplace=True)
        
        # 删除关键特征仍有缺失的行
        essential_cols = ['孕周', 'BMI', 'Y染色体浓度']
        before_drop = len(processed_df)
        processed_df = processed_df.dropna(subset=[col for col in essential_cols if col in processed_df.columns])
        print(f"\n删除关键列缺失值: {before_drop} -> {len(processed_df)}")
        
        if len(processed_df) == 0:
            print("警告：处理后没有有效数据！")
            return pd.DataFrame()
        
        # 过滤异常值 - 使用更宽松的标准
        initial_count = len(processed_df)
        
        # 基于IQR方法过滤异常值，但使用2.5倍IQR（更宽松）
        for col in ['BMI', 'Y染色体浓度']:
            if col in processed_df.columns:
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 2.5 * IQR
                upper_bound = Q3 + 2.5 * IQR
                
                before_filter = len(processed_df)
                processed_df = processed_df[(processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
                after_filter = len(processed_df)
                
                if before_filter != after_filter:
                    print(f"  IQR过滤 {col}: {before_filter} -> {after_filter} (移除{before_filter-after_filter}条)")
        
        # 确保合理范围
        before = len(processed_df)
        processed_df = processed_df[(processed_df['孕周'] >= 10) & (processed_df['孕周'] <= 25)]
        if len(processed_df) != before:
            print(f"  孕周范围过滤: {before} -> {len(processed_df)} (移除{before-len(processed_df)}条)")
        
        before = len(processed_df)
        processed_df = processed_df[(processed_df['BMI'] >= 15) & (processed_df['BMI'] <= 50)]
        if len(processed_df) != before:
            print(f"  BMI范围过滤: {before} -> {len(processed_df)} (移除{before-len(processed_df)}条)")
        
        before = len(processed_df)  
        processed_df = processed_df[(processed_df['Y染色体浓度'] >= 0) & (processed_df['Y染色体浓度'] <= 100)]
        if len(processed_df) != before:
            print(f"  Y浓度范围过滤: {before} -> {len(processed_df)} (移除{before-len(processed_df)}条)")
        
        if len(processed_df) > 0:
            print(f"\n最终数据统计:")
            print(f"  样本数: {len(processed_df)}")
            print(f"  Y浓度范围: {processed_df['Y染色体浓度'].min():.2f}% - {processed_df['Y染色体浓度'].max():.2f}%")
            print(f"  BMI范围: {processed_df['BMI'].min():.1f} - {processed_df['BMI'].max():.1f}")
            print(f"  孕周范围: {processed_df['孕周'].min():.1f} - {processed_df['孕周'].max():.1f}周")
        else:
            print("\n警告：所有数据被过滤！")
        
        print(f"\n预处理完成，保留{len(processed_df)}条有效数据")
        
        return processed_df
    
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