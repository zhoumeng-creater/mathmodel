"""
NIPT两阶段优化算法实现
外层：遗传算法优化BMI分组
内层：网格搜索优化检测时点
"""

import numpy as np
import pandas as pd
from models import YConcentrationPredictor, RiskCalculator, BMIGrouping
import random
from typing import List, Tuple, Dict


class GeneticAlgorithm:
    """遗传算法优化BMI分组边界"""
    
    def __init__(self, 
                 k: int,
                 bmi_range: Tuple[float, float],
                 pop_size: int = 30,
                 elite_size: int = 5,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 random_state: int = 42):
        """
        初始化遗传算法
        k: 分组数量
        bmi_range: BMI范围 (min, max)
        """
        self.k = k
        self.bmi_min, self.bmi_max = bmi_range
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        np.random.seed(random_state)
        random.seed(random_state)
        
    def initialize_population(self) -> List[np.ndarray]:
        """
        初始化种群
        染色体编码：[b1, b2, ..., b_{k-1}]，不包含b0和bk
        """
        population = []
        
        for _ in range(self.pop_size):
            # 在BMI范围内随机生成k-1个分割点
            boundaries = np.random.uniform(
                self.bmi_min + 1, 
                self.bmi_max - 1, 
                self.k - 1
            )
            boundaries.sort()
            population.append(boundaries)
            
        return population
    
    def decode_chromosome(self, chromosome: np.ndarray) -> np.ndarray:
        """
        解码染色体为完整边界
        添加b0和bk
        """
        return np.concatenate([[self.bmi_min], chromosome, [self.bmi_max]])
    
    def fitness(self, chromosome: np.ndarray, evaluator) -> float:
        """
        适应度函数（负风险值，因为要最大化适应度）
        evaluator: 用于评估的函数
        """
        boundaries = self.decode_chromosome(chromosome)
        total_risk = evaluator(boundaries)
        
        # 返回负风险作为适应度（风险越小，适应度越大）
        return -total_risk
    
    def selection(self, population: List, fitness_scores: List) -> List:
        """
        轮盘赌选择
        """
        # 检查并处理无效的适应度值
        valid_scores = []
        for i, score in enumerate(fitness_scores):
            if np.isnan(score) or np.isinf(score):
                valid_scores.append(-1e10)  # 给无效值一个很差的适应度
            else:
                valid_scores.append(score)
        
        # 将适应度转换为正值（偏移）
        min_fitness = min(valid_scores)
        adjusted_fitness = [f - min_fitness + 1e-6 for f in valid_scores]
        
        # 计算选择概率
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            # 如果总适应度为0，使用均匀概率
            probs = [1.0/len(population) for _ in population]
        else:
            probs = [f / total_fitness for f in adjusted_fitness]
        
        # 选择
        selected = []
        for _ in range(self.pop_size):
            selected.append(population[np.random.choice(len(population), p=probs)])
            
        return selected
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        单点交叉
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if len(parent1) == 1:
            # 只有一个基因，交换
            return parent2.copy(), parent1.copy()
        
        # 选择交叉点
        point = random.randint(1, len(parent1) - 1)
        
        # 交叉
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        # 确保边界有序
        child1.sort()
        child2.sort()
        
        return child1, child2
    
    def mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """
        高斯变异
        """
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # 高斯扰动
                delta = np.random.normal(0, (self.bmi_max - self.bmi_min) * 0.05)
                mutated[i] += delta
                
                # 确保在范围内
                mutated[i] = np.clip(mutated[i], self.bmi_min + 1, self.bmi_max - 1)
        
        # 确保有序
        mutated.sort()
        
        return mutated
    
    def run(self, evaluator, max_generations: int = 50) -> Tuple[np.ndarray, float]:
        """
        运行遗传算法
        返回最佳边界和对应的风险值
        """
        # 初始化种群
        population = self.initialize_population()
        
        best_chromosome = None
        best_fitness = float('-inf')
        
        for generation in range(max_generations):
            # 评估适应度
            fitness_scores = [self.fitness(chrom, evaluator) for chrom in population]
            
            # 记录最佳个体
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_chromosome = population[max_idx].copy()
            
            # 精英保留
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite = [population[i] for i in elite_indices]
            
            # 选择
            selected = self.selection(population, fitness_scores)
            
            # 交叉和变异
            new_population = elite.copy()  # 保留精英
            
            while len(new_population) < self.pop_size:
                # 随机选择两个父代
                p1, p2 = random.sample(selected, 2)
                
                # 交叉
                c1, c2 = self.crossover(p1, p2)
                
                # 变异
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)
                
                new_population.extend([c1, c2])
            
            # 限制种群大小
            population = new_population[:self.pop_size]
        
        # 返回最佳边界和对应的风险（注意适应度是负风险）
        return self.decode_chromosome(best_chromosome), -best_fitness


class TimePointOptimizer:
    """检测时点优化器"""
    
    def __init__(self, 
                 predictor: YConcentrationPredictor,
                 risk_calculator: RiskCalculator):
        """
        初始化
        predictor: Y浓度预测器
        risk_calculator: 风险计算器
        """
        self.predictor = predictor
        self.risk_calculator = risk_calculator
        
    def check_constraints(self, t: int, group_data: pd.DataFrame, 
                         threshold_prob: float = 0.8) -> bool:
        """
        检查约束条件
        1. 时间窗口约束: 10 <= t <= 25
        2. 达标概率约束: P(Y >= 4%) >= threshold_prob
        """
        # 时间窗口约束
        if t < 10 or t > 25:
            return False
        
        # 达标概率约束
        success_prob = self.predictor.get_success_probability(t, group_data)
        if success_prob < threshold_prob:
            return False
        
        return True
    
    def optimize_single_group(self, 
                             group_data: pd.DataFrame,
                             bmi_mean: float,
                             time_range: Tuple[int, int] = (10, 25)) -> Tuple[int, float]:
        """
        优化单个组的检测时点（网格搜索）
        返回：(最佳时点, 最小风险)
        """
        best_t = None
        min_risk = float('inf')
        
        # 网格搜索所有可能的时点
        for t in range(time_range[0], time_range[1] + 1):
            # 检查约束
            if not self.check_constraints(t, group_data):
                continue
            
            # 计算该时点的总风险
            p_not_reach = 1 - self.predictor.get_success_probability(t, group_data)
            total_risk = 0
            
            # 对组内每个样本计算风险
            for _, sample in group_data.iterrows():
                risk = self.risk_calculator.total_risk(
                    t, sample['BMI'], p_not_reach
                )
                total_risk += risk
            
            # 平均风险
            avg_risk = total_risk / len(group_data)
            
            # 更新最佳时点
            if avg_risk < min_risk:
                min_risk = avg_risk
                best_t = t
        
        # 如果没找到满足约束的时点，返回默认值
        if best_t is None:
            best_t = 15  # 默认中间时点
            min_risk = float('inf')
            
        return best_t, min_risk
    
    def check_monotonicity_constraint(self, time_points: List[int]) -> bool:
        """
        检查单调性约束
        BMI越高，检测时点应该越晚: t_k <= t_{k+1}
        """
        for i in range(len(time_points) - 1):
            if time_points[i] > time_points[i+1]:
                return False
        return True
    
    def enforce_monotonicity(self, time_points: List[int]) -> List[int]:
        """
        强制满足单调性约束
        使用简单的调整策略
        """
        adjusted = time_points.copy()
        
        for i in range(1, len(adjusted)):
            if adjusted[i] < adjusted[i-1]:
                adjusted[i] = adjusted[i-1]
        
        return adjusted


class NIPTOptimizer:
    """主优化器 - 协调两阶段优化"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 predictor: YConcentrationPredictor,
                 risk_weights: Dict[str, float] = None):
        """
        初始化主优化器
        data: 男胎数据
        predictor: Y浓度预测器
        risk_weights: 风险权重字典
        """
        self.data = data
        self.predictor = predictor
        
        # 设置风险权重（默认值）
        if risk_weights is None:
            risk_weights = {'alpha': 0.4, 'beta': 0.4, 'gamma': 0.2}
        
        self.risk_calculator = RiskCalculator(**risk_weights)
        self.time_optimizer = TimePointOptimizer(predictor, self.risk_calculator)
        
        # BMI范围
        self.bmi_min = data['BMI'].min()
        self.bmi_max = data['BMI'].max()
        
    def evaluate_solution(self, boundaries: np.ndarray) -> float:
        """
        评估一个分组方案的总风险
        这是遗传算法的评估函数
        """
        k = len(boundaries) - 1
        grouping = BMIGrouping(k, boundaries)
        
        # 检查分组平衡约束
        if not grouping.check_balance_constraint(self.data, n_min=10):
            return float('inf')  # 违反约束，返回极大风险
        
        # 分配样本到各组
        grouped_data = grouping.assign_groups(self.data)
        
        # 优化各组时点并计算总风险
        total_risk = 0
        time_points = []
        
        for g in range(k):
            group_data = grouped_data[grouped_data['group'] == g]
            if len(group_data) == 0:
                return float('inf')
            
            # 优化该组的检测时点
            bmi_mean = group_data['BMI'].mean()
            best_t, group_risk = self.time_optimizer.optimize_single_group(
                group_data, bmi_mean
            )
            
            time_points.append(best_t)
            total_risk += group_risk * len(group_data)
        
        # 检查并调整单调性约束
        if not self.time_optimizer.check_monotonicity_constraint(time_points):
            # 强制满足单调性（作为软约束，添加惩罚）
            total_risk *= 1.2  # 20%惩罚
        
        # 返回平均风险
        return total_risk / len(self.data)
    
    def optimize(self, 
                k_range: List[int] = [3, 4, 5],
                max_iterations: int = 50) -> Dict:
        """
        执行两阶段优化
        k_range: 尝试的分组数范围
        返回最优解
        """
        best_solution = None
        min_risk = float('inf')
        
        # 对每个可能的分组数进行优化
        for k in k_range:
            print(f"\n优化K={k}分组...")
            
            # 使用遗传算法优化分组边界
            ga = GeneticAlgorithm(
                k=k,
                bmi_range=(self.bmi_min, self.bmi_max),
                pop_size=30,
                elite_size=5
            )
            
            # 运行遗传算法
            boundaries, risk = ga.run(
                evaluator=self.evaluate_solution,
                max_generations=max_iterations
            )
            
            print(f"K={k}的最小风险: {risk:.4f}")
            
            # 更新最优解
            if risk < min_risk:
                min_risk = risk
                
                # 获取详细的解
                grouping = BMIGrouping(k, boundaries)
                grouped_data = grouping.assign_groups(self.data)
                
                # 获取各组的最优时点
                time_points = []
                group_info = []
                
                for g in range(k):
                    group_data = grouped_data[grouped_data['group'] == g]
                    if len(group_data) > 0:
                        bmi_mean = group_data['BMI'].mean()
                        best_t, _ = self.time_optimizer.optimize_single_group(
                            group_data, bmi_mean
                        )
                        time_points.append(best_t)
                        
                        group_info.append({
                            'group': g + 1,
                            'bmi_range': f"[{boundaries[g]:.2f}, {boundaries[g+1]:.2f})",
                            'optimal_week': best_t,
                            'sample_size': len(group_data),
                            'mean_bmi': bmi_mean
                        })
                
                # 确保满足单调性约束
                time_points = self.time_optimizer.enforce_monotonicity(time_points)
                
                # 更新group_info中的时点
                for i, info in enumerate(group_info):
                    info['optimal_week'] = time_points[i]
                
                best_solution = {
                    'k': k,
                    'boundaries': boundaries,
                    'time_points': time_points,
                    'total_risk': risk,
                    'group_info': pd.DataFrame(group_info),
                    'grouping': grouping
                }
        
        return best_solution
    
    def sensitivity_analysis(self, solution: Dict, n_simulations: int = 100) -> Dict:
        """
        敏感性分析 - 测量误差影响
        使用蒙特卡洛模拟
        """
        print("\n进行敏感性分析...")
        
        risks = []
        success_rates = []
        
        for sim in range(n_simulations):
            # 对Y浓度添加5%的测量误差
            data_with_error = self.data.copy()
            error = np.random.normal(0, 0.05 * data_with_error['Y_concentration'])
            data_with_error['Y_concentration'] = data_with_error['Y_concentration'] + error
            
            # 重新评估风险
            risk = self.evaluate_solution(solution['boundaries'])
            risks.append(risk)
            
            # 计算达标率
            grouping = solution['grouping']
            grouped_data = grouping.assign_groups(data_with_error)
            
            total_success = 0
            for g, t in enumerate(solution['time_points']):
                group_data = grouped_data[grouped_data['group'] == g]
                if len(group_data) > 0:
                    success_prob = self.predictor.get_success_probability(t, group_data)
                    total_success += success_prob * len(group_data)
            
            success_rates.append(total_success / len(data_with_error))
        
        return {
            'risk_mean': np.mean(risks),
            'risk_std': np.std(risks),
            'risk_ci': np.percentile(risks, [2.5, 97.5]),
            'success_rate_mean': np.mean(success_rates),
            'success_rate_std': np.std(success_rates),
            'robust_score': 1 / (1 + np.std(risks))  # 鲁棒性评分
        }