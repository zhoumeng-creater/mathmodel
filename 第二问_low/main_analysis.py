"""
NIPT分析主程序
数据加载、执行分析、可视化和性能测试
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nipt_solver import NIPTSolver
import time
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class NIPTAnalyzer:
    """NIPT分析和可视化工具"""
    
    def __init__(self, data_path: str):
        """
        初始化分析器
        
        参数:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.solver = None
        self.solution = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载Excel数据
        
        返回:
            原始数据DataFrame
        """
        print(f"加载数据: {self.data_path}")
        try:
            self.raw_data = pd.read_excel(self.data_path)
            print(f"成功加载 {len(self.raw_data)} 条记录")
            
            # 显示数据基本信息
            male_count = self.raw_data['Y染色体浓度'].notna().sum()
            female_count = self.raw_data['Y染色体浓度'].isna().sum()
            print(f"男胎数据: {male_count} 条")
            print(f"女胎数据: {female_count} 条")
            
            return self.raw_data
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def run_analysis(self, risk_weights: Dict[str, float] = None) -> Dict:
        """
        执行完整分析流程
        
        参数:
            risk_weights: 风险权重配置
            
        返回:
            分析结果
        """
        print("\n" + "="*50)
        print("开始NIPT分析")
        print("="*50)
        
        # 初始化求解器
        self.solver = NIPTSolver(risk_weights)
        
        # 数据预处理
        print("\n数据预处理...")
        self.processed_data = self.solver.preprocess_data(self.raw_data)
        print(f"预处理后数据: {len(self.processed_data)} 条")
        
        # 两阶段优化
        print("\n执行两阶段优化...")
        start_time = time.time()
        self.solution = self.solver.two_stage_optimization(self.processed_data)
        optimization_time = time.time() - start_time
        print(f"\n优化完成，耗时: {optimization_time:.2f} 秒")
        
        # 敏感性分析
        sensitivity = self.solver.sensitivity_analysis(self.processed_data)
        print(f"\n敏感性分析结论: {sensitivity['conclusion']}")
        
        # 输出最终方案
        self._print_solution()
        
        return {
            'solution': self.solution,
            'sensitivity': sensitivity,
            'optimization_time': optimization_time
        }
    
    def _print_solution(self):
        """打印最优解方案"""
        print("\n" + "="*50)
        print("最优NIPT检测方案")
        print("="*50)
        
        print(f"\n分组数: {self.solution['K']}")
        print(f"总风险: {self.solution['total_risk']:.4f}")
        
        print("\n各组详细方案:")
        print("-"*50)
        for info in self.solution['group_info']:
            print(f"\n组 {info['group_id']}:")
            print(f"  BMI范围: {info['bmi_range']}")
            print(f"  最佳检测时点: {info['detection_week']} 周")
            print(f"  样本数: {info['sample_size']}")
            print(f"  平均成功概率: {info['mean_success_prob']:.2%}")
            print(f"  成功概率范围: [{info['min_success_prob']:.2%}, {info['max_success_prob']:.2%}]")
    
    def visualize_results(self):
        """结果可视化 - 每张图单独展示"""
        if self.solution is None:
            print("请先运行分析")
            return
        
        # 确保中文显示正常
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        boundaries = self.solution['boundaries']
        
        # 图1: BMI分组分布图
        plt.figure(figsize=(10, 6))
        plt.hist(self.processed_data['BMI'], bins=30, alpha=0.7, 
                color='skyblue', edgecolor='black', label='BMI分布')
        
        # 添加分组边界线
        for i, b in enumerate(boundaries[1:-1]):
            plt.axvline(x=b, color='red', linestyle='--', linewidth=2, 
                    label='分组边界' if i == 0 else '')
        
        plt.xlabel('BMI值', fontsize=14)
        plt.ylabel('样本数量', fontsize=14)
        plt.title('BMI分组分布图', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 图2: Y浓度-孕周关系图
        plt.figure(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(boundaries)-1))
        
        for i in range(len(boundaries)-1):
            bmi_range = (boundaries[i], boundaries[i+1])
            group_data = self.processed_data[
                (self.processed_data['BMI'] >= bmi_range[0]) & 
                (self.processed_data['BMI'] < bmi_range[1])
            ]
            
            if len(group_data) > 0:
                plt.scatter(group_data['孕周'], group_data['Y染色体浓度'], 
                        alpha=0.6, color=colors[i], s=30,
                        label=f'BMI [{bmi_range[0]:.0f}, {bmi_range[1]:.0f})')
        
        # 添加4%阈值线
        plt.axhline(y=4, color='red', linestyle='-', linewidth=2.5, 
                    label='4%达标阈值', alpha=0.8)
        
        plt.xlabel('孕周', fontsize=14)
        plt.ylabel('Y染色体浓度 (%)', fontsize=14)
        plt.title('Y染色体浓度随孕周变化关系图', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.xlim(9.5, 25.5)
        plt.ylim(0, max(self.processed_data['Y染色体浓度'].max() * 1.1, 10))
        plt.tight_layout()
        plt.show()
        
        # 图3: 风险热力图
        plt.figure(figsize=(12, 6))
        weeks = list(range(10, 26))
        bmi_groups = [f'组{i+1}\nBMI[{boundaries[i]:.0f}-{boundaries[i+1]:.0f})' 
                    for i in range(len(boundaries)-1)]
        
        # 创建风险矩阵
        risk_matrix = np.zeros((len(bmi_groups), len(weeks)))
        for i in range(len(bmi_groups)):
            for j, week in enumerate(weeks):
                # 计算延迟风险
                delay_risk = self.evaluator.calculate_delay_risk(week, 12)
                risk_matrix[i, j] = delay_risk
        
        # 绘制热力图
        im = plt.imshow(risk_matrix, cmap='RdYlGn_r', aspect='auto', 
                        interpolation='nearest')
        
        # 设置坐标轴
        plt.xticks(range(len(weeks)), weeks)
        plt.yticks(range(len(bmi_groups)), bmi_groups)
        plt.xlabel('孕周', fontsize=14)
        plt.ylabel('BMI分组', fontsize=14)
        plt.title('检测风险热力图（★标记为最优检测时点）', 
                fontsize=16, fontweight='bold', pad=20)
        
        # 添加最优时点标记
        for i, time in enumerate(self.solution['detection_times']):
            plt.text(time-10, i, '★', ha='center', va='center', 
                    color='yellow', fontsize=24, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, label='风险值', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        plt.show()
        
        # 图4: 最优检测时点对比图
        plt.figure(figsize=(10, 6))
        
        groups = []
        for i in range(len(self.solution['detection_times'])):
            bmi_range = f"[{boundaries[i]:.0f}, {boundaries[i+1]:.0f})"
            sample_size = self.solution['group_info'][i]['sample_size']
            groups.append(f'组{i+1}\nBMI{bmi_range}\n(n={sample_size})')
        
        times = self.solution['detection_times']
        
        # 创建条形图
        bars = plt.bar(range(len(groups)), times, color='steelblue', 
                    edgecolor='navy', linewidth=2, alpha=0.8)
        
        # 添加数值标签
        for i, (bar, time) in enumerate(zip(bars, times)):
            height = bar.get_height()
            success_prob = self.solution['group_info'][i]['mean_success_prob']
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{int(time)}周\n({success_prob:.1%})', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xticks(range(len(groups)), groups, fontsize=10)
        plt.ylabel('最佳检测孕周', fontsize=14)
        plt.title('各BMI组最佳NIPT检测时点方案', fontsize=16, fontweight='bold', pad=20)
        plt.ylim(8, max(times) + 3)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加说明文字
        plt.text(0.02, 0.98, f'总体风险值: {self.solution["total_risk"]:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
    
    def performance_test(self, n_iterations: int = 5):
        """
        性能测试
        
        参数:
            n_iterations: 测试迭代次数
        """
        print("\n" + "="*50)
        print("性能测试")
        print("="*50)
        
        times = []
        risks = []
        
        for i in range(n_iterations):
            print(f"\n迭代 {i+1}/{n_iterations}")
            
            # 重新初始化求解器
            solver = NIPTSolver()
            processed_data = solver.preprocess_data(self.raw_data)
            
            # 计时
            start = time.time()
            solution = solver.two_stage_optimization(processed_data)
            elapsed = time.time() - start
            
            times.append(elapsed)
            risks.append(solution['total_risk'])
            
            print(f"  耗时: {elapsed:.2f}秒")
            print(f"  总风险: {solution['total_risk']:.4f}")
        
        print("\n性能统计:")
        print(f"  平均耗时: {np.mean(times):.2f} ± {np.std(times):.2f} 秒")
        print(f"  平均风险: {np.mean(risks):.4f} ± {np.std(risks):.4f}")
        print(f"  最快: {np.min(times):.2f} 秒")
        print(f"  最慢: {np.max(times):.2f} 秒")
        
        return {
            'times': times,
            'risks': risks,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'mean_risk': np.mean(risks),
            'std_risk': np.std(risks)
        }
    
    def save_results(self, output_path: str = 'nipt_results.xlsx'):
        """
        保存结果到Excel文件
        
        参数:
            output_path: 输出文件路径
        """
        if self.solution is None:
            print("无结果可保存，请先运行分析")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 保存方案总结
            summary_df = pd.DataFrame([{
                '分组数': self.solution['K'],
                '总风险': self.solution['total_risk'],
                'BMI边界': str(self.solution['boundaries']),
                '检测时点': str(self.solution['detection_times'])
            }])
            summary_df.to_excel(writer, sheet_name='方案总结', index=False)
            
            # 保存各组详情
            group_df = pd.DataFrame(self.solution['group_info'])
            group_df.to_excel(writer, sheet_name='分组详情', index=False)
            
            # 保存敏感性分析
            if self.solver.sensitivity_results:
                sensitivity_df = pd.DataFrame(self.solver.sensitivity_results['results'])
                sensitivity_df.to_excel(writer, sheet_name='敏感性分析', index=False)
        
        print(f"\n结果已保存至: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "附件.xlsx"  # 数据文件路径
    
    # 风险权重配置（可调参数）
    RISK_WEIGHTS = {
        'delay': 0.5,  # 延迟风险权重 α
        'fail': 0.3,   # 失败风险权重 β
        'cost': 0.2    # 成本风险权重 γ
    }
    
    try:
        # 创建分析器
        analyzer = NIPTAnalyzer(DATA_PATH)
        
        # 加载数据
        analyzer.load_data()
        
        # 运行分析
        results = analyzer.run_analysis(risk_weights=RISK_WEIGHTS)
        
        # 可视化结果
        analyzer.visualize_results()
        
        # 性能测试（可选）
        print("\n是否进行性能测试？(y/n): ", end="")
        if input().lower() == 'y':
            perf_results = analyzer.performance_test(n_iterations=3)
        
        # 保存结果
        analyzer.save_results('nipt_optimal_solution.xlsx')
        
        # 测试个体建议功能
        print("\n" + "="*50)
        print("个体检测建议示例")
        print("="*50)
        
        test_bmis = [22, 30, 35, 42]
        for bmi in test_bmis:
            recommendation = analyzer.solver.get_recommendation(bmi)
            print(f"\nBMI={bmi}的孕妇:")
            print(f"  建议组别: 第{recommendation.get('group', 'N/A')}组")
            print(f"  建议检测时间: {recommendation.get('recommended_week', 'N/A')}周")
            if 'note' in recommendation:
                print(f"  备注: {recommendation['note']}")
        
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()