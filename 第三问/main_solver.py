"""
NIPT问题3主求解程序
整合数据处理、模型训练、优化和可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_processor import DataProcessor
from model_optimizer import (
    YConcentrationPredictor,
    AdaptiveGrouping,
    RiskOptimizer,
    MonteCarloSimulator
)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class NIPT_Solver:
    """NIPT问题3主求解器"""
    
    def __init__(self, data_path='..\附件.xlsx'):
        """
        初始化求解器
        
        参数:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        
        # 初始化各组件
        self.data_processor = DataProcessor(data_path)
        self.predictor = YConcentrationPredictor()
        self.grouper = AdaptiveGrouping(
            max_depth=4,
            min_samples_leaf=50,
            min_groups=3,
            max_groups=7
        )
        self.optimizer = RiskOptimizer(
            weights={
                'alpha': 0.3,   # 时间风险
                'beta': 0.3,    # 失败风险
                'gamma': 0.2,   # 成本风险
                'delta': 0.2    # 异质性风险
            }
        )
        self.simulator = MonteCarloSimulator(
            n_simulations=1000,
            error_rate=0.05
        )
        
        # 存储结果
        self.results = {}
        
    def solve(self):
        """
        主求解流程
        
        返回:
            results: 包含分组方案、最优时点、风险评估的字典
        """
        print("="*60)
        print("NIPT问题3 - 多因素分层优化求解")
        print("="*60)
        
        start_time = time.time()
        
        # 1. 数据准备
        print("\n[步骤1] 数据加载与预处理...")
        data_dict = self.data_processor.prepare_dataset(test_size=0.2)
        print(f"  - 训练样本数: {len(data_dict['y_train'])}")
        print(f"  - 测试样本数: {len(data_dict['y_test'])}")
        print(f"  - 特征维度: {len(data_dict['feature_names'])}")
        print(f"  - 特征列表: {data_dict['feature_names']}")
        
        # 2. Y浓度预测模型训练
        print("\n[步骤2] 训练Y浓度集成预测模型...")
        
        # 分割验证集用于权重优化
        X_train, X_val, y_train, y_val = train_test_split(
            data_dict['X_train'], data_dict['y_train'],
            test_size=0.2, random_state=42
        )
        
        self.predictor.train(X_train, y_train, X_val, y_val)
        
        # 评估预测性能
        y_pred_test = self.predictor.predict(data_dict['X_test'])
        mse = np.mean((y_pred_test - data_dict['y_test']) ** 2)
        mae = np.mean(np.abs(y_pred_test - data_dict['y_test']))
        
        print(f"  - 集成权重: CART={self.predictor.weights['cart']:.3f}, "
              f"RF={self.predictor.weights['rf']:.3f}, "
              f"LGB={self.predictor.weights['lgb']:.3f}")
        print(f"  - 测试集MSE: {mse:.4f}")
        print(f"  - 测试集MAE: {mae:.4f}")
        
        # 3. 自适应分组
        print("\n[步骤3] 基于CART的自适应分组...")
        
        # 使用原始特征进行分组（未标准化）
        self.grouper.fit(
            data_dict['X_train_original'],
            data_dict['y_train']
        )
        
        groups = self.grouper.get_groups()
        print(f"  - 分组数量: {len(groups)}")
        
        for i, group in enumerate(groups):
            print(f"  - 第{i+1}组: 样本数={group['size']}, "
                  f"平均BMI={group['mean_bmi']:.2f}, "
                  f"平均Y浓度={group['mean_y']:.2f}%")
        
        # 4. 时点优化
        print("\n[步骤4] 优化各组检测时点...")
        
        optimal_timepoints, risks = self.optimizer.optimize_timepoints(
            groups,
            lambda x: self.predictor.predict(self.data_processor.scaler.transform(x))
        )
        
        print("  优化结果:")
        for i, (t, r) in enumerate(zip(optimal_timepoints, risks)):
            success_rate = np.mean(groups[i]['y_values'] >= 4.0)
            print(f"  - 第{i+1}组: 最优时点={t}周, "
                  f"风险值={r:.4f}, "
                  f"当前达标率={success_rate:.2%}")
        
        # 5. 误差分析
        print("\n[步骤5] 蒙特卡洛误差分析...")
        
        robustness = self.simulator.analyze_robustness(
            optimal_timepoints,
            groups,
            lambda x: self.predictor.predict(self.data_processor.scaler.transform(x)),
            self.optimizer
        )
        
        print(f"  - 时点稳定性: {robustness['timepoint_stability']:.3f}")
        print(f"  - 风险变异系数: {robustness['risk_cv']:.3f}")
        print(f"  - 平均风险变化: {robustness['mean_risk_change']:.4f}")
        
        print("\n  各组时点95%置信区间:")
        for group_id, ci in robustness['confidence_intervals'].items():
            print(f"  - {group_id}: [{ci[0]:.1f}, {ci[1]:.1f}]周")
        
        # 6. 保存结果
        self.results = {
            'groups': groups,
            'optimal_timepoints': optimal_timepoints,
            'risks': risks,
            'robustness': robustness,
            'predictor': self.predictor,
            'data_dict': data_dict
        }
        
        elapsed_time = time.time() - start_time
        print(f"\n求解完成! 总用时: {elapsed_time:.2f}秒")
        
        return self.results
    
    def visualize_results(self):
        """可视化结果 - 每张图单独展示"""
        if not self.results:
            print("请先运行solve()方法")
            return
        
        # 设置图表风格
        plt.style.use('seaborn-v0_8-darkgrid')
        
        groups = self.results['groups']
        optimal_t = self.results['optimal_timepoints']
        
        # ========== 图1: 分组样本数量分布 ==========
        plt.figure(figsize=(10, 6))
        group_sizes = [g['size'] for g in groups]
        group_bmis = [g['mean_bmi'] for g in groups]
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(groups)))
        bars = plt.bar(range(len(groups)), group_sizes, color=colors, edgecolor='navy', linewidth=2)
        
        plt.xlabel('分组编号', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        plt.title('NIPT多因素自适应分组 - 样本数量分布', fontsize=14, fontweight='bold')
        plt.xticks(range(len(groups)), 
                [f'组{i+1}\n(BMI:{bmi:.1f})' for i, bmi in enumerate(group_bmis)])
        
        # 添加数值标签
        for bar, size in zip(bars, group_sizes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(size)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # ========== 图2: Y浓度预测效果 ==========
        plt.figure(figsize=(10, 6))
        data_dict = self.results['data_dict']
        y_true = data_dict['y_test'][:100]
        y_pred = self.predictor.predict(data_dict['X_test'][:100])
        
        plt.scatter(y_true, y_pred, alpha=0.6, s=50, c=y_true, cmap='viridis', edgecolors='black', linewidth=0.5)
        plt.plot([0, max(y_true)], [0, max(y_true)], 'r--', alpha=0.8, linewidth=2, label='理想预测线')
        plt.axhline(y=4, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='达标线(4%)')
        plt.axvline(x=4, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.xlabel('真实Y染色体浓度(%)', fontsize=12)
        plt.ylabel('预测Y染色体浓度(%)', fontsize=12)
        plt.title('集成学习模型 - Y染色体浓度预测效果', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 添加性能指标文本
        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        plt.text(0.95, 0.05, f'MSE: {mse:.3f}\nMAE: {mae:.3f}', 
                transform=plt.gca().transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.colorbar(label='Y浓度(%)')
        plt.tight_layout()
        plt.show()
        
        # ========== 图3: 各组最优检测时点 ==========
        plt.figure(figsize=(12, 7))
        
        x = np.arange(len(optimal_t))
        width = 0.6
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(optimal_t)))
        bars = plt.bar(x, optimal_t, width, color=colors, edgecolor='darkred', linewidth=2)
        
        # 添加关键时期参考线
        plt.axhspan(10, 12, alpha=0.2, color='green', label='早期(10-12周)')
        plt.axhspan(12, 20, alpha=0.2, color='yellow', label='中期(12-20周)')
        plt.axhspan(20, 25, alpha=0.2, color='red', label='晚期(20-25周)')
        
        plt.xlabel('分组编号', fontsize=12)
        plt.ylabel('最优检测时点(孕周)', fontsize=12)
        plt.title('各组最优NIPT检测时点方案', fontsize=14, fontweight='bold')
        plt.xticks(x, [f'组{i+1}\nBMI:{groups[i]["mean_bmi"]:.1f}' for i in range(len(optimal_t))])
        plt.ylim(9, 26)
        
        # 添加数值标签和风险值
        for i, (bar, t, risk) in enumerate(zip(bars, optimal_t, self.results['risks'])):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{int(t)}周', ha='center', va='bottom', fontsize=11, fontweight='bold')
            plt.text(bar.get_x() + bar.get_width()/2., height - 1,
                    f'风险:{risk:.3f}', ha='center', va='top', fontsize=9, style='italic')
        
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        
        # ========== 图4: 风险分布热力图 ==========
        plt.figure(figsize=(14, 8))
        
        time_points = range(10, 26)
        risk_matrix = np.zeros((len(groups), len(time_points)))
        
        for i, group in enumerate(groups):
            for j, t in enumerate(time_points):
                risk_matrix[i, j] = self.optimizer.calculate_total_risk(
                    t, group['features'],
                    lambda x: self.predictor.predict(self.data_processor.scaler.transform(x))
                )
        
        im = plt.imshow(risk_matrix, aspect='auto', cmap='RdYlGn_r', interpolation='bilinear')
        
        plt.xlabel('检测时点(孕周)', fontsize=12)
        plt.ylabel('分组', fontsize=12)
        plt.title('多因素风险评估热力图', fontsize=14, fontweight='bold')
        
        plt.xticks(range(len(time_points)), time_points)
        plt.yticks(range(len(groups)), 
                [f'组{i+1} (BMI:{g["mean_bmi"]:.1f})' for i, g in enumerate(groups)])
        
        # 标记最优时点
        for i, t in enumerate(optimal_t):
            plt.plot(t-10, i, 'b*', markersize=20, markeredgecolor='white', markeredgewidth=2)
        
        plt.colorbar(im, label='综合风险值')
        
        # 添加图例
        plt.plot([], [], 'b*', markersize=15, label='最优时点')
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # ========== 图5: 误差稳定性分析 ==========
        plt.figure(figsize=(12, 7))
        
        robustness = self.results['robustness']
        group_ids = list(range(len(groups)))
        lower_bounds = []
        upper_bounds = []
        
        for i in range(len(groups)):
            ci = robustness['confidence_intervals'][f'group_{i}']
            lower_bounds.append(ci[0])
            upper_bounds.append(ci[1])
        
        # 计算误差条
        errors = [[optimal_t[i] - lower_bounds[i], 
                upper_bounds[i] - optimal_t[i]] 
                for i in range(len(group_ids))]
        errors = np.array(errors).T
        
        plt.errorbar(group_ids, optimal_t, yerr=errors, 
                    fmt='o', capsize=8, capthick=3, markersize=10,
                    color='darkblue', ecolor='gray', alpha=0.8, linewidth=2)
        
        # 添加背景色区分风险等级
        plt.axhspan(10, 12, alpha=0.15, color='green')
        plt.axhspan(12, 20, alpha=0.15, color='yellow')
        plt.axhspan(20, 25, alpha=0.15, color='red')
        
        plt.xlabel('分组编号', fontsize=12)
        plt.ylabel('检测时点(孕周)', fontsize=12)
        plt.title('测量误差影响下的时点稳定性分析(95%置信区间)', fontsize=14, fontweight='bold')
        plt.xticks(group_ids, [f'组{i+1}' for i in group_ids])
        plt.grid(True, alpha=0.3)
        
        # 添加稳定性评分
        plt.text(0.02, 0.98, f'稳定性得分: {robustness["timepoint_stability"]:.3f}',
                transform=plt.gca().transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # ========== 图6: 综合性能雷达图 ==========
        plt.figure(figsize=(10, 10))
        
        # 计算各项指标
        metrics = {
            '平均达标率': np.mean([np.mean(g['y_values'] >= 4.0) for g in groups]),
            '时点稳定性': robustness['timepoint_stability'],
            '风险降低率': 1 - np.mean(self.results['risks']),
            '分组均衡度': 1 - np.std([g['size'] for g in groups]) / np.mean([g['size'] for g in groups]),
            '预测准确度': 1 - np.mean(np.abs(self.predictor.predict(data_dict['X_test'][:100]) - data_dict['y_test'][:100])) / 10
        }
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values = list(metrics.values())
        values += values[:1]
        angles += angles[:1]
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='darkblue', markersize=8)
        ax.fill(angles, values, alpha=0.25, color='skyblue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(list(metrics.keys()), size=12)
        ax.set_ylim(0, 1)
        ax.set_title('NIPT优化方案综合性能评估', size=16, fontweight='bold', pad=20)
        ax.grid(True, linewidth=0.5)
        
        # 添加数值标签
        for angle, value, name in zip(angles[:-1], values[:-1], metrics.keys()):
            ax.text(angle, value + 0.05, f'{value:.3f}', ha='center', va='center', size=10)
        
        plt.tight_layout()
        plt.show()
        
        print("\n所有图表已生成完成！")
    
    def evaluate_performance(self):
        """评估模型性能并与基准对比"""
        if not self.results:
            print("请先运行solve()方法")
            return
        
        print("\n" + "="*60)
        print("性能评估报告")
        print("="*60)
        
        groups = self.results['groups']
        optimal_t = self.results['optimal_timepoints']
        risks = self.results['risks']
        
        # 1. 达标率分析
        print("\n[1] 达标率分析:")
        total_samples = sum(g['size'] for g in groups)
        success_samples = sum(np.sum(g['y_values'] >= 4.0) for g in groups)
        overall_success_rate = success_samples / total_samples
        
        print(f"  - 总体达标率: {overall_success_rate:.2%}")
        for i, group in enumerate(groups):
            group_success = np.mean(group['y_values'] >= 4.0)
            print(f"  - 第{i+1}组达标率: {group_success:.2%} "
                  f"(时点={optimal_t[i]}周)")
        
        # 2. 风险分析
        print("\n[2] 风险分析:")
        avg_risk = np.mean(risks)
        print(f"  - 平均风险值: {avg_risk:.4f}")
        print(f"  - 最小风险值: {min(risks):.4f} (组{risks.index(min(risks))+1})")
        print(f"  - 最大风险值: {max(risks):.4f} (组{risks.index(max(risks))+1})")
        
        # 3. 与基准方案对比（简单BMI分组）
        print("\n[3] 与基准方案对比:")
        
        # 基准方案：固定BMI分组[20,28), [28,32), [32,36), [36,40), 40+
        baseline_timepoints = [12, 13, 14, 15, 16]  # 经验时点
        
        # 计算基准风险
        baseline_risks = []
        for i, group in enumerate(groups[:min(5, len(groups))]):
            baseline_t = baseline_timepoints[min(i, 4)]
            baseline_risk = self.optimizer.calculate_total_risk(
                baseline_t, group['features'],
                lambda x: self.predictor.predict(self.data_processor.scaler.transform(x))
            )
            baseline_risks.append(baseline_risk)
        
        if baseline_risks:
            baseline_avg_risk = np.mean(baseline_risks)
            improvement = (baseline_avg_risk - avg_risk) / baseline_avg_risk * 100
            
            print(f"  - 基准方案平均风险: {baseline_avg_risk:.4f}")
            print(f"  - 优化方案平均风险: {avg_risk:.4f}")
            print(f"  - 风险降低率: {improvement:.1f}%")
        
        # 4. 计算效率
        print("\n[4] 计算效率:")
        print(f"  - 特征维度: {self.results['data_dict']['X_train'].shape[1]}")
        print(f"  - 训练样本数: {len(self.results['data_dict']['y_train'])}")
        print(f"  - 分组数量: {len(groups)}")
        print(f"  - 模型复杂度: O(N log N)")
        
        # 5. 鲁棒性评估
        print("\n[5] 鲁棒性评估:")
        robustness = self.results['robustness']
        print(f"  - 时点稳定性得分: {robustness['timepoint_stability']:.3f}/1.000")
        print(f"  - 风险变异系数: {robustness['risk_cv']:.3f}")
        
        # 判定鲁棒性等级
        if robustness['timepoint_stability'] > 0.9:
            robustness_level = "优秀"
        elif robustness['timepoint_stability'] > 0.7:
            robustness_level = "良好"
        else:
            robustness_level = "一般"
        
        print(f"  - 鲁棒性等级: {robustness_level}")
        
        print("\n" + "="*60)


# 使用示例和测试
if __name__ == "__main__":
    # 创建求解器实例
    solver = NIPT_Solver(data_path='附件.xlsx')
    
    # 求解问题
    results = solver.solve()
    
    # 可视化结果
    solver.visualize_results()
    
    # 性能评估
    solver.evaluate_performance()
    
    # 输出最终建议
    print("\n" + "="*60)
    print("最终建议方案")
    print("="*60)
    
    for i, (group, timepoint, risk) in enumerate(zip(
        results['groups'], 
        results['optimal_timepoints'],
        results['risks']
    )):
        print(f"\n【第{i+1}组】")
        print(f"  - BMI范围: 约{group['mean_bmi']-2:.1f} ~ {group['mean_bmi']+2:.1f}")
        print(f"  - 样本数量: {group['size']}")
        print(f"  - 建议检测时点: 孕{timepoint}周")
        print(f"  - 预期达标率: {np.mean(group['y_values'] >= 4.0):.1%}")
        print(f"  - 综合风险值: {risk:.4f}")