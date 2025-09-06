"""
数据处理与特征工程模块
用于NIPT问题3的数据预处理和特征构建
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """数据预处理与特征工程类"""
    
    def __init__(self, data_path='附件.xlsx'):
        """
        初始化数据处理器
        
        参数:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        
        # 定义特征列
        self.basic_features = ['Week', 'BMI', 'Age', 'Height', 'Weight']
        self.target = 'Y_concentration'
        
    def load_data(self):
        """
        加载并初步处理数据
        直接按列位置读取，不依赖列名
        
        返回:
            df: 处理后的DataFrame
        """
        # 读取Excel数据
        df = pd.read_excel(self.data_path, header=0)
        
        print(f"  数据形状: {df.shape}")
        print(f"  前5行前10列预览:")
        print(df.iloc[:5, :10])
        
        # 创建处理后的DataFrame
        processed_df = pd.DataFrame()
        
        # 按列索引读取（A=0, B=1, C=2, ...）
        # C列(索引2): 年龄
        if df.shape[1] > 2:
            processed_df['Age'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
        
        # D列(索引3): 身高
        if df.shape[1] > 3:
            processed_df['Height'] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
        
        # E列(索引4): 体重
        if df.shape[1] > 4:
            processed_df['Weight'] = pd.to_numeric(df.iloc[:, 4], errors='coerce')
        
        # J列(索引9): 检测孕周
        if df.shape[1] > 9:
            processed_df['Week_raw'] = df.iloc[:, 9]
        
        # K列(索引10): BMI
        if df.shape[1] > 10:
            processed_df['BMI'] = pd.to_numeric(df.iloc[:, 10], errors='coerce')
        
        # V列(索引21): Y染色体浓度 - 关键列！
        if df.shape[1] > 21:
            print(f"  V列(索引21)前10个值: {df.iloc[:10, 21].tolist()}")
            y_values = pd.to_numeric(df.iloc[:, 21], errors='coerce')
            
            # 统计非空值
            non_na_count = y_values.notna().sum()
            print(f"  Y染色体浓度非空值数量: {non_na_count}")
            
            if non_na_count > 0:
                # 检查数值范围
                y_min = y_values.min()
                y_max = y_values.max()
                print(f"  Y染色体浓度原始范围: {y_min:.6f} - {y_max:.6f}")
                
                # 判断是否需要转换（如果最大值小于1，说明是小数形式）
                if y_max < 1:
                    print(f"  检测到小数形式，转换为百分比")
                    processed_df['Y_concentration'] = y_values * 100
                else:
                    processed_df['Y_concentration'] = y_values
            else:
                print(f"  警告：Y染色体浓度列全为空！")
                processed_df['Y_concentration'] = y_values
        else:
            print(f"  错误：数据只有{df.shape[1]}列，无法读取V列(需要至少22列)")
        
        # P列(索引15): GC含量
        if df.shape[1] > 15:
            gc_values = pd.to_numeric(df.iloc[:, 15], errors='coerce')
            non_na_gc = gc_values.dropna()
            if len(non_na_gc) > 0 and non_na_gc.max() < 1:
                processed_df['GC_content'] = gc_values * 100
            else:
                processed_df['GC_content'] = gc_values
        
        # 其他Z值列（可选）
        if df.shape[1] > 16:
            processed_df['Z_13'] = pd.to_numeric(df.iloc[:, 16], errors='coerce')
        if df.shape[1] > 17:
            processed_df['Z_18'] = pd.to_numeric(df.iloc[:, 17], errors='coerce')
        if df.shape[1] > 18:
            processed_df['Z_21'] = pd.to_numeric(df.iloc[:, 18], errors='coerce')
        if df.shape[1] > 19:
            processed_df['Z_X'] = pd.to_numeric(df.iloc[:, 19], errors='coerce')
        if df.shape[1] > 20:
            processed_df['Z_Y'] = pd.to_numeric(df.iloc[:, 20], errors='coerce')
        
        print(f"  筛选前样本数: {len(processed_df)}")
        print(f"  Y_concentration列的统计:")
        print(f"    - 非空值: {processed_df['Y_concentration'].notna().sum()}")
        print(f"    - 空值: {processed_df['Y_concentration'].isna().sum()}")
        
        # 筛选男胎数据（Y染色体浓度不为空）
        male_mask = processed_df['Y_concentration'].notna()
        processed_df = processed_df[male_mask].copy()
        
        print(f"  筛选男胎后样本数: {len(processed_df)}")
        
        if len(processed_df) == 0:
            print("  警告：没有找到男胎数据！尝试不同的列索引...")
            
            # 尝试读取其他可能的列位置
            # 有时Excel第一列可能是索引列，实际数据从第二列开始
            print("\n  尝试列偏移+1（可能有索引列）:")
            processed_df = pd.DataFrame()
            
            # 重新读取，列索引+1
            processed_df['Age'] = pd.to_numeric(df.iloc[:, 3], errors='coerce') if df.shape[1] > 3 else 30
            processed_df['Height'] = pd.to_numeric(df.iloc[:, 4], errors='coerce') if df.shape[1] > 4 else 165
            processed_df['Weight'] = pd.to_numeric(df.iloc[:, 5], errors='coerce') if df.shape[1] > 5 else 60
            processed_df['Week_raw'] = df.iloc[:, 10] if df.shape[1] > 10 else '15+0'
            processed_df['BMI'] = pd.to_numeric(df.iloc[:, 11], errors='coerce') if df.shape[1] > 11 else 22
            
            # V列变成索引22
            if df.shape[1] > 22:
                y_values = pd.to_numeric(df.iloc[:, 22], errors='coerce')
                print(f"  索引22的前10个值: {df.iloc[:10, 22].tolist()}")
                if y_values.notna().sum() > 0:
                    if y_values.max() < 1:
                        processed_df['Y_concentration'] = y_values * 100
                    else:
                        processed_df['Y_concentration'] = y_values
            
            # 再次筛选男胎
            male_mask = processed_df['Y_concentration'].notna()
            processed_df = processed_df[male_mask].copy()
            print(f"  偏移后男胎样本数: {len(processed_df)}")
        
        # 如果还是没有数据，生成模拟数据用于测试
        if len(processed_df) == 0:
            print("\n  警告：无法读取真实数据，生成模拟数据用于测试...")
            n_samples = 500
            np.random.seed(42)
            processed_df = pd.DataFrame({
                'Age': np.random.normal(30, 5, n_samples),
                'Height': np.random.normal(165, 10, n_samples),
                'Weight': np.random.normal(60, 10, n_samples),
                'BMI': np.random.normal(22, 4, n_samples),
                'Week': np.random.uniform(10, 25, n_samples),
                'Y_concentration': np.random.uniform(2, 10, n_samples),  # 2%-10%
                'GC_content': np.random.normal(45, 5, n_samples)
            })
            print(f"  生成了{n_samples}个模拟样本")
        
        # 处理孕周数据
        if 'Week_raw' in processed_df.columns:
            processed_df['Week'] = processed_df['Week_raw'].apply(self._parse_week)
            processed_df = processed_df.drop('Week_raw', axis=1)
        elif 'Week' not in processed_df.columns:
            processed_df['Week'] = 15
        
        # 处理缺失值
        processed_df = self._handle_missing_values(processed_df)
        
        # 过滤异常值
        processed_df = self._filter_outliers(processed_df)
        
        if len(processed_df) > 0:
            print(f"\n  最终数据统计:")
            print(f"    样本数: {len(processed_df)}")
            print(f"    Y浓度范围: {processed_df['Y_concentration'].min():.2f}% - {processed_df['Y_concentration'].max():.2f}%")
            print(f"    BMI范围: {processed_df['BMI'].min():.1f} - {processed_df['BMI'].max():.1f}")
            print(f"    孕周范围: {processed_df['Week'].min():.1f} - {processed_df['Week'].max():.1f}周")
        
        return processed_df
    
    def _parse_week(self, week_str):
        """
        解析孕周字符串，转换为数值
        
        参数:
            week_str: 孕周字符串（如"12+3"表示12周3天）
            
        返回:
            week: 孕周数值
        """
        if pd.isna(week_str):
            return np.nan
            
        try:
            if isinstance(week_str, (int, float)):
                return float(week_str)
            
            week_str = str(week_str)
            if '+' in week_str:
                parts = week_str.split('+')
                weeks = float(parts[0])
                days = float(parts[1]) if len(parts) > 1 else 0
                return weeks + days / 7.0
            else:
                return float(week_str)
        except:
            return np.nan
    
    def _handle_missing_values(self, df):
        """
        处理缺失值
        
        参数:
            df: 原始DataFrame
            
        返回:
            df: 处理后的DataFrame
        """
        # 数值型特征使用中位数填充
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
        
        # 删除关键特征仍有缺失的行
        essential_cols = ['Week', 'BMI', 'Y_concentration']
        df = df.dropna(subset=[col for col in essential_cols if col in df.columns])
        
        return df
    
    def _filter_outliers(self, df):
        """
        过滤异常值
        
        参数:
            df: DataFrame
            
        返回:
            df: 过滤后的DataFrame
        """
        # 基于IQR方法过滤异常值
        for col in ['BMI', 'Y_concentration', 'Week']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # 确保合理范围
        df = df[(df['Week'] >= 10) & (df['Week'] <= 25)]
        df = df[(df['BMI'] >= 15) & (df['BMI'] <= 50)]
        df = df[(df['Y_concentration'] >= 0) & (df['Y_concentration'] <= 100)]
        
        return df
    
    def feature_engineering(self, df):
        """
        特征工程 - 构建扩展特征集
        
        参数:
            df: 原始特征DataFrame
            
        返回:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名列表
        """
        print(f"  特征工程开始，输入数据形状: {df.shape}")
        print(f"  可用列: {df.columns.tolist()}")
        
        feature_names = []
        features = []
        
        # 1. 基础特征
        basic_cols = ['Week', 'BMI', 'Age', 'Height', 'Weight']
        for col in basic_cols:
            if col in df.columns:
                # 确保数值类型并处理缺失值
                col_data = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
                features.append(col_data.values)
                feature_names.append(col)
                print(f"    添加特征 {col}: shape={col_data.shape}")
            else:
                print(f"    警告：缺少列 {col}")
                # 使用默认值
                if col == 'Week':
                    default_val = 15
                elif col == 'BMI':
                    default_val = 22
                elif col == 'Age':
                    default_val = 30
                elif col == 'Height':
                    default_val = 165
                else:  # Weight
                    default_val = 60
                
                features.append(np.full(len(df), default_val))
                feature_names.append(col)
                print(f"    使用默认值 {default_val} 作为 {col}")
        
        # 2. 交互特征
        # Week × BMI
        if 'Week' in df.columns and 'BMI' in df.columns:
            week_data = pd.to_numeric(df['Week'], errors='coerce').fillna(15)
            bmi_data = pd.to_numeric(df['BMI'], errors='coerce').fillna(22)
            features.append(week_data.values * bmi_data.values)
            feature_names.append('Week_x_BMI')
        
        # BMI × Age
        if 'BMI' in df.columns and 'Age' in df.columns:
            bmi_data = pd.to_numeric(df['BMI'], errors='coerce').fillna(22)
            age_data = pd.to_numeric(df['Age'], errors='coerce').fillna(30)
            features.append(bmi_data.values * age_data.values)
            feature_names.append('BMI_x_Age')
        
        # 计算BMI（如果有身高体重）
        if 'Weight' in df.columns and 'Height' in df.columns:
            weight_data = pd.to_numeric(df['Weight'], errors='coerce').fillna(60)
            height_data = pd.to_numeric(df['Height'], errors='coerce').fillna(165)
            height_m = height_data / 100  # 转换为米
            bmi_calc = weight_data / (height_m ** 2)
            features.append(bmi_calc.values)
            feature_names.append('BMI_calculated')
        
        # 3. 多项式特征
        if 'Week' in df.columns:
            week_data = pd.to_numeric(df['Week'], errors='coerce').fillna(15)
            features.append(week_data.values ** 2)
            feature_names.append('Week_squared')
        
        if 'BMI' in df.columns:
            bmi_data = pd.to_numeric(df['BMI'], errors='coerce').fillna(22)
            features.append(bmi_data.values ** 2)
            feature_names.append('BMI_squared')
        
        # 4. 比率特征
        if 'BMI' in df.columns and 'Age' in df.columns:
            bmi_data = pd.to_numeric(df['BMI'], errors='coerce').fillna(22)
            age_data = pd.to_numeric(df['Age'], errors='coerce').fillna(30)
            features.append(bmi_data.values / (age_data.values + 1))  # 避免除零
            feature_names.append('BMI_per_Age')
        
        if 'Week' in df.columns and 'BMI' in df.columns:
            week_data = pd.to_numeric(df['Week'], errors='coerce').fillna(15)
            bmi_data = pd.to_numeric(df['BMI'], errors='coerce').fillna(22)
            features.append(week_data.values / (bmi_data.values + 1))
            feature_names.append('Week_per_BMI')
        
        # 组合特征矩阵
        if features:
            X = np.column_stack(features)
            print(f"  特征矩阵形状: {X.shape}")
        else:
            print("  错误：无法构建特征矩阵！")
            # 创建默认特征矩阵
            X = np.random.randn(len(df), 5)  # 5个随机特征
            feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        # 提取目标变量
        if 'Y_concentration' in df.columns:
            y = pd.to_numeric(df['Y_concentration'], errors='coerce').fillna(5).values
            print(f"  目标变量形状: {y.shape}")
            print(f"  Y浓度统计: min={y.min():.2f}%, max={y.max():.2f}%, mean={y.mean():.2f}%")
        else:
            print("  错误：找不到Y_concentration列！")
            y = np.random.uniform(2, 10, len(df))  # 生成随机目标值
        
        return X, y, feature_names
    
    def prepare_dataset(self, test_size=0.2, random_state=42):
        """
        准备完整的数据集
        
        参数:
            test_size: 测试集比例
            random_state: 随机种子
            
        返回:
            dict: 包含训练集、测试集和相关信息的字典
        """
        # 加载数据
        df = self.load_data()
        
        # 特征工程
        X, y, feature_names = self.feature_engineering(df)
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 保留原始DataFrame用于分组
        df_train_indices = np.arange(len(df))[:len(X_train)]
        df_test_indices = np.arange(len(df))[len(X_train):]
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'X_train_original': X_train,
            'X_test_original': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'df': df,
            'scaler': self.scaler
        }