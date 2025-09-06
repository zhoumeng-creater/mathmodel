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
        
        返回:
            df: 处理后的DataFrame
        """
        # 读取Excel数据
        df = pd.read_excel(self.data_path)
        
        # 打印列名以便调试
        print(f"  原始数据列名: {df.columns.tolist()[:10]}...")  # 显示前10个列名
        
        # 根据实际的中文列名进行处理
        # 创建一个新的DataFrame，只包含需要的列
        processed_df = pd.DataFrame()
        
        # 基础信息列
        if '孕妇年龄' in df.columns:
            processed_df['Age'] = df['孕妇年龄']
        elif 'C' in df.columns:  # 兼容字母列名
            processed_df['Age'] = df['C']
        
        if '孕妇身高' in df.columns:
            processed_df['Height'] = df['孕妇身高']
        elif 'D' in df.columns:
            processed_df['Height'] = df['D']
        
        if '孕妇体重' in df.columns:
            processed_df['Weight'] = df['孕妇体重']
        elif 'E' in df.columns:
            processed_df['Weight'] = df['E']
        
        if '孕妇本次检测时的孕周' in df.columns:
            processed_df['Week_raw'] = df['孕妇本次检测时的孕周']
        elif 'J' in df.columns:
            processed_df['Week_raw'] = df['J']
        
        if '孕妇BMI指标' in df.columns:
            processed_df['BMI'] = df['孕妇BMI指标']
        elif 'K' in df.columns:
            processed_df['BMI'] = df['K']
        
        # Y染色体浓度（关键列）
        if 'Y染色体浓度' in df.columns:
            # 如果是百分比形式（0.05表示5%），转换为百分数
            y_values = df['Y染色体浓度']
            if y_values.dropna().max() < 1:  # 判断是否为小数形式
                processed_df['Y_concentration'] = y_values * 100  # 转换为百分比
            else:
                processed_df['Y_concentration'] = y_values
        elif 'V' in df.columns:
            y_values = df['V']
            if pd.notna(y_values).any() and y_values.dropna().max() < 1:
                processed_df['Y_concentration'] = y_values * 100
            else:
                processed_df['Y_concentration'] = y_values
        
        # 其他可能用到的列
        if 'GC含量' in df.columns:
            processed_df['GC_content'] = df['GC含量']
        elif 'P' in df.columns:
            processed_df['GC_content'] = df['P']
        
        # 筛选男胎数据（Y染色体浓度不为空）
        processed_df = processed_df[processed_df['Y_concentration'].notna()].copy()
        
        # 处理孕周数据
        if 'Week_raw' in processed_df.columns:
            processed_df['Week'] = processed_df['Week_raw'].apply(self._parse_week)
        
        # 处理缺失值
        processed_df = self._handle_missing_values(processed_df)
        
        # 过滤异常值
        processed_df = self._filter_outliers(processed_df)
        
        print(f"  数据加载完成: {len(processed_df)}个有效样本")
        
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
        feature_names = []
        features = []
        
        # 1. 基础特征
        basic_cols = ['Week', 'BMI', 'Age', 'Height', 'Weight']
        for col in basic_cols:
            if col in df.columns:
                features.append(df[col].values)
                feature_names.append(col)
        
        # 2. 交互特征
        if 'Week' in df.columns and 'BMI' in df.columns:
            features.append(df['Week'].values * df['BMI'].values)
            feature_names.append('Week_x_BMI')
        
        if 'BMI' in df.columns and 'Age' in df.columns:
            features.append(df['BMI'].values * df['Age'].values)
            feature_names.append('BMI_x_Age')
        
        if 'Weight' in df.columns and 'Height' in df.columns:
            features.append(df['Weight'].values / (df['Height'].values / 100) ** 2)
            feature_names.append('BMI_calculated')
        
        # 3. 多项式特征
        if 'Week' in df.columns:
            features.append(df['Week'].values ** 2)
            feature_names.append('Week_squared')
        
        if 'BMI' in df.columns:
            features.append(df['BMI'].values ** 2)
            feature_names.append('BMI_squared')
        
        if 'Age' in df.columns:
            features.append(df['Age'].values ** 2)
            feature_names.append('Age_squared')
        
        # 4. 比率特征
        if 'BMI' in df.columns and 'Age' in df.columns:
            features.append(df['BMI'].values / (df['Age'].values + 1))  # 避免除零
            feature_names.append('BMI_per_Age')
        
        if 'Week' in df.columns and 'BMI' in df.columns:
            features.append(df['Week'].values / (df['BMI'].values + 1))
            feature_names.append('Week_per_BMI')
        
        if 'BMI' in df.columns and 'Week' in df.columns:
            features.append(df['BMI'].values * np.sqrt(df['Week'].values))
            feature_names.append('BMI_x_sqrt_Week')
        
        # 组合特征矩阵
        X = np.column_stack(features) if features else np.array([[]])
        
        # 提取目标变量
        y = df['Y_concentration'].values if 'Y_concentration' in df.columns else np.array([])
        
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