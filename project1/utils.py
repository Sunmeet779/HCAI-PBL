import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


def preprocess_dataframe(df, target_column):
    df = df.copy()
    df = df.dropna()

    feature_cols = [col for col in df.columns if col != target_column]

    # Encode categorical features
    for col in feature_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Encode target if categorical
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column].astype(str))

    return df, feature_cols


def train_and_evaluate(df, feature_cols, target_col, model_name, test_size, metric,
                       c_values=None, n_estimators_values=None, max_depth_values=None):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    best_score = -1
    best_params = {}
    best_model = None
    
    # Determine number of unique classes for metric calculation
    n_classes = len(np.unique(y))
    
    # Define metric function based on the selected metric and number of classes
    if metric == 'accuracy':
        metric_func = accuracy_score
    elif metric == 'f1':
        if n_classes == 2:
            metric_func = lambda y_true, y_pred: f1_score(y_true, y_pred, average='binary')
        else:
            metric_func = lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
    elif metric == 'precision':
        if n_classes == 2:
            metric_func = lambda y_true, y_pred: precision_score(y_true, y_pred, average='binary', zero_division=0)
        else:
            metric_func = lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0)
    elif metric == 'recall':
        if n_classes == 2:
            metric_func = lambda y_true, y_pred: recall_score(y_true, y_pred, average='binary', zero_division=0)
        else:
            metric_func = lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if model_name == 'logreg':
        c_list = c_values if c_values else [1.0]
        for c in c_list:
            model = LogisticRegression(C=c, max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = metric_func(y_test, preds)
            
            if score > best_score:
                best_score = score
                best_params = {'C': c}
                best_model = model

    elif model_name == 'rf':
        n_list = n_estimators_values if n_estimators_values else [100]
        depth_list = max_depth_values if max_depth_values else [None]
        for n in n_list:
            for d in depth_list:
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = metric_func(y_test, preds)
                
                if score > best_score:
                    best_score = score
                    best_params = {'n_estimators': n, 'max_depth': d}
                    best_model = model

    elif model_name == 'svm':
        c_list = c_values if c_values else [1.0]
        for c in c_list:
            model = SVC(C=c, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = metric_func(y_test, preds)
            
            if score > best_score:
                best_score = score
                best_params = {'C': c}
                best_model = model
    else:
        raise ValueError("Unsupported model selected")

    return {
        'best_score': best_score,
        'best_params': best_params,
        'model': best_model
    }


def generate_visualizations(df, target_column):
    """Generate enhanced visualizations for the dataset"""
    plt.switch_backend('Agg')
    plt.style.use('default')
    
    # Set up color palette
    colors = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#db2777']
    sns.set_palette(colors)
    
    viz_cards = []

    try:
        # 1. Target Distribution
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig1.patch.set_facecolor('white')
        
        target_counts = df[target_column].value_counts()
        bars = ax1.bar(range(len(target_counts)), target_counts.values, 
                      color=colors[:len(target_counts)], alpha=0.8, edgecolor='black', linewidth=1)
        
        ax1.set_xlabel('Target Classes', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax1.set_title(f'Distribution of {target_column}', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xticks(range(len(target_counts)))
        ax1.set_xticklabels(target_counts.index, fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, target_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buf1.seek(0)
        img1 = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)
        
        viz_cards.append(f'''
        <div class="viz-card">
            <h4><i class="fas fa-chart-bar"></i> Target Distribution</h4>
            <img src="data:image/png;base64,{img1}" alt="Target Distribution" />
        </div>
        ''')

        # 2. Dataset Statistics
        fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(12, 10))
        fig2.patch.set_facecolor('white')
        fig2.suptitle('Dataset Statistics Overview', fontsize=16, fontweight='bold', y=0.98)
        
        # Missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0].head(10)
        if len(missing) > 0:
            missing.plot(kind='barh', ax=ax2a, color='#dc2626', alpha=0.7)
            ax2a.set_title('Missing Values by Column', fontweight='bold')
            ax2a.set_xlabel('Missing Count')
        else:
            ax2a.text(0.5, 0.5, 'No Missing Values!', transform=ax2a.transAxes, 
                     ha='center', va='center', fontsize=14, fontweight='bold', color='green')
            ax2a.set_title('Missing Values Check', fontweight='bold')
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        wedges, texts, autotexts = ax2b.pie(dtype_counts.values, labels=dtype_counts.index, 
                                           autopct='%1.1f%%', colors=colors[:len(dtype_counts)])
        ax2b.set_title('Data Types Distribution', fontweight='bold')
        
        # Dataset shape info
        ax2c.text(0.5, 0.7, f'Rows: {df.shape[0]:,}', transform=ax2c.transAxes, 
                 ha='center', va='center', fontsize=16, fontweight='bold')
        ax2c.text(0.5, 0.5, f'Columns: {df.shape[1]:,}', transform=ax2c.transAxes, 
                 ha='center', va='center', fontsize=16, fontweight='bold')
        ax2c.text(0.5, 0.3, f'Memory: {df.memory_usage(deep=True).sum()/1024:.1f} KB', 
                 transform=ax2c.transAxes, ha='center', va='center', fontsize=14, fontweight='bold')
        ax2c.set_title('Dataset Overview', fontweight='bold')
        ax2c.set_xlim(0, 1)
        ax2c.set_ylim(0, 1)
        ax2c.set_xticks([])
        ax2c.set_yticks([])
        
        # Unique values
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            unique_counts = [df[col].nunique() for col in numeric_cols[:10]]
            ax2d.bar(range(len(unique_counts)), unique_counts, color=colors[2], alpha=0.7)
            ax2d.set_title('Unique Values (Numeric Columns)', fontweight='bold')
            ax2d.set_xlabel('Columns')
            ax2d.set_ylabel('Unique Count')
            ax2d.set_xticks(range(len(unique_counts)))
            ax2d.set_xticklabels([col[:8] + '...' if len(col) > 8 else col 
                                for col in numeric_cols[:10]], rotation=45)
        else:
            ax2d.text(0.5, 0.5, 'No Numeric\nColumns', transform=ax2d.transAxes, 
                     ha='center', va='center', fontsize=14, fontweight='bold')
            ax2d.set_title('Numeric Columns Analysis', fontweight='bold')
        
        plt.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buf2.seek(0)
        img2 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)
        
        viz_cards.append(f'''
        <div class="viz-card">
            <h4><i class="fas fa-chart-pie"></i> Dataset Statistics</h4>
            <img src="data:image/png;base64,{img2}" alt="Dataset Statistics" />
        </div>
        ''')

        # 3. Correlation Heatmap (only for numeric columns)
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] > 1:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            fig3.patch.set_facecolor('white')
            
            corr = numeric_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            sns.heatmap(corr, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax3)
            ax3.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format='png', dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            buf3.seek(0)
            img3 = base64.b64encode(buf3.read()).decode('utf-8')
            plt.close(fig3)
            
            viz_cards.append(f'''
            <div class="viz-card">
                <h4><i class="fas fa-project-diagram"></i> Feature Correlations</h4>
                <img src="data:image/png;base64,{img3}" alt="Correlation Heatmap" />
            </div>
            ''')

        # 4. Feature Analysis (scatter plots for top features)
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_cols = [col for col in numeric_cols if col != target_column]
        
        if len(numeric_cols) >= 2:
            fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig4.patch.set_facecolor('white')
            fig4.suptitle('Feature Relationships', fontsize=16, fontweight='bold', y=0.98)
            axes = axes.ravel()
            
            # Select top 4 numeric features (by correlation with target if numeric, else first 4)
            if target_column in numeric_df.columns:
                target_corr = numeric_df.corr()[target_column].abs().sort_values(ascending=False)
                top_features = target_corr.index[1:5].tolist()  # Skip target itself
            else:
                top_features = numeric_cols[:4]
            
            for i, feature in enumerate(top_features[:4]):
                if i < len(axes):
                    if target_column in numeric_df.columns:
                        # Scatter plot for numeric target
                        axes[i].scatter(df[feature], df[target_column], alpha=0.6, 
                                      color=colors[i], s=30, edgecolors='black', linewidth=0.5)
                        axes[i].set_xlabel(feature, fontweight='bold')
                        axes[i].set_ylabel(target_column, fontweight='bold')
                    else:
                        # Box plot for categorical target
                        df_plot = df[[feature, target_column]].dropna()
                        unique_targets = df_plot[target_column].unique()
                        bp = axes[i].boxplot([df_plot[df_plot[target_column]==target][feature].values 
                                            for target in unique_targets], 
                                           labels=unique_targets, patch_artist=True)
                        for patch, color in zip(bp['boxes'], colors[:len(unique_targets)]):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        axes[i].set_xlabel(target_column, fontweight='bold')
                        axes[i].set_ylabel(feature, fontweight='bold')
                    
                    axes[i].set_title(f'{feature} vs {target_column}', fontweight='bold')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(top_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            buf4 = io.BytesIO()
            fig4.savefig(buf4, format='png', dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            buf4.seek(0)
            img4 = base64.b64encode(buf4.read()).decode('utf-8')
            plt.close(fig4)
            
            viz_cards.append(f'''
            <div class="viz-card">
                <h4><i class="fas fa-scatter-chart"></i> Feature Relationships</h4>
                <img src="data:image/png;base64,{img4}" alt="Feature Analysis" />
            </div>
            ''')

    except Exception as e:
        print(f"Error generating visualizations: {e}")
        viz_cards.append(f'''
        <div class="viz-card">
            <h4><i class="fas fa-exclamation-triangle"></i> Visualization Error</h4>
            <p style="text-align: center; color: #dc2626; padding: 2rem;">
                Could not generate visualizations: {str(e)}
            </p>
        </div>
        ''')

    return '\n'.join(viz_cards)
