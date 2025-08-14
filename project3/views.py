from palmerpenguins import load_penguins
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for rendering
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from django.shortcuts import render

# Set global style for all plots
plt.style.use('default')  # Using default style as base
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'axes.labelcolor': '#1e293b',
    'text.color': '#1e293b',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#f1f5f9',
    'grid.linewidth': 0.5,
})

# Common data loading
def load_penguin_data():
    df = load_penguins().dropna()
    X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    X.loc[:, 'sex'] = X['sex'].map({'male': 0, 'female': 1})  # Encode sex
    y = df['species']
    return X, y, df

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='white')
    plt.close(fig)  # Close the figure to free memory
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_modern_tree(tree_model, feature_names, class_names):
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    ax.set_facecolor('white')
    
    # Define a color palette for the classes
    class_colors = {
        'Adelie': '#4e79a7',    # Blue
        'Gentoo': '#59a14f',    # Green
        'Chinstrap': '#e15759'  # Red
    }
    
    # Plot tree with custom styling
    plot_tree(tree_model,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              proportion=True,
              ax=ax,
              fontsize=10,
              impurity=False,
              node_ids=False,
              precision=2)
    
    # Custom coloring of nodes after plotting
    for i, text in enumerate(ax.texts):
        text.set_color('#2e3440')  # Dark text for readability
        
        # Color nodes based on class
        if 'class =' in text.get_text():
            for cls, color in class_colors.items():
                if cls in text.get_text():
                    # Get the parent artist (the node box)
                    for artist in ax.artists:
                        if artist.get_bbox().contains(text.get_position()):
                            artist.set_facecolor(color)
                            artist.set_edgecolor('#2e3440')
                            artist.set_alpha(0.7)
                            break
        
        # Style decision nodes differently
        if 'samples =' in text.get_text() and 'class =' not in text.get_text():
            text.set_bbox(dict(
                facecolor='#f8f9fa',
                edgecolor='#d1d9e0',
                boxstyle='round,pad=0.5',
                alpha=0.8
            ))
    
    # Remove spines and add subtle grid
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.grid(True, color='#e5e9f0', linestyle='-', linewidth=0.5)
    
    # Add title with species color indicators
    ax.set_title('Penguin Species Decision Tree', pad=20, fontsize=14)
    
    # Add color legend manually
    for i, (cls, color) in enumerate(class_colors.items()):
        ax.text(0.95, 0.95 - (i*0.03), cls,
                transform=ax.transAxes,
                color=color,
                fontsize=10,
                ha='right',
                bbox=dict(facecolor='white', edgecolor=color, pad=3))
    
    return fig

# Task 1: Simple Tree
def simple_tree_view(request):
    X, y, df = load_penguin_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    
    fig = plot_modern_tree(tree, X.columns, y.unique().tolist())
    
    return render(request, 'project3/simple_tree.html', {
        'accuracy': round(accuracy_score(y_test, tree.predict(X_test)), 3),
        'n_leaves': tree.get_n_leaves(),
        'tree_image': plot_to_base64(fig),
        'features': X.columns.tolist()
    })

# Task 2: Sparse Tree
def sparse_tree_view(request):
    X, y, df = load_penguin_data()
    ccp_alpha = float(request.GET.get('ccp_alpha', 0.02))
    
    tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha, max_depth=5, random_state=42)
    tree.fit(X, y)
    
    fig = plot_modern_tree(tree, X.columns, y.unique())
    
    return render(request, 'project3/sparse_tree.html', {
        'tree_image': plot_to_base64(fig),
        'accuracy': round(accuracy_score(y, tree.predict(X)), 3),
        'n_leaves': tree.get_n_leaves(),
        'ccp_alpha': ccp_alpha,
        'features': X.columns.tolist()
    })

# Task 3: Logistic Regression
def logistic_regression_view(request):
    X, y, df = load_penguin_data()
    C = float(request.GET.get('C', 1.0))  # Inverse of regularization strength
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model with L1 penalty
    lr = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=C,
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_scaled, y)
    
    # Get metrics
    accuracy = round(accuracy_score(y, lr.predict(X_scaled)), 3)
    n_features = np.sum(np.any(lr.coef_ != 0, axis=0))
    selected_features = [f for f, used in zip(X.columns, np.any(lr.coef_ != 0, axis=0)) if used]
    
    # Create modern coefficient plot
    fig, ax = plt.subplots(figsize=(10, 5))
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': np.mean(lr.coef_, axis=0)
    }).sort_values('coefficient', ascending=True)
    
    colors = ['#ef4444' if x < 0 else '#10b981' for x in coef_df['coefficient']]
    bars = ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        label_x = width if width > 0 else width
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                va='center',
                ha='left' if width > 0 else 'right',
                color='#1e293b')
    
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Importance', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.tick_params(axis='both', colors='#64748b')
    
    return render(request, 'project3/logistic_regression.html', {
        'accuracy': accuracy,
        'n_features': n_features,
        'selected_features': selected_features,
        'all_features': X.columns.tolist(),
        'C': C,
        'coef_plot': plot_to_base64(fig)
    })

# Task 4: Counterfactual Explanations
def counterfactual_explanations(request):
    # Load and preprocess data
    df = load_penguins().dropna()
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['island'] = df['island'].astype('category').cat.codes
    # Define features explicitly to match expected order
    X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    y = df['species']

    # Identify columns
    categorical_cols = ['island', 'sex']
    continuous_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

    # Scale only continuous features
    scaler = StandardScaler()
    X_continuous = X[continuous_cols].copy()
    X_scaled_continuous = scaler.fit_transform(X_continuous)
    
    # Combine scaled continuous and unscaled categorical features
    X_scaled = np.hstack([
        X[['island']].values,  # island first
        X_scaled_continuous,   # continuous features
        X[['sex']].values      # sex last
    ])
    # print("X_scaled shape:", X_scaled.shape)  # Debugging
    # print("X.columns:", X.columns.tolist())   # Debugging

    # Fit classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)

    # Read parameters from request
    selected_id = int(request.GET.get('selected_id', 0))
    target_class = request.GET.get('target_class', y.unique()[0])
    n_samples = int(request.GET.get('n_samples', 1000))
    k_results = int(request.GET.get('k_results', 3))

    # Get the selected instance
    x_original = X_scaled[selected_id]
    original_class = y.iloc[selected_id]
    # Reconstruct original features
    original_features = dict(zip(
        X.columns,
        np.hstack([
            x_original[0:1],  # island (unscaled)
            scaler.inverse_transform([x_original[1:1+len(continuous_cols)]])[0],  # continuous
            x_original[-1:]   # sex
        ])
    ))
    # print("Original features:", original_features)  # Debugging

    # Calculate MAD for continuous features only
    mad_continuous = median_abs_deviation(X_scaled[:, 1:1+len(continuous_cols)], axis=0)
    mad = np.hstack([np.ones(1), mad_continuous, np.ones(1)])  # island, continuous, sex
    # print("MAD values:", mad)  # Debugging

    # Generate counterfactuals
    counterfactuals = []
    rng = np.random.default_rng(42)
    X_df = X.reset_index(drop=True)
    valid_predictions = 0

    for _ in range(n_samples):
        x_candidate = x_original.copy()
        # Add noise to continuous features
        for i in range(1, 1 + len(continuous_cols)):
            x_candidate[i] += rng.normal(0, 1.0)
        # For categorical features
        for col in categorical_cols:
            idx = X.columns.get_loc(col)
            possible_vals = X_df[col].unique().tolist()
            orig_val = original_features[col]
            possible_vals = [v for v in possible_vals if v != round(orig_val)]
            if possible_vals:
                x_candidate[idx] = rng.choice(possible_vals)
        pred = clf.predict([x_candidate])[0]
        if pred == target_class:
            valid_predictions += 1
            dist_continuous = np.sum(
                np.abs(x_candidate[1:1+len(continuous_cols)] - x_original[1:1+len(continuous_cols)]) / 
                (mad_continuous + 1e-9)
            )
            dist_categorical = np.sum(
                x_candidate[[0, -1]] != x_original[[0, -1]]
            )
            dist = dist_continuous + dist_categorical
            inv_candidate = np.hstack([
                x_candidate[0:1],  # island
                scaler.inverse_transform([x_candidate[1:1+len(continuous_cols)]])[0],  # continuous
                x_candidate[-1:]   # sex
            ])
            changes = inv_candidate - np.hstack([
                x_original[0:1],
                scaler.inverse_transform([x_original[1:1+len(continuous_cols)]])[0],
                x_original[-1:]
            ])
            counterfactual_features = dict(zip(X.columns, inv_candidate))
            # print("Counterfactual features:", counterfactual_features)  # Debugging
            counterfactuals.append({
                'features': counterfactual_features,
                'distance': round(dist, 2),
                'changes': dict(zip(X.columns, changes)),
            })
    
    # print(f"Valid counterfactuals found: {valid_predictions}/{n_samples}")

    # Sort and select top-k
    counterfactuals.sort(key=lambda x: x['distance'])
    top_counterfactuals = counterfactuals[:k_results]

    # Create plot
    if top_counterfactuals:
        fig, ax = plt.subplots(figsize=(12, 6))
        features = list(original_features.keys())
        original_values = list(original_features.values())
        ax.plot(features, original_values, 
                marker='o', 
                linewidth=3, 
                color='#3b82f6',
                label='Original',
                markersize=8)
        explanation = (
            "This plot shows how features must change to transform a "
            f"{original_class} penguin into a {target_class} penguin.\n"
            "The blue line represents the original penguin's features, while "
            "green lines show possible counterfactual variations.\n"
            "The distance value indicates how much change is required "
            "(lower = more realistic transformation)."
        )
        plt.figtext(0.5, -0.1, explanation, 
                   ha='center', 
                   fontsize=10,
                   bbox=dict(facecolor='#f8f9fa', edgecolor='#e5e7eb', boxstyle='round,pad=0.5'))
        for i, cf in enumerate(top_counterfactuals):
            cf_values = list(cf['features'].values())
            ax.plot(features, cf_values,
                    marker='o',
                    linewidth=1.5,
                    color='#10b981',
                    alpha=0.6,
                    label=f'Counterfactual {i+1} (d={cf["distance"]})',
                    markersize=6)
        ax.set_title(f'Counterfactual Feature Changes: {original_class} → {target_class}', pad=20)
        ax.legend(frameon=False, bbox_to_anchor=(1, 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.tick_params(axis='both', colors='#64748b')
        ax.grid(True, axis='y', color='#f1f5f9', linestyle='-', linewidth=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close(fig)
        plot_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    else:
        plot_url = None

    return render(request, 'project3/counterfactual.html', {
        'original': original_features,
        'original_class': original_class,
        'target_class': target_class,
        'counterfactuals': top_counterfactuals,
        'plot_url': plot_url,
        'parameters': {
            'all_classes': y.unique().tolist(),
            'all_instances': df.to_dict('records'),
            'selected_id': selected_id,
            'target_class': target_class,
            'n_samples': n_samples,
            'k_results': k_results
        }
    })

def index(request):
    tasks = [
        {
            'url': 'project3:simple_tree',
            'title': 'Standard Tree',
            'description': 'Fixed complexity decision tree',
            'label': 'Task 1',
            'color': 'bg-blue-100 text-blue-800'
        },
        {
            'url': 'project3:sparse_tree',
            'title': 'Pruned Tree',
            'description': 'Complexity-controlled with CCP',
            'label': 'Task 2',
            'color': 'bg-green-100 text-green-800'
        },
        {
            'url': 'project3:logistic_regression',
            'title': 'Logistic Regression',
            'description': 'Feature selection via L1 regularization',
            'label': 'Task 3',
            'color': 'bg-purple-100 text-purple-800'
        },
        {
            'url': 'project3:counterfactual',
            'title': 'Counterfactuals',
            'description': 'What-if explanations for predictions',
            'label': 'Task 4',
            'color': 'bg-red-100 text-red-800'
        }
    ]
    return render(request, 'project3/index.html', {'tasks': tasks})