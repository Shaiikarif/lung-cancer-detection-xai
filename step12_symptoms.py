# step12_symptoms.py
"""
STEP 12: Symptom-Based Lung Cancer Risk Assessment
Complete Fixed Version - No Errors

Trains: MLP, Random Forest, Gradient Boosting, Logistic Regression,
        SVM, LightGBM, CatBoost, XGBoost
"""

import os
import sys

# Fix Windows console encoding for Unicode characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Try importing optional libraries
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False
XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
    print("  [OK] LightGBM available")
except ImportError:
    print("  [WARN] LightGBM not installed. Run: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    print("  [OK] CatBoost available")
except ImportError:
    print("  [WARN] CatBoost not installed. Run: pip install catboost")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("  [OK] XGBoost available")
except ImportError:
    print("  [WARN] XGBoost not installed. Run: pip install xgboost")


##################################################################
#                     DATASET CLASS
##################################################################

class LungCancerSymptomDataset:
    """Handles loading or creating the symptom dataset"""

    def __init__(self, data_path='dataset/lung_cancer_symptoms.csv'):
        self.data_path = data_path
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_or_create_dataset(self):
        """Load existing CSV or create synthetic dataset"""

        if os.path.exists(self.data_path):
            print(f"  Loading dataset from: {self.data_path}")
            df = pd.read_csv(self.data_path)
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        else:
            print(f"  Dataset not found at: {self.data_path}")
            print(f"  Creating comprehensive synthetic dataset...")
            df = self.create_synthetic_dataset()
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            df.to_csv(self.data_path, index=False)
            print(f"  [OK] Saved {len(df)} samples to: {self.data_path}")
            return df

    def create_synthetic_dataset(self):
        """Create realistic synthetic lung cancer symptom dataset"""

        np.random.seed(42)
        n = 5000

        data = {}

        # Demographics
        data['Age'] = np.random.choice(range(25, 85), n)
        data['Gender'] = np.random.choice([0, 1], n)

        # Lifestyle
        data['Smoking'] = np.random.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])
        data['Smoking_Years'] = np.zeros(n)
        data['Alcohol_Use'] = np.random.choice([0, 1, 2], n, p=[0.3, 0.45, 0.25])
        data['Obesity'] = np.random.choice([0, 1], n, p=[0.65, 0.35])

        # Environmental
        data['Air_Pollution_Exposure'] = np.random.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2])
        data['Occupational_Hazards'] = np.random.choice([0, 1], n, p=[0.7, 0.3])
        data['Dust_Allergy'] = np.random.choice([0, 1], n, p=[0.6, 0.4])

        # Family and Genetics
        data['Family_History'] = np.random.choice([0, 1], n, p=[0.75, 0.25])
        data['Genetic_Risk'] = np.random.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1])

        # Symptoms
        data['Persistent_Cough'] = np.random.choice([0, 1], n)
        data['Coughing_Blood'] = np.random.choice([0, 1], n, p=[0.85, 0.15])
        data['Shortness_of_Breath'] = np.random.choice([0, 1], n)
        data['Chest_Pain'] = np.random.choice([0, 1], n)
        data['Wheezing'] = np.random.choice([0, 1], n)
        data['Hoarseness'] = np.random.choice([0, 1], n, p=[0.7, 0.3])
        data['Fatigue'] = np.random.choice([0, 1], n)
        data['Weight_Loss'] = np.random.choice([0, 1], n, p=[0.65, 0.35])
        data['Loss_of_Appetite'] = np.random.choice([0, 1], n, p=[0.6, 0.4])
        data['Difficulty_Swallowing'] = np.random.choice([0, 1], n, p=[0.75, 0.25])
        data['Finger_Clubbing'] = np.random.choice([0, 1], n, p=[0.85, 0.15])
        data['Frequent_Cold'] = np.random.choice([0, 1], n, p=[0.55, 0.45])
        data['Dry_Cough'] = np.random.choice([0, 1], n)
        data['Snoring'] = np.random.choice([0, 1], n, p=[0.6, 0.4])
        data['Swallowing_Difficulty'] = np.random.choice([0, 1], n, p=[0.7, 0.3])

        # Medical History
        data['Chronic_Lung_Disease'] = np.random.choice([0, 1], n, p=[0.8, 0.2])
        data['Previous_Cancer'] = np.random.choice([0, 1], n, p=[0.9, 0.1])

        df = pd.DataFrame(data)

        # Set smoking years based on smoking status
        for i in range(n):
            smoking_status = df.loc[i, 'Smoking']
            if smoking_status == 0:
                df.loc[i, 'Smoking_Years'] = 0
            elif smoking_status == 1:
                df.loc[i, 'Smoking_Years'] = np.random.randint(1, 20)
            else:
                df.loc[i, 'Smoking_Years'] = np.random.randint(5, 40)

        # Calculate risk score based on medical knowledge
        risk = np.zeros(n, dtype=float)

        # Major risk factors
        risk = risk + (df['Age'].values > 55).astype(float) * 2.0
        risk = risk + (df['Age'].values > 65).astype(float) * 1.0
        risk = risk + df['Smoking'].values.astype(float) * 3.0
        risk = risk + (df['Smoking_Years'].values > 20).astype(float) * 2.0
        risk = risk + df['Family_History'].values.astype(float) * 2.0
        risk = risk + df['Genetic_Risk'].values.astype(float) * 1.5

        # Symptom factors
        risk = risk + df['Persistent_Cough'].values.astype(float) * 1.5
        risk = risk + df['Coughing_Blood'].values.astype(float) * 3.0
        risk = risk + df['Shortness_of_Breath'].values.astype(float) * 1.5
        risk = risk + df['Chest_Pain'].values.astype(float) * 2.0
        risk = risk + df['Wheezing'].values.astype(float) * 1.0
        risk = risk + df['Hoarseness'].values.astype(float) * 1.0
        risk = risk + df['Fatigue'].values.astype(float) * 1.0
        risk = risk + df['Weight_Loss'].values.astype(float) * 2.0
        risk = risk + df['Loss_of_Appetite'].values.astype(float) * 1.5
        risk = risk + df['Finger_Clubbing'].values.astype(float) * 2.5
        risk = risk + df['Difficulty_Swallowing'].values.astype(float) * 1.5

        # Environmental factors
        risk = risk + df['Air_Pollution_Exposure'].values.astype(float) * 1.0
        risk = risk + df['Occupational_Hazards'].values.astype(float) * 1.5
        risk = risk + df['Chronic_Lung_Disease'].values.astype(float) * 2.0
        risk = risk + df['Previous_Cancer'].values.astype(float) * 2.0

        # Add noise for realism
        risk = risk + np.random.normal(0, 1.5, n)

        # Convert to 3 classes
        low_threshold = np.percentile(risk, 40)
        high_threshold = np.percentile(risk, 75)

        labels = []
        for r in risk:
            if r <= low_threshold:
                labels.append('Low')
            elif r <= high_threshold:
                labels.append('Medium')
            else:
                labels.append('High')

        df['Risk_Level'] = labels

        return df

    def prepare_data(self, df):
        """Prepare data for ML training"""

        # Find target column
        target_col = None
        possible_names = ['Risk_Level', 'LUNG_CANCER', 'Result', 'Level']

        for col_name in possible_names:
            if col_name in df.columns:
                target_col = col_name
                break

        if target_col is None:
            target_col = df.columns[-1]

        print(f"  Target column: {target_col}")
        print(f"  Target values: {df[target_col].unique()}")

        # Separate features and target
        X = df.drop(columns=[target_col]).copy()
        y_raw = df[target_col].copy()

        # Convert X categorical columns to numeric
        for col in X.columns:
            col_dtype = str(X[col].dtype)
            if col_dtype == 'object' or col_dtype == 'category':
                temp_encoder = LabelEncoder()
                X[col] = temp_encoder.fit_transform(X[col].astype(str))

        # Convert target to numeric integers
        y_raw_str = y_raw.astype(str)
        self.label_encoder.fit(y_raw_str)
        y_encoded = self.label_encoder.transform(y_raw_str)
        y = y_encoded.astype(int)

        print(f"  Label mapping:")
        for original, encoded in zip(self.label_encoder.classes_,
                                      range(len(self.label_encoder.classes_))):
            print(f"    '{original}' -> {encoded}")

        # Save feature names
        self.feature_names = X.columns.tolist()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Print dataset info
        print(f"\n  Dataset Info:")
        print(f"    Total samples:  {len(X)}")
        print(f"    Total features: {len(self.feature_names)}")
        print(f"    Train samples:  {len(X_train)}")
        print(f"    Test samples:   {len(X_test)}")
        print(f"\n  Class Distribution:")

        class_display_names = ['Low Risk', 'Medium Risk', 'High Risk']
        unique_values = np.unique(y)
        value_counts = np.bincount(y)

        for val in unique_values:
            count = value_counts[val]
            total = len(y)
            if val < len(class_display_names):
                display = class_display_names[val]
            else:
                display = f"Class {val}"
            print(f"    {display}: {count} ({count / total * 100:.1f}%)")

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test


##################################################################
#                     MODEL TRAINER CLASS
##################################################################

class SymptomModelTrainer:
    """Create, train, and compare multiple ML models"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.class_names = ['Low Risk', 'Medium Risk', 'High Risk']

    def create_models(self):
        """Initialize all models"""

        print("\n  Creating models...")

        # 1. MLP Neural Network
        self.models['MLP'] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            random_state=42,
            verbose=False
        )

        # 2. Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 3. Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        # 4. Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial'
        )

        # 5. SVM
        self.models['SVM'] = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )

        # 6. LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )

        # 7. CatBoost (if available)
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=0
            )

        # 8. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )

        print(f"  [OK] Created {len(self.models)} models:")
        for model_name in self.models:
            print(f"    - {model_name}")

    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train every model and collect metrics"""

        print("\n" + "=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)

        for model_name, model in self.models.items():
            print(f"\n  Training {model_name}...")

            try:
                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Get probabilities if available
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except Exception:
                        y_proba = None

                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

                # ROC AUC
                roc = 0.0
                if y_proba is not None:
                    try:
                        roc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                    except Exception:
                        roc = 0.0

                # Store results
                self.results[model_name] = {
                    'accuracy': float(acc),
                    'f1_score': float(f1),
                    'precision': float(prec),
                    'recall': float(rec),
                    'roc_auc': float(roc),
                    'y_pred': y_pred,
                    'y_proba': y_proba
                }

                print(f"    Acc: {acc:.4f} | F1: {f1:.4f} | "
                      f"Prec: {prec:.4f} | Rec: {rec:.4f} | AUC: {roc:.4f}")

            except Exception as e:
                print(f"    ERROR training {model_name}: {e}")
                self.results[model_name] = {
                    'accuracy': 0.0, 'f1_score': 0.0,
                    'precision': 0.0, 'recall': 0.0,
                    'roc_auc': 0.0,
                    'y_pred': None, 'y_proba': None
                }

        return self.results

    def save_best_model(self, save_dir='checkpoints'):
        """Find and save the best model"""

        if not self.results:
            print("  No results available!")
            return None

        # Find best by F1 score
        best_name = None
        best_f1 = -1.0

        for name, res in self.results.items():
            if res['f1_score'] > best_f1:
                best_f1 = res['f1_score']
                best_name = name

        if best_name is None:
            print("  Could not determine best model!")
            return None

        best_model = self.models[best_name]

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'best_symptom_model.joblib')
        joblib.dump(best_model, save_path)

        print(f"\n  [OK] Best model: {best_name}")
        print(f"       F1 Score:   {best_f1:.4f}")
        print(f"       Saved to:   {save_path}")

        return best_name


##################################################################
#                     VISUALIZATION CLASS
##################################################################

class SymptomVisualizer:
    """Generate all charts and plots"""

    def __init__(self):
        self.class_names = ['Low Risk', 'Medium Risk', 'High Risk']
        os.makedirs('outputs', exist_ok=True)

    def plot_dataset_analysis(self, df, save_path='outputs/symptom_dataset_analysis.png'):
        """Visualize the dataset"""

        target_col = 'Risk_Level' if 'Risk_Level' in df.columns else df.columns[-1]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. Target distribution
        target_counts = df[target_col].value_counts()
        target_counts.plot(kind='bar', ax=axes[0, 0], color=colors[:len(target_counts)],
                           edgecolor='black')
        axes[0, 0].set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=0)

        # 2. Age distribution by risk
        if 'Age' in df.columns:
            unique_levels = sorted(df[target_col].unique())
            for i, level in enumerate(unique_levels):
                mask = df[target_col] == level
                color = colors[i % len(colors)]
                axes[0, 1].hist(df.loc[mask, 'Age'], bins=20, alpha=0.5,
                                label=str(level), color=color, edgecolor='black')
            axes[0, 1].set_title('Age by Risk Level', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Age')
            axes[0, 1].legend()

        # 3. Smoking vs Risk
        if 'Smoking' in df.columns:
            smoking_ct = pd.crosstab(df['Smoking'], df[target_col])
            smoking_ct.plot(kind='bar', ax=axes[0, 2], color=colors[:smoking_ct.shape[1]],
                            edgecolor='black')
            axes[0, 2].set_title('Smoking vs Risk', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Smoking (0=Never, 1=Former, 2=Current)')
            axes[0, 2].tick_params(axis='x', rotation=0)

        # 4. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_data = df[numeric_cols].corr()
            if corr_data.shape[0] > 12:
                top_cols = corr_data.iloc[:, -1].abs().nlargest(12).index
                corr_data = corr_data.loc[top_cols, top_cols]
            sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r',
                        ax=axes[1, 0], center=0, annot_kws={'size': 7})
            axes[1, 0].set_title('Correlations', fontsize=14, fontweight='bold')

        # 5. Symptom prevalence
        skip_cols = {'Age', 'Gender', 'Smoking_Years', target_col, 'Smoking',
                     'Alcohol_Use', 'Genetic_Risk', 'Air_Pollution_Exposure'}
        symptom_cols = [c for c in df.columns if c not in skip_cols]
        if symptom_cols:
            means = df[symptom_cols[:10]].mean().sort_values(ascending=True)
            means.plot(kind='barh', ax=axes[1, 1], color='#3498db', edgecolor='black')
            axes[1, 1].set_title('Symptom Prevalence', fontsize=14, fontweight='bold')

        # 6. Summary text
        axes[1, 2].axis('off')
        total = len(df)
        text = f"Dataset Summary\n{'=' * 25}\n"
        text += f"Samples: {total}\n"
        text += f"Features: {df.shape[1] - 1}\n\n"
        for level in sorted(df[target_col].unique()):
            cnt = int((df[target_col] == level).sum())
            text += f"{level}: {cnt} ({cnt / total * 100:.1f}%)\n"
        axes[1, 2].text(0.1, 0.5, text, transform=axes[1, 2].transAxes,
                        fontsize=13, verticalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.suptitle('Lung Cancer Symptom Dataset Analysis',
                      fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Dataset analysis saved: {save_path}")
        plt.show()

    def plot_model_comparison(self, results, save_path='outputs/symptom_model_comparison.png'):
        """Compare all models side by side"""

        if not results:
            print("  [WARN] No results to plot")
            return

        model_names = list(results.keys())
        metric_keys = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
        metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC-AUC']
        bar_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Bar chart
        x = np.arange(len(model_names))
        bar_width = 0.15

        for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            values = [results[m][key] for m in model_names]
            axes[0].bar(x + i * bar_width, values, bar_width,
                        label=label, color=bar_colors[i],
                        edgecolor='black', alpha=0.85)

        axes[0].set_xlabel('Model', fontsize=13)
        axes[0].set_ylabel('Score', fontsize=13)
        axes[0].set_title('Model Performance', fontsize=16, fontweight='bold')
        axes[0].set_xticks(x + bar_width * 2)
        axes[0].set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
        axes[0].set_ylim(0, 1.1)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')

        # Heatmap
        heatmap_data = []
        for m in model_names:
            row = [results[m][k] for k in metric_keys]
            heatmap_data.append(row)

        sns.heatmap(np.array(heatmap_data), annot=True, fmt='.4f',
                    xticklabels=metric_labels, yticklabels=model_names,
                    cmap='YlOrRd', ax=axes[1], annot_kws={'size': 11})
        axes[1].set_title('Metrics Heatmap', fontsize=16, fontweight='bold')

        plt.suptitle('ML Model Comparison for Symptom Prediction',
                      fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Model comparison saved: {save_path}")
        plt.show()

    def plot_feature_importance(self, models, feature_names,
                                 save_path='outputs/symptom_feature_importance.png'):
        """Plot feature importance from tree-based models"""

        tree_models = []
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                tree_models.append((name, model))

        if not tree_models:
            print("  [WARN] No tree-based models for feature importance")
            return

        n_plots = min(3, len(tree_models))
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))

        if n_plots == 1:
            axes = [axes]

        for idx in range(n_plots):
            name, model = tree_models[idx]
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-15:]

            top_importances = importances[top_indices]
            top_names = [feature_names[i] for i in top_indices]

            color_values = top_importances / (top_importances.max() + 1e-10)
            bar_colors = plt.cm.viridis(color_values)

            axes[idx].barh(range(len(top_indices)), top_importances,
                           color=bar_colors, edgecolor='black')
            axes[idx].set_yticks(range(len(top_indices)))
            axes[idx].set_yticklabels(top_names, fontsize=10)
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{name}\nFeature Importance',
                                 fontsize=14, fontweight='bold')

        plt.suptitle('Top Features for Risk Prediction',
                      fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Feature importance saved: {save_path}")
        plt.show()

    def plot_confusion_matrices(self, results, y_test,
                                  save_path='outputs/symptom_confusion_matrices.png'):
        """Plot confusion matrices for top models"""

        valid_models = []
        for name, res in results.items():
            if res['y_pred'] is not None:
                valid_models.append((name, res))

        # Sort by F1
        valid_models.sort(key=lambda x: x[1]['f1_score'], reverse=True)
        top_models = valid_models[:4]

        if not top_models:
            print("  [WARN] No valid predictions for confusion matrix")
            return

        n_plots = len(top_models)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

        if n_plots == 1:
            axes = [axes]

        for idx, (name, res) in enumerate(top_models):
            cm = confusion_matrix(y_test, res['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names,
                        ax=axes[idx], annot_kws={'size': 14})
            axes[idx].set_title(f'{name}\nAcc: {res["accuracy"]:.3f}',
                                 fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.suptitle('Confusion Matrices - Top Models',
                      fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Confusion matrices saved: {save_path}")
        plt.show()

    def plot_roc_curves(self, results, y_test,
                         save_path='outputs/symptom_roc_curves.png'):
        """Plot ROC curves for all models"""

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        n_classes = len(self.class_names)

        # One-hot encode y_test
        y_test_int = np.array(y_test, dtype=int)
        y_onehot = np.zeros((len(y_test_int), n_classes))
        for i, label in enumerate(y_test_int):
            if label < n_classes:
                y_onehot[i, label] = 1

        for (name, res), color in zip(results.items(), plot_colors):
            if res['y_proba'] is not None:
                try:
                    proba = res['y_proba']
                    fpr, tpr, _ = roc_curve(y_onehot.ravel(), proba.ravel())
                    roc_auc_val = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, linewidth=2,
                            label=f'{name} (AUC = {roc_auc_val:.4f})')
                except Exception:
                    pass

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('ROC Curves - All Models', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] ROC curves saved: {save_path}")
        plt.show()


##################################################################
#                     MAIN EXECUTION
##################################################################

def run_symptom_analysis():
    """Run the complete symptom analysis pipeline"""

    print("\n" + "=" * 60)
    print("STEP 12: SYMPTOM-BASED LUNG CANCER RISK ASSESSMENT")
    print("=" * 60)

    # ---- 1. Load Dataset ----
    print("\n[1/6] Loading Dataset...")
    dataset = LungCancerSymptomDataset()
    df = dataset.load_or_create_dataset()

    # ---- 2. Visualize Dataset ----
    print("\n[2/6] Visualizing Dataset...")
    visualizer = SymptomVisualizer()
    visualizer.plot_dataset_analysis(df)

    # ---- 3. Prepare Data ----
    print("\n[3/6] Preparing Data...")
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw = dataset.prepare_data(df)

    # ---- 4. Train Models ----
    print("\n[4/6] Training Models...")
    trainer = SymptomModelTrainer()
    trainer.create_models()
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)

    # ---- 5. Generate Visualizations ----
    print("\n[5/6] Generating Visualizations...")
    visualizer.plot_model_comparison(results)
    visualizer.plot_confusion_matrices(results, y_test)
    visualizer.plot_roc_curves(results, y_test)
    visualizer.plot_feature_importance(trainer.models, dataset.feature_names)

    # ---- 6. Save Everything ----
    print("\n[6/6] Saving Models & Results...")
    best_model_name = trainer.save_best_model()

    # Save scaler, features, encoder
    os.makedirs('checkpoints', exist_ok=True)
    joblib.dump(dataset.scaler, 'checkpoints/symptom_scaler.joblib')
    joblib.dump(dataset.feature_names, 'checkpoints/symptom_features.joblib')
    joblib.dump(dataset.label_encoder, 'checkpoints/symptom_label_encoder.joblib')
    print("  [OK] Scaler, features, and label encoder saved")

    # Save results JSON
    summary = {}
    for name, res in results.items():
        summary[name] = {
            'accuracy': float(res['accuracy']),
            'f1_score': float(res['f1_score']),
            'precision': float(res['precision']),
            'recall': float(res['recall']),
            'roc_auc': float(res['roc_auc'])
        }

    json_path = 'outputs/symptom_model_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"  [OK] Results saved to: {json_path}")

    # ---- Print Final Comparison Table ----
    print("\n" + "=" * 90)
    print("FINAL MODEL COMPARISON TABLE")
    print("=" * 90)
    print(f"\n  {'Model':<25} {'Accuracy':<12} {'F1 Score':<12} "
          f"{'Precision':<12} {'Recall':<12} {'ROC-AUC':<12}")
    print(f"  {'-' * 85}")

    sorted_results = sorted(
        results.items(),
        key=lambda item: item[1]['f1_score'],
        reverse=True
    )

    for name, res in sorted_results:
        marker = " <-- BEST" if name == best_model_name else ""
        print(f"  {name:<25} {res['accuracy']:<12.4f} {res['f1_score']:<12.4f} "
              f"{res['precision']:<12.4f} {res['recall']:<12.4f} "
              f"{res['roc_auc']:<12.4f}{marker}")

    print(f"\n  [OK] Step 12 complete!")
    print(f"  Check 'outputs/' folder for all visualizations.")
    print(f"  Next: python step13_training_visualizer.py")


##################################################################
#                     RUN
##################################################################

if __name__ == "__main__":
    run_symptom_analysis()