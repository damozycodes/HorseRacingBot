import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# CSV_FOLDER = os.getenv("CSV_FOLDER", r"C:\Users\damozy\Projects\HorseRacingBot\results")
# OUTPUT_DIR = CSV_FOLDER 
# MODEL_PATH = os.path.join(OUTPUT_DIR, "racing_model.joblib")

# Directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = SCRIPT_DIR  # or os.path.join(SCRIPT_DIR, "results") if you want a subfolder
MODEL_PATH = os.path.join(OUTPUT_DIR, "racing_model.joblib")

class HorseRacingPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, df):
        """
        Load and prepare the horse racing data for training
        """
        # Create target variable (1 if position == 1, 0 otherwise)
        df['position'] = pd.to_numeric(df['position'], errors='coerce')
        df['won'] = (df['position'] == 1).astype(int)   
        
        # Remove non-runners and invalid positions
        
        df = df[df['position'].notna() & (df['position'] > 0)]
        
        print(f"Dataset shape: {df.shape}")
        print(f"Win rate: {df['won'].mean():.3f}")
        print(df['position'].value_counts())
        
        return df
    
    def create_course_form_features(self, df):
        """
        Create course form features based on historical performance at specific courses
        """
        course_form = defaultdict(lambda: defaultdict(list))
        
        # Sort by date to ensure chronological order
        df_sorted = df.sort_values(['horse_id', 'date'])
        
        # Calculate course-specific form for each horse
        for idx, row in df_sorted.iterrows():
            horse_id = row['horse_id']
            course_id = row['course_id']
            
            # Get historical performance at this course (before current race)
            prev_course_races = course_form[horse_id][course_id]
            
            if len(prev_course_races) > 0:
                df.at[idx, 'course_wins'] = sum(prev_course_races)
                df.at[idx, 'course_runs'] = len(prev_course_races)
                df.at[idx, 'course_win_rate'] = sum(prev_course_races) / len(prev_course_races)
            else:
                df.at[idx, 'course_wins'] = 0
                df.at[idx, 'course_runs'] = 0
                df.at[idx, 'course_win_rate'] = 0
            
            # Add current race result to history
            course_form[horse_id][course_id].append(row['won'])
        
        return df
    
    def create_distance_form_features(self, df):
        """
        Create distance form features based on performance at similar distances
        """
        distance_form = defaultdict(lambda: defaultdict(list))
        
        # Define distance bands for grouping similar distances
        def get_distance_band(distance):
            if distance < 1200:
                return 'sprint'
            elif distance < 1800:
                return 'mile'
            elif distance < 2400:
                return 'middle'
            else:
                return 'staying'
        
        df['distance_band'] = df['dist_m'].apply(get_distance_band)
        df_sorted = df.sort_values(['horse_id', 'date'])
        
        for idx, row in df_sorted.iterrows():
            horse_id = row['horse_id']
            dist_band = row['distance_band']
            
            prev_dist_races = distance_form[horse_id][dist_band]
            
            if len(prev_dist_races) > 0:
                df.at[idx, 'distance_wins'] = sum(prev_dist_races)
                df.at[idx, 'distance_runs'] = len(prev_dist_races)
                df.at[idx, 'distance_win_rate'] = sum(prev_dist_races) / len(prev_dist_races)
            else:
                df.at[idx, 'distance_wins'] = 0
                df.at[idx, 'distance_runs'] = 0
                df.at[idx, 'distance_win_rate'] = 0
            
            distance_form[horse_id][dist_band].append(row['won'])
        
        return df
    
    def create_going_form_features(self, df):
        """
        Create going (ground condition) form features
        """
        going_form = defaultdict(lambda: defaultdict(list))
        df_sorted = df.sort_values(['horse_id', 'date'])
        
        for idx, row in df_sorted.iterrows():
            horse_id = row['horse_id']
            going = row['going']
            
            prev_going_races = going_form[horse_id][going]
            
            if len(prev_going_races) > 0:
                df.at[idx, 'going_wins'] = sum(prev_going_races)
                df.at[idx, 'going_runs'] = len(prev_going_races)
                df.at[idx, 'going_win_rate'] = sum(prev_going_races) / len(prev_going_races)
            else:
                df.at[idx, 'going_wins'] = 0
                df.at[idx, 'going_runs'] = 0
                df.at[idx, 'going_win_rate'] = 0
            
            going_form[horse_id][going].append(row['won'])
        
        return df
    
    def create_past_performance_features(self, df):
        """
        Create features based on recent past performances
        """
        df_sorted = df.sort_values(['horse_id', 'date'])
        
        # Initialize past performance tracking
        horse_history = defaultdict(list)
        
        for idx, row in df_sorted.iterrows():
            horse_id = row['horse_id']
            history = horse_history[horse_id]
            
            # Features based on last 5 runs
            if len(history) > 0:
                last_5_runs = history[-5:]
                df.at[idx, 'recent_wins'] = sum([r['won'] for r in last_5_runs])
                df.at[idx, 'recent_runs'] = len(last_5_runs)
                df.at[idx, 'recent_win_rate'] = sum([r['won'] for r in last_5_runs]) / len(last_5_runs)
                df.at[idx, 'avg_recent_position'] = np.mean([r['position'] for r in last_5_runs])
                df.at[idx, 'best_recent_position'] = min([r['position'] for r in last_5_runs])
                
                # Days since last run
                df.at[idx, 'days_since_last_run'] = (pd.to_datetime(row['date']) - 
                                                   pd.to_datetime(history[-1]['date'])).days
            else:
                df.at[idx, 'recent_wins'] = 0
                df.at[idx, 'recent_runs'] = 0
                df.at[idx, 'recent_win_rate'] = 0
                df.at[idx, 'avg_recent_position'] = 999
                df.at[idx, 'best_recent_position'] = 999
                df.at[idx, 'days_since_last_run'] = 999
            
            # Add current race to history
            horse_history[horse_id].append({
                'date': row['date'],
                'won': row['won'],
                'position': row['position']
            })
        
        return df
    
    def create_headgear_form_features(self, df):
        """
        Create headgear form features based on previous wins with specific headgear
        """
        headgear_form = defaultdict(lambda: defaultdict(list))
        df_sorted = df.sort_values(['horse_id', 'date'])
        
        # Handle missing headgear values
        df['headgear'] = df['headgear'].fillna('none')
        
        for idx, row in df_sorted.iterrows():
            horse_id = row['horse_id']
            headgear = row['headgear']
            
            prev_headgear_races = headgear_form[horse_id][headgear]
            
            if len(prev_headgear_races) > 0:
                df.at[idx, 'headgear_wins'] = sum(prev_headgear_races)
                df.at[idx, 'headgear_runs'] = len(prev_headgear_races)
                df.at[idx, 'headgear_win_rate'] = sum(prev_headgear_races) / len(prev_headgear_races)
                df.at[idx, 'won_with_headgear_before'] = 1 if sum(prev_headgear_races) > 0 else 0
            else:
                df.at[idx, 'headgear_wins'] = 0
                df.at[idx, 'headgear_runs'] = 0
                df.at[idx, 'headgear_win_rate'] = 0
                df.at[idx, 'won_with_headgear_before'] = 0
            
            headgear_form[horse_id][headgear].append(row['won'])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:

        # 1) Baseline numeric/categorical defaults
        numeric_defaults = {
            'weight_lbs': 0,
            'course_runs': 0,
            'days_since_last_run': 0,
            'avg_recent_position': 0,
            'best_recent_position': 0,
            'dist_m': 0,
            'ofr': 0,
            'rpr': 0,
            'ts': 0,
        }
        for col, default in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default

        # Categorical defaults present before building features
        for col in ['headgear', 'going', 'surface', 'type', 'sex']:
            if col not in df.columns:
                df[col] = 'Unknown'

        # 2) Build historical/form features 
        print("Creating course form features...")
        df = self.create_course_form_features(df)

        print("Creating distance form features...")
        df = self.create_distance_form_features(df)

        print("Creating going form features...")
        df = self.create_going_form_features(df)

        print("Creating past performance features...")
        if "position" in df.columns:
            print("Creating past performance features...")
            df = self.create_past_performance_features(df)
        else:
            print("Skipping past performance features (no 'position' column).")

        print("Creating headgear form features...")
        df = self.create_headgear_form_features(df)

        # 3) Define the canonical feature list
        feature_columns_master = [
            # Course form
            'course_wins', 'course_runs', 'course_win_rate',

            # Distance form
            'distance_wins', 'distance_runs', 'distance_win_rate',

            # Going form
            'going_wins', 'going_runs', 'going_win_rate',

            # Past performance
            'recent_wins', 'recent_runs', 'recent_win_rate',
            'avg_recent_position', 'best_recent_position', 'days_since_last_run',

            # Headgear form
            'headgear_wins', 'headgear_runs', 'headgear_win_rate', 'won_with_headgear_before',

            # Additional features
            'sp_dec',  # Starting price
            'age',
            'weight_lbs',
            'draw',

            # Categorical features
            'going', 'distance_band', 'headgear', 'sex', 'surface'
        ]

        # Only set once during training; keep fixed for prediction
        if self.feature_columns is None:
            self.feature_columns = feature_columns_master

        # 4) Ensure every expected feature exists (safe defaults)
        categorical_features = {'going', 'distance_band', 'headgear', 'sex', 'surface'}

        for col in self.feature_columns:
            if col not in df.columns:
                # Sensible defaults by name pattern/type
                if col in categorical_features:
                    df[col] = 'unknown'
                    print(f"Filling missing categorical feature '{col}' with 'unknown'")
                elif col.endswith('rate'):
                    df[col] = 0.0
                    print(f"Filling missing numeric rate feature '{col}' with 0.0")
                elif col.endswith('wins') or col.endswith('runs'):
                    df[col] = 0
                    print(f"Filling missing count feature '{col}' with 0")
                elif col in ['avg_recent_position', 'best_recent_position', 'days_since_last_run']:
                    df[col] = 999
                    print(f"Filling missing recency/position feature '{col}' with 999")
                elif col == 'sp_dec':
                    # If sp_dec exists partially, try its median; else constant fallback
                    if 'sp_dec' in df.columns:
                        df['sp_dec'] = pd.to_numeric(df['sp_dec'], errors='coerce')
                        default_sp = df['sp_dec'].median()
                        df[col] = default_sp if not np.isnan(default_sp) else 10.0
                    else:
                        df[col] = 10.0
                    print(f"Filling missing odds feature 'sp_dec' with {df[col] if 'sp_dec' in df.columns else 10.0}")
                else:
                    df[col] = 0
                    print(f"Filling missing numeric feature '{col}' with 0")

        # 5) Final type coercion & NA handling
        # Coerce numerics
        numeric_cols = [c for c in self.feature_columns if c not in categorical_features]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Median-fill numeric NaNs (robust to empty/constant columns)
        if numeric_cols:
            medians = df[numeric_cols].median(numeric_only=True)
            df[numeric_cols] = df[numeric_cols].fillna(medians)

        # Clean categorical
        for c in categorical_features:
            if c in df.columns:
                df[c] = df[c].astype(str).fillna('unknown').replace('', 'unknown')
            else:
                df[c] = 'unknown'

        # 6) Output aligned frame
        out_cols = [c for c in self.feature_columns if c in df.columns]
        if 'won' in df.columns:
            return df[out_cols + ['won']]
        return df[out_cols]

    
    def create_preprocessor(self, df):
        """
        Create preprocessing pipeline
        """
        # Identify categorical and numerical columns
        categorical_features = []
        numerical_features = []
        
        for col in self.feature_columns:
            if df[col].dtype == 'object':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Create preprocessing steps
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def train_model(self, df):
        """
        Train the prediction model
        """
        # Prepare data
        df_processed = self.prepare_features(df)
        
        # Separate features and target
        X = df_processed.drop('won', axis=1)
        y = df_processed['won']
        
        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {y.value_counts()}")
        
        # Create preprocessor
        self.preprocessor = self.create_preprocessor(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Try different models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        best_score = 0
        best_model_name = None
        results = {}
        
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)
            
            # Get probabilities for AUC score
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'train_score': train_score,
                'test_score': test_score,
                'auc_score': auc_score,
                'pipeline': pipeline
            }
            
            print(f"{name}:")
            print(f"  Train Accuracy: {train_score:.4f}")
            print(f"  Test Accuracy: {test_score:.4f}")
            print(f"  AUC Score: {auc_score:.4f}")
            print()
            
            if auc_score > best_score:
                best_score = auc_score
                best_model_name = name
        
        # Select best model
        self.model = results[best_model_name]['pipeline']
        print(f"Best model: {best_model_name} (AUC: {best_score:.4f})")
        
        # Detailed evaluation of best model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("DETAILED EVALUATION OF BEST MODEL")
        print(classification_report(y_test, y_pred))
        
        return self.model, results
    
    def predict_win_probability(self, df):
        """
        Predict win probabilities for new data (does not require 'won' column).
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Add dummy 'won' column if missing (required for form feature calculations)
        # Convert distance -> meters (handles strings like "2m" = 2 miles)
        if 'dist_m' not in df.columns:
            if 'distance' in df.columns:
                def parse_distance(d):
                    if pd.isna(d):
                        return 0.0
                    s = str(d).strip().lower()
                    # If it's something like "2m" treat as miles
                    if s.endswith('m') and s.replace('m', '').replace('.', '', 1).isdigit():
                        return float(s.replace('m', '')) * 1609.34
                    # else try direct float (already meters)
                    try:
                        return float(s)
                    except Exception:
                        return 0.0
                df['dist_m'] = df['distance'].apply(parse_distance)
            else:
                df['dist_m'] = 0.0

        if 'last_run' in df.columns and 'days_since_last_run' not in df.columns:
            def parse_last_run(x):
                if pd.isna(x):
                    return 0
                s = str(x).strip()
                # Extract numeric part only (ignore things like "(30J)")
                num = ''.join(ch for ch in s if (ch.isdigit() or ch == '.'))
                return float(num) if num else 0

            df['days_since_last_run'] = df['last_run'].apply(parse_last_run)
        else:
            df['days_since_last_run'] = 0

        if 'lbs' in df.columns and 'weight_lbs' not in df.columns:
            df['weight_lbs'] = pd.to_numeric(df['lbs'], errors='coerce').fillna(0).astype(float)
        
        if 'ofr' in df.columns and 'or' not in df.columns:
            df['or'] = pd.to_numeric(df['ofr'], errors='coerce').fillna(0).astype(float)

        if 'ts' in df.columns and 'tsr' not in df.columns:
            df['tsr'] = pd.to_numeric(df['ts'], errors='coerce').fillna(0).astype(float)


        if 'won' not in df.columns:
            df['won'] = 0

        # Prepare features
        df_processed = self.prepare_features(df)
        X = df_processed.drop('won', axis=1, errors='ignore')
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X)[:, 1]

        df['win_probability_normalized'] = df['win_probability'] / df['win_probability'].sum()
        
        # Return as a DataFrame with horse identifiers
        horse_col = 'horse_name' if 'horse_name' in df.columns else 'horse'
        col = [horse_col, "race_id", "horse_id", "win_probability_normalized"]
        result_df = df[col].copy()
        result_df['win_probability'] = probabilities 

        # return race_df[[horse_col, 'horse_id', 'race_id', 'win_probability', 'win_probability_normalized']].sort_values(
        #     'win_probability_normalized', ascending=False
        # )   
        
        return result_df.sort_values(
            'win_probability_normalized', ascending=False
        )

    def predict_race_probabilities(self, df, race_id):
        """
        Predict normalized win probabilities for each horse in a specific race.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Filter for the given race
    
        race_df = df[df['race_id'] == race_id].copy()
        if race_df.empty:
            raise ValueError(f"No horses found for race_id {race_id}")

        if 'dist_m' not in df.columns:
            if 'distance' in df.columns:
                def parse_distance(d):
                    if pd.isna(d):
                        return 0.0
                    s = str(d).strip().lower()
                    # If it's something like "2m" treat as miles
                    if s.endswith('m') and s.replace('m', '').replace('.', '', 1).isdigit():
                        return float(s.replace('m', '')) * 1609.34
                    # else try direct float (already meters)
                    try:
                        return float(s)
                    except Exception:
                        return 0.0
                df['dist_m'] = df['distance'].apply(parse_distance)
            else:
                df['dist_m'] = 0.0

        if 'last_run' in df.columns and 'days_since_last_run' not in df.columns:
            def parse_last_run(x):
                if pd.isna(x):
                    return 0
                s = str(x).strip()
                # Extract numeric part only (ignore things like "(30J)")
                num = ''.join(ch for ch in s if (ch.isdigit() or ch == '.'))
                return float(num) if num else 0

            df['days_since_last_run'] = df['last_run'].apply(parse_last_run)
        else:
            df['days_since_last_run'] = 0

        if 'lbs' in df.columns and 'weight_lbs' not in df.columns:
            df['weight_lbs'] = df['lbs'].astype(float)
        
        if 'ofr' in df.columns and 'or' not in df.columns:
            df['or'] = pd.to_numeric(df['ofr'], errors='coerce').fillna(0).astype(float)

        if 'ts' in df.columns and 'tsr' not in df.columns:
            df['tsr'] = pd.to_numeric(df['ts'], errors='coerce').fillna(0).astype(float)

        # Add dummy 'won' column if missing
        if 'won' not in race_df.columns:
            race_df['won'] = 0

        # Prepare features
        race_features = self.prepare_features(race_df)
        X = race_features.drop('won', axis=1, errors='ignore')
        if X.empty:
            raise ValueError(f"Race {race_id} has no usable feature columns after preprocessing.")
        
        # Predict probabilities
        race_df['win_probability'] = self.model.predict_proba(X)[:, 1]
        
        # Normalize to sum to 1 within the race
        race_df['win_probability_normalized'] = race_df['win_probability'] / race_df['win_probability'].sum()
        
        # Return horse identifier + probabilities
        horse_col = 'horse_name' if 'horse_name' in race_df.columns else 'horse'
        return race_df[[horse_col, 'horse_id', 'race_id', 'win_probability', 'win_probability_normalized']].sort_values(
            'win_probability_normalized', ascending=False
        )
    
    
    def feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Get feature names after preprocessing
        feature_names = (self.model.named_steps['preprocessor']
                        .named_transformers_['num'].feature_names_in_.tolist() +
                        self.model.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .named_steps['onehot'].get_feature_names_out().tolist())
        
        # Get importance scores
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importances = self.model.named_steps['classifier'].feature_importances_
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(20)
        else:
            print("Feature importance not available for this model type")
            return None
    

    def save_model(self, path: str = MODEL_PATH):
        """
        Save trained model and feature columns
        """
        if self.model is None:
            raise ValueError("No model trained to save.")
        
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns
        }
        joblib.dump(payload, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = MODEL_PATH):
        """
        Load trained model and feature columns
        """
        if not os.path.exists(path):
            print("No saved model found.")
            return False
        
        payload = joblib.load(path)
        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]
        print(f"Model loaded from {path}")
        return True

def usage():

    # Initialize predictor
    predictor = HorseRacingPredictor()
    
    # Load your data (replace with your actual data loading)
    df = pd.read_csv('merged_results.csv')
    df = predictor.load_and_prepare_data(df)
    
    # Train the model
    model, results = predictor.train_model(df)
    
    
    # View feature importance
    importance = predictor.feature_importance()
    print(importance)

if __name__ == "__main__":
    usage()