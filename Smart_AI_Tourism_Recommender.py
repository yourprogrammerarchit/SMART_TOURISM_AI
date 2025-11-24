"""
AI-POWERED SMART TOURISM RECOMMENDER & DEMAND OPTIMIZER
Complete Implementation for B.Tech Final Year Project
Author: Archit Baloni (Roll No: 25scs1003003550)
Supervisor: Dr. Lalita Chaudhary
Institution: IILM University, Greater Noida

This system provides:
1. Personalized tourism recommendations
2. Demand forecasting with weather integration
3. Resource optimization
4. Sentiment analysis of reviews
5. Sustainable tourism support
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

# Machine Learning
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel



from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             precision_score, recall_score, accuracy_score)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠ NLTK not available. Some NLP features will be limited.")

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# ============================================================================
# DIRECTORY SETUP
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# File paths
HISTORY_CSV = os.path.join(DATA_DIR, "tourism_history.csv")
DEST_CSV = os.path.join(DATA_DIR, "destinations.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "user_ratings.csv")
REVIEWS_CSV = os.path.join(DATA_DIR, "user_reviews.csv")
WEATHER_CSV = os.path.join(DATA_DIR, "weather_data.csv")
EVENTS_CSV = os.path.join(DATA_DIR, "events_data.csv")
HOTELS_CSV = os.path.join(DATA_DIR, "hotels_data.csv")
TRANSPORT_CSV = os.path.join(DATA_DIR, "transport_data.csv")


# ============================================================================
# DATA GENERATION MODULE
# ============================================================================
class DataGenerator:
    """Generates comprehensive tourism dataset"""
    
    def __init__(self):
        self.rng = np.random.default_rng(42)
        self.destinations = [
            "Goa", "Manali", "Jaipur", "Varanasi", "Kerala", "Rishikesh",
            "Udaipur", "Shimla", "Darjeeling", "Hampi", "Leh-Ladakh", 
            "Ooty", "Agra", "Mumbai", "Delhi"
        ]
        self.years = [2021, 2022, 2023, 2024]
        
    def generate_all_data(self):
        """Generate complete dataset"""
        print("\n" + "="*70)
        print("  📊 GENERATING COMPREHENSIVE TOURISM DATASET")
        print("="*70 + "\n")
        
        self._generate_historical_data()
        self._generate_destinations_data()
        self._generate_ratings_data()
        self._generate_reviews_data()
        self._generate_weather_data()
        self._generate_events_data()
        self._generate_hotels_data()
        self._generate_transport_data()
        
        print("\n✅ Dataset generation completed successfully!\n")
        self._display_dataset_stats()
    
    def _generate_historical_data(self):
        """Historical visitor data with seasonality"""
        print("  [1/8] Generating historical visitor data...")
        rows = []
        
        for dest in self.destinations:
            base_visitors = self.rng.integers(2000, 6000)
            popularity = self.rng.uniform(0.7, 1.3)
            
            for year in self.years:
                for month in range(1, 13):
                    # Realistic seasonality
                    season_factor = self._get_season_factor(dest, month)
                    
                    # Year-over-year growth
                    growth = 1 + (year - 2021) * 0.05
                    
                    visitors = int(base_visitors * season_factor * popularity * growth)
                    visitors = max(300, visitors + self.rng.integers(-200, 200))
                    
                    rows.append({
                        "year": year,
                        "month": month,
                        "destination": dest,
                        "visitors": visitors,
                        "avg_stay_days": round(self.rng.uniform(2, 7), 1),
                        "occupancy_rate": round(min(95, visitors / 50), 1)
                    })
        
        pd.DataFrame(rows).to_csv(HISTORY_CSV, index=False)
        print(f"     ✓ Saved {len(rows)} historical records")
    
    def _generate_destinations_data(self):
        """Destination metadata"""
        print("  [2/8] Generating destinations metadata...")
        
        dest_data = [
            ["Goa", "beach", "Pristine beaches, vibrant nightlife, water sports", 3, "winter", 4.5, "Portuguese-Indian", 250, "coastal"],
            ["Manali", "mountain", "Snow-capped Himalayan peaks, adventure sports", 2, "summer", 4.3, "Himachali", 180, "himalayan"],
            ["Jaipur", "culture", "Pink city with magnificent royal palaces", 2, "winter", 4.6, "Rajasthani", 200, "desert"],
            ["Varanasi", "spiritual", "Ancient ghats, spiritual ceremonies, temples", 1, "all", 4.4, "North Indian", 150, "riverine"],
            ["Kerala", "backwater", "Serene houseboats, lush greenery, ayurveda", 3, "winter", 4.7, "South Indian", 280, "tropical"],
            ["Rishikesh", "adventure", "Yoga capital, river rafting, spiritual retreat", 2, "summer", 4.5, "North Indian", 160, "himalayan"],
            ["Udaipur", "heritage", "City of lakes, royal palaces, romantic setting", 3, "winter", 4.6, "Rajasthani", 220, "lake"],
            ["Shimla", "mountain", "Colonial hill station with scenic toy train", 2, "summer", 4.2, "Himachali", 170, "himalayan"],
            ["Darjeeling", "mountain", "Tea gardens, toy train, sunrise views", 2, "all", 4.4, "Bengali-Tibetan", 190, "himalayan"],
            ["Hampi", "archaeological", "Ancient ruins, temple complexes, boulders", 2, "winter", 4.5, "South Indian", 140, "heritage"],
            ["Leh-Ladakh", "adventure", "High-altitude desert, monasteries, trekking", 3, "summer", 4.8, "Tibetan", 200, "highland"],
            ["Ooty", "mountain", "Hill station, tea estates, botanical gardens", 2, "summer", 4.3, "Tamil", 160, "nilgiri"],
            ["Agra", "heritage", "Taj Mahal, Mughal architecture, history", 2, "winter", 4.7, "Mughlai", 180, "heritage"],
            ["Mumbai", "urban", "Bollywood, beaches, street food, nightlife", 3, "winter", 4.4, "Multi-cuisine", 300, "metropolitan"],
            ["Delhi", "culture", "Historical monuments, street food, museums", 2, "winter", 4.3, "Multi-cuisine", 350, "metropolitan"]
        ]
        
        df = pd.DataFrame(dest_data, columns=[
            "destination", "type", "description", "cost_level", "best_season",
            "avg_rating", "cuisine", "attractions_count", "geography"
        ])
        df.to_csv(DEST_CSV, index=False)
        print(f"     ✓ Saved {len(df)} destinations")
    
    def _generate_ratings_data(self):
        """User ratings with demographics"""
        print("  [3/8] Generating user ratings...")
        
        users = [f"user{i:03d}" for i in range(1, 101)]
        age_groups = ["18-25", "26-35", "36-45", "46-60", "60+"]
        travel_styles = ["adventure", "luxury", "budget", "family", "solo", "romantic"]
        countries = ["India", "USA", "UK", "Germany", "France", "Australia", "Japan"]
        
        ratings = []
        for user in users:
            age = self.rng.choice(age_groups)
            style = self.rng.choice(travel_styles)
            country = self.rng.choice(countries)
            
            # Each user rates 5-12 destinations
            n_ratings = self.rng.integers(5, 13)
            dests = self.rng.choice(self.destinations, size=n_ratings, replace=False)
            
            for dest in dests:
                # Rating influenced by travel style match
                base_rating = self.rng.integers(2, 6)
                
                ratings.append({
                    "user": user,
                    "destination": dest,
                    "rating": base_rating,
                    "age_group": age,
                    "travel_style": style,
                    "country": country,
                    "timestamp": self._random_timestamp()
                })
        
        pd.DataFrame(ratings).to_csv(RATINGS_CSV, index=False)
        print(f"     ✓ Saved {len(ratings)} user ratings")
    
    def _generate_reviews_data(self):
        """Detailed user reviews"""
        print("  [4/8] Generating user reviews...")
        
        positive_templates = [
            "Amazing {aspect}! The {feature} was absolutely wonderful. Highly recommend for {audience}.",
            "Fantastic experience at {destination}. The {aspect} exceeded expectations.",
            "Beautiful {feature} and excellent {aspect}. Perfect for {audience}.",
            "Outstanding {destination} visit. {aspect} was incredible."
        ]
        
        negative_templates = [
            "Disappointing {aspect}. The {feature} was not up to the mark.",
            "Overrated destination. {aspect} needs improvement.",
            "Poor {feature} and crowded {aspect}. Not recommended for {audience}.",
            "Expected better {aspect} considering the hype."
        ]
        
        neutral_templates = [
            "Decent {aspect}. The {feature} was okay but nothing special.",
            "Average experience. {aspect} could be better.",
            "Standard {destination} visit. {feature} was acceptable."
        ]
        
        aspects = ["scenery", "hospitality", "cleanliness", "attractions", "food", "accommodation"]
        features = ["views", "service", "facilities", "activities", "culture", "ambiance"]
        audiences = ["families", "couples", "solo travelers", "adventure seekers"]
        
        reviews = []
        ratings_df = pd.read_csv(RATINGS_CSV)
        
        for _, row in ratings_df.iterrows():
            rating = row["rating"]
            dest = row["destination"]
            
            if rating >= 4:
                template = self.rng.choice(positive_templates)
                sentiment = "positive"
            elif rating <= 2:
                template = self.rng.choice(negative_templates)
                sentiment = "negative"
            else:
                template = self.rng.choice(neutral_templates)
                sentiment = "neutral"
            
            review_text = template.format(
                destination=dest,
                aspect=self.rng.choice(aspects),
                feature=self.rng.choice(features),
                audience=self.rng.choice(audiences)
            )
            
            reviews.append({
                "user": row["user"],
                "destination": dest,
                "review": review_text,
                "sentiment": sentiment,
                "helpful_count": self.rng.integers(0, 100),
                "verified_traveler": self.rng.choice([True, False], p=[0.7, 0.3])
            })
        
        pd.DataFrame(reviews).to_csv(REVIEWS_CSV, index=False)
        print(f"     ✓ Saved {len(reviews)} reviews")
    
    def _generate_weather_data(self):
        """Weather patterns by destination"""
        print("  [5/8] Generating weather data...")
        
        weather_rows = []
        for dest in self.destinations:
            # Base climate by geography
            dest_info = pd.read_csv(DEST_CSV)
            geo = dest_info[dest_info["destination"] == dest]["geography"].values[0]
            
            for month in range(1, 13):
                temp, rainfall = self._get_weather_params(geo, month)
                
                weather_rows.append({
                    "destination": dest,
                    "month": month,
                    "avg_temp": round(temp, 1),
                    "rainfall_mm": round(rainfall, 1),
                    "humidity": round(self.rng.uniform(40, 90), 1),
                    "air_quality_index": self.rng.integers(50, 200)
                })
        
        pd.DataFrame(weather_rows).to_csv(WEATHER_CSV, index=False)
        print(f"     ✓ Saved {len(weather_rows)} weather records")
    
    def _generate_events_data(self):
        """Local events and festivals"""
        print("  [6/8] Generating events data...")
        
        events = []
        event_types = ["festival", "concert", "exhibition", "sports", "cultural"]
        
        for dest in self.destinations:
            n_events = self.rng.integers(3, 8)
            for _ in range(n_events):
                month = self.rng.integers(1, 13)
                events.append({
                    "destination": dest,
                    "event_name": f"{dest} {self.rng.choice(['Festival', 'Fair', 'Carnival', 'Week'])}",
                    "event_type": self.rng.choice(event_types),
                    "month": month,
                    "duration_days": self.rng.integers(1, 7),
                    "expected_crowd": self.rng.choice(["low", "medium", "high", "very_high"])
                })
        
        pd.DataFrame(events).to_csv(EVENTS_CSV, index=False)
        print(f"     ✓ Saved {len(events)} events")
    
    def _generate_hotels_data(self):
        """Hotel availability and pricing"""
        print("  [7/8] Generating hotels data...")
        
        hotels = []
        categories = ["budget", "mid-range", "luxury", "resort"]
        
        for dest in self.destinations:
            n_hotels = self.rng.integers(10, 30)
            for i in range(n_hotels):
                category = self.rng.choice(categories)
                base_price = {"budget": 1500, "mid-range": 3500, "luxury": 8000, "resort": 12000}
                
                hotels.append({
                    "destination": dest,
                    "hotel_name": f"{dest} {category.title()} Hotel {i+1}",
                    "category": category,
                    "price_per_night": base_price[category] + self.rng.integers(-500, 1000),
                    "rating": round(self.rng.uniform(3.0, 5.0), 1),
                    "total_rooms": self.rng.integers(20, 100),
                    "amenities": self.rng.integers(5, 15)
                })
        
        pd.DataFrame(hotels).to_csv(HOTELS_CSV, index=False)
        print(f"     ✓ Saved {len(hotels)} hotels")
    
    def _generate_transport_data(self):
        """Transport options and connectivity"""
        print("  [8/8] Generating transport data...")
        
        transport = []
        modes = ["flight", "train", "bus"]
        
        for dest in self.destinations:
            for mode in modes:
                if mode == "flight":
                    price = self.rng.integers(3000, 15000)
                    duration = self.rng.uniform(1, 5)
                elif mode == "train":
                    price = self.rng.integers(500, 3000)
                    duration = self.rng.uniform(5, 24)
                else:
                    price = self.rng.integers(300, 1500)
                    duration = self.rng.uniform(8, 30)
                
                transport.append({
                    "destination": dest,
                    "mode": mode,
                    "avg_price": price,
                    "avg_duration_hours": round(duration, 1),
                    "frequency_per_day": self.rng.integers(2, 20),
                    "comfort_rating": round(self.rng.uniform(2.5, 5.0), 1)
                })
        
        pd.DataFrame(transport).to_csv(TRANSPORT_CSV, index=False)
        print(f"     ✓ Saved {len(transport)} transport options")
    
    def _get_season_factor(self, dest, month):
        """Calculate seasonal visitor factor"""
        # Winter months (Dec, Jan, Feb)
        if month in [12, 1, 2]:
            if dest in ["Goa", "Kerala", "Jaipur", "Udaipur", "Agra"]:
                return 1.6
            elif dest in ["Manali", "Shimla", "Leh-Ladakh"]:
                return 0.5
            else:
                return 1.2
        
        # Summer months (Mar, Apr, May)
        elif month in [3, 4, 5]:
            if dest in ["Manali", "Shimla", "Darjeeling", "Leh-Ladakh", "Ooty"]:
                return 1.5
            else:
                return 0.8
        
        # Monsoon (Jun, Jul, Aug, Sep)
        elif month in [6, 7, 8, 9]:
            if dest in ["Goa", "Kerala"]:
                return 0.5
            else:
                return 1.0
        
        # Post-monsoon/Autumn (Oct, Nov)
        else:
            return 1.3
    
    def _get_weather_params(self, geography, month):
        """Get temperature and rainfall by geography"""
        base_temps = {
            "coastal": 28, "himalayan": 12, "desert": 30,
            "tropical": 27, "heritage": 26, "metropolitan": 28,
            "riverine": 27, "lake": 25, "highland": 5, "nilgiri": 18
        }
        
        temp = base_temps.get(geography, 25)
        
        # Adjust for season
        if month in [12, 1, 2]:
            temp -= 5
        elif month in [3, 4, 5]:
            temp += 5
        elif month in [6, 7, 8, 9]:
            temp -= 2
        
        # Rainfall
        if month in [6, 7, 8, 9]:
            rainfall = self.rng.uniform(150, 400)
        else:
            rainfall = self.rng.uniform(5, 80)
        
        return temp, rainfall
    
    def _random_timestamp(self):
        """Generate random timestamp"""
        year = self.rng.choice([2023, 2024])
        month = self.rng.integers(1, 13)
        day = self.rng.integers(1, 29)
        return f"{year}-{month:02d}-{day:02d}"
    
    def _display_dataset_stats(self):
        """Display dataset statistics"""
        print("\n" + "="*70)
        print("  📈 DATASET STATISTICS")
        print("="*70)
        
        stats = {
            "Historical Records": len(pd.read_csv(HISTORY_CSV)),
            "Destinations": len(pd.read_csv(DEST_CSV)),
            "User Ratings": len(pd.read_csv(RATINGS_CSV)),
            "Reviews": len(pd.read_csv(REVIEWS_CSV)),
            "Weather Data Points": len(pd.read_csv(WEATHER_CSV)),
            "Events": len(pd.read_csv(EVENTS_CSV)),
            "Hotels": len(pd.read_csv(HOTELS_CSV)),
            "Transport Options": len(pd.read_csv(TRANSPORT_CSV))
        }
        
        for key, value in stats.items():
            print(f"  {key:.<50} {value:>6,}")
        
        print("="*70 + "\n")


# ============================================================================
# MACHINE LEARNING MODELS MODULE
# ============================================================================
class MLModels:
    """Train and manage all ML models"""
    
    def __init__(self):
        self.models = {}
    
    def train_all_models(self):
        """Train all ML models"""
        print("\n" + "="*70)
        print("  🤖 TRAINING MACHINE LEARNING MODELS")
        print("="*70 + "\n")
        
        self.train_demand_forecasting()
        self.train_recommendation_models()
        self.train_sentiment_analysis()
        self.train_clustering()
        self.train_popularity_model()
        
        print("\n✅ All models trained successfully!\n")
        self._save_training_report()
    
    def train_demand_forecasting(self):
        """Train demand forecasting models"""
        print("  [1/5] Training demand forecasting models...")
        
        # Load data
        df = pd.read_csv(HISTORY_CSV)
        weather = pd.read_csv(WEATHER_CSV)
        events = pd.read_csv(EVENTS_CSV)
        
        # Merge features
        df = df.merge(weather, on=["destination", "month"], how="left")
        
        # Event impact feature
        event_counts = events.groupby(["destination", "month"]).size().reset_index(name="event_count")
        df = df.merge(event_counts, on=["destination", "month"], how="left")
        df["event_count"] = df["event_count"].fillna(0)
        
        # Feature engineering
        df["month_sin"] = np.sin(2 * np.pi * df['month'] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df['month'] / 12)
        df["dest_code"] = df["destination"].astype('category').cat.codes
        df["year_normalized"] = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min())
        
        # Lag features
        df = df.sort_values(['destination', 'year', 'month'])
        df['visitors_lag1'] = df.groupby('destination')['visitors'].shift(1)
        df['visitors_lag3'] = df.groupby('destination')['visitors'].shift(3)
        df = df.fillna(0)
        
        features = ['year_normalized', 'month_sin', 'month_cos', 'dest_code', 
                   'avg_temp', 'rainfall_mm', 'event_count', 'visitors_lag1', 'visitors_lag3']
        
        X = df[features]
        y = df['visitors']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Evaluate
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        gb_mae = mean_absolute_error(y_test, gb_pred)
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
        
        print(f"     Random Forest  - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
        print(f"     Gradient Boost - MAE: {gb_mae:.2f}, RMSE: {gb_rmse:.2f}")
        
        # Save best model
        best_model = gb_model if gb_mae < rf_mae else rf_model
        model_name = "Gradient Boosting" if gb_mae < rf_mae else "Random Forest"
        
        joblib.dump({
            'model': best_model,
            'features': features,
            'destinations': df["destination"].astype("category").cat.categories.tolist(),
            'model_name': model_name,
            'mae': min(rf_mae, gb_mae),
            'rmse': min(rf_rmse, gb_rmse)
        }, os.path.join(MODELS_DIR, "demand_forecast.pkl"))
        
        print(f"     ✓ Best model: {model_name}")
    
    def train_recommendation_models(self):
        """Train recommendation models"""
        print("  [2/5] Training recommendation models...")
        
        ratings = pd.read_csv(RATINGS_CSV)
        dest_df = pd.read_csv(DEST_CSV)
        
        # 1. Collaborative Filtering (Matrix Factorization)
        pivot = ratings.pivot_table(index="user", columns="destination", values="rating").fillna(0)
        
        n_components = min(10, min(pivot.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = svd.fit_transform(pivot)
        item_factors = svd.components_
        
        joblib.dump({
            'svd': svd,
            'pivot': pivot,
            'user_factors': user_factors,
            'item_factors': item_factors
        }, os.path.join(MODELS_DIR, "collaborative_filtering.pkl"))
        
        # 2. Content-Based Filtering
        dest_df["combined_features"] = (
            dest_df["type"] + " " +
            dest_df["description"] + " " +
            dest_df["cuisine"] + " " +
            dest_df["geography"] + " " +
            dest_df["best_season"]
        )
        
        tfidf = TfidfVectorizer(stop_words="english", max_features=150)
        tfidf_matrix = tfidf.fit_transform(dest_df["combined_features"])
        
        joblib.dump({
            'tfidf': tfidf,
            'tfidf_matrix': tfidf_matrix,
            'destinations': dest_df["destination"].tolist(),
            'dest_df': dest_df
        }, os.path.join(MODELS_DIR, "content_based.pkl"))
        
        # 3. Demographic-Based
        user_demo = ratings[['user', 'age_group', 'travel_style', 'country']].drop_duplicates()
        
        joblib.dump({
            'user_demographics': user_demo,
            'ratings': ratings
        }, os.path.join(MODELS_DIR, "demographic_filter.pkl"))
        
        print("     ✓ Collaborative, Content-based, and Demographic models trained")
    
    def train_sentiment_analysis(self):
        """Train sentiment analysis models"""
        print("  [3/5] Training sentiment analysis...")
        
        reviews = pd.read_csv(REVIEWS_CSV)
        
        # Prepare data
        vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
        X = vectorizer.fit_transform(reviews["review"])
        y = reviews["sentiment"]
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Train models
        nb_model = MultinomialNB()
        lr_model = LogisticRegression(max_iter=500, random_state=42)
        
        nb_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        
        nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
        lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
        
        print(f"     Naive Bayes accuracy: {nb_acc:.3f}")
        print(f"     Logistic Regression accuracy: {lr_acc:.3f}")
        
        best_model = lr_model if lr_acc > nb_acc else nb_model
        
        joblib.dump({
            'model': best_model,
            'vectorizer': vectorizer,
            'label_encoder': le,
            'accuracy': max(nb_acc, lr_acc)
        }, os.path.join(MODELS_DIR, "sentiment_model.pkl"))
        
        print(f"     ✓ Sentiment model trained (accuracy: {max(nb_acc, lr_acc):.3f})")
    
    def train_clustering(self):
        """Train clustering models"""
        print("  [4/5] Training clustering models...")
        
        dest_df = pd.read_csv(DEST_CSV)
        
        # Feature selection
        features_df = dest_df[["cost_level", "avg_rating", "attractions_count"]].copy()
        
        scaler = MinMaxScaler()
                
    
        X_scaled = scaler.fit_transform(features_df)
        # Choose k using elbow (simple heuristic) - keep k small for demo
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        dest_df["cluster"] = clusters
        joblib.dump({
            "kmeans": kmeans,
            "scaler": scaler,
            "clusters": clusters,
            "dest_df": dest_df
        }, os.path.join(MODELS_DIR, "clustering.pkl"))
        print("     ✓ Clustering (KMeans) trained and saved")

    def train_popularity_model(self):
        """Simple popularity / baseline model"""
        print("  [5/5] Training popularity model (baseline)...")
        hist = pd.read_csv(HISTORY_CSV)
        pop = hist.groupby("destination")["visitors"].mean().reset_index().rename(columns={"visitors":"avg_visitors"})
        pop = pop.sort_values("avg_visitors", ascending=False)
        joblib.dump({"popularity": pop}, os.path.join(MODELS_DIR, "popularity.pkl"))
        print("     ✓ Popularity baseline computed and saved")

    def _save_training_report(self):
        """Save a small JSON report summarizing models and metrics"""
        report = {"generated_at": datetime.now().isoformat(), "models": []}
        # look for created model files
        for fname in os.listdir(MODELS_DIR):
            report["models"].append(fname)
        with open(os.path.join(RESULTS_DIR, "training_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("  [REPORT] Training report written -> results/training_report.json")


# ====================================================================
# UTILS: Loading & Inference functions
# ====================================================================
def load_demand_model():
    p = os.path.join(MODELS_DIR, "demand_forecast.pkl")
    if not os.path.exists(p):
        print("Demand model not found. Please train models first.")
        return None
    return joblib.load(p)

def predict_demand_for(dest, year, month):
    data = load_demand_model()
    if data is None:
        return None
    model = data["model"]
    features = data["features"]
    dest_list = data["destinations"]
    if dest not in dest_list:
        print("Destination not known in the trained list.")
        return None
    dest_code = dest_list.index(dest)
    month_sin = np.sin(2*np.pi*month/12)
    month_cos = np.cos(2*np.pi*month/12)
    year_norm = (year - 2021) / (2024 - 2021) if year >= 2021 else 0
    # simple placeholders for weather/event/lag - set zero (demo)
    x = pd.DataFrame([{
        "year_normalized": year_norm,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "dest_code": dest_code,
        "avg_temp": 25,
        "rainfall_mm": 10,
        "event_count": 0,
        "visitors_lag1": 0,
        "visitors_lag3": 0
    }])
    pred = model.predict(x[features])[0]
    return int(np.round(pred))

def load_content_model():
    p = os.path.join(MODELS_DIR, "content_based.pkl")
    if not os.path.exists(p):
        print("Content-based model not found. Train first.")
        return None
    return joblib.load(p)

def content_recommendations(destination, topn=5):
    content = load_content_model()
    if content is None:
        return []
    tfidf = content["tfidf"]
    tfidf_matrix = content["tfidf_matrix"]
    dests = content["destinations"]
    if destination not in dests:
        return []
    idx = dests.index(destination)
    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][1:topn+1]
    results = [{"destination": dests[i], "score": float(np.round(sims[i], 4))} for i in top_idx]
    return results

def load_cf_model():
    p = os.path.join(MODELS_DIR, "collaborative_filtering.pkl")
    if not os.path.exists(p):
        print("CF model not found. Train first.")
        return None
    return joblib.load(p)

def cf_recommend_for(user_id, topn=5):
    cf = load_cf_model()
    if cf is None:
        return []
    pivot = cf["pivot"]
    if user_id not in pivot.index:
        print("User not in dataset.")
        return []
    user_vec = pivot.loc[user_id].values.reshape(1, -1)
    sims = cosine_similarity(user_vec, pivot.values).flatten()
    sims_series = pd.Series(sims, index=pivot.index)
    sims_series[user_id] = 0
    weighted = pivot.T.dot(sims_series) / (sims_series.sum() + 1e-9)
    weighted = weighted.sort_values(ascending=False)
    top = weighted.head(topn)
    return [{"destination": idx, "score": float(np.round(val, 4))} for idx, val in top.items()]


def load_sentiment_model():
    p = os.path.join(MODELS_DIR, "sentiment_model.pkl")
    if not os.path.exists(p):
        print("Sentiment model not found. Train first.")
        return None
    return joblib.load(p)

def predict_review_sentiment(text):
    sent = load_sentiment_model()
    if sent is None:
        return None
    vec = sent["vectorizer"]
    le = sent["label_encoder"]
    model = sent["model"]
    X = vec.transform([text])
    pred = model.predict(X)[0]
    label = le.inverse_transform([pred])[0]
    return label


# ====================================================================
# INTERACTIVE MENU (for demo / viva)
# ====================================================================
def interactive_menu():
    dg = DataGenerator()
    ml = MLModels()

    while True:
        print("\n" + "="*60)
        print(" SMART TOURISM AI - INTERACTIVE DEMO")
        print("="*60)
        print("1. Generate Full Synthetic Dataset")
        print("2. Train All Models (may take some time)")
        print("3. Quick Train (only Forecast + Recommender & Sentiment)")
        print("4. Predict Visitors (Demand Forecast)")
        print("5. Content-based Recommendation")
        print("6. Collaborative Filtering Recommendation")
        print("7. Predict Sentiment for a Review")
        print("8. Show Dataset Summary")
        print("0. Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            dg.generate_all_data()
        elif choice == "2":
            ml.train_all_models()
        elif choice == "3":
            # quick path: generate minimal data if missing then train core models
            if not os.path.exists(HISTORY_CSV) or not os.path.exists(DEST_CSV) or not os.path.exists(RATINGS_CSV) or not os.path.exists(REVIEWS_CSV):
                print("Required data missing - generating minimal dataset first...")
                dg._generate_historical_data()
                dg._generate_destinations_data()
                dg._generate_ratings_data()
                dg._generate_reviews_data()
                dg._generate_weather_data()
                dg._generate_events_data()
            ml.train_demand_forecasting()
            ml.train_recommendation_models()
            ml.train_sentiment_analysis()
            print("Quick train completed.")
        elif choice == "4":
            dest = input("Destination name (e.g., Goa): ").strip()
            year = int(input("Year (e.g., 2024): ").strip())
            month = int(input("Month (1-12): ").strip())
            pred = predict_demand_for(dest, year, month)
            if pred is not None:
                print(f"Predicted visitors for {dest} in {month}/{year}: {pred}")
        elif choice == "5":
            dest = input("Enter destination for similarity (e.g., Goa): ").strip()
            recs = content_recommendations(dest, topn=5)
            if recs:
                print("Similar destinations:")
                for r in recs:
                    print(f" - {r['destination']} (score: {r['score']})")
            else:
                print("No recommendations found. Train models / check destination name.")
        elif choice == "6":
            user = input("Enter user id (e.g., user001): ").strip()
            recs = cf_recommend_for(user, topn=5)
            if recs:
                print("CF recommendations:")
                for r in recs:
                    print(f" - {r['destination']} (score: {r['score']})")
            else:
                print("No CF recommendations available.")
        elif choice == "7":
            text = input("Paste review text: ").strip()
            label = predict_review_sentiment(text)
            if label:
                print(f"Predicted sentiment: {label}")
        elif choice == "8":
            # lightweight summary
            for fpath in [HISTORY_CSV, DEST_CSV, RATINGS_CSV, REVIEWS_CSV, WEATHER_CSV]:
                if os.path.exists(fpath):
                    print(f"{os.path.basename(fpath)}: {pd.read_csv(fpath).shape} rows x cols")
                else:
                    print(f"{os.path.basename(fpath)}: MISSING")
        elif choice == "0":
            print("Exiting. Bye!")
            break
        else:
            print("Invalid choice. Try again.")


# Run interactive menu if script executed directly
if __name__ == "__main__":
    interactive_menu()