from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value, PULP_CBC_CMD
import re
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the preprocessed dataframe
df = None

# --- Category Mapping using C column (item_category) ---
def map_main_group(row):
    item_category = str(row['item_category']).lower()
    name = str(row['name']).lower()
    
    # Exclude granola items
    if 'granola' in item_category or 'granola' in name:
        return 'exclude'
    
    # Vegetables
    if any(keyword in item_category for keyword in ['sebze', 'domates', 'biber', 'salatalık', 'patates', 'soğan', 'havuç']):
        return 'vegetables'
    if any(keyword in name for keyword in ['domates', 'biber', 'salatalık', 'patates', 'soğan', 'havuç', 'kabak']):
        return 'vegetables'
    
    # Fruits
    if any(keyword in item_category for keyword in ['meyve', 'elma', 'muz', 'portakal', 'armut', 'çilek']):
        return 'fruits'
    if any(keyword in name for keyword in ['elma', 'muz', 'portakal', 'armut', 'çilek', 'kayısı', 'şeftali']):
        return 'fruits'
    
    # Dairy
    if any(keyword in item_category for keyword in ['süt', 'kahvalt', 'peynir', 'yoğurt', 'süt ürünleri']):
        return 'dairy'
    if any(keyword in name for keyword in ['peynir', 'yoğurt', 'süt', 'kaymak', 'krema']):
        return 'dairy'
    
    # Legumes
    if any(keyword in item_category for keyword in ['bakliyat', 'fasulye', 'mercimek', 'nohut', 'bezelye']):
        return 'legumes'
    if any(keyword in name for keyword in ['fasulye', 'mercimek', 'nohut', 'bezelye', 'barbunya']):
        return 'legumes'
    
    # Meat/Fish
    if any(keyword in item_category for keyword in ['et', 'balık', 'tavuk', 'kıyma', 'sucuk']):
        return 'meat_fish'
    if any(keyword in name for keyword in ['tavuk', 'balık', 'kıyma', 'sucuk', 'salam', 'pastırma']):
        return 'meat_fish'
    
    # Grains
    if any(keyword in item_category for keyword in ['temel gıda', 'ekmek', 'bulgur', 'pirinç', 'makarna', 'un']):
        return 'grains'
    if any(keyword in name for keyword in ['ekmek', 'bulgur', 'pirinç', 'makarna', 'un', 'börek']):
        return 'grains'
    
    return 'other'

# --- TDEE Calculation ---
def calculate_tdee(age, gender, weight, height, activity):
    # Basal Metabolic Rate (BMR) calculation
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    # Activity level multipliers
    activity_factors = {
        "sedentary": 1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active": 1.725,
        "extra active": 1.9
    }
    
    tdee = bmr * activity_factors.get(activity, 1.2)
    return tdee

# --- Macro Targets ---
def get_macro_targets(tdee, goal):
    # Adjust TDEE based on goal
    if "gain" in goal:
        tdee += 200
    elif "lose" in goal:
        tdee -= 200
    
    # Macro ratios based on goal
    if "sport" in goal:
        protein_ratio = 0.20
        fat_ratio = 0.25
        carb_ratio = 0.55
    else:
        protein_ratio = 0.15
        fat_ratio = 0.25
        carb_ratio = 0.60
    
    # Calculate macro targets in grams
    protein_g = (tdee * protein_ratio) / 4  # 4 calories per gram of protein
    fat_g = (tdee * fat_ratio) / 9          # 9 calories per gram of fat
    carb_g = (tdee * carb_ratio) / 4        # 4 calories per gram of carbs
    
    return tdee, protein_g, fat_g, carb_g

def extract_weight(name):
    """Extract weight from product name"""
    match = re.search(r"(\d+[.,]?\d*)\s*(kg|g|gr)", name.lower())
    if match:
        value = float(match.group(1).replace(",", "."))
        unit = match.group(2)
        if "kg" in unit:
            return int(value * 1000)  # Convert kg to grams
        else:
            return int(value)  # Already in grams
    return 1000  # Default weight in grams

# --- Data Preprocessing ---
def preprocess_data(df):
    print("Preprocessing data...")
    
    # Clean price column
    df["price"] = df["price"].astype(str).str.replace(" TL", "", regex=False)
    df["price"] = df["price"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    
    # Clean nutrition columns
    for col in ["calories", "protein", "carbs", "fat"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Remove rows with missing or invalid data
    df = df.dropna(subset=["price", "calories", "protein", "carbs", "fat"])
    df = df[(df["price"] > 0) & (df["calories"] >= 0) & (df["protein"] >= 0) & 
            (df["carbs"] >= 0) & (df["fat"] >= 0)]
    
    # Exclude beverages
    df = df[~df["category"].str.lower().str.contains("içecek")]
    
    # Exclude noodles
    df = df[~df["name"].str.lower().str.contains("noodle")]
    df = df[~df["item_category"].str.lower().str.contains("noodle")]
    
    # Exclude liver and heart products
    df = df[~df["name"].str.lower().str.contains("ciğer")]
    df = df[~df["name"].str.lower().str.contains("yürek")]
    df = df[~df["name"].str.lower().str.contains("liver")]
    df = df[~df["name"].str.lower().str.contains("heart")]
    
    # Exclude products containing 'çabuk' or 'bardak' (Turkish only)
    df = df[~df["name"].str.lower().str.contains("çabuk")]
    df = df[~df["name"].str.lower().str.contains("bardak")]
    
    # Exclude Berliner and Kruvasan products
    df = df[~df["name"].str.lower().str.contains("berliner")]
    df = df[~df["name"].str.lower().str.contains("kruvasan")]
    df = df[~df["name"].str.lower().str.contains("croissant")]
    
    # Exclude products containing 'pilavı' and 'çikolata'
    df = df[~df["name"].str.lower().str.contains("pilavı")]
    df = df[~df["name"].str.lower().str.contains("çikolata")]
    
    # Extract weight from product names
    df["weight_g"] = df["name"].apply(extract_weight)
    
    # Apply filters
    df = df[df["weight_g"] <= 5000]  # Max 5kg per item
    df = df[df["price"] <= 1000]     # Max 1000 TL per item
    df = df[df["calories"] > 0]      # Must have calories
    
    # Map categories using C column (item_category)
    df['main_group'] = df.apply(map_main_group, axis=1)
    
    # Exclude granola items
    df = df[df['main_group'] != 'exclude']
    
    print(f"Data preprocessing complete: {len(df)} products available")
    return df

# --- Optimization ---
def optimize_shopping(df, tdee, protein_g, fat_g, carb_g, budget, days=30):
    print(f"Starting optimization with budget: {budget} TL")
    
    # Create optimization problem
    prob = LpProblem("ShoppingList", LpMinimize)
    n = len(df)
    
    # Decision variables: number of each item to buy (0-5)
    items = [LpVariable(f"x_{i}", lowBound=0, upBound=5, cat='Integer') for i in range(n)]
    
    # Binary variables for counting different items
    y = [LpVariable(f"y_{i}", cat='Binary') for i in range(n)]
    
    # Objective: minimize total cost
    prob += lpSum([items[i] * df.iloc[i]["price"] for i in range(n)])
    
    # Nutrition constraints (scaled for days)
    prob += lpSum([items[i] * df.iloc[i]["calories"] for i in range(n)]) >= tdee * days
    prob += lpSum([items[i] * df.iloc[i]["protein"] for i in range(n)]) >= protein_g * days
    prob += lpSum([items[i] * df.iloc[i]["fat"] for i in range(n)]) >= fat_g * days
    prob += lpSum([items[i] * df.iloc[i]["carbs"] for i in range(n)]) >= carb_g * days
    
    # Budget constraints: use at least 70% of budget
    prob += lpSum([items[i] * df.iloc[i]["price"] for i in range(n)]) >= budget * 0.70
    prob += lpSum([items[i] * df.iloc[i]["price"] for i in range(n)]) <= budget
    
    # Category diversity: at least 1 from each main group
    for group in ['vegetables', 'fruits', 'dairy', 'legumes', 'meat_fish', 'grains']:
        indices = [i for i in range(n) if df.iloc[i]['main_group'] == group]
        if indices:
            prob += lpSum([items[i] for i in indices]) >= 1
    
    # Meat/Fish weight constraint: at least 7.5 kg
    meat_indices = [i for i in range(n) if df.iloc[i]['main_group'] == 'meat_fish']
    if meat_indices:
        prob += lpSum([items[i] * df.iloc[i]["weight_g"] for i in meat_indices]) >= 7500
    
    # Pasta weight constraint: maximum 2.5 kg total
    pasta_terms = ['makarna', 'pasta', 'spaghetti', 'penne', 'farfalle', 'rigatoni', 'şehriye', 'erişte']
    pasta_indices = [i for i in range(n) if any(term in df.iloc[i]['name'].lower() for term in pasta_terms)]
    if pasta_indices:
        prob += lpSum([items[i] * df.iloc[i]["weight_g"] for i in pasta_indices]) <= 2500
    
    # Bulgur constraints: maximum 2.5 kg total and maximum 3 different items
    bulgur_terms = ['bulgur', 'bulguru', 'bulgurlu']
    bulgur_indices = [i for i in range(n) if any(term in df.iloc[i]['name'].lower() for term in bulgur_terms)]
    if bulgur_indices:
        prob += lpSum([items[i] * df.iloc[i]["weight_g"] for i in bulgur_indices]) <= 2500
        prob += lpSum([y[i] for i in bulgur_indices]) <= 3
    
    # Pirinç constraints: maximum 2.5 kg total and maximum 3 different items
    pirinc_terms = ['pirinç', 'pirinçli', 'rice']
    pirinc_indices = [i for i in range(n) if any(term in df.iloc[i]['name'].lower() for term in pirinc_terms)]
    if pirinc_indices:
        prob += lpSum([items[i] * df.iloc[i]["weight_g"] for i in pirinc_indices]) <= 2500
        prob += lpSum([y[i] for i in pirinc_indices]) <= 3
    
    # Weight constraint: maximum 50kg total
    prob += lpSum([items[i] * df.iloc[i]["weight_g"] for i in range(n)]) <= 50000
    
    # Product count constraint: maximum 200 products total
    prob += lpSum([items[i] for i in range(n)]) <= 200
    
    # Product variety constraint: at least 10 different items
    for i in range(n):
        prob += items[i] >= y[i]
    prob += lpSum(y) >= 10
    
    # Solve the optimization problem
    try:
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=30))
    except Exception as e:
        print(f"Solver error: {e}")
        return None
    
    # Check solution status
    if LpStatus[prob.status] != "Optimal":
        print(f"No optimal solution found. Status: {LpStatus[prob.status]}")
        return None
    
    # Extract results
    total_cost = sum([value(items[i]) * df.iloc[i]["price"] for i in range(n)])
    total_weight = sum([value(items[i]) * df.iloc[i]["weight_g"] for i in range(n)])
    total_items = sum([value(items[i]) for i in range(n)])
    
    # Prepare results
    results = {
        'items': [],
        'total_cost': total_cost,
        'total_weight': total_weight,
        'total_items': total_items,
        'budget_usage': (total_cost / budget) * 100
    }
    
    # Collect items that were selected
    for i, var in enumerate(items):
        qty = value(var)
        if qty and qty >= 1:
            item_info = {
                'name': df.iloc[i]['name'],
                'market': df.iloc[i]['market'],
                'quantity': int(qty),
                'price_per_unit': df.iloc[i]['price'],
                'total_price': df.iloc[i]['price'] * qty,
                'weight_per_unit': df.iloc[i]['weight_g'],
                'total_weight': df.iloc[i]['weight_g'] * qty,
                'calories': df.iloc[i]['calories'],
                'protein': df.iloc[i]['protein'],
                'carbs': df.iloc[i]['carbs'],
                'fat': df.iloc[i]['fat'],
                'category': df.iloc[i]['main_group']
            }
            results['items'].append(item_info)
    
    return results

# Load data on startup
def load_data():
    global df
    try:
        print("Loading CSV data...")
        df = pd.read_csv("enriched_2025_05_21.csv")
        df = preprocess_data(df)
        print(f"✅ Data loaded successfully: {len(df)} products")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

@app.route('/')
def home():
    return jsonify({
        "message": "Shopping Optimizer API v2.0",
        "status": "running",
        "products_loaded": len(df) if df is not None else 0,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "data_loaded": df is not None,
        "products_count": len(df) if df is not None else 0
    })

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        age = data.get('age')
        gender = data.get('gender')
        weight = data.get('weight')
        height = data.get('height')
        activity = data.get('activity')
        goal = data.get('goal')
        budget = data.get('budget')
        days = data.get('days', 30)
        
        # Validate required parameters
        required_params = ['age', 'gender', 'weight', 'height', 'activity', 'goal', 'budget']
        missing_params = [param for param in required_params if data.get(param) is None]
        
        if missing_params:
            return jsonify({
                "error": f"Missing required parameters: {', '.join(missing_params)}"
            }), 400
        
        # Validate parameter ranges
        if not (0 < age < 120):
            return jsonify({"error": "Age must be between 1 and 120"}), 400
        
        if gender not in ['male', 'female']:
            return jsonify({"error": "Gender must be 'male' or 'female'"}), 400
        
        if not (20 < weight < 300):