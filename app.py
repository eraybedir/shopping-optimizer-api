from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value, PULP_CBC_CMD
import re
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the preprocessed data
products = None

# --- Category Mapping using C column (item_category) ---
def map_main_group(item_category, name):
    item_category = str(item_category).lower()
    name = str(name).lower()
    
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

def safe_float(value, default=0.0):
    """Safely convert string to float"""
    try:
        if isinstance(value, str):
            # Clean price format
            value = value.replace(" TL", "").replace(".", "").replace(",", ".")
        return float(value)
    except (ValueError, TypeError):
        return default

# --- Data Preprocessing ---
def preprocess_data():
    print("Preprocessing data...")
    
    products_list = []
    
    try:
        with open("enriched_2025_05_21.csv", 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                # Clean and validate price
                price = safe_float(row.get('price', 0))
                if price <= 0:
                    continue
                
                # Clean and validate nutrition data
                calories = safe_float(row.get('calories', 0))
                protein = safe_float(row.get('protein', 0))
                carbs = safe_float(row.get('carbs', 0))
                fat = safe_float(row.get('fat', 0))
                
                if calories < 0 or protein < 0 or carbs < 0 or fat < 0:
                    continue
                
                # Exclude beverages
                category = row.get('category', '').lower()
                if 'içecek' in category:
                    continue
                
                # Exclude noodles
                name = row.get('name', '').lower()
                item_category = row.get('item_category', '').lower()
                if 'noodle' in name or 'noodle' in item_category:
                    continue
                
                # Exclude liver and heart products
                if any(term in name for term in ['ciğer', 'yürek', 'liver', 'heart']):
                    continue
                
                # Exclude products containing 'çabuk' or 'bardak'
                if any(term in name for term in ['çabuk', 'bardak']):
                    continue
                
                # Exclude Berliner and Kruvasan products
                if any(term in name for term in ['berliner', 'kruvasan', 'croissant']):
                    continue
                
                # Exclude products containing 'pilavı' and 'çikolata'
                if any(term in name for term in ['pilavı', 'çikolata']):
                    continue
                
                # Extract weight
                weight_g = extract_weight(row.get('name', ''))
                
                # Apply filters
                if weight_g > 5000 or price > 1000 or calories <= 0:
                    continue
                
                # Map category
                main_group = map_main_group(item_category, name)
                if main_group == 'exclude':
                    continue
                
                # Create product dict
                product = {
                    'name': row.get('name', ''),
                    'market': row.get('market', ''),
                    'price': price,
                    'calories': calories,
                    'protein': protein,
                    'carbs': carbs,
                    'fat': fat,
                    'weight_g': weight_g,
                    'main_group': main_group
                }
                
                products_list.append(product)
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
    
    print(f"Data preprocessing complete: {len(products_list)} products available")
    return products_list

# --- Optimization ---
def optimize_shopping(products, tdee, protein_g, fat_g, carb_g, budget, days=30):
    print(f"Starting optimization with budget: {budget} TL")
    
    # Create optimization problem
    prob = LpProblem("ShoppingList", LpMinimize)
    n = len(products)
    
    # Decision variables: number of each item to buy (0-5)
    items = [LpVariable(f"x_{i}", lowBound=0, upBound=5, cat='Integer') for i in range(n)]
    
    # Binary variables for counting different items
    y = [LpVariable(f"y_{i}", cat='Binary') for i in range(n)]
    
    # Objective: minimize total cost
    prob += lpSum([items[i] * products[i]["price"] for i in range(n)])
    
    # Nutrition constraints (scaled for days)
    prob += lpSum([items[i] * products[i]["calories"] for i in range(n)]) >= tdee * days
    prob += lpSum([items[i] * products[i]["protein"] for i in range(n)]) >= protein_g * days
    prob += lpSum([items[i] * products[i]["fat"] for i in range(n)]) >= fat_g * days
    prob += lpSum([items[i] * products[i]["carbs"] for i in range(n)]) >= carb_g * days
    
    # Budget constraints: use at least 70% of budget
    prob += lpSum([items[i] * products[i]["price"] for i in range(n)]) >= budget * 0.70
    prob += lpSum([items[i] * products[i]["price"] for i in range(n)]) <= budget
    
    # Category diversity: at least 1 from each main group
    for group in ['vegetables', 'fruits', 'dairy', 'legumes', 'meat_fish', 'grains']:
        indices = [i for i in range(n) if products[i]['main_group'] == group]
        if indices:
            prob += lpSum([items[i] for i in indices]) >= 1
    
    # Meat/Fish weight constraint: at least 7.5 kg
    meat_indices = [i for i in range(n) if products[i]['main_group'] == 'meat_fish']
    if meat_indices:
        prob += lpSum([items[i] * products[i]["weight_g"] for i in meat_indices]) >= 7500
    
    # Pasta weight constraint: maximum 2.5 kg total
    pasta_terms = ['makarna', 'pasta', 'spaghetti', 'penne', 'farfalle', 'rigatoni', 'şehriye', 'erişte']
    pasta_indices = [i for i in range(n) if any(term in products[i]['name'].lower() for term in pasta_terms)]
    if pasta_indices:
        prob += lpSum([items[i] * products[i]["weight_g"] for i in pasta_indices]) <= 2500
    
    # Bulgur constraints: maximum 2.5 kg total and maximum 3 different items
    bulgur_terms = ['bulgur', 'bulguru', 'bulgurlu']
    bulgur_indices = [i for i in range(n) if any(term in products[i]['name'].lower() for term in bulgur_terms)]
    if bulgur_indices:
        prob += lpSum([items[i] * products[i]["weight_g"] for i in bulgur_indices]) <= 2500
        prob += lpSum([y[i] for i in bulgur_indices]) <= 3
    
    # Pirinç constraints: maximum 2.5 kg total and maximum 3 different items
    pirinc_terms = ['pirinç', 'pirinçli', 'rice']
    pirinc_indices = [i for i in range(n) if any(term in products[i]['name'].lower() for term in pirinc_terms)]
    if pirinc_indices:
        prob += lpSum([items[i] * products[i]["weight_g"] for i in pirinc_indices]) <= 2500
        prob += lpSum([y[i] for i in pirinc_indices]) <= 3
    
    # Weight constraint: maximum 50kg total
    prob += lpSum([items[i] * products[i]["weight_g"] for i in range(n)]) <= 50000
    
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
    total_cost = sum([value(items[i]) * products[i]["price"] for i in range(n)])
    total_weight = sum([value(items[i]) * products[i]["weight_g"] for i in range(n)])
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
                'name': products[i]['name'],
                'market': products[i]['market'],
                'quantity': int(qty),
                'price_per_unit': products[i]['price'],
                'total_price': products[i]['price'] * qty,
                'weight_per_unit': products[i]['weight_g'],
                'total_weight': products[i]['weight_g'] * qty,
                'calories': products[i]['calories'],
                'protein': products[i]['protein'],
                'carbs': products[i]['carbs'],
                'fat': products[i]['fat'],
                'category': products[i]['main_group']
            }
            results['items'].append(item_info)
    
    return results

# Load data on startup
def load_data():
    global products
    try:
        print("Loading CSV data...")
        products = preprocess_data()
        print(f"✅ Data loaded successfully: {len(products)} products")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

@app.route('/')
def home():
    return jsonify({
        "message": "Shopping Optimizer API v2.0 (No Pandas)",
        "status": "running",
        "products_loaded": len(products) if products is not None else 0,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "data_loaded": products is not None,
        "products_count": len(products) if products is not None else 0
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
            return jsonify({"error": "Weight must be between 20 and 300 kg"}), 400
        
        if not (100 < height < 250):
            return jsonify({"error": "Height must be between 100 and 250 cm"}), 400
        
        valid_activities = ['sedentary', 'lightly active', 'moderately active', 'very active', 'extra active']
        if activity not in valid_activities:
            return jsonify({"error": f"Activity must be one of: {', '.join(valid_activities)}"}), 400
        
        valid_goals = ['gaining weight', 'doing sports', 'losing weight', 'being healthy']
        if goal not in valid_goals:
            return jsonify({"error": f"Goal must be one of: {', '.join(valid_goals)}"}), 400
        
        if budget <= 0:
            return jsonify({"error": "Budget must be positive"}), 400
        
        # Check if data is loaded
        if products is None:
            return jsonify({"error": "Product data not loaded"}), 500
        
        # Calculate nutrition targets
        tdee = calculate_tdee(age, gender, weight, height, activity)
        tdee, protein_g, fat_g, carb_g = get_macro_targets(tdee, goal)
        
        # Run optimization
        results = optimize_shopping(products, tdee, protein_g, fat_g, carb_g, budget, days)
        
        if results is None:
            return jsonify({
                "error": "Optimization failed - no solution found. Try increasing budget or relaxing constraints."
            }), 400
        
        # Calculate nutrition summary
        total_calories = sum([item['calories'] * item['quantity'] for item in results['items']])
        total_protein = sum([item['protein'] * item['quantity'] for item in results['items']])
        total_fat = sum([item['fat'] * item['quantity'] for item in results['items']])
        total_carbs = sum([item['carbs'] * item['quantity'] for item in results['items']])
        
        # Prepare response
        response = {
            "success": True,
            "optimization_results": results,
            "nutrition_targets": {
                "calories_target": tdee * days,
                "protein_target": protein_g * days,
                "fat_target": fat_g * days,
                "carbs_target": carb_g * days
            },
            "nutrition_achieved": {
                "calories_achieved": total_calories,
                "protein_achieved": total_protein,
                "fat_achieved": total_fat,
                "carbs_achieved": total_carbs
            },
            "parameters": {
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "activity": activity,
                "goal": goal,
                "budget": budget,
                "days": days
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Load data on startup
    if load_data():
        print("🚀 Shopping Optimizer API v2.0 (No Pandas) is ready!")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    else:
        print("❌ Failed to load data. Exiting.")
        exit(1)
