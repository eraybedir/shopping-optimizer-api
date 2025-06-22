from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value, PULP_CBC_CMD
import re
import os
import json
from datetime import datetime
import signal
import time

app = Flask(__name__)
CORS(app)

products = None

def load_data():
    global products
    try:
        print("Loading CSV data...")
        products = preprocess_data()
        print(f"✅ Data loaded successfully: {len(products)} products")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

def map_main_group(item_category, name):
    item_category = str(item_category).lower()
    name = str(name).lower()
    
    if 'granola' in item_category or 'granola' in name:
        return 'exclude'
    
    if any(keyword in item_category for keyword in ['sebze', 'domates', 'biber', 'salatalık', 'patates', 'soğan', 'havuç']):
        return 'vegetables'
    if any(keyword in name for keyword in ['domates', 'biber', 'salatalık', 'patates', 'soğan', 'havuç', 'kabak']):
        return 'vegetables'
    
    if any(keyword in item_category for keyword in ['meyve', 'elma', 'muz', 'portakal', 'armut', 'çilek']):
        return 'fruits'
    if any(keyword in name for keyword in ['elma', 'muz', 'portakal', 'armut', 'çilek', 'kayısı', 'şeftali']):
        return 'fruits'
    
    if any(keyword in item_category for keyword in ['süt', 'kahvalt', 'peynir', 'yoğurt', 'süt ürünleri']):
        return 'dairy'
    if any(keyword in name for keyword in ['peynir', 'yoğurt', 'süt', 'kaymak', 'krema']):
        return 'dairy'
    
    if any(keyword in item_category for keyword in ['bakliyat', 'fasulye', 'mercimek', 'nohut', 'bezelye']):
        return 'legumes'
    if any(keyword in name for keyword in ['fasulye', 'mercimek', 'nohut', 'bezelye', 'barbunya']):
        return 'legumes'
    
    if any(keyword in item_category for keyword in ['et', 'balık', 'tavuk', 'kıyma', 'sucuk']):
        return 'meat_fish'
    if any(keyword in name for keyword in ['tavuk', 'balık', 'kıyma', 'sucuk', 'salam', 'pastırma']):
        return 'meat_fish'
    
    if any(keyword in item_category for keyword in ['temel gıda', 'ekmek', 'bulgur', 'pirinç', 'makarna', 'un']):
        return 'grains'
    if any(keyword in name for keyword in ['ekmek', 'bulgur', 'pirinç', 'makarna', 'un', 'börek']):
        return 'grains'
    
    return 'other'

def calculate_tdee(age, gender, weight, height, activity):
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    activity_factors = {
        "sedentary": 1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active": 1.725,
        "extra active": 1.9
    }
    
    tdee = bmr * activity_factors.get(activity, 1.2)
    return tdee

def get_macro_targets(tdee, goal):
    if "gain" in goal:
        tdee += 200
    elif "lose" in goal:
        tdee -= 200
    
    if "sport" in goal:
        protein_ratio = 0.20
        fat_ratio = 0.25
        carb_ratio = 0.55
    else:
        protein_ratio = 0.15
        fat_ratio = 0.25
        carb_ratio = 0.60
    
    protein_g = (tdee * protein_ratio) / 4
    fat_g = (tdee * fat_ratio) / 9
    carb_g = (tdee * carb_ratio) / 4
    
    return tdee, protein_g, fat_g, carb_g

def extract_weight(name):
    match = re.search(r"(\d+[.,]?\d*)\s*(kg|g|gr)", name.lower())
    if match:
        value = float(match.group(1).replace(",", "."))
        unit = match.group(2)
        if "kg" in unit:
            return int(value * 1000)
        else:
            return int(value)
    return 1000

def safe_float(value, default=0.0):
    try:
        if isinstance(value, str):
            value = value.replace(" TL", "").replace(".", "").replace(",", ".")
        return float(value)
    except (ValueError, TypeError):
        return default

def preprocess_data():
    print("Preprocessing data...")
    products_list = []
    
    # Check if CSV file exists
    csv_file = "enriched_2025_05_21.csv"
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        return []
    
    print(f"✅ CSV file found: {csv_file}")
    print(f"File size: {os.path.getsize(csv_file)} bytes")
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            print(f"CSV headers: {reader.fieldnames}")
            
            row_count = 0
            for row in reader:
                row_count += 1
                if row_count % 5000 == 0:
                    print(f"Processed {row_count} rows...")
                
                price = safe_float(row.get('price', 0))
                if price <= 0:
                    continue
                
                calories = safe_float(row.get('calories', 0))
                protein = safe_float(row.get('protein', 0))
                carbs = safe_float(row.get('carbs', 0))
                fat = safe_float(row.get('fat', 0))
                
                if calories < 0 or protein < 0 or carbs < 0 or fat < 0:
                    continue
                
                category = row.get('category', '').lower()
                if 'içecek' in category:
                    continue
                
                name = row.get('name', '').lower()
                item_category = row.get('item_category', '').lower()
                if 'noodle' in name or 'noodle' in item_category:
                    continue
                
                if any(term in name for term in ['ciğer', 'yürek', 'liver', 'heart']):
                    continue
                
                if any(term in name for term in ['çabuk', 'bardak']):
                    continue
                
                if any(term in name for term in ['berliner', 'kruvasan', 'croissant']):
                    continue
                
                if any(term in name for term in ['pilavı', 'çikolata']):
                    continue
                
                weight_g = extract_weight(row.get('name', ''))
                
                if weight_g > 5000 or price > 1000 or calories <= 0:
                    continue
                
                main_group = map_main_group(item_category, name)
                if main_group == 'exclude':
                    continue
                
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
        print(f"❌ Error reading CSV: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    print(f"✅ Data preprocessing complete: {len(products_list)} products available")
    print(f"Total rows processed: {row_count}")
    return products_list

def optimize_shopping(products, tdee, protein_g, fat_g, carb_g, budget, days=30):
    print(f"Starting optimization with budget: {budget} TL")
    print(f"Processing {len(products)} products...")
    
    # Add timeout handling
    def timeout_handler(signum, frame):
        raise TimeoutError("Optimization timed out after 30 seconds")
    
    # Set timeout for 30 seconds
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        prob = LpProblem("ShoppingList", LpMinimize)
        n = len(products)
        
        items = [LpVariable(f"x_{i}", lowBound=0, upBound=5, cat='Integer') for i in range(n)]
        y = [LpVariable(f"y_{i}", cat='Binary') for i in range(n)]
        
        prob += lpSum([items[i] * products[i]["price"] for i in range(n)])
        
        prob += lpSum([items[i] * products[i]["calories"] for i in range(n)]) >= tdee * days
        prob += lpSum([items[i] * products[i]["protein"] for i in range(n)]) >= protein_g * days
        prob += lpSum([items[i] * products[i]["fat"] for i in range(n)]) >= fat_g * days
        prob += lpSum([items[i] * products[i]["carbs"] for i in range(n)]) >= carb_g * days
        
        prob += lpSum([items[i] * products[i]["price"] for i in range(n)]) >= budget * 0.70
        prob += lpSum([items[i] * products[i]["price"] for i in range(n)]) <= budget
        
        for group in ['vegetables', 'fruits', 'dairy', 'legumes', 'meat_fish', 'grains']:
            indices = [i for i in range(n) if products[i]['main_group'] == group]
            if indices:
                prob += lpSum([items[i] for i in indices]) >= 1
        
        meat_indices = [i for i in range(n) if products[i]['main_group'] == 'meat_fish']
        if meat_indices:
            prob += lpSum([items[i] * products[i]["weight_g"] for i in meat_indices]) >= 7500
        
        pasta_terms = ['makarna', 'pasta', 'spaghetti', 'penne', 'farfalle', 'rigatoni', 'şehriye', 'erişte']
        pasta_indices = [i for i in range(n) if any(term in products[i]['name'].lower() for term in pasta_terms)]
        if pasta_indices:
            prob += lpSum([items[i] * products[i]["weight_g"] for i in pasta_indices]) <= 2500
        
        bulgur_terms = ['bulgur', 'bulguru', 'bulgurlu']
        bulgur_indices = [i for i in range(n) if any(term in products[i]['name'].lower() for term in bulgur_terms)]
        if bulgur_indices:
            prob += lpSum([items[i] * products[i]["weight_g"] for i in bulgur_indices]) <= 2500
            prob += lpSum([y[i] for i in bulgur_indices]) <= 3
        
        pirinc_terms = ['pirinç', 'pirinçli', 'rice']
        pirinc_indices = [i for i in range(n) if any(term in products[i]['name'].lower() for term in pirinc_terms)]
        if pirinc_indices:
            prob += lpSum([items[i] * products[i]["weight_g"] for i in pirinc_indices]) <= 2500
            prob += lpSum([y[i] for i in pirinc_indices]) <= 3
        
        prob += lpSum([items[i] * products[i]["weight_g"] for i in range(n)]) <= 50000
        prob += lpSum([items[i] for i in range(n)]) <= 200
        
        for i in range(n):
            prob += items[i] >= y[i]
        prob += lpSum(y) >= 10
        
        print("Solving optimization problem...")
        start_time = time.time()
        
        try:
            prob.solve(PULP_CBC_CMD(msg=False, timeLimit=25))  # 25 seconds for solver, 5 seconds buffer
        except Exception as e:
            print(f"Solver error: {e}")
            signal.alarm(0)  # Cancel timeout
            return None
        
        solve_time = time.time() - start_time
        print(f"Optimization completed in {solve_time:.2f} seconds")
        
        signal.alarm(0)  # Cancel timeout
        
        if LpStatus[prob.status] != "Optimal":
            print(f"No optimal solution found. Status: {LpStatus[prob.status]}")
            return None
        
        total_cost = sum([value(items[i]) * products[i]["price"] for i in range(n)])
        total_weight = sum([value(items[i]) * products[i]["weight_g"] for i in range(n)])
        total_items = sum([value(items[i]) for i in range(n)])
        
        results = {
            'items': [],
            'total_cost': total_cost,
            'total_weight': total_weight,
            'total_items': total_items,
            'budget_usage': (total_cost / budget) * 100
        }
        
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
        
    except TimeoutError:
        print("❌ Optimization timed out after 30 seconds")
        signal.alarm(0)  # Cancel timeout
        return None
    except Exception as e:
        print(f"❌ Unexpected error in optimization: {e}")
        signal.alarm(0)  # Cancel timeout
        return None

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

@app.route('/debug')
def debug():
    csv_file = "enriched_2025_05_21.csv"
    return jsonify({
        "csv_exists": os.path.exists(csv_file),
        "csv_size": os.path.getsize(csv_file) if os.path.exists(csv_file) else 0,
        "current_directory": os.getcwd(),
        "files_in_directory": os.listdir('.'),
        "products_loaded": len(products) if products is not None else 0
    })

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        age = data.get('age')
        gender = data.get('gender')
        weight = data.get('weight')
        height = data.get('height')
        activity = data.get('activity')
        goal = data.get('goal')
        budget = data.get('budget')
        days = data.get('days', 30)
        
        required_params = ['age', 'gender', 'weight', 'height', 'activity', 'goal', 'budget']
        missing_params = [param for param in required_params if data.get(param) is None]
        
        if missing_params:
            return jsonify({
                "error": f"Missing required parameters: {', '.join(missing_params)}"
            }), 400
        
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
        
        if products is None:
            return jsonify({"error": "Product data not loaded"}), 500
        
        print(f"Starting optimization request for budget: {budget} TL")
        start_time = time.time()
        
        tdee = calculate_tdee(age, gender, weight, height, activity)
        tdee, protein_g, fat_g, carb_g = get_macro_targets(tdee, goal)
        
        results = optimize_shopping(products, tdee, protein_g, fat_g, carb_g, budget, days)
        
        total_time = time.time() - start_time
        print(f"Total optimization time: {total_time:.2f} seconds")
        
        if results is None:
            return jsonify({
                "error": "Optimization failed - no solution found within time limit. Try increasing budget or relaxing constraints."
            }), 400
        
        total_calories = sum([item['calories'] * item['quantity'] for item in results['items']])
        total_protein = sum([item['protein'] * item['quantity'] for item in results['items']])
        total_fat = sum([item['fat'] * item['quantity'] for item in results['items']])
        total_carbs = sum([item['carbs'] * item['quantity'] for item in results['items']])
        
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
            "performance": {
                "total_time_seconds": round(total_time, 2),
                "products_processed": len(products)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

# Initialize data loading
print("🚀 Initializing Shopping Optimizer API v2.0 (No Pandas)...")
if not load_data():
    print("❌ Failed to load data. API may not function properly.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False) 
