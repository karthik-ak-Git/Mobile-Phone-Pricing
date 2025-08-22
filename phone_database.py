"""
Mobile Phone Database with Real Phone Models and Specifications
Enhanced recommendation system with actual phone data
"""

import json
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PhoneModel:
    """Phone model data structure"""
    brand: str
    model: str
    price: int
    currency: str
    specifications: Dict[str, Any]
    availability: str
    launch_date: str
    image_url: str
    rating: float
    reviews_count: int
    key_features: List[str]

class PhoneDatabase:
    """Database of real mobile phones with specifications"""
    
    def __init__(self):
        self.phones = self._load_phone_data()
    
    def _load_phone_data(self) -> List[PhoneModel]:
        """Load comprehensive phone database"""
        phones_data = [
            # Budget Phones (₹5,000 - ₹10,000)
            {
                "brand": "Redmi",
                "model": "A3",
                "price": 7999,
                "currency": "₹",
                "specifications": {
                    "battery_power": 5000, "blue": 1, "clock_speed": 1.8, "dual_sim": 1,
                    "fc": 5, "four_g": 1, "int_memory": 64, "m_dep": 0.8, "mobile_wt": 193,
                    "n_cores": 8, "pc": 8, "px_height": 1600, "px_width": 720,
                    "ram": 3072, "sc_h": 6.7, "sc_w": 3.1, "talk_time": 22,
                    "three_g": 1, "touch_screen": 1, "wifi": 1
                },
                "availability": "In Stock",
                "launch_date": "2024-02-15",
                "image_url": "/static/images/redmi_a3.jpg",
                "rating": 4.2,
                "reviews_count": 1250,
                "key_features": ["5000mAh Battery", "6.71″ Display", "Dual Camera", "Android 14"]
            },
            {
                "brand": "Samsung",
                "model": "Galaxy M14 5G",
                "price": 9999,
                "currency": "₹",
                "specifications": {
                    "battery_power": 6000, "blue": 1, "clock_speed": 2.0, "dual_sim": 1,
                    "fc": 13, "four_g": 1, "int_memory": 128, "m_dep": 0.9, "mobile_wt": 206,
                    "n_cores": 8, "pc": 50, "px_height": 2408, "px_width": 1080,
                    "ram": 4096, "sc_h": 6.6, "sc_w": 3.0, "talk_time": 25,
                    "three_g": 1, "touch_screen": 1, "wifi": 1
                },
                "availability": "In Stock",
                "launch_date": "2024-03-20",
                "image_url": "/static/images/samsung_m14.jpg",
                "rating": 4.3,
                "reviews_count": 2100,
                "key_features": ["6000mAh Battery", "50MP Camera", "5G Ready", "6.6″ sAMOLED"]
            },
            
            # Low-Mid Range Phones (₹10,000 - ₹20,000)
            {
                "brand": "Realme",
                "model": "Narzo 70 5G",
                "price": 15999,
                "currency": "₹",
                "specifications": {
                    "battery_power": 5000, "blue": 1, "clock_speed": 2.2, "dual_sim": 1,
                    "fc": 16, "four_g": 1, "int_memory": 128, "m_dep": 0.8, "mobile_wt": 188,
                    "n_cores": 8, "pc": 50, "px_height": 2412, "px_width": 1080,
                    "ram": 6144, "sc_h": 6.7, "sc_w": 3.1, "talk_time": 20,
                    "three_g": 1, "touch_screen": 1, "wifi": 1
                },
                "availability": "In Stock",
                "launch_date": "2024-05-10",
                "image_url": "/static/images/realme_narzo70.jpg",
                "rating": 4.4,
                "reviews_count": 890,
                "key_features": ["120Hz Display", "50MP AI Camera", "5G Connectivity", "45W Fast Charging"]
            },
            {
                "brand": "Xiaomi",
                "model": "Redmi Note 13 Pro",
                "price": 18999,
                "currency": "₹",
                "specifications": {
                    "battery_power": 5100, "blue": 1, "clock_speed": 2.4, "dual_sim": 1,
                    "fc": 16, "four_g": 1, "int_memory": 128, "m_dep": 0.8, "mobile_wt": 187,
                    "n_cores": 8, "pc": 200, "px_height": 2712, "px_width": 1220,
                    "ram": 8192, "sc_h": 6.7, "sc_w": 3.1, "talk_time": 22,
                    "three_g": 1, "touch_screen": 1, "wifi": 1
                },
                "availability": "In Stock",
                "launch_date": "2024-01-25",
                "image_url": "/static/images/redmi_note13pro.jpg",
                "rating": 4.5,
                "reviews_count": 3200,
                "key_features": ["200MP Camera", "120Hz AMOLED", "67W Fast Charging", "IP54 Rating"]
            },
            
            # Medium Range Phones (₹20,000 - ₹40,000)
            {
                "brand": "OnePlus",
                "model": "Nord CE 4",
                "price": 24999,
                "currency": "₹",
                "specifications": {
                    "battery_power": 5500, "blue": 1, "clock_speed": 2.4, "dual_sim": 1,
                    "fc": 16, "four_g": 1, "int_memory": 128, "m_dep": 0.8, "mobile_wt": 186,
                    "n_cores": 8, "pc": 50, "px_height": 2412, "px_width": 1080,
                    "ram": 8192, "sc_h": 6.7, "sc_w": 3.1, "talk_time": 24,
                    "three_g": 1, "touch_screen": 1, "wifi": 1
                },
                "availability": "In Stock",
                "launch_date": "2024-04-12",
                "image_url": "/static/images/oneplus_nord_ce4.jpg",
                "rating": 4.6,
                "reviews_count": 1560,
                "key_features": ["100W SuperVOOC", "120Hz Fluid Display", "50MP Main Camera", "OxygenOS 14"]
            },
            {
                "brand": "Samsung",
                "model": "Galaxy A55 5G",
                "price": 32999,
                "currency": "₹",
                "specifications": {
                    "battery_power": 5000, "blue": 1, "clock_speed": 2.7, "dual_sim": 1,
                    "fc": 32, "four_g": 1, "int_memory": 128, "m_dep": 0.8, "mobile_wt": 213,
                    "n_cores": 8, "pc": 50, "px_height": 2340, "px_width": 1080,
                    "ram": 8192, "sc_h": 6.6, "sc_w": 3.1, "talk_time": 20,
                    "three_g": 1, "touch_screen": 1, "wifi": 1
                },
                "availability": "In Stock",
                "launch_date": "2024-03-15",
                "image_url": "/static/images/samsung_a55.jpg",
                "rating": 4.5,
                "reviews_count": 2800,
                "key_features": ["Super AMOLED Display", "50MP Triple Camera", "25W Fast Charging", "Gorilla Glass Victus+"]
            },
            
            # High-End Phones (₹40,000+)
            {
                "brand": "Samsung",
                "model": "Galaxy S24",
                "price": 74999,
                "currency": "₹",
                "specifications": {
                    "battery_power": 4000, "blue": 1, "clock_speed": 3.2, "dual_sim": 1,
                    "fc": 12, "four_g": 1, "int_memory": 256, "m_dep": 0.8, "mobile_wt": 167,
                    "n_cores": 8, "pc": 50, "px_height": 2340, "px_width": 1080,
                    "ram": 8192, "sc_h": 6.2, "sc_w": 2.9, "talk_time": 18,
                    "three_g": 1, "touch_screen": 1, "wifi": 1
                },
                "availability": "In Stock",
                "launch_date": "2024-01-24",
                "image_url": "/static/images/samsung_s24.jpg",
                "rating": 4.8,
                "reviews_count": 5200,
                "key_features": ["AI-powered Camera", "Dynamic AMOLED 2X", "Galaxy AI", "45W Fast Charging"]
            },
            {
                "brand": "Apple",
                "model": "iPhone 15",
                "price": 79900,
                "currency": "₹",
                "specifications": {
                    "battery_power": 3349, "blue": 1, "clock_speed": 3.8, "dual_sim": 1,
                    "fc": 12, "four_g": 1, "int_memory": 128, "m_dep": 0.8, "mobile_wt": 171,
                    "n_cores": 6, "pc": 48, "px_height": 2556, "px_width": 1179,
                    "ram": 6144, "sc_h": 6.1, "sc_w": 2.8, "talk_time": 20,
                    "three_g": 1, "touch_screen": 1, "wifi": 1
                },
                "availability": "In Stock",
                "launch_date": "2023-09-15",
                "image_url": "/static/images/iphone_15.jpg",
                "rating": 4.9,
                "reviews_count": 8900,
                "key_features": ["A16 Bionic Chip", "48MP Main Camera", "USB-C", "Dynamic Island"]
            },
            {
                "brand": "Google",
                "model": "Pixel 8",
                "price": 62999,
                "currency": "₹",
                "specifications": {
                    "battery_power": 4575, "blue": 1, "clock_speed": 3.0, "dual_sim": 1,
                    "fc": 10.5, "four_g": 1, "int_memory": 128, "m_dep": 0.9, "mobile_wt": 187,
                    "n_cores": 8, "pc": 50, "px_height": 2400, "px_width": 1080,
                    "ram": 8192, "sc_h": 6.2, "sc_w": 2.9, "talk_time": 24,
                    "three_g": 1, "touch_screen": 1, "wifi": 1
                },
                "availability": "In Stock",
                "launch_date": "2023-10-12",
                "image_url": "/static/images/pixel_8.jpg",
                "rating": 4.7,
                "reviews_count": 2100,
                "key_features": ["Google Tensor G3", "Magic Eraser", "7 Years Updates", "AI Photography"]
            }
        ]
        
        return [PhoneModel(**phone) for phone in phones_data]
    
    def calculate_similarity(self, input_specs: Dict, phone_specs: Dict) -> float:
        """Calculate similarity between input specs and phone specs"""
        feature_weights = {
            'battery_power': 0.15,
            'ram': 0.20,
            'int_memory': 0.15,
            'pc': 0.10,
            'fc': 0.05,
            'clock_speed': 0.10,
            'px_height': 0.05,
            'px_width': 0.05,
            'mobile_wt': 0.05,
            'sc_h': 0.05,
            'sc_w': 0.05
        }
        
        total_similarity = 0
        for feature, weight in feature_weights.items():
            if feature in input_specs and feature in phone_specs:
                # Normalize the difference
                max_val = max(input_specs[feature], phone_specs[feature], 1)
                diff = abs(input_specs[feature] - phone_specs[feature]) / max_val
                similarity = 1 - min(diff, 1)
                total_similarity += similarity * weight
        
        return total_similarity
    
    def recommend_phones(self, input_specs: Dict, price_category: int, limit: int = 5) -> List[Dict]:
        """Recommend phones based on specifications and price category"""
        
        # Filter phones by price category
        price_ranges = {
            0: (5000, 10000),    # Budget
            1: (10000, 20000),   # Low
            2: (20000, 40000),   # Medium
            3: (40000, 150000)   # High
        }
        
        min_price, max_price = price_ranges.get(price_category, (0, 150000))
        filtered_phones = [phone for phone in self.phones 
                          if min_price <= phone.price <= max_price]
        
        # Calculate similarity for each phone
        recommendations = []
        for phone in filtered_phones:
            similarity = self.calculate_similarity(input_specs, phone.specifications)
            
            recommendations.append({
                "brand": phone.brand,
                "model": phone.model,
                "full_name": f"{phone.brand} {phone.model}",
                "price": phone.price,
                "currency": phone.currency,
                "formatted_price": f"{phone.currency}{phone.price:,}",
                "similarity_score": round(similarity * 100, 2),
                "rating": phone.rating,
                "reviews_count": phone.reviews_count,
                "availability": phone.availability,
                "launch_date": phone.launch_date,
                "image_url": phone.image_url,
                "key_features": phone.key_features,
                "specifications": {
                    "RAM": f"{phone.specifications['ram']}MB",
                    "Storage": f"{phone.specifications['int_memory']}GB",
                    "Battery": f"{phone.specifications['battery_power']}mAh",
                    "Camera": f"{phone.specifications['pc']}MP + {phone.specifications['fc']}MP",
                    "Display": f"{phone.specifications['sc_h']}\" x {phone.specifications['sc_w']}\"",
                    "Weight": f"{phone.specifications['mobile_wt']}g"
                }
            })
        
        # Sort by similarity score and rating
        recommendations.sort(key=lambda x: (x['similarity_score'], x['rating']), reverse=True)
        
        return recommendations[:limit]
    
    def get_phone_by_name(self, brand: str, model: str) -> Dict:
        """Get detailed information about a specific phone"""
        for phone in self.phones:
            if phone.brand.lower() == brand.lower() and phone.model.lower() == model.lower():
                return {
                    "brand": phone.brand,
                    "model": phone.model,
                    "full_name": f"{phone.brand} {phone.model}",
                    "price": phone.price,
                    "currency": phone.currency,
                    "formatted_price": f"{phone.currency}{phone.price:,}",
                    "rating": phone.rating,
                    "reviews_count": phone.reviews_count,
                    "availability": phone.availability,
                    "launch_date": phone.launch_date,
                    "image_url": phone.image_url,
                    "key_features": phone.key_features,
                    "detailed_specs": phone.specifications
                }
        return None
    
    def get_trending_phones(self, limit: int = 6) -> List[Dict]:
        """Get trending/popular phones"""
        trending = sorted(self.phones, key=lambda x: (x.rating, x.reviews_count), reverse=True)
        
        return [{
            "brand": phone.brand,
            "model": phone.model,
            "full_name": f"{phone.brand} {phone.model}",
            "price": phone.price,
            "currency": phone.currency,
            "formatted_price": f"{phone.currency}{phone.price:,}",
            "rating": phone.rating,
            "reviews_count": phone.reviews_count,
            "image_url": phone.image_url,
            "key_features": phone.key_features[:3]  # Top 3 features
        } for phone in trending[:limit]]
