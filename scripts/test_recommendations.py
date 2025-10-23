"""
Test script to verify the phone recommendation system
"""

import requests
import json

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_phone_recommendations():
    """Test the phone recommendation endpoint"""
    
    # Test phone specifications
    test_specs = {
        "battery_power": 3000,
        "blue": 1,
        "clock_speed": 2.0,
        "dual_sim": 1,
        "fc": 8,
        "four_g": 1,
        "int_memory": 64,
        "m_dep": 0.7,
        "mobile_wt": 180,
        "n_cores": 4,
        "pc": 12,
        "px_height": 1920,
        "px_width": 1080,
        "ram": 3000,
        "sc_h": 6.0,
        "sc_w": 3.0,
        "talk_time": 18,
        "three_g": 1,
        "touch_screen": 1,
        "wifi": 1
    }
    
    print("🧪 Testing Phone Recommendation System...")
    print("=" * 50)
    
    try:
        # Test recommendation endpoint
        response = requests.post(
            f"{BASE_URL}/api/recommend",
            json=test_specs,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Prediction successful!")
            print(f"📱 Predicted Category: {data['predicted_category']}")
            print(f"💰 Price Range: {data['price_range']}")
            print(f"🎯 Confidence: {data['confidence']:.2%}")
            print(f"⚡ Processing Time: {data['processing_time_ms']}ms")
            print(f"🏆 Model Accuracy: {data['model_accuracy']}")
            
            # Check recommendations
            if 'recommended_phones' in data and data['recommended_phones']:
                print(f"\n📋 Recommended Phones ({data['total_recommendations']}):")
                print("-" * 50)
                
                for i, phone in enumerate(data['recommended_phones'], 1):
                    print(f"{i}. {phone['full_name']}")
                    print(f"   💵 Price: {phone['formatted_price']}")
                    print(f"   🔍 Similarity: {phone['similarity_score']}%")
                    print(f"   ⭐ Rating: {phone['rating']}/5 ({phone['reviews_count']} reviews)")
                    print(f"   🛍️ Status: {phone['availability']}")
                    print(f"   🔧 Key Features: {', '.join(phone['key_features'][:3])}")
                    print()
            else:
                print("❌ No phone recommendations returned")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_trending_phones():
    """Test the trending phones endpoint"""
    
    print("\n🔥 Testing Trending Phones...")
    print("=" * 30)
    
    try:
        response = requests.get(f"{BASE_URL}/api/phones/trending", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Trending phones loaded successfully!")
            print(f"📱 Total trending phones: {data['total_count']}")
            print(f"📅 Last updated: {data['last_updated']}")
            
            print("\n🌟 Top Trending Phones:")
            print("-" * 30)
            
            for i, phone in enumerate(data['trending_phones'], 1):
                print(f"{i}. {phone['full_name']}")
                print(f"   💵 {phone['formatted_price']}")
                print(f"   ⭐ {phone['rating']}/5 ({phone['reviews_count']} reviews)")
                print()
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_health_check():
    """Test the API health check"""
    
    print("\n💓 Testing API Health...")
    print("=" * 25)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"🤖 Model Loaded: {data['model_loaded']}")
            print(f"🎯 Accuracy: {data['accuracy']}")
            print(f"🔢 Version: {data['version']}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    # Run all tests
    test_health_check()
    test_trending_phones() 
    test_phone_recommendations()
    
    print("\n" + "=" * 60)
    print("🎉 Testing completed!")
    print("💡 You can now access the web interface at: http://127.0.0.1:8000")
    print("📚 API documentation available at: http://127.0.0.1:8000/docs")
