import streamlit as st
import requests
import os

st.title("📱 Mobile Price Predictor")

# API Configuration
API_URL = "https://mobile-price-api.onrender.com"
st.sidebar.info(f"🔗 API: {API_URL}")

# Input fields
battery_power = st.number_input("Battery Power (mAh)", 500, 5000, value=1500)
ram = st.number_input("RAM (MB)", 256, 8000, value=2000)
px_height = st.number_input("Pixel Height", 0, 2000, value=1080)
px_width = st.number_input("Pixel Width", 0, 3000, value=1440)
clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.5, 2.0)

if st.button("🔮 Predict Price Range"):
    # Format data as list of 20 features (in correct order)
    features = [
        int(battery_power),    # 0: battery_power
        1,                      # 1: blue
        clock_speed,            # 2: clock_speed
        1,                      # 3: dual_sim
        5,                      # 4: fc
        1,                      # 5: four_g
        32,                     # 6: int_memory
        0.5,                    # 7: m_dep
        150,                    # 8: mobile_wt
        4,                      # 9: n_cores
        12,                     # 10: pc
        int(px_height),         # 11: px_height
        int(px_width),          # 12: px_width
        int(ram),               # 13: ram
        12,                     # 14: sc_h
        7,                      # 15: sc_w
        10,                     # 16: talk_time
        1,                      # 17: three_g
        1,                      # 18: touch_screen
        1                       # 19: wifi
    ]
    
    try:
        with st.spinner('Making prediction...'):
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                price_range = result.get('price_range', 'Unknown')
                
                # Map price range to description
                range_names = {
                    0: "Budget (0-15k)",
                    1: "Mid-range (15-30k)",
                    2: "Premium (30-50k)",
                    3: "Ultra-premium (50k+)"
                }
                
                st.success(f"✅ Predicted Price Range: **{range_names.get(price_range, 'Unknown')}** (Range: {price_range})")
                st.balloons()
            else:
                st.error(f"❌ API Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Unable to connect to API. Please check if the API is running.")
    except requests.exceptions.Timeout:
        st.error("❌ API request timed out. Please try again.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")