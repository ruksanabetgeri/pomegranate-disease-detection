import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import plotly.graph_objects as go

st.markdown("""
<style>
.disease-card { background: white; padding: 1.5rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

model = load_model("improved_efficientnet.h5")

class_names = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Colletotrichum spp', 'Ectomyelois ceratoniae', 'Healthy', 'Sunburn']

disease_descriptions = {
    'Alternaria': "Fungal leaf spot disease causing dark lesions on leaves and fruit rot. Highly contagious in humid conditions. Apply broad-spectrum fungicides early.",
    'Anthracnose': "Devastating fruit rot with sunken lesions. Thrives in warm, wet weather. Use protective copper-based fungicides and prune for airflow.",
    'Bacterial_Blight': "Water-soaked spots on leaves turning dark. Bacterial infection spread by rain. Improve drainage and avoid overhead watering.",
    'Cercospora': "Common leaf spot fungus creating circular brown spots. Destroy infected debris and apply foliar fungicides preventatively.",
    'Colletotrichum spp': "Anthracnose family causing black spots and fruit decay. Multiple strains; integrated management with sanitation and fungicides essential.",
    'Ectomyelois ceratoniae': "Pomegranate fruit moth - larvae bore into fruit causing internal damage. Use pheromone traps, bagging, and sanitation.",
    'Healthy': "Excellent! No disease detected. Continue best practices: proper irrigation, balanced fertilization, and monitoring.",
    'Sunburn': "Non-infectious physiological disorder from excessive direct sunlight. Protect with shade cloth, whitewash, or anti-sunburn sprays."
}

st.title("🍎 Pomegranate Disease Detection")
st.markdown("**Model:** EfficientNetB0 Fine-tuned")

# SIDEBAR DISEASE INFO
st.sidebar.title("🍎 Disease Classes")
for class_name in class_names:
    if st.sidebar.button(class_name, key=class_name):
        st.sidebar.info(disease_descriptions[class_name])

# INITIALIZE STATE
for key in ['pred', 'confidence', 'pred_probs', 'image']:
    if key not in st.session_state:
        st.session_state[key] = None



# UPLOAD & PREDICT
uploaded_file = st.file_uploader("📁 Upload pomegranate image", type=['jpg','jpeg','png'])
if uploaded_file is not None:
    st.session_state.image = Image.open(uploaded_file).convert('RGB')
    st.image(st.session_state.image, caption="Uploaded", width=400)
    
    if st.button("🔍 Detect Disease", type="primary"):
        try:
            img = st.session_state.image.resize((224, 224))
            img_array = np.array(img, dtype=np.uint8)
            img_array = np.expand_dims(preprocess_input(img_array), 0)
            
            probs = model.predict(img_array, verbose=0)[0] * 100
            idx = np.argmax(probs)
            
            st.session_state.pred = class_names[idx]
            st.session_state.confidence = probs[idx]
            st.session_state.pred_probs = probs
            
            # HIGHLIGHT SIDEBAR
            st.sidebar.success(f"Predicted: {st.session_state.pred}")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.pred is None:
    st.info("👆 Upload image and click Detect Disease")
    
# RESULTS BELOW UPLOAD
if st.session_state.pred is not None:
    st.markdown("---")
    
    st.markdown(f"""
    <div style='background: linear-gradient(45deg, #FF6B6B, #FF8E8E); padding: 2rem; border-radius: 25px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 1rem;'>
        <h1 style='color: white; margin: 0;'>{st.session_state.pred}</h1>
        <h2 style='color: white; margin: 0.5rem 0; font-size: 3.5rem;'>{st.session_state.confidence:.1f}%</h2>
        <h3 style='color: white; margin: 0;'>Confidence</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='disease-card'>
        <h3>📝 About {st.session_state.pred}</h3>
        <p style='line-height: 1.8; font-size: 16px; color: #555;'>{disease_descriptions[st.session_state.pred]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📊 Prediction Graph")
    idx = class_names.index(st.session_state.pred)
    fig = go.Figure(data=[go.Bar(
        x=class_names, 
        y=st.session_state.pred_probs,
        marker_color=['gold' if i==idx else '#4ECDC4' for i in range(len(class_names))],
        text=[f'{p:.1f}%' for p in st.session_state.pred_probs],
        textposition='outside'
    )])
    fig.update_layout(
        title=f"Predicted Disease: {st.session_state.pred}", 
        height=400, 
        xaxis_title="Diseases", 
        yaxis_title="Confidence %"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ New Prediction"):
            for key in ['pred', 'confidence', 'pred_probs', 'image']:
                st.session_state[key] = None
            st.rerun()

