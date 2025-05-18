import streamlit as st
import pandas as pd
from doctor_recommender import DoctorRecommender

# =============================================
# WHITE THEME CONFIGURATION
# =============================================
def set_white_theme():
    # Set page config with light theme
    st.set_page_config(
        page_title="Doctor Recommendation System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for white theme with cursor fix
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    
    /* Text color */
    body, h1, h2, h3, h4, h5, h6, p {
        color: #333333 !important;
    }
    
    /* Input text styling - FIXED CURSOR VISIBILITY */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        color: #333333 !important;
        caret-color: #4CAF50 !important;  /* Custom cursor color */
    }
    
    /* Input field background */
    .stTextInput>div>div,
    .stTextArea>div>div {
        background-color: white !important;
        border: 1px solid #dddddd !important;
        border-radius: 4px !important;
    }
    
    /* Placeholder text */
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #999999 !important;
    }
    
    /* Focus state - makes cursor more visible */
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 1px #4CAF50 !important;
    }
    
    /* Labels */
    .stTextInput label,
    .stTextArea label {
        color: #333333 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: white !important;
    }
    
    /* Expander headers */
    .stExpander .streamlit-expanderHeader {
        background-color: #f1f1f1;
    }
    </style>
    """, unsafe_allow_html=True)

set_white_theme()

# =============================================
# APP TITLE AND DESCRIPTION
# =============================================
st.title("🏥 Doctor Recommendation System")
st.markdown("""
Find the best doctors based on your symptoms, location, budget and preferences.
""", unsafe_allow_html=True)

# =============================================
# LOAD RECOMMENDER SYSTEM (WITH CACHING)
# =============================================
@st.cache_resource
def load_recommender():
    try:
        recommender = DoctorRecommender("DoctorsDatasets.xlsx")
        st.success("✅ Recommendation system loaded successfully!")
        return recommender
    except Exception as e:
        st.error(f"❌ Error loading recommendation system: {str(e)}")
        return None

recommender = load_recommender()

if recommender is None:
    st.stop()

# =============================================
# SIDEBAR FILTERS (WHITE THEME STYLE)
# =============================================
with st.sidebar:
    st.header("🔍 Search Filters")
    
    # Symptoms input
    symptoms = st.text_input(
        "Describe your symptoms:", 
        placeholder="e.g. back pain, fever, headache",
        help="Describe what health issues you're experiencing"
    )
    
    # Location filter
    location = st.text_input(
        "Preferred location:", 
        placeholder="e.g. Mumbai, Delhi",
        help="Enter city or area where you want to find doctors"
    )
    
    # Fee range slider
    st.subheader("💰 Fee Range")
    min_fee, max_fee = st.slider(
        "Select affordable fee range (₹):",
        min_value=0,
        max_value=int(recommender.df['fees'].max()) if not recommender.df.empty else 2000,
        value=(500, 1500),
        step=100,
        help="Set your budget range for consultation fees"
    )
    
    # Gender selection
    gender = st.radio(
        "👨‍⚕️ Preferred doctor gender:",
        ["Any", "Male", "Female"],
        index=0,
        horizontal=True
    )
    
    # Specialty selection (optional)
    specialty = st.text_input(
        "Specific specialty (optional):",
        placeholder="e.g. Cardiologist, Dermatologist",
        help="If you know the type of specialist you need"
    )
    
    # Search button in sidebar
    search_clicked = st.button(
        "Find Doctors", 
        type="primary",
        use_container_width=True
    )

# =============================================
# MAIN CONTENT AREA
# =============================================
st.header("👨‍⚕️ Recommended Doctors")

if search_clicked or symptoms or location or specialty:
    with st.spinner("Searching for the best doctors..."):
        # Convert "Any" gender to None
        gender_filter = None if gender == "Any" else gender
        
        # Get recommendations
        recommendations, inferred_spec = recommender.recommend_doctors(
            symptoms=symptoms,
            location=location,
            min_fee=min_fee,
            max_fee=max_fee,
            gender=gender_filter,
            specialty=specialty
        )
        
        # Show results
        if not recommendations.empty:
            st.success(f"Found {len(recommendations)} matching doctors")
            
            # Display inferred specialty if available
            if inferred_spec and not specialty:
                st.info(f"💡 Based on your symptoms, you might want a **{inferred_spec}** specialist")
            
            # Show results in a nice table
            st.dataframe(
                recommendations[[
                    'Doctor_Name', 'Specialty', 'Location', 
                    'Experience_Years', 'Rating', 'fees','phoneNumber', 'email', 'Availability'
                ]].rename(columns={
                    'Doctor_Name': 'Name',
                    'Experience_Years': 'Experience (yrs)',
                    'fees': 'Fee (₹)',
                    'phoneNumber': 'Contact',
                    'email': 'Email'
                }),
                column_config={
                    "Rating": st.column_config.NumberColumn(
                        format="%.1f ⭐",
                    ),
                    "Experience (yrs)": st.column_config.NumberColumn(
                        format="%d yrs",
                    ),
                    "Fee (₹)": st.column_config.NumberColumn(
                        format="₹%d",
                    ),
                    "Contact": st.column_config.TextColumn(
                        width="medium"
                    ),
                    "Email": st.column_config.TextColumn(
                        width="large"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Show additional details in expandable sections
            for _, doctor in recommendations.iterrows():
                with st.expander(f"🔍 Detailed info for Dr. {doctor['Doctor_Name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Specialty:** {doctor['Specialty']}")
                        st.markdown(f"**Experience:** {doctor['Experience_Years']} years")
                        st.markdown(f"**Rating:** {doctor['Rating']} ⭐")
                        st.markdown(f"**Fee:** ₹{doctor['fees']}")
                    
                    with col2:
                        st.markdown(f"**Location:** {doctor['Location']}")
                        st.markdown(f"**Availability:** {doctor['Availability']}")
                        st.markdown(f"**Contact:** {doctor['phoneNumber']}")
                        st.markdown(f"**Email:** {doctor['email']}")
                    
                    st.markdown("**Symptoms Handled:**")
                    st.info(doctor['Symptoms_Handled'])
        else:
            st.warning("No doctors found matching all your criteria. Try adjusting your filters.")

# Initial state before search
elif not search_clicked:
    st.info("💡 Enter your symptoms and preferences in the sidebar, then click **Find Doctors**")

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    text-align: center;
    color: #666666;
    margin-top: 2rem;
}
</style>
<div class="footer">
    <p>Doctor Recommendation System © 2023 | For demonstration purposes only</p>
</div>
""", unsafe_allow_html=True)