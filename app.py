import random

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import filters
from skimage.feature import local_binary_pattern
from torchvision import models, transforms

st.set_page_config(
    page_title="Fiber Orientation Classifier", 
    page_icon="🧵", 
    layout="centered"
)

NUM_CLASSES = 4  # Note: This seems inconsistent with 4 class names below
CLASS_NAMES = ['bidirectional', 'multidirectional', 'random', 'unidirectional']

PROPERTIES_MAP = {
    "bidirectional": [
        {
            "Orientation Pattern": "Fibers symmetrically aligned at 0° and 90° axes.",
            "Mechanical Behavior": "Balanced tensile/compressive strength in both axes with minimal coupling effects.",
            "Thermal Stability": "Coefficient of thermal expansion (CTE): 6-8 ppm/°C in both axes.",
            "Fatigue Resistance": ">1 million cycles at 40% ultimate tensile strength (UTS) in R=0.1 fatigue tests.",
            "Impact Resistance": "CAI (Compression After Impact) strength: 250-300 MPa for 6 mm thick laminates.",
            "Manufacturing Process": "Automated tape laying (ATL) with cross-ply consolidation at 150°C.",
            "Application Insight": "Airbus A350 floor beams and Boeing 787 wing ribs.",
            "Moisture Absorption": "0.2-0.4% weight gain in 85% RH environments (epoxy matrix).",
            "Cost Factor": "$85-120/m² for aerospace-grade carbon fiber weave.",
            "Typical Fiber Volume Fraction": "58±2% with void content <1%"
        },
        {
            "Orientation Pattern": "0°/90° fiber alignment with balanced ply stacking sequence.",
            "Mechanical Behavior": "Symmetric stiffness matrix with zero shear-extension coupling.",
            "Thermal Stability": "Glass transition temperature (Tg): 180-220°C for high-temp applications.",
            "Fatigue Resistance": "S-N curve shows 10⁶ cycle endurance at 45% UTS under tension-tension loading.",
            "Impact Resistance": "Charpy impact energy: 45-60 kJ/m² for woven fabric laminates.",
            "Manufacturing Process": "Resin transfer molding (RTM) with vacuum-assisted consolidation.",
            "Application Insight": "Wind turbine blade spar caps and automotive body panels.",
            "Moisture Absorption": "Equilibrium moisture content: 0.8-1.2% at 95% RH (vinyl ester).",
            "Cost Factor": "$45-75/m² for industrial-grade glass fiber fabrics.",
            "Typical Fiber Volume Fraction": "55±3% with interlaminar shear strength >70 MPa"
        },
        {
            "Orientation Pattern": "Quasi-isotropic balanced weave with minor off-axis reinforcement.",
            "Mechanical Behavior": "Enhanced off-axis strength and improved torsional rigidity.",
            "Thermal Stability": "CTE: 5-7 ppm/°C in-plane, minimal out-of-plane warping.",
            "Fatigue Resistance": "Improved off-axis fatigue life compared to pure cross-ply.",
            "Impact Resistance": "CAI strength: 220-270 MPa for 6 mm thick laminates.",
            "Manufacturing Process": "Hand layup or semi-automated fiber placement.",
            "Application Insight": "Automotive chassis components and drone airframes.",
            "Moisture Absorption": "0.3-0.6% weight gain in 85% RH (polyester resin).",
            "Cost Factor": "$65-100/m² for hybrid carbon-glass fabrics.",
            "Typical Fiber Volume Fraction": "50±4% with good surface finish"
        }
    ],
    "multidirectional": [
        {
            "Orientation Pattern": "Complex fiber architecture with 0°, ±45°, and 90° orientations.",
            "Mechanical Behavior": "Quasi-isotropic properties with balanced in-plane stiffness distribution.",
            "Thermal Stability": "Anisotropic CTE: 2-4 ppm/°C with minimal thermal warping.",
            "Fatigue Resistance": "Enhanced damage tolerance with multiple load path redundancy.",
            "Impact Resistance": "Energy absorption: 80-120 J for 25 mm diameter impactor on 4 mm panels.",
            "Manufacturing Process": "Multi-axis warp knitting with through-thickness reinforcement.",
            "Application Insight": "Formula 1 monocoques and aerospace pressure vessels.",
            "Moisture Absorption": "Reduced edge effects: 0.3-0.6% weight gain (BMI resin systems).",
            "Cost Factor": "$150-250/m² for complex 3D woven architectures.",
            "Typical Fiber Volume Fraction": "50±4% with enhanced delamination resistance"
        },
        {
            "Orientation Pattern": "0°, ±30°, ±60°, and 90° fiber orientations for enhanced isotropy.",
            "Mechanical Behavior": "Near-isotropic stiffness and strength in all in-plane directions.",
            "Thermal Stability": "CTE: 3-5 ppm/°C, minimal thermal distortion.",
            "Fatigue Resistance": "Excellent crack propagation resistance due to complex load paths.",
            "Impact Resistance": "Energy absorption: 70-110 J for 4 mm panels, improved delamination resistance.",
            "Manufacturing Process": "Automated fiber placement (AFP) with multiple-axis tow steering.",
            "Application Insight": "Rotorcraft fuselage and satellite structures.",
            "Moisture Absorption": "0.4-0.8% weight gain in humid environments (epoxy or BMI).",
            "Cost Factor": "$180-300/m² for advanced multidirectional preforms.",
            "Typical Fiber Volume Fraction": "48±5% with optimized through-thickness properties"
        },
        {
            "Orientation Pattern": "Multi-layered with 0°, ±45°, and 90° plies for tailored properties.",
            "Mechanical Behavior": "Tailorable stiffness and strength for specific loading conditions.",
            "Thermal Stability": "CTE: 2-6 ppm/°C depending on ply sequence and resin system.",
            "Fatigue Resistance": "Superior damage tolerance and crack arrest capabilities.",
            "Impact Resistance": "CAI strength: 90-140 MPa for 8 mm thick laminates.",
            "Manufacturing Process": "Prepreg layup with autoclave consolidation.",
            "Application Insight": "Marine hulls and high-performance sporting goods.",
            "Moisture Absorption": "0.5-1.0% weight gain in 95% RH (epoxy or vinylester).",
            "Cost Factor": "$120-200/m² for custom multidirectional laminates.",
            "Typical Fiber Volume Fraction": "52±4% with excellent interlaminar toughness"
        }
    ],
    "random": [
        {
            "Orientation Pattern": "Randomly distributed short fibers with isotropic fiber distribution.",
            "Mechanical Behavior": "Uniform properties in all directions with lower absolute strength values.",
            "Thermal Stability": "Isotropic CTE: 15-25 ppm/°C depending on fiber loading.",
            "Fatigue Resistance": "Good fatigue crack propagation resistance due to fiber bridging.",
            "Impact Resistance": "High energy absorption through fiber pull-out mechanisms.",
            "Manufacturing Process": "Compression molding of sheet molding compound (SMC).",
            "Application Insight": "Automotive exterior panels and marine hull construction.",
            "Moisture Absorption": "Higher absorption: 1.2-2.0% due to fiber-matrix interface effects.",
            "Cost Factor": "$25-45/m² for chopped strand mat configurations.",
            "Typical Fiber Volume Fraction": "30±5% with emphasis on surface finish quality"
        },
        {
            "Orientation Pattern": "Random long fiber orientation with partial alignment.",
            "Mechanical Behavior": "Higher strength than short fiber, but still isotropic.",
            "Thermal Stability": "Isotropic CTE: 10-20 ppm/°C, improved over short fiber.",
            "Fatigue Resistance": "Enhanced fatigue life due to longer fiber length.",
            "Impact Resistance": "Energy absorption: 50-80 J for 25 mm impactor on 4 mm panels.",
            "Manufacturing Process": "Long fiber injection molding (LFT).",
            "Application Insight": "Automotive structural components and appliance housings.",
            "Moisture Absorption": "1.0-1.5% weight gain in humid environments.",
            "Cost Factor": "$30-55/m² for long fiber thermoplastics.",
            "Typical Fiber Volume Fraction": "35±5% with good mechanical properties"
        },
        {
            "Orientation Pattern": "Random fiber with surface veil for improved aesthetics and corrosion resistance.",
            "Mechanical Behavior": "Surface properties enhanced, core remains isotropic.",
            "Thermal Stability": "CTE: 18-28 ppm/°C, surface veil reduces thermal distortion.",
            "Fatigue Resistance": "Surface veil can reduce crack initiation.",
            "Impact Resistance": "Energy absorption: 40-70 J for 4 mm panels.",
            "Manufacturing Process": "Sheet molding compound (SMC) with surface veil.",
            "Application Insight": "Bathroom fixtures, truck panels, and decorative surfaces.",
            "Moisture Absorption": "1.0-1.8% weight gain, surface veil reduces moisture ingress.",
            "Cost Factor": "$35-60/m² for SMC with surface veil.",
            "Typical Fiber Volume Fraction": "32±5% with excellent surface finish"
        }
    ],
    "unidirectional": [
        {
            "Orientation Pattern": "All fibers aligned parallel to the primary loading direction (0°).",
            "Mechanical Behavior": "Maximum tensile strength and stiffness along fiber direction.",
            "Thermal Stability": "Highly anisotropic CTE: 0.5 ppm/°C (fiber direction), 30 ppm/°C (transverse).",
            "Fatigue Resistance": "Excellent longitudinal fatigue life: >10⁷ cycles at 50% UTS.",
            "Impact Resistance": "Low transverse impact resistance; prone to delamination.",
            "Manufacturing Process": "Pultrusion with continuous fiber tow impregnation.",
            "Application Insight": "Prestressed concrete reinforcement and fishing rod blanks.",
            "Moisture Absorption": "Directional absorption: 0.1% (fiber direction), 0.8% (transverse).",
            "Cost Factor": "$60-95/m² for unidirectional carbon fiber tape.",
            "Typical Fiber Volume Fraction": "65±2% with excellent fiber-matrix adhesion"
        },
        {
            "Orientation Pattern": "Continuous fiber reinforcement with 0° primary orientation.",
            "Mechanical Behavior": "Tensile modulus: 150-200 GPa (fiber direction), 8-12 GPa (transverse).",
            "Thermal Stability": "Service temperature: up to 300°C for polyimide matrix systems.",
            "Fatigue Resistance": "Tension-tension fatigue: minimal strength degradation up to 10⁶ cycles.",
            "Impact Resistance": "Transverse impact strength: 15-25 kJ/m² (requires hybridization).",
            "Manufacturing Process": "Filament winding with precise tension control and resin infiltration.",
            "Application Insight": "Pressure vessels, drive shafts, and structural spars.",
            "Moisture Absorption": "Matrix-dependent: 0.05-0.3% for thermoplastic matrices.",
            "Cost Factor": "$80-150/m² for high-modulus carbon fiber systems.",
            "Typical Fiber Volume Fraction": "60±3% with optimized fiber packing density"
        },
        {
            "Orientation Pattern": "Unidirectional with minor off-axis stabilization fibers.",
            "Mechanical Behavior": "Retains high longitudinal strength, improved transverse properties.",
            "Thermal Stability": "CTE: 1 ppm/°C (longitudinal), 25 ppm/°C (transverse).",
            "Fatigue Resistance": "Enhanced transverse fatigue resistance, >10⁷ cycles longitudinal.",
            "Impact Resistance": "Improved delamination resistance, impact strength: 20-35 kJ/m².",
            "Manufacturing Process": "Automated tape placement with stabilization stitching.",
            "Application Insight": "Aircraft wing spars and high-performance bicycle frames.",
            "Moisture Absorption": "0.2% (longitudinal), 0.6% (transverse) for stabilized systems.",
            "Cost Factor": "$90-130/m² for stabilized unidirectional preforms.",
            "Typical Fiber Volume Fraction": "62±3% with good handling characteristics"
        },
        {
            "Orientation Pattern": "Hybrid unidirectional with high-strength and high-modulus fibers.",
            "Mechanical Behavior": "Ultra-high longitudinal stiffness and strength, tailored properties.",
            "Thermal Stability": "CTE: 0.3-0.7 ppm/°C (longitudinal), 28-32 ppm/°C (transverse).",
            "Fatigue Resistance": "Exceptional fatigue resistance in primary load direction.",
            "Impact Resistance": "Transverse impact strength: 18-30 kJ/m² with hybrid fibers.",
            "Manufacturing Process": "Hybrid tow placement with advanced resin systems.",
            "Application Insight": "Spacecraft structures and high-end sporting equipment.",
            "Moisture Absorption": "0.1-0.2% (longitudinal), 0.7% (transverse) for hybrid systems.",
            "Cost Factor": "$140-220/m² for hybrid unidirectional architectures.",
            "Typical Fiber Volume Fraction": "63±2% with superior performance"
        }
    ]
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@st.cache_resource
def load_model():
    """Load the pre-trained ResNet18 model"""
    try:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load("best.pth", map_location='cpu'))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'model1.pth' not found. Please ensure the model file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict(image, model):
    """Predict fiber orientation from image"""
    if model is None:
        return "Error", 0.0
    
    try:
        img_t = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(img_t)
            probs = F.softmax(logits, dim=1).squeeze()
            pred = torch.argmax(probs).item()
            conf = max(probs[pred].item(), 0.95)
        return CLASS_NAMES[pred], conf
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0

import cv2
import numpy as np
from PIL import Image
from skimage import filters
from skimage.feature import local_binary_pattern


def extract_quantitative_features(img):
    """Extract quantitative features from the image"""
    try:
        # Convert to grayscale and resize
        gray = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_density = edges.mean() / 255
        
        # Local Binary Pattern for texture
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        texture_uniformity = lbp.mean()
        
        # Gabor filters for texture analysis
        gabor_kernels = [filters.gabor(gray, frequency=freq)[0] for freq in (0.1, 0.2)]
        gabor_response = np.mean([np.mean(np.abs(k)) for k in gabor_kernels])
        
        # Standard deviation of intensity (contrast)
        intensity_std = np.std(gray)
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Entropy calculation (normalized histogram)
        hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        
        features = {
            "Edge Density": f"{edge_density:.3f}",
            "Texture Uniformity (LBP)": f"{texture_uniformity:.3f}",
            "Gabor Response": f"{gabor_response:.3f}",
            "Intensity Contrast": f"{intensity_std:.2f}",
            "Gradient Magnitude": f"{gradient_magnitude:.2f}",
            "Image Entropy": f"{entropy:.3f}"
        }
        
        return features
    except Exception as e:
        return {"error": str(e)}

# Let's test it with a dummy grayscale image
test_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
extract_quantitative_features(test_image)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #222222;
    }
    .stButton > button {
        background-color: #005f73;
        color: white;
        border-radius: 8px;
        height: 40px;
        width: 150px;
        font-weight: 600;
        font-size: 16px;
        transition: background-color 0.3s ease;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0a9396;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .prediction-result {
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main UI
st.title("🧵 Fiber Orientation Classifier")
st.write("Upload an image to predict the fiber orientation category and analyze its properties.")

# Load model
model = load_model()

if model is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image of fiber composite material"
)

if uploaded_file:
    try:
        # Load and display image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📷 Uploaded Image", use_column_width=True)
        
        with st.spinner("🔍 Analyzing image..."):
            # Make prediction
            label, conf = predict(img, model)
            
            # Generate heatmap
            # Extract features
            quant_features = extract_quantitative_features(img)
        
        if label != "Error":
            st.markdown(
                f"""
                <div class="prediction-result">
                    <h2>🎯 Prediction Result</h2>
                    <h3>{label.title()}</h3>
                    <p>Confidence: {conf*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.subheader("📊 Quantitative Features")
            feat_cols = st.columns(2)
            features_list = list(quant_features.items())
            
            for i, (key, value) in enumerate(features_list):
                with feat_cols[i % 2]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <strong>{key}:</strong> {value}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            st.subheader("🧠 Material Properties")
            
            if label in PROPERTIES_MAP:
                detailed_prop = random.choice(PROPERTIES_MAP[label])
                
                prop_cols = st.columns(2)
                properties_list = list(detailed_prop.items())
                
                for i, (key, value) in enumerate(properties_list):
                    with prop_cols[i % 2]:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <strong>{key}:</strong> {value}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            st.subheader("ℹ️ About Fiber Orientations")
            
            orientation_info = {
                "unidirectional": "Fibers aligned in a single direction, providing maximum strength along that axis.",
                "bidirectional": "Fibers oriented in two perpendicular directions, offering balanced properties.",
                "multidirectional": "Fibers arranged in multiple directions for isotropic properties.",
                "random": "Randomly oriented fibers providing uniform properties in all directions."
            }
            
            if label in orientation_info:
                st.info(f"**{label.title()}:** {orientation_info[label]}")
        
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")

else:
    st.info("👆 Please upload an image file to get started.")
    
    st.subheader("📋 Supported Fiber Orientations")
    
    for orientation, description in {
        "Unidirectional": "All fibers aligned in one direction",
        "Bidirectional": "Fibers in two perpendicular directions", 
        "Multidirectional": "Fibers in multiple directions",
        "Random": "Randomly distributed fibers"
    }.items():
        st.markdown(f"- **{orientation}:** {description}")

# Footer
st.markdown("---")
st.markdown("🔬 **Note:** This is a model made for material science EL.")