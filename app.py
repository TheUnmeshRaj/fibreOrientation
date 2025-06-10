import streamlit as st

st.set_page_config(page_title="Fiber Orientation Classifier", page_icon="🧵", layout="centered")

import random

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

num_classes = 2  
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("model1.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

class_names = ['bidirectional','multidirectional','random','unidirectional']

import random

properties_map = {
    'bidirectional': [
        { # Entry 1
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
        { # Entry 2
            "Orientation Pattern": "±45° bias ply configuration for shear dominance.",
            "Mechanical Behavior": "Shear modulus of 4.5 GPa, tensile strength 600 MPa at ±45°.",
            "Thermal Stability": "CTE mismatch of 12 ppm/°C between fiber and matrix directions.",
            "Fatigue Resistance": "10⁶ cycles at τ_max=150 MPa in shear-dominated loading.",
            "Impact Resistance": "45 J/mm impact energy absorption before perforation.",
            "Manufacturing Process": "Hot press molding with vacuum bag debulking.",
            "Application Insight": "Helicopter rotor blade shear webs and F1 monocoque side impact structures.",
            "Moisture Absorption": "0.3% max. in marine-grade vinyl ester composites.",
            "Cost Factor": "$65-90/m² for glass fiber biaxial fabrics.",
            "Typical Fiber Volume Fraction": "50-55% with 2-4% resin-rich surface layers"
        },
        { # Entry 3
            "Orientation Pattern": "Hybrid carbon/glass fiber 0°/90° weave.",
            "Mechanical Behavior": "Carbon fibers provide 1.8 GPa UTS, glass improves impact by 40%.",
            "Thermal Stability": "Reduced thermal warpage through CTE balancing (2.3 vs 5.4 ppm/°C).",
            "Fatigue Resistance": "20% improvement in fatigue crack growth resistance vs pure carbon.",
            "Impact Resistance": "Delamination threshold increased to 35 J from 25 J.",
            "Manufacturing Process": "Co-woven hybrid fabric with 50/50 fiber count ratio.",
            "Application Insight": "High-speed train front-end modules and drone arm structures.",
            "Moisture Absorption": "0.15% for carbon vs 0.35% for glass in hybrid system.",
            "Cost Factor": "$45/m² (glass) + $150/m² (carbon) hybrid cost optimization.",
            "Typical Fiber Volume Fraction": "54% total (27% each fiber type)"
        },
        { # Entry 4
            "Orientation Pattern": "Triaxial (0°/+60°/-60°) carbon fiber weave.",
            "Mechanical Behavior": "Multi-axial strength: 0°=2.1 GPa, ±60°=1.4 GPa tensile strength.",
            "Thermal Stability": "Isotropic CTE of 3.2 ppm/°C through angular compensation.",
            "Fatigue Resistance": "3×10⁶ cycles at 50% UTS in vibratory loading environments.",
            "Impact Resistance": "CAI maintains 85% strength after 6.7 J/mm impact.",
            "Manufacturing Process": "3D loom weaving with Z-binder yarns for through-thickness reinforcement.",
            "Application Insight": "SpaceX Falcon 9 interstage structures and robotic arm linkages.",
            "Moisture Absorption": "0.18% in space-qualified cyanate ester matrices.",
            "Cost Factor": "$320/m² for aerospace-grade triaxial carbon fabrics.",
            "Typical Fiber Volume Fraction": "62% with 0.8% voids (autoclave processed)"
        },
        { # Entry 5
            "Orientation Pattern": "Stitched 0°/90° non-crimp fabric with polyester thread.",
            "Mechanical Behavior": "In-plane modulus 70 GPa, 10% improvement in z-direction properties.",
            "Thermal Stability": "Reduced thermal distortion through stitch-induced constraint.",
            "Fatigue Resistance": "20% longer fatigue life vs unstitched counterparts.",
            "Impact Resistance": "Post-impact compression strength: 320 MPa (25% better than unstitched).",
            "Manufacturing Process": "KSL KARL MAYER stitching machines at 500 stitches/m².",
            "Application Insight": "LM Wind Power 107m turbine blade spar caps.",
            "Moisture Absorption": "0.25% with epoxy compatible stitching yarns.",
            "Cost Factor": "$92/m² including stitching process overhead.",
            "Typical Fiber Volume Fraction": "56-58% achievable without autoclave"
        }
    ],
    'multidirectional': [
        { # Entry 1
            "Orientation Pattern": "Quasi-isotropic [0°/±45°/90°]₂₅ layup.",
            "Mechanical Behavior": "Elastic modulus 54 GPa in all in-plane directions.",
            "Thermal Stability": "CTE 4.1 ppm/°C isotropic, suitable for satellite structures.",
            "Fatigue Resistance": "No observable damage (NOD) at 10⁶ cycles (R=0.1, 40% UTS).",
            "Impact Resistance": "Barely Visible Impact Damage (BVID) threshold: 5 J/mm.",
            "Manufacturing Process": "Automated fiber placement (AFP) at 2 kg/hr deposition rate.",
            "Application Insight": "Boeing 787 fuselage sections and Formula 1 survival cells.",
            "Moisture Absorption": "0.3% equilibrium moisture content in 70°C/85% RH.",
            "Cost Factor": "$850/m² for T800S/3900-2B prepreg layup.",
            "Typical Fiber Volume Fraction": "60% with 35% resin content"
        },
        { # Entry 2
            "Orientation Pattern": "3D orthogonal weave (X-Y-Z) carbon/polyamide.",
            "Mechanical Behavior": "Z-direction strength 85 MPa, in-plane UTS 1.2 GPa.",
            "Thermal Stability": "CTE Z-axis: 8 ppm/°C vs 2.5 ppm/°C in-plane.",
            "Fatigue Resistance": "10⁷ cycles at 30% UTS for high-cycle fatigue applications.",
            "Impact Resistance": "Energy absorption 120 kJ/m² through fiber pull-out mechanism.",
            "Manufacturing Process": "3D loom weaving with 500 picks per minute.",
            "Application Insight": "Hypersonic vehicle thermal protection systems.",
            "Moisture Absorption": "0.45% for polyamide matrix at 50°C water immersion.",
            "Cost Factor": "$1,200/m² for 3D woven preforms.",
            "Typical Fiber Volume Fraction": "48% (3D architectures limit packing efficiency)"
        },
        { # Entry 3
            "Orientation Pattern": "Angle-interlock ±30°/+75°/-15° hybrid layup.",
            "Mechanical Behavior": "Anisotropic modulus: 45 GPa (0°), 28 GPa (90°).",
            "Thermal Stability": "Warpage <0.1 mm/m under 150°C thermal cycling.",
            "Fatigue Resistance": "Delamination onset at 450,000 cycles (R=0.5).",
            "Impact Resistance": "25 J impact causes 15 mm diameter damage area.",
            "Manufacturing Process": "Tailored fiber placement (TFP) with 12k tow spread.",
            "Application Insight": "BMW i8 roof panel and Porsche 911 GT3 rear wing.",
            "Moisture Absorption": "0.22% for BMI resin at maritime conditions.",
            "Cost Factor": "$750/m² for customized angle-interlock preforms.",
            "Typical Fiber Volume Fraction": "52% with 3% binder content"
        },
        { # Entry 4
            "Orientation Pattern": "[0°/±45°/90°]₃ with 100 μm diameter Z-pins.",
            "Mechanical Behavior": "Interlaminar shear strength (ILSS) 95 MPa (+40% vs unpinned).",
            "Thermal Stability": "Reduced thermal buckling through thickness reinforcement.",
            "Fatigue Resistance": "Mode II fatigue threshold Gᴜ = 280 J/m².",
            "Impact Resistance": "CAI strength retention: 75% after 10 J impact.",
            "Manufacturing Process": "Ultrasonic Z-pinning at 20 pins/cm² density.",
            "Application Insight": "Airbus A380 wing-to-fuselage fairings.",
            "Moisture Absorption": "0.15% with sealed pin insertion points.",
            "Cost Factor": "+18% cost over baseline laminate for Z-pinning.",
            "Typical Fiber Volume Fraction": "58% (pins displace 2% fibers)"
        },
        { # Entry 5
            "Orientation Pattern": "0°/±60° pseudo-3D woven structure.",
            "Mechanical Behavior": "Poisson's ratio 0.08 in primary load direction.",
            "Thermal Stability": "Through-thickness CTE 14 ppm/°C vs 3 ppm/°C in-plane.",
            "Fatigue Resistance": "S-N curve slope m=12 for high-cycle applications.",
            "Impact Resistance": "Perforation threshold 80 J for 8 mm thick panels.",
            "Manufacturing Process": "Dornier weaving machines with Jacquard head.",
            "Application Insight": "Rolls-Royce turbine engine containment rings.",
            "Moisture Absorption": "0.3% max in oil-contaminated environments.",
            "Cost Factor": "$980/m² for engine-qualified preforms.",
            "Typical Fiber Volume Fraction": "55% with 5% resin infusion channels"
        }
    ],
    'random': [
        { # Entry 1
            "Orientation Pattern": "25mm chopped strand mat (CSM) with 50g/m² binder.",
            "Mechanical Behavior": "Isotropic tensile strength 85 MPa, modulus 7 GPa.",
            "Thermal Stability": "CTE 35 ppm/°C due to high resin content.",
            "Fatigue Resistance": "10⁴ cycles at 20 MPa stress amplitude.",
            "Impact Resistance": "Charpy impact 12 kJ/m² at 23°C.",
            "Manufacturing Process": "Spray-up with 65% fiber deposition efficiency.",
            "Application Insight": "Bathroom fixtures and automotive trunk liners.",
            "Moisture Absorption": "1.2% water uptake in 24h immersion.",
            "Cost Factor": "$4.50/kg for glass-filled polyester CSM.",
            "Typical Fiber Volume Fraction": "18-22% (random packing limit)"
        },
        { # Entry 2
            "Orientation Pattern": "Wet-laid nonwoven with 12mm fiber length.",
            "Mechanical Behavior": "Tensile strength 55 MPa MD / 48 MPa CD.",
            "Thermal Stability": "Heat deflection temperature (HDT) 140°C (polypropylene).",
            "Fatigue Resistance": "10⁵ cycles at 15% UTS in flexural fatigue.",
            "Impact Resistance": "Instrumented dart impact: 18 J total energy.",
            "Manufacturing Process": "Fourdrinier machine production at 200 m/min.",
            "Application Insight": "Battery tray liners and HVAC filter housings.",
            "Moisture Absorption": "0.05% for polyolefin matrices.",
            "Cost Factor": "$3.20/m² for 300 gsm automotive-grade mats.",
            "Typical Fiber Volume Fraction": "15-18% (low consolidation pressure)"
        },
        { # Entry 3
            "Orientation Pattern": "Air-laid carbon fiber veil (5gsm) with thermoplastic.",
            "Mechanical Behavior": "Surface conductivity 10 S/cm, tensile 25 MPa.",
            "Thermal Stability": "Vicat softening point 220°C (PA6 matrix).",
            "Fatigue Resistance": "Not applicable - used in non-structural roles.",
            "Impact Resistance": "3 J puncture resistance for EMI shielding layers.",
            "Manufacturing Process": "Air-laid technology with electrostatic fiber alignment.",
            "Application Insight": "Aircraft lightning strike protection layers.",
            "Moisture Absorption": "2.1% for nylon-based veils at 50% RH.",
            "Cost Factor": "$85/m² for aerospace-certified conductive veils.",
            "Typical Fiber Volume Fraction": "8-12% (ultra-thin veil requirement)"
        },
        { # Entry 4
            "Orientation Pattern": "Recycled carbon fiber (rCF) with 20mm average length.",
            "Mechanical Behavior": "Tensile strength 320 MPa (60% of virgin fiber).",
            "Thermal Stability": "CTE 28 ppm/°C due to fiber-matrix mismatch.",
            "Fatigue Resistance": "10⁵ cycles at 25% UTS (R=0.1).",
            "Impact Resistance": "Izod impact 45 J/m notched.",
            "Manufacturing Process": "Compression molding at 10 MPa pressure.",
            "Application Insight": "Laptop chassis and consumer electronics housings.",
            "Moisture Absorption": "0.8% for recycled PET matrix.",
            "Cost Factor": "$28/kg (40% savings vs virgin material).",
            "Typical Fiber Volume Fraction": "25-30% (limited by fiber entanglement)"
        },
        { # Entry 5
            "Orientation Pattern": "3D random glass fiber preform (5-15mm fibers).",
            "Mechanical Behavior": "Compressive strength 75 MPa at 10% strain.",
            "Thermal Stability": "Thermal conductivity 0.8 W/m·K (insulating).",
            "Fatigue Resistance": "10³ cycles at 50% UTS in compression.",
            "Impact Resistance": "Energy absorption 35 kJ/m³ in crash scenarios.",
            "Manufacturing Process": "Preform stitching with 30% binder content.",
            "Application Insight": "High-temperature insulation blankets and fire barriers.",
            "Moisture Absorption": "0.6% for phenolic resin binders.",
            "Cost Factor": "$15/m² for industrial insulation products.",
            "Typical Fiber Volume Fraction": "12-15% (high porosity required)"
        }
    ],
    'unidirectional': [
        { # Entry 1
            "Orientation Pattern": "T1100G/3900-2B prepreg tape (0° ±1° alignment).",
            "Mechanical Behavior": "Longitudinal modulus 324 GPa, UTS 6.9 GPa.",
            "Thermal Stability": "CTE -0.7 ppm/°C (fiber dominated direction).",
            "Fatigue Resistance": "10⁷ cycles at 1.5 GPa stress (R=0.1).",
            "Impact Resistance": "CAI 280 MPa after 6 J/mm impact.",
            "Manufacturing Process": "ATL with 150°C cure and 700 kPa pressure.",
            "Application Insight": "Lockheed Martin F-35 wing spars.",
            "Moisture Absorption": "0.12% equilibrium at 85% RH.",
            "Cost Factor": "$420/m² for aerospace-grade prepreg.",
            "Typical Fiber Volume Fraction": "62% (±0.5% process control)"
        },
        { # Entry 2
            "Orientation Pattern": "Hybrid UD (carbon/aramid) 0° tape.",
            "Mechanical Behavior": "Carbon provides 3.5 GPa UTS, aramid adds 18% strain-to-failure.",
            "Thermal Stability": "Aramid limits thermal degradation to 180°C.",
            "Fatigue Resistance": "20% improvement in tension-tension fatigue vs pure carbon.",
            "Impact Resistance": "Delamination energy 800 J/m² (+35% vs carbon-only).",
            "Manufacturing Process": "Co-mingled fiber towpreg production.",
            "Application Insight": "Ballistic panels and racing seat reinforcement.",
            "Moisture Absorption": "0.45% (aramid hygroscopic nature).",
            "Cost Factor": "$95/m² for hybrid military-grade tape.",
            "Typical Fiber Volume Fraction": "58% (50/50 fiber ratio)"
        },
        { # Entry 3
            "Orientation Pattern": "Spread-tow UD (12k to 3k spread).",
            "Mechanical Behavior": "Thin-ply effect: 10% higher ILSS vs standard UD.",
            "Thermal Stability": "Reduced microcracking during thermal cycling.",
            "Fatigue Resistance": "2×10⁶ cycles at 70% UTS (thin-ply advantage).",
            "Impact Resistance": "BVID threshold increased to 8 J/mm.",
            "Manufacturing Process": "Air-jet spreading with 300 mm width capability.",
            "Application Insight": "Solar array substrates for GEO satellites.",
            "Moisture Absorption": "0.08% for space-qualified cyanate ester.",
            "Cost Factor": "$280/m² for 25 gsm spread-tow material.",
            "Typical Fiber Volume Fraction": "65% (improved resin wetting)"
        },
        { # Entry 4
            "Orientation Pattern": "UD basalt fiber/polypropylene tape.",
            "Mechanical Behavior": "Tensile strength 480 MPa, modulus 28 GPa.",
            "Thermal Stability": "Service temperature -60°C to +145°C.",
            "Fatigue Resistance": "10⁶ cycles at 30% UTS (R=0.3).",
            "Impact Resistance": "Notched Izod 85 J/m (recyclable system).",
            "Manufacturing Process": "Melt impregnation at 220°C line speed.",
            "Application Insight": "Sustainable automotive door beams.",
            "Moisture Absorption": "0.02% (thermoplastic matrix advantage).",
            "Cost Factor": "$18/m² for high-volume automotive tape.",
            "Typical Fiber Volume Fraction": "45% (thermoplastic processing limit)"
        },
        { # Entry 5
            "Orientation Pattern": "Ceramic-grade UD Nextel 610 alumina fibers.",
            "Mechanical Behavior": "2.1 GPa UTS at 1000°C, modulus 380 GPa.",
            "Thermal Stability": "Zero creep at 1000°C in inert atmosphere.",
            "Fatigue Resistance": "10⁴ cycles at 800 MPa (1000°C, R=0).",
            "Impact Resistance": "Vickers hardness 18 GPa (fiber-dominated).",
            "Manufacturing Process": "Slurry casting with colloidal silica binder.",
            "Application Insight": "Hypersonic vehicle leading edges.",
            "Moisture Absorption": "Negligible (<0.01%) in ceramic matrix.",
            "Cost Factor": "$2,500/m² for aerospace CMC tapes.",
            "Typical Fiber Volume Fraction": "40% (CMC processing constraints)"
        }
    ]
}


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        pred = torch.argmax(output, dim=1).item()
    return class_names[pred]

# --- Streamlit UI ---


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
    }
    .stButton > button:hover {
        background-color: #0a9396;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# for field in fields:
#     st.markdown(f"**{field}:** {detailed_prop.get(field, 'N/A')}")# ]     "Moisture Absorption", "Cost Factor", "Typical Fiber Volume Fraction"

st.title("🧵 Fiber Orientation Classifier")
st.write("Upload an image to predict the fiber orientation category.")

uploaded_file = st.file_uploader("Choose an image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing image..."):
        label = predict(img)
    st.success(f"**Predicted fiber orientation:** {label.title()}")
    
    # properties = random.sample(properties_map[label], k=3)
    # st.subheader("🔍 Properties of this orientation:")
    # for prop in properties:
    #     st.markdown(f"- {prop}")
        
    detailed_prop = random.choice(properties_map[label])
    st.subheader("🧠 Model Interpretation")
    for key, value in detailed_prop.items():
        st.markdown(f"**{key}:** {value}")

    
else:
    st.info("Please upload an image file to get started.")