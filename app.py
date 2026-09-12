import streamlit as st
from PIL import Image
import numpy as np
import cv2
import hashlib

# ============================================================
# DOSA DNA
# Frontend redesign + connection to existing dosa_analysis.py
# IMPORTANT: dosa_analysis.py is NOT changed.
# ============================================================

from dosa_analysis import (
    detect_dosa,
    calculate_geometry,
    calculate_roundness,
    calculate_symmetry,
    calculate_center_accuracy,
    calculate_edge_quality,
    calculate_browning,
    calculate_pores,
    calculate_overall_score,
)

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="DOSA DNA",
    page_icon="🥞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# HELPERS
# ============================================================

def clamp_score(value):
    """Convert NumPy/Python values to a safe float from 0 to 100."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, value))


def safe_progress(value):
    """Streamlit progress requires a normal Python float."""
    return max(0.0, min(1.0, float(value) / 100.0))


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700;800&display=swap');

:root {
    --cream: #fffaf0;
    --cream2: #fff4dc;
    --brown: #3d2815;
    --brown2: #6b4a28;
    --gold: #d97706;
    --gold-light: #fff0c7;
    --border: #eadcc8;
    --muted: #8b735b;
}

.stApp {
    background: var(--cream);
}

.block-container {
    max-width: 1200px;
    padding-top: 25px;
    padding-bottom: 55px;
}

/* HERO */
.hero {
    background:
        radial-gradient(circle at 10% 10%,
            rgba(255,224,157,0.55), transparent 32%),
        radial-gradient(circle at 90% 80%,
            rgba(255,210,120,0.35), transparent 35%),
        #fffaf0;
    border-radius: 30px;
    padding: 55px 25px 50px;
    text-align: center;
    margin-bottom: 20px;
}

.hero-badge {
    display: inline-block;
    background: #fff0c7;
    border: 1px solid #ead19b;
    color: #704214;
    padding: 9px 18px;
    border-radius: 50px;
    font-family: "DM Mono", monospace;
    font-size: 13px;
    letter-spacing: 0.8px;
    margin-bottom: 22px;
}

.hero-title {
    font-family: "Playfair Display", serif;
    font-size: 70px;
    line-height: 1;
    font-weight: 800;
    color: var(--brown);
    margin-bottom: 18px;
}

.hero-title span {
    color: var(--gold);
}

.hero-subtitle {
    font-family: "Playfair Display", serif;
    font-size: 24px;
    color: var(--brown2);
    margin-bottom: 18px;
}

.hero-description {
    max-width: 730px;
    margin: auto;
    color: var(--muted);
    font-family: "DM Mono", monospace;
    font-size: 13px;
    line-height: 1.8;
}

/* FEATURE PILLS */
.feature-area {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    margin: 25px 0 45px;
}

.feature-pill {
    background: white;
    border: 1px solid var(--border);
    color: #654321;
    border-radius: 50px;
    padding: 10px 17px;
    font-family: "DM Mono", monospace;
    font-size: 12px;
    box-shadow: 0 5px 15px rgba(100,70,30,0.05);
}

/* SECTION */
.section-title {
    font-family: "Playfair Display", serif;
    color: var(--brown);
    font-size: 31px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: var(--muted);
    font-family: "DM Mono", monospace;
    font-size: 12px;
    margin-bottom: 20px;
}

/* UPLOAD */
.upload-card {
    background: white;
    border: 2px dashed #e4c985;
    border-radius: 25px;
    padding: 32px 25px;
    text-align: center;
    margin-bottom: 15px;
    box-shadow: 0 10px 30px rgba(100,70,30,0.06);
}

.upload-icon {
    font-size: 52px;
    margin-bottom: 8px;
}

.upload-title {
    font-family: "Playfair Display", serif;
    font-size: 25px;
    font-weight: 700;
    color: var(--brown);
    margin-bottom: 7px;
}

.upload-text {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.7;
}

[data-testid="stFileUploaderDropzone"] {
    background: #fffdf8 !important;
    border-radius: 18px !important;
    border: 1px solid #eadcc8 !important;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    background: #d97706;
    color: white;
    border: none;
    border-radius: 15px;
    min-height: 54px;
    font-family: "DM Mono", monospace;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #b45309;
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(180,83,9,0.22);
}

/* IMAGE CARD */
.image-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 23px;
    padding: 18px;
    box-shadow: 0 8px 25px rgba(100,70,30,0.06);
    margin-bottom: 10px;
}

.image-card-title {
    font-family: "Playfair Display", serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--brown);
    margin-bottom: 12px;
}

/* METRIC CARD */
.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 19px;
    padding: 18px;
    min-height: 120px;
    box-shadow: 0 6px 20px rgba(100,70,30,0.05);
    margin-bottom: 10px;
}

.metric-icon {
    font-size: 25px;
}

.metric-name {
    font-family: "DM Mono", monospace;
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 1px;
    margin-top: 7px;
}

.metric-value {
    font-family: "Playfair Display", serif;
    color: var(--brown);
    font-size: 27px;
    font-weight: 800;
    margin-top: 2px;
}

/* SCORE */
.score-wrapper {
    display: flex;
    justify-content: center;
    margin: 30px 0;
}

.score-box {
    width: 100%;
    max-width: 470px;
    background:
        radial-gradient(circle at top right,
            rgba(255,255,255,0.7), transparent 40%),
        linear-gradient(135deg, #fff4d6, #ffe5a5);
    border: 1px solid #e2c47e;
    border-radius: 28px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 12px 35px rgba(140,90,20,0.10);
}

.score-label {
    color: #805b2a;
    font-family: "DM Mono", monospace;
    font-size: 12px;
    letter-spacing: 1.5px;
}

.score-number {
    color: #b45309;
    font-family: "Playfair Display", serif;
    font-size: 76px;
    font-weight: 800;
    line-height: 1;
    margin: 10px 0;
}

.score-outof {
    color: #805b2a;
    font-size: 13px;
}

.classification {
    margin-top: 15px;
    display: inline-block;
    padding: 8px 17px;
    background: rgba(255,255,255,0.65);
    border-radius: 30px;
    color: #704214;
    font-family: "DM Mono", monospace;
    font-size: 12px;
    font-weight: 600;
}

/* PERSONALITY */
.personality-box {
    background: white;
    border: 1px solid var(--border);
    border-radius: 25px;
    text-align: center;
    padding: 30px;
    margin-top: 20px;
    box-shadow: 0 8px 25px rgba(100,70,30,0.06);
}

.personality-icon {
    font-size: 48px;
}

.personality-title {
    color: var(--brown);
    font-family: "Playfair Display", serif;
    font-size: 31px;
    font-weight: 800;
    margin-top: 8px;
}

.personality-description {
    color: var(--muted);
    font-family: "DM Mono", monospace;
    font-size: 12px;
    margin-top: 8px;
}

/* COMMENT */
.comment-box {
    background: #35230f;
    border-radius: 22px;
    padding: 28px;
    margin-top: 20px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(50,30,10,0.12);
}

.comment-label {
    color: #e8c77e;
    font-family: "DM Mono", monospace;
    font-size: 10px;
    letter-spacing: 2px;
}

.comment-text {
    color: #fff8e9;
    font-family: "Playfair Display", serif;
    font-size: 21px;
    line-height: 1.5;
    margin-top: 10px;
}

/* PASSPORT */
.passport {
    background: linear-gradient(135deg, white, #fff8e5);
    border: 2px solid #d9bd82;
    border-radius: 28px;
    padding: 30px;
    margin-top: 28px;
    box-shadow: 0 12px 35px rgba(100,70,30,0.07);
}

.passport-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    border-bottom: 1px dashed #d8c29b;
    padding-bottom: 17px;
    margin-bottom: 12px;
}

.passport-title {
    color: var(--brown);
    font-family: "Playfair Display", serif;
    font-size: 28px;
    font-weight: 800;
}

.passport-id {
    background: var(--gold-light);
    color: #704214;
    padding: 7px 12px;
    border-radius: 20px;
    font-family: "DM Mono", monospace;
    font-size: 10px;
}

.passport-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid #f0e6d5;
    font-family: "DM Mono", monospace;
    font-size: 12px;
}

.passport-key {
    color: var(--muted);
}

.passport-value {
    color: var(--brown);
    font-weight: 600;
}

/* INFO */
.info-card {
    background: #fff7e5;
    border-left: 5px solid #d97706;
    border-radius: 12px;
    padding: 15px 18px;
    color: #6b4b27;
    font-family: "DM Mono", monospace;
    font-size: 12px;
    line-height: 1.7;
    margin: 15px 0;
}

/* FOOTER */
.footer {
    text-align: center;
    border-top: 1px solid var(--border);
    margin-top: 55px;
    padding-top: 25px;
    color: #9a836a;
    font-family: "DM Mono", monospace;
    font-size: 11px;
    line-height: 1.8;
}

/* MOBILE */
@media (max-width: 700px) {
    .hero-title {
        font-size: 48px;
    }

    .hero-subtitle {
        font-size: 18px;
    }

    .hero {
        padding: 35px 15px;
    }

    .score-number {
        font-size: 58px;
    }

    .passport-header {
        flex-direction: column;
        align-items: flex-start;
    }
}

</style>
""")

# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-badge">
        🧬 COMPUTER VISION × DOSA
    </div>

    <div class="hero-title">
        🥞 DOSA <span>DNA</span>
    </div>

    <div class="hero-subtitle">
        The Dosa Personality & Geometry Lab
    </div>

    <div class="hero-description">
        Because every dosa deserves a scientific evaluation.
        Upload a plain dosa and let our highly unnecessary
        computer vision system investigate its geometry,
        symmetry, browning and personality.
    </div>

</div>
""")

# ============================================================
# FEATURES
# ============================================================

st.html("""
<div class="feature-area">

    <div class="feature-pill">⭕ Roundness</div>
    <div class="feature-pill">⚖️ Symmetry</div>
    <div class="feature-pill">〰️ Edge Quality</div>
    <div class="feature-pill">🎯 Center Accuracy</div>
    <div class="feature-pill">🔥 Browning</div>
    <div class="feature-pill">⚫ Visible Pores</div>

</div>
""")

# ============================================================
# UPLOAD
# ============================================================

st.html("""
<div class="section-title">
    🥞 Submit Your Dosa
</div>

<div class="section-subtitle">
    Give us a plain dosa. We will take it far too seriously.
</div>

<div class="upload-card">

    <div class="upload-icon">🥞</div>

    <div class="upload-title">
        Upload a Dosa
    </div>

    <div class="upload-text">
        For best results, use a clear top-view photo
        where the complete dosa is visible.
    </div>

</div>
""")

# Keep a changing uploader key so TEST ANOTHER DOSA clears the old file.
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

if "show_new_dosa" not in st.session_state:
    st.session_state.show_new_dosa = False

if st.session_state.show_new_dosa:
    st.html("""
    <div class="info-card">
        🥞 <b>Ready for a new dosa!</b>
        Upload another photo below and analyze it.
    </div>
    """)

uploaded_file = st.file_uploader(
    "Choose your dosa image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key=f"dosa_uploader_{st.session_state.uploader_version}",
)

# Once a new file is selected, return to the normal analysis flow.
if uploaded_file is not None:
    st.session_state.show_new_dosa = False

# ============================================================
# BEFORE UPLOAD
# ============================================================

if uploaded_file is None:

    st.html("""
    <div class="info-card">

        💡 <b>Tip:</b>
        Use a top-view photo where the entire plain dosa is visible.
        A clear boundary gives better computer vision results.

    </div>
    """)

# ============================================================
# AFTER UPLOAD
# ============================================================

else:

    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    st.html("""
    <div class="info-card">
        📸 <b>Dosa image loaded.</b>
        Your dosa is ready for unnecessary scientific investigation.
    </div>
    """)

    st.html("""
    <div class="image-card-title">
        📷 Your Dosa
    </div>
    """)

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.image(
            image_rgb,
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze_button = st.button(
        "🔬 ANALYZE THIS DOSA",
        use_container_width=True,
    )

    if analyze_button:

        with st.spinner(
            "🔬 Measuring the dosa's completely unnecessary DNA..."
        ):

            # IMPORTANT:
            # detect_dosa() expects BGR because it uses cv2.cvtColor.
            image_bgr = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR,
            )

            # =================================================
            # DETECTION
            # =================================================

            detected_image, mask, contour, detection_info = (
                detect_dosa(image_bgr)
            )

            if contour is None or mask is None:
                st.error(
                    "❌ I could not detect a clear dosa. "
                    "Please try a clearer top-view image."
                )
                st.stop()

            # =================================================
            # GEOMETRY
            # =================================================

            # Your actual function is:
            # calculate_geometry(contour, mask)

            geometry = calculate_geometry(
                contour,
                mask,
            )

            area = float(geometry["area"])
            perimeter = float(geometry["perimeter"])
            diameter = float(geometry["diameter"])
            radius = float(geometry["radius"])

            centroid_x = float(geometry["centroid_x"])
            centroid_y = float(geometry["centroid_y"])

            circle_x = float(geometry["circle_center_x"])
            circle_y = float(geometry["circle_center_y"])

            # =================================================
            # ROUNDNESS
            # =================================================

            roundness = clamp_score(
                calculate_roundness(
                    area,
                    perimeter,
                )
            )

            # =================================================
            # SYMMETRY
            # =================================================

            lr_symmetry, tb_symmetry, symmetry = (
                calculate_symmetry(mask)
            )

            lr_symmetry = clamp_score(lr_symmetry)
            tb_symmetry = clamp_score(tb_symmetry)
            symmetry = clamp_score(symmetry)

            # =================================================
            # CENTER ACCURACY
            # =================================================

            center_deviation, center_accuracy = (
                calculate_center_accuracy(
                    centroid_x,
                    centroid_y,
                    circle_x,
                    circle_y,
                    radius,
                )
            )

            center_deviation = float(center_deviation)
            center_accuracy = clamp_score(center_accuracy)

            # =================================================
            # EDGE QUALITY
            # =================================================

            edge_quality, edge_irregularity = (
                calculate_edge_quality(
                    contour,
                    centroid_x,
                    centroid_y,
                    radius,
                )
            )

            edge_quality = clamp_score(edge_quality)
            edge_irregularity = float(edge_irregularity)

            # =================================================
            # BROWNING
            # =================================================

            browning, browning_heatmap = (
                calculate_browning(
                    image_bgr,
                    mask,
                )
            )

            browning = clamp_score(browning)

            # =================================================
            # PORES
            # =================================================

            pore_score, pore_visual, pore_count = (
                calculate_pores(
                    image_bgr,
                    mask,
                )
            )

            pore_score = clamp_score(pore_score)
            pore_count = int(pore_count)

            # =================================================
            # OVERALL SCORE
            # =================================================

            overall_score = calculate_overall_score(
                roundness,
                symmetry,
                edge_quality,
                center_accuracy,
                browning,
                pore_score,
            )

            overall_score = clamp_score(overall_score)

        # =====================================================
        # RESULTS TITLE
        # =====================================================

        st.html("""
        <div class="section-title">
            🔬 Dosa Analysis Complete
        </div>

        <div class="section-subtitle">
            The results are in. Your dosa has officially been judged.
        </div>
        """)

        # =====================================================
        # ORIGINAL + DETECTED
        # =====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.html("""
            <div class="image-card-title">
                📷 Original Dosa
            </div>
            """)

            st.image(
                image_rgb,
                use_container_width=True,
            )

        with col2:

            st.html("""
            <div class="image-card-title">
                🧠 Computer Vision Detection
            </div>
            """)

            detected_rgb = cv2.cvtColor(
                detected_image,
                cv2.COLOR_BGR2RGB,
            )

            st.image(
                detected_rgb,
                use_container_width=True,
            )

        # =====================================================
        # GEOMETRY
        # =====================================================

        st.html("""
        <div class="section-title">
            📐 Dosa Geometry
        </div>

        <div class="section-subtitle">
            Because apparently a dosa needs mathematical measurements.
        </div>
        """)

        g1, g2, g3, g4 = st.columns(4)

        with g1:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-icon">⭕</div>
                <div class="metric-name">DIAMETER</div>
                <div class="metric-value">{diameter:.1f}px</div>
            </div>
            """)

        with g2:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-icon">📐</div>
                <div class="metric-name">AREA</div>
                <div class="metric-value">{area:,.0f}</div>
            </div>
            """)

        with g3:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-icon">〰️</div>
                <div class="metric-name">PERIMETER</div>
                <div class="metric-value">{perimeter:,.1f}</div>
            </div>
            """)

        with g4:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-name">CENTER DEVIATION</div>
                <div class="metric-value">{center_deviation:.1f}px</div>
            </div>
            """)

        # =====================================================
        # DNA PROFILE
        # =====================================================

        st.html("""
        <div class="section-title">
            🧬 Dosa DNA Profile
        </div>

        <div class="section-subtitle">
            Six completely serious characteristics of your dosa.
        </div>
        """)

        # ROW 1
        m1, m2, m3 = st.columns(3)

        with m1:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-icon">⭕</div>
                <div class="metric-name">ROUNDNESS</div>
                <div class="metric-value">{roundness:.1f}%</div>
            </div>
            """)
            st.progress(safe_progress(roundness))

        with m2:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-icon">⚖️</div>
                <div class="metric-name">SYMMETRY</div>
                <div class="metric-value">{symmetry:.1f}%</div>
            </div>
            """)
            st.progress(safe_progress(symmetry))

        with m3:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-icon">〰️</div>
                <div class="metric-name">EDGE QUALITY</div>
                <div class="metric-value">{edge_quality:.1f}%</div>
            </div>
            """)
            st.progress(safe_progress(edge_quality))

        # ROW 2
        m4, m5, m6 = st.columns(3)

        with m4:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-name">CENTER ACCURACY</div>
                <div class="metric-value">{center_accuracy:.1f}%</div>
            </div>
            """)
            st.progress(safe_progress(center_accuracy))

        with m5:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-icon">🔥</div>
                <div class="metric-name">BROWNING</div>
                <div class="metric-value">{browning:.1f}%</div>
            </div>
            """)
            st.progress(safe_progress(browning))

        with m6:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-icon">⚫</div>
                <div class="metric-name">VISIBLE PORES</div>
                <div class="metric-value">{pore_score:.1f}%</div>
            </div>
            """)
            st.progress(safe_progress(pore_score))

        # =====================================================
        # SYMMETRY DETAILS
        # =====================================================

        with st.expander("⚖️ View detailed symmetry analysis"):

            s1, s2, s3 = st.columns(3)

            with s1:
                st.metric(
                    "Left ↔ Right",
                    f"{lr_symmetry:.1f}%",
                )

            with s2:
                st.metric(
                    "Top ↕ Bottom",
                    f"{tb_symmetry:.1f}%",
                )

            with s3:
                st.metric(
                    "Overall",
                    f"{symmetry:.1f}%",
                )

        # =====================================================
        # EDGE DETAILS
        # =====================================================

        with st.expander("〰️ View edge analysis"):

            e1, e2 = st.columns(2)

            with e1:
                st.metric(
                    "Edge Quality",
                    f"{edge_quality:.1f}%",
                )

            with e2:
                st.metric(
                    "Edge Irregularity",
                    f"{edge_irregularity:.1f}%",
                )

        # =====================================================
        # BROWNING
        # =====================================================

        with st.expander("🔥 View browning analysis"):

            st.write(
                "This is a visual brightness-based estimate, "
                "not a temperature measurement."
            )

            if browning_heatmap is not None:

                heatmap_rgb = cv2.cvtColor(
                    browning_heatmap,
                    cv2.COLOR_BGR2RGB,
                )

                st.image(
                    heatmap_rgb,
                    caption="Browning / brightness heatmap",
                    use_container_width=True,
                )

        # =====================================================
        # PORES
        # =====================================================

        with st.expander("⚫ View visible pore analysis"):

            st.write(
                f"Candidate visible dark regions detected: "
                f"**{pore_count}**"
            )

            st.write(
                "These are image-based dark regions and may include "
                "cooking spots as well as actual pores."
            )

            if pore_visual is not None:

                pore_rgb = cv2.cvtColor(
                    pore_visual,
                    cv2.COLOR_BGR2RGB,
                )

                st.image(
                    pore_rgb,
                    caption="Candidate pore regions",
                    use_container_width=True,
                )

        # =====================================================
        # DOSA VERDICT
        # Final judgement is based mainly on symmetry + browning.
        # =====================================================

        if symmetry >= 85 and 45 <= browning <= 80:
            classification = "GOOD"
            verdict_title = "READY TO SERVE"
            verdict_icon = "✅"
            personality = "THE ROYAL DOSA"
            personality_icon = "👑"
            verdict = "Good symmetry and balanced browning. No major improvement is needed."
            comment = "Symmetry approved. Browning approved. This dosa understood the assignment."

        elif symmetry >= 85 and browning > 80:
            classification = "GOOD — NEEDS LESS BROWNING"
            verdict_title = "GOOD SHAPE, TOO MUCH BROWNING"
            verdict_icon = "🔥"
            personality = "THE BOLD DOSA"
            personality_icon = "🔥"
            verdict = "The shape is excellent, but the browning is high. Reduce cooking time slightly next time."
            comment = "The geometry is excellent, but the tawa was a little too enthusiastic."

        elif symmetry >= 85 and browning < 45:
            classification = "GOOD — NEEDS MORE BROWNING"
            verdict_title = "GOOD SHAPE, LIGHTLY COOKED"
            verdict_icon = "🌤️"
            personality = "THE PALE BUT PROPER DOSA"
            personality_icon = "🥞"
            verdict = "The shape is excellent, but the browning is low. Cook it a little longer."
            comment = "The circle is excellent. The browning department needs a little more confidence."

        elif 70 <= symmetry < 85 and 45 <= browning <= 80:
            classification = "NORMAL"
            verdict_title = "NORMAL DOSA"
            verdict_icon = "🥞"
            personality = "THE NORMAL DOSA"
            personality_icon = "🥞"
            verdict = "The dosa is normal and reasonably well-made. Browning is balanced, but symmetry can improve."
            comment = "Pretty decent dosa. A little more attention to the shape could make it better."

        elif symmetry < 70 and 45 <= browning <= 80:
            classification = "NEEDS IMPROVEMENT"
            verdict_title = "SHAPE NEEDS IMPROVEMENT"
            verdict_icon = "⚠️"
            personality = "THE CONFUSED DOSA"
            personality_icon = "😵"
            verdict = "Browning is reasonably balanced, but symmetry is low. Spread the batter more evenly."
            comment = "The browning is fine, but the left side and right side have different plans."

        elif symmetry < 70 and browning > 80:
            classification = "NEEDS IMPROVEMENT"
            verdict_title = "SHAPE + BROWNING NEED IMPROVEMENT"
            verdict_icon = "🚨"
            personality = "THE EXPERIMENTAL DOSA"
            personality_icon = "🧪"
            verdict = "Both symmetry and browning need improvement. Spread the batter evenly and reduce cooking time."
            comment = "The shape is confused and the tawa was clearly having a very productive day."

        elif symmetry < 70 and browning < 45:
            classification = "NEEDS IMPROVEMENT"
            verdict_title = "SHAPE + BROWNING NEED IMPROVEMENT"
            verdict_icon = "⚠️"
            personality = "THE EXPERIMENTAL DOSA"
            personality_icon = "🧪"
            verdict = "Both symmetry and browning are low. Spread the batter evenly and cook it a little longer."
            comment = "The circle needs work and the browning needs another meeting with the tawa."

        else:
            classification = "NEEDS IMPROVEMENT"
            verdict_title = "SLIGHT IMPROVEMENT NEEDED"
            verdict_icon = "🔧"
            personality = "THE ALMOST PERFECT DOSA"
            personality_icon = "✨"
            verdict = "The dosa is not bad, but one or more characteristics could be improved."
            comment = "Almost there. A few small improvements could make this dosa much better."

        # =====================================================
        # SCORE
        # =====================================================

        st.html(f"""
        <div class="score-wrapper">

            <div class="score-box">

                <div class="score-label">
                    OVERALL DOSA SCORE
                </div>

                <div class="score-number">
                    {overall_score:.1f}
                </div>

                <div class="score-outof">
                    OUT OF 100
                </div>

                <div class="classification">
                    {classification}
                </div>

            </div>

        </div>
        """)

        # =====================================================
        # DOSA VERDICT
        # =====================================================

        st.html(f"""
        <div class="personality-box">
            <div class="personality-icon">
                {verdict_icon}
            </div>
            <div class="personality-title">
                {verdict_title}
            </div>
            <div class="personality-description">
                {verdict}
            </div>
        </div>
        """)

        # =====================================================
        # PERSONALITY
        # =====================================================

        st.html(f"""
        <div class="personality-box">

            <div class="personality-icon">
                {personality_icon}
            </div>

            <div class="personality-title">
                {personality}
            </div>

            <div class="personality-description">
                Your dosa has officially developed a personality.
            </div>

        </div>
        """)

        # =====================================================
        # FUNNY COMMENT
        # =====================================================

        st.html(f"""
        <div class="comment-box">

            <div class="comment-label">
                🔬 SCIENTIFICALLY UNNECESSARY COMMENT
            </div>

            <div class="comment-text">
                “{comment}”
            </div>

        </div>
        """)

        # =====================================================
        # DOSA PASSPORT
        # =====================================================

        image_hash = hashlib.md5(
            image_rgb.tobytes()
        ).hexdigest()[:6].upper()

        dosa_id = f"DSA-{image_hash}"

        st.html(f"""
        <div class="passport">

            <div class="passport-header">

                <div class="passport-title">
                    🥞 DOSA PASSPORT
                </div>

                <div class="passport-id">
                    ID: {dosa_id}
                </div>

            </div>

            <div class="passport-row">
                <span class="passport-key">Diameter</span>
                <span class="passport-value">{diameter:.1f}px</span>
            </div>

            <div class="passport-row">
                <span class="passport-key">Area</span>
                <span class="passport-value">{area:,.0f}px²</span>
            </div>

            <div class="passport-row">
                <span class="passport-key">Perimeter</span>
                <span class="passport-value">{perimeter:,.1f}px</span>
            </div>

            <div class="passport-row">
                <span class="passport-key">Roundness</span>
                <span class="passport-value">{roundness:.1f}%</span>
            </div>

            <div class="passport-row">
                <span class="passport-key">Symmetry</span>
                <span class="passport-value">{symmetry:.1f}%</span>
            </div>

            <div class="passport-row">
                <span class="passport-key">Edge Quality</span>
                <span class="passport-value">{edge_quality:.1f}%</span>
            </div>

            <div class="passport-row">
                <span class="passport-key">Center Accuracy</span>
                <span class="passport-value">{center_accuracy:.1f}%</span>
            </div>

            <div class="passport-row">
                <span class="passport-key">Browning</span>
                <span class="passport-value">{browning:.1f}%</span>
            </div>

            <div class="passport-row">
                <span class="passport-key">Visible Pores</span>
                <span class="passport-value">{pore_count}</span>
            </div>

            <div class="passport-row">
                <span class="passport-key">Overall Score</span>
                <span class="passport-value">{overall_score:.1f}/100</span>
            </div>

            <div class="passport-row">
                <span class="passport-key">Personality</span>
                <span class="passport-value">{personality}</span>
            </div>

        </div>
        """)

        # =====================================================
        # FINAL MESSAGE
        # =====================================================

        st.html("""
        <div class="info-card">

            🥞 <b>Analysis complete.</b>

            Your dosa has been measured, judged, classified,
            assigned a personality and given a passport.

            <br><br>

            Was any of this necessary?

            <b>Absolutely not.</b>

        </div>
        """)

        # =====================================================
        # NEW TEST BUTTON
        # =====================================================

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "🥞 TEST ANOTHER DOSA",
            use_container_width=True,
        ):
            # Clear only the current dosa and prepare a fresh uploader.
            # The analysis module and all other frontend content stay unchanged.
            st.session_state.uploader_version += 1
            st.session_state.show_new_dosa = True
            st.rerun()

# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    🥞 DOSA DNA

    <br>

    The Dosa Personality & Geometry Lab

    <br><br>

    Built with Python • Streamlit • OpenCV

    <br>

    <i>
        Scientifically unnecessary. Technically impressive.
    </i>

</div>
""")
