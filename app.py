import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

from dosa_analysis import (
    detect_dosa,
    calculate_geometry,
    calculate_roundness,
    calculate_symmetry,
    calculate_center_accuracy,
    calculate_edge_quality,
    calculate_browning,
    calculate_pores,
    calculate_overall_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DOSA DNA",
    page_icon="🥞",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #666666;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .score-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #fff7e6;
        border: 2px solid #f0d49a;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .big-score {
        font-size: 60px;
        font-weight: 800;
    }

    .metric-label {
        font-size: 16px;
        color: #666666;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🥞 DOSA DNA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'The Dosa Personality & Geometry Lab'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Because every dosa deserves a scientific evaluation."
)

st.write(
    "Upload a plain dosa photo and let our highly "
    "unnecessary scientific system evaluate its geometry, "
    "symmetry and appearance."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload your dosa image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "Unable to read the image. "
            "Please upload a valid JPG or PNG image."
        )

        st.stop()

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    st.success(
        "Dosa image uploaded successfully! 🥞"
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "🔬 Analyze My Dosa",
        type="primary"
    )


    if analyze:

        # ====================================================
        # STEP 3 - DETECTION
        # ====================================================

        with st.spinner(
            "Scientifically examining your dosa..."
        ):

            annotated, mask, contour, detection_info = (
                detect_dosa(image)
            )

        if contour is None:

            st.error(
                "We couldn't find a dosa. "
                "Please upload a clearer top-view dosa image."
            )

            st.stop()


        # ====================================================
        # ORIGINAL + DETECTED IMAGE
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🔍 Dosa Detection'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                rgb_image,
                caption="Original Image",
                use_container_width=True
            )

        with col2:

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                annotated_rgb,
                caption="Detected Dosa Boundary",
                use_container_width=True
            )


        # ====================================================
        # STEP 4 - GEOMETRY
        # ====================================================

        geometry = calculate_geometry(
            contour,
            mask
        )

        st.markdown(
            '<div class="section-title">'
            '📐 Dosa Geometry'
            '</div>',
            unsafe_allow_html=True
        )

        g1, g2, g3, g4 = st.columns(4)

        with g1:

            st.metric(
                "Area",
                f"{geometry['area']:.0f} px²"
            )

        with g2:

            st.metric(
                "Perimeter",
                f"{geometry['perimeter']:.0f} px"
            )

        with g3:

            st.metric(
                "Diameter",
                f"{geometry['diameter']:.0f} px"
            )

        with g4:

            st.metric(
                "Radius",
                f"{geometry['radius']:.0f} px"
            )


        # ====================================================
        # DOSA CENTER
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🎯 Dosa Center'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:

            st.write(
                f"**Centroid X:** "
                f"{geometry['centroid_x']:.1f} px"
            )

            st.write(
                f"**Centroid Y:** "
                f"{geometry['centroid_y']:.1f} px"
            )

        with c2:

            st.write(
                f"**Circle Center X:** "
                f"{geometry['circle_center_x']:.1f} px"
            )

            st.write(
                f"**Circle Center Y:** "
                f"{geometry['circle_center_y']:.1f} px"
            )


        # ====================================================
        # STEP 5 - ROUNDNESS
        # ====================================================

        roundness_score = calculate_roundness(
            geometry["area"],
            geometry["perimeter"]
        )

        st.markdown(
            '<div class="section-title">'
            '⭕ Dosa Roundness'
            '</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Roundness Score",
            f"{roundness_score:.1f}%"
        )

        if roundness_score >= 90:

            st.success(
                "⭕ Excellent! This dosa has "
                "very respectable geometry."
            )

        elif roundness_score >= 75:

            st.info(
                "⭕ Pretty round! This dosa is doing well."
            )

        else:

            st.warning(
                "⭕ The circle has taken a creative direction."
            )


        # ====================================================
        # STEP 6 - SYMMETRY
        # ====================================================

        (
            left_right,
            top_bottom,
            overall_symmetry
        ) = calculate_symmetry(mask)

        st.markdown(
            '<div class="section-title">'
            '⚖️ Dosa Symmetry'
            '</div>',
            unsafe_allow_html=True
        )

        s1, s2, s3 = st.columns(3)

        with s1:

            st.metric(
                "Left-Right Symmetry",
                f"{left_right:.1f}%"
            )

        with s2:

            st.metric(
                "Top-Bottom Symmetry",
                f"{top_bottom:.1f}%"
            )

        with s3:

            st.metric(
                "Overall Symmetry",
                f"{overall_symmetry:.1f}%"
            )

        if overall_symmetry >= 90:

            st.success(
                "⚖️ Highly symmetrical! "
                "Both sides seem to agree with each other."
            )

        elif overall_symmetry >= 75:

            st.info(
                "⚖️ Fairly symmetrical. "
                "The dosa is mostly cooperating."
            )

        else:

            st.warning(
                "⚖️ The left side and right side "
                "appear to have different career goals."
            )


        # ====================================================
        # STEP 7 - CENTER ACCURACY
        # ====================================================

        (
            center_deviation,
            center_accuracy
        ) = calculate_center_accuracy(
            geometry["centroid_x"],
            geometry["centroid_y"],
            geometry["circle_center_x"],
            geometry["circle_center_y"],
            geometry["radius"]
        )

        st.markdown(
            '<div class="section-title">'
            '🎯 Center Accuracy'
            '</div>',
            unsafe_allow_html=True
        )

        ca1, ca2 = st.columns(2)

        with ca1:

            st.metric(
                "Center Deviation",
                f"{center_deviation:.2f} px"
            )

        with ca2:

            st.metric(
                "Center Accuracy",
                f"{center_accuracy:.1f}%"
            )

        if center_accuracy >= 90:

            st.success(
                "🎯 Almost perfectly centered!"
            )

        elif center_accuracy >= 75:

            st.info(
                "🎯 Slightly off-center, but acceptable."
            )

        else:

            st.warning(
                "🎯 The dosa center appears to have "
                "gone on a small adventure."
            )


        # ====================================================
        # STEP 8 - EDGE QUALITY
        # ====================================================

        (
            edge_quality,
            edge_irregularity
        ) = calculate_edge_quality(
            contour,
            geometry["centroid_x"],
            geometry["centroid_y"],
            geometry["radius"]
        )

        st.markdown(
            '<div class="section-title">'
            '〰️ Dosa Edge Quality'
            '</div>',
            unsafe_allow_html=True
        )

        e1, e2 = st.columns(2)

        with e1:

            st.metric(
                "Edge Quality",
                f"{edge_quality:.1f}%"
            )

        with e2:

            st.metric(
                "Edge Irregularity",
                f"{edge_irregularity:.1f}%"
            )

        if edge_quality >= 90:

            st.success(
                "〰️ Excellent edge! "
                "The boundary is behaving itself."
            )

        elif edge_quality >= 75:

            st.info(
                "〰️ Decent edge quality."
            )

        else:

            st.warning(
                "〰️ The edges have chosen chaos."
            )


        # ====================================================
        # STEP 9 - BROWNING
        # ====================================================

        (
            browning_score,
            browning_heatmap
        ) = calculate_browning(
            image,
            mask
        )

        st.markdown(
            '<div class="section-title">'
            '🔥 Browning Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Browning Score",
            f"{browning_score:.1f}%"
        )

        st.caption(
            "This is a visual image-based estimate of "
            "browning distribution. It is not a cooking "
            "temperature measurement."
        )

        if browning_heatmap is not None:

            heatmap_rgb = cv2.cvtColor(
                browning_heatmap,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                heatmap_rgb,
                caption="Browning Distribution Heatmap",
                use_container_width=True
            )


        # ====================================================
        # STEP 10 - VISIBLE PORES
        # ====================================================

        (
            pore_score,
            pore_visual,
            pore_count
        ) = calculate_pores(
            image,
            mask
        )

        st.markdown(
            '<div class="section-title">'
            '🕳️ Visible Pores'
            '</div>',
            unsafe_allow_html=True
        )

        p1, p2 = st.columns(2)

        with p1:

            st.metric(
                "Candidate Visible Pores",
                str(pore_count)
            )

        with p2:

            st.metric(
                "Pore Score",
                f"{pore_score:.1f}%"
            )

        st.caption(
            "Pores are detected as candidate dark regions "
            "inside the dosa. Some dark cooking spots may "
            "also be detected."
        )

        if pore_visual is not None:

            pore_visual_rgb = cv2.cvtColor(
                pore_visual,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                pore_visual_rgb,
                caption="Detected Dark/Pore-like Regions",
                use_container_width=True
            )


        # ====================================================
        # STEP 11 - OVERALL DOSA SCORE
        # ====================================================

        overall_score = calculate_overall_score(
            roundness_score,
            overall_symmetry,
            edge_quality,
            center_accuracy,
            browning_score,
            pore_score
        )

        st.markdown(
            '<div class="section-title">'
            '🏆 Overall Dosa Score'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="score-box">
                <div class="metric-label">
                    DOSA DNA SCORE
                </div>
                <div class="big-score">
                    {overall_score}/100
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # SCORE BREAKDOWN
        # ====================================================

        st.subheader(
            "📊 Score Breakdown"
        )

        b1, b2, b3 = st.columns(3)

        with b1:

            st.write(
                f"⭕ **Roundness:** "
                f"{roundness_score:.1f}%"
            )

            st.write(
                f"⚖️ **Symmetry:** "
                f"{overall_symmetry:.1f}%"
            )

        with b2:

            st.write(
                f"〰️ **Edge Quality:** "
                f"{edge_quality:.1f}%"
            )

            st.write(
                f"🎯 **Center Accuracy:** "
                f"{center_accuracy:.1f}%"
            )

        with b3:

            st.write(
                f"🔥 **Browning:** "
                f"{browning_score:.1f}%"
            )

            st.write(
                f"🕳️ **Visible Pores:** "
                f"{pore_score:.1f}%"
            )


        # ====================================================
        # FORMULA
        # ====================================================

        st.info(
            "Final score = "
            "Roundness × 25% + "
            "Symmetry × 20% + "
            "Edge Quality × 15% + "
            "Center Accuracy × 15% + "
            "Browning × 15% + "
            "Visible Pores × 10%"
        )


        # ====================================================
        # CURRENT STATUS
        # ====================================================

        if overall_score >= 90:

            st.success(
                f"🏆 Outstanding Dosa! "
                f"Final Score: {overall_score}/100"
            )

        elif overall_score >= 70:

            st.info(
                f"🥞 Decent Dosa! "
                f"Final Score: {overall_score}/100"
            )

        else:

            st.warning(
                f"🥴 This dosa needs some explanation. "
                f"Final Score: {overall_score}/100"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🥞 DOSA DNA — Because measuring dosa geometry "
    "is definitely more important than it sounds."
)