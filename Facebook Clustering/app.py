import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import base64


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Facebook Live Engagement Clustering",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "kmeans_live_model.pkl"
FEATURES_PATH = BASE_DIR / "kmeans_live_features.pkl"
DATA_PATH = BASE_DIR / "Live.csv"
BACKGROUND_PATH = BASE_DIR / "background.jpg"


# ============================================================
# FEATURE LABELS
# ============================================================

FEATURE_LABELS = {
    "num_reactions": "Total Reactions",
    "num_comments": "Comments",
    "num_shares": "Shares",
    "num_likes": "Likes",
    "num_loves": "Loves",
    "num_wows": "Wows",
    "num_hahas": "Hahas",
    "num_sads": "Sads",
    "num_angrys": "Angrys"
}


# ============================================================
# OPTIONAL BACKGROUND
# ============================================================

def get_background_css():
    if not BACKGROUND_PATH.exists():
        return ".stApp { background: #f4f7fb; }"

    try:
        with open(BACKGROUND_PATH, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()

        return f"""
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255, 255, 255, 0.90),
                    rgba(255, 255, 255, 0.90)
                ),
                url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    except Exception:
        return ".stApp { background: #f4f7fb; }"


background_css = get_background_css()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
<style>

{background_css}

.main-title {{
    background: linear-gradient(135deg, #1877f2, #00a896);
    padding: 32px 38px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(24, 119, 242, 0.18);
}}

.main-title h1 {{
    color: white;
    margin: 0;
    font-size: 38px;
    font-weight: 750;
}}

.main-title p {{
    color: #eef7ff;
    font-size: 17px;
    margin-top: 10px;
}}

.card {{
    background: rgba(255, 255, 255, 0.96);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.06);
    margin-bottom: 18px;
}}

.card-title {{
    color: #1877f2;
    font-size: 21px;
    font-weight: 700;
}}

.card-text {{
    color: #64748b;
    font-size: 15px;
    margin-top: 7px;
}}

.cluster-zero {{
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-left: 6px solid #1877f2;
    padding: 25px;
    border-radius: 15px;
    margin-top: 15px;
}}

.cluster-one {{
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border-left: 6px solid #00a896;
    padding: 25px;
    border-radius: 15px;
    margin-top: 15px;
}}

.cluster-title {{
    font-size: 27px;
    font-weight: 750;
}}

.cluster-description {{
    color: #475569;
    margin-top: 8px;
    font-size: 15px;
}}

.stButton > button {{
    width: 100%;
    min-height: 50px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #1877f2, #00a896);
    color: white;
    font-size: 16px;
    font-weight: 700;
}}

.stButton > button:hover {{
    background: linear-gradient(135deg, #145db2, #008f80);
    color: white;
}}

section[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.97);
}}

div[data-testid="stMetric"] {{
    background: rgba(255, 255, 255, 0.96);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}}

.footer {{
    text-align: center;
    color: #64748b;
    padding: 30px 0 10px 0;
    font-size: 13px;
}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():
    st.error("❌ kmeans_live_model.pkl was not found.")
    st.info(f"Put the pickle file here:\n\n{MODEL_PATH}")
    st.stop()

if not FEATURES_PATH.exists():
    st.error("❌ kmeans_live_features.pkl was not found.")
    st.info(f"Put the feature pickle file here:\n\n{FEATURES_PATH}")
    st.stop()

if not DATA_PATH.exists():
    st.warning(
        "⚠️ Live.csv was not found. Prediction will still work, "
        "but dataset graphs will not be available."
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_features():
    with open(FEATURES_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    feature_names = load_features()
    df = load_dataset()

except Exception as error:
    st.error("❌ Could not load the model, features, or dataset.")
    st.exception(error)
    st.stop()


# ============================================================
# VALIDATE FEATURES
# ============================================================

if not isinstance(feature_names, list):
    feature_names = list(feature_names)

if df is not None:
    missing_features = [
        feature for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:
        st.error(
            "❌ These model features are missing from Live.csv:"
        )
        st.write(missing_features)
        st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="main-title">
    <h1>📊 Facebook Live Engagement Clustering</h1>
    <p>
        K-Means Machine Learning model for grouping Facebook Live
        posts according to audience engagement patterns.
    </p>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📊 K-Means AI")

    st.markdown("---")

    st.markdown("### About")

    st.write(
        """
        This application uses a trained K-Means clustering model
        to group Facebook Live posts based on their engagement.
        """
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.success("K-Means Clustering")

    st.markdown("---")

    st.markdown("### 📌 Features")

    for feature in feature_names:
        st.write(f"• {FEATURE_LABELS.get(feature, feature)}")

    st.markdown("---")

    st.caption(
        "Cluster labels are machine-learning groups. "
        "They do not automatically represent good/bad performance."
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

if df is not None:

    st.markdown("### 📈 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Posts", f"{len(df):,}")

    with c2:
        st.metric("Features", len(feature_names))

    with c3:
        st.metric(
            "Average Reactions",
            f"{df['num_reactions'].mean():,.0f}"
        )

    with c4:
        st.metric(
            "Average Comments",
            f"{df['num_comments'].mean():,.0f}"
        )


# ============================================================
# PREDICTION INPUT
# ============================================================

st.markdown("## 🔮 Predict Cluster")

st.markdown(
    """
<div class="card">
    <div class="card-title">Enter Facebook Live Engagement</div>
    <div class="card-text">
        Enter the engagement values for a post. The trained K-Means
        model will assign the post to Cluster 0 or Cluster 1.
    </div>
</div>
""",
    unsafe_allow_html=True
)


# Default values based on the uploaded dataset when available.
defaults = {}

if df is not None:
    for feature in feature_names:
        defaults[feature] = float(df[feature].median())
else:
    for feature in feature_names:
        defaults[feature] = 0.0


left, right = st.columns(2)

input_values = {}


for index, feature in enumerate(feature_names):

    column = left if index < 5 else right

    with column:

        label = FEATURE_LABELS.get(feature, feature)

        max_value = 1000000.0

        if df is not None:
            dataset_max = float(df[feature].max())
            max_value = max(100.0, dataset_max * 2)

        input_values[feature] = st.number_input(
            label,
            min_value=0.0,
            max_value=max_value,
            value=max(0.0, defaults[feature]),
            step=1.0,
            format="%.0f",
            key=f"input_{feature}"
        )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    [[input_values[feature] for feature in feature_names]],
    columns=feature_names
)


# ============================================================
# INPUT PREVIEW
# ============================================================

with st.expander("🔍 View Input Data"):

    display_input = input_data.T.reset_index()
    display_input.columns = ["Feature", "Value"]

    display_input["Feature"] = display_input["Feature"].map(
        lambda x: FEATURE_LABELS.get(x, x)
    )

    st.dataframe(
        display_input,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

predict = st.button(
    "🔎  PREDICT CLUSTER"
)


# ============================================================
# CLUSTER PREDICTION
# ============================================================

if predict:

    try:

        prediction = int(model.predict(input_data)[0])

        # Distance from each cluster center, if available.
        distances = None

        if hasattr(model, "named_steps"):
            kmeans_step = model.named_steps.get("kmeans")

            if kmeans_step is not None:
                transformed_input = model.named_steps[
                    "scaler"
                ].transform(input_data)

                distances = kmeans_step.transform(
                    transformed_input
                )[0]

        elif hasattr(model, "transform"):
            distances = model.transform(input_data)[0]


        st.markdown("---")

        st.markdown("## 🎯 Prediction Result")


        if prediction == 0:

            st.markdown(
                """
<div class="cluster-zero">
    <div class="cluster-title">🔵 Cluster 0</div>
    <div class="cluster-description">
        This Facebook Live post has been assigned to
        <b>Cluster 0</b> by the trained K-Means model.
    </div>
</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
<div class="cluster-one">
    <div class="cluster-title">🟢 Cluster 1</div>
    <div class="cluster-description">
        This Facebook Live post has been assigned to
        <b>Cluster 1</b> by the trained K-Means model.
    </div>
</div>
""",
                unsafe_allow_html=True
            )


        # ====================================================
        # DISTANCE TO CLUSTERS
        # ====================================================

        if distances is not None:

            st.markdown("### 📏 Distance from Cluster Centers")

            distance_df = pd.DataFrame({
                "Cluster": [
                    "Cluster 0",
                    "Cluster 1"
                ],
                "Distance": [
                    float(distances[0]),
                    float(distances[1])
                ]
            })

            distance_fig = px.bar(
                distance_df,
                x="Cluster",
                y="Distance",
                text="Distance"
            )

            distance_fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            distance_fig.update_layout(
                height=350,
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="rgba(255,255,255,0.92)",
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis_title="Distance",
                xaxis_title="Cluster"
            )

            st.plotly_chart(
                distance_fig,
                use_container_width=True
            )


        # ====================================================
        # MODEL CENTERS
        # ====================================================

        if hasattr(model, "named_steps"):

            kmeans_step = model.named_steps.get("kmeans")
            scaler_step = model.named_steps.get("scaler")

            if (
                kmeans_step is not None
                and scaler_step is not None
                and hasattr(kmeans_step, "cluster_centers_")
            ):

                st.markdown("### 🎯 Cluster Center Comparison")

                centers_scaled = kmeans_step.cluster_centers_

                centers_original = scaler_step.inverse_transform(
                    centers_scaled
                )

                centers_df = pd.DataFrame(
                    centers_original,
                    columns=feature_names
                )

                centers_df.index = [
                    "Cluster 0",
                    "Cluster 1"
                ]

                selected_feature = st.selectbox(
                    "Select feature for cluster comparison",
                    feature_names,
                    format_func=lambda x: FEATURE_LABELS.get(x, x)
                )

                center_plot_df = pd.DataFrame({
                    "Cluster": [
                        "Cluster 0",
                        "Cluster 1"
                    ],
                    "Average Value": [
                        centers_df.loc[
                            "Cluster 0",
                            selected_feature
                        ],
                        centers_df.loc[
                            "Cluster 1",
                            selected_feature
                        ]
                    ]
                })

                center_fig = px.bar(
                    center_plot_df,
                    x="Cluster",
                    y="Average Value",
                    text="Average Value"
                )

                center_fig.update_traces(
                    texttemplate="%{text:.2f}",
                    textposition="outside"
                )

                center_fig.update_layout(
                    height=350,
                    showlegend=False,
                    plot_bgcolor="white",
                    paper_bgcolor="rgba(255,255,255,0.92)",
                    yaxis_title=FEATURE_LABELS.get(
                        selected_feature,
                        selected_feature
                    ),
                    xaxis_title="Cluster"
                )

                st.plotly_chart(
                    center_fig,
                    use_container_width=True
                )


    except Exception as error:

        st.error("❌ Cluster prediction failed.")
        st.exception(error)


# ============================================================
# DATASET CLUSTER ANALYSIS
# ============================================================

if df is not None:

    st.markdown("---")
    st.markdown("## 📊 Dataset Cluster Analysis")


    # Generate clusters for the complete dataset using the
    # same trained model.
    try:

        dataset_for_model = df[feature_names].copy()

        df_analysis = df.copy()

        df_analysis["Cluster"] = model.predict(
            dataset_for_model
        )

        # ----------------------------------------------------
        # CLUSTER DISTRIBUTION
        # ----------------------------------------------------

        st.markdown("### 🧩 Cluster Distribution")

        cluster_counts = (
            df_analysis["Cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        cluster_counts.columns = [
            "Cluster",
            "Posts"
        ]

        cluster_counts["Cluster"] = (
            cluster_counts["Cluster"]
            .astype(str)
            .apply(lambda x: f"Cluster {x}")
        )

        distribution_fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Posts",
            text="Posts"
        )

        distribution_fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside"
        )

        distribution_fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="rgba(255,255,255,0.92)",
            xaxis_title="Cluster",
            yaxis_title="Number of Posts"
        )

        st.plotly_chart(
            distribution_fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # CLUSTER PIE CHART
        # ----------------------------------------------------

        st.markdown("### 🥧 Cluster Share")

        pie_fig = px.pie(
            cluster_counts,
            names="Cluster",
            values="Posts",
            hole=0.45
        )

        pie_fig.update_layout(
            height=400,
            paper_bgcolor="rgba(255,255,255,0.92)"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # ENGAGEMENT COMPARISON
        # ----------------------------------------------------

        st.markdown("### 📈 Average Engagement by Cluster")

        comparison_features = [
            "num_reactions",
            "num_comments",
            "num_shares",
            "num_likes",
            "num_loves",
            "num_wows",
            "num_hahas",
            "num_sads",
            "num_angrys"
        ]

        available_comparison_features = [
            feature
            for feature in comparison_features
            if feature in df_analysis.columns
        ]

        avg_cluster = (
            df_analysis
            .groupby("Cluster")[available_comparison_features]
            .mean()
            .reset_index()
        )

        selected_avg_feature = st.selectbox(
            "Select engagement feature",
            available_comparison_features,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            key="average_feature"
        )

        avg_plot_df = pd.DataFrame({
            "Cluster": avg_cluster["Cluster"].apply(
                lambda x: f"Cluster {x}"
            ),
            "Average": avg_cluster[selected_avg_feature]
        })

        avg_fig = px.bar(
            avg_plot_df,
            x="Cluster",
            y="Average",
            text="Average"
        )

        avg_fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        avg_fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="rgba(255,255,255,0.92)",
            xaxis_title="Cluster",
            yaxis_title=FEATURE_LABELS.get(
                selected_avg_feature,
                selected_avg_feature
            )
        )

        st.plotly_chart(
            avg_fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # SCATTER PLOT
        # ----------------------------------------------------

        st.markdown("### 🔵 Engagement Relationship")

        scatter_left, scatter_right = st.columns(2)

        with scatter_left:

            x_feature = st.selectbox(
                "X-axis feature",
                available_comparison_features,
                index=0,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key="scatter_x"
            )

        with scatter_right:

            y_feature = st.selectbox(
                "Y-axis feature",
                available_comparison_features,
                index=1 if len(available_comparison_features) > 1 else 0,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key="scatter_y"
            )

        scatter_fig = px.scatter(
            df_analysis,
            x=x_feature,
            y=y_feature,
            color=df_analysis["Cluster"].astype(str),
            hover_data=feature_names,
            opacity=0.65,
            labels={
                x_feature: FEATURE_LABELS.get(
                    x_feature,
                    x_feature
                ),
                y_feature: FEATURE_LABELS.get(
                    y_feature,
                    y_feature
                ),
                "color": "Cluster"
            }
        )

        scatter_fig.update_layout(
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="rgba(255,255,255,0.92)"
        )

        st.plotly_chart(
            scatter_fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # DATA TABLE
        # ----------------------------------------------------

        with st.expander("🔍 View Dataset with Cluster Labels"):

            display_df = df_analysis.copy()

            display_df["Cluster"] = display_df["Cluster"].apply(
                lambda x: f"Cluster {x}"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


    except Exception as error:

        st.error("❌ Dataset cluster analysis could not be generated.")
        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    📊 Facebook Live Engagement Clustering<br>
    K-Means Machine Learning • Python • Streamlit • Plotly
</div>
""",
    unsafe_allow_html=True
)
