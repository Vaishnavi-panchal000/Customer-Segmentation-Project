import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation Dashboard")
st.write(
    "Segment customers based on demographics and purchasing behavior "
    "using K-Means clustering."
)

# Load data
df = pd.read_csv("customer_data.csv")

# Sidebar filters
st.sidebar.header("🔎 Dashboard Filters")

genders = st.sidebar.multiselect(
    "Gender",
    sorted(df["Gender"].unique()),
    default=sorted(df["Gender"].unique())
)

cities = st.sidebar.multiselect(
    "City",
    sorted(df["City"].unique()),
    default=sorted(df["City"].unique())
)

categories = st.sidebar.multiselect(
    "Preferred Category",
    sorted(df["Preferred_Category"].unique()),
    default=sorted(df["Preferred_Category"].unique())
)

filtered = df[
    df["Gender"].isin(genders)
    & df["City"].isin(cities)
    & df["Preferred_Category"].isin(categories)
].copy()

if len(filtered) < 4:
    st.warning("Please select enough customers to perform clustering.")
    st.stop()

# K-Means features
features = [
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency",
    "Avg_Order_Value",
    "Website_Visits"
]

scaler = StandardScaler()
X = scaler.fit_transform(filtered[features])

# Four customer segments
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
filtered["Cluster"] = kmeans.fit_predict(X)

# Create meaningful names based on cluster profile
profile = filtered.groupby("Cluster")[features].mean()
profile["Value_Index"] = (
    profile["Annual_Income"].rank(pct=True)
    + profile["Spending_Score"].rank(pct=True)
    + profile["Purchase_Frequency"].rank(pct=True)
) / 3

ordered = profile.sort_values("Value_Index").index.tolist()
name_map = {
    ordered[0]: "Budget Customers",
    ordered[1]: "Potential Customers",
    ordered[2]: "Regular Customers",
    ordered[3]: "High-Value Customers"
}
filtered["Segment"] = filtered["Cluster"].map(name_map)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 Customers", f"{len(filtered):,}")
c2.metric("💰 Avg. Income", f"₹{filtered['Annual_Income'].mean():,.0f}")
c3.metric("🛒 Avg. Spend", f"₹{filtered['Total_Spend'].mean():,.0f}")
c4.metric("⭐ Avg. Spending Score", f"{filtered['Spending_Score'].mean():.1f}")

st.divider()

# Segment distribution
segment_counts = (
    filtered["Segment"].value_counts()
    .rename_axis("Segment")
    .reset_index(name="Customers")
)

col1, col2 = st.columns(2)

fig_segments = px.bar(
    segment_counts,
    x="Segment",
    y="Customers",
    title="👥 Customers by Segment",
    text_auto=True
)
col1.plotly_chart(fig_segments, use_container_width=True)

fig_income_spend = px.scatter(
    filtered,
    x="Annual_Income",
    y="Spending_Score",
    color="Segment",
    size="Total_Spend",
    hover_data=[
        "Customer_ID", "Age", "Gender", "City",
        "Purchase_Frequency", "Preferred_Category"
    ],
    title="💰 Income vs Spending Score"
)
col2.plotly_chart(fig_income_spend, use_container_width=True)

# Purchase behavior
col1, col2 = st.columns(2)

fig_frequency = px.scatter(
    filtered,
    x="Purchase_Frequency",
    y="Avg_Order_Value",
    color="Segment",
    size="Total_Spend",
    hover_data=["Customer_ID", "City", "Preferred_Category"],
    title="🛒 Purchase Frequency vs Average Order Value"
)
col1.plotly_chart(fig_frequency, use_container_width=True)

category_segment = (
    filtered.groupby(["Segment", "Preferred_Category"])
    .size()
    .reset_index(name="Customers")
)

fig_category = px.bar(
    category_segment,
    x="Segment",
    y="Customers",
    color="Preferred_Category",
    barmode="group",
    title="🛍️ Preferred Categories by Segment"
)
col2.plotly_chart(fig_category, use_container_width=True)

# Segment profile table
st.subheader("📊 Segment Profile")

segment_profile = (
    filtered.groupby("Segment")[features]
    .mean()
    .round(1)
    .reset_index()
)

st.dataframe(segment_profile, use_container_width=True)

# Business insights
st.subheader("💡 Business Insights")

segment_avg = filtered.groupby("Segment")["Total_Spend"].mean()
top_segment = segment_avg.idxmax()
top_category = filtered["Preferred_Category"].value_counts().idxmax()
top_city = filtered["City"].value_counts().idxmax()

st.write(f"• **Highest average-spending segment:** {top_segment}")
st.write(f"• **Most preferred category:** {top_category}")
st.write(f"• **City with the most customers:** {top_city}")
st.write("• **High-Value Customers:** focus on loyalty rewards, premium offers, and personalized recommendations.")
st.write("• **Potential Customers:** use targeted promotions and product recommendations to increase purchase frequency.")
st.write("• **Budget Customers:** offer discounts, bundles, and value-oriented products.")
st.write("• **Regular Customers:** use loyalty programs and cross-selling to improve retention and spending.")

# Filtered customer data
st.subheader("📋 Customer Data")
st.dataframe(
    filtered[
        [
            "Customer_ID", "Age", "Gender", "City", "Annual_Income",
            "Spending_Score", "Purchase_Frequency", "Avg_Order_Value",
            "Total_Spend", "Website_Visits", "Preferred_Category", "Segment"
        ]
    ],
    use_container_width=True
)

st.divider()
st.caption("Customer Segmentation Project | Python, Pandas, Scikit-learn, Streamlit and Plotly")
