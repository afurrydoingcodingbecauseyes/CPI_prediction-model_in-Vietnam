import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Predicting CPI of Vietnam", layout="wide")
st.title("Predicting CPI by nmber of disasters")


# 1. Load model
def load_model_year():
  return joblib.load("pred_CPI_year.pkl")


def load_model_disasters():
  return joblib.load("predCPI_disasterAmount.pkl")


year_model = load_model_year()
disasters_model = load_model_disasters()


# 2. Disaster Values (Moved up so the slider value is available for the graph)
number_of_disasters = st.slider(
    label="Number of disasters since 1989", min_value=0, max_value=1000, step=1
)

# 3. Predict calculation (Calculated early so we can plot it)
predicted_cpi = disasters_model.predict([[number_of_disasters]])[0]


# 4. First Graph
# 4.1 DataFrame
df = pd.read_csv("data/data.csv")
df = df[df["Start Year"] > 1967]
df = df[["Start Year", "CPI", "Total Damage, Adjusted ('000 US$)"]]
df = df.dropna(subset="Total Damage, Adjusted ('000 US$)")
df[["Sum"]] = df.groupby(["Start Year"])[["CPI"]].transform("count")
df["Total Damage, Adjusted ('000 US$)"] = df.groupby(["Start Year"])[
    "Total Damage, Adjusted ('000 US$)"
].transform("sum")
df["cumsum"] = df["Sum"].cumsum()

# 4.2 Graph drawing
fig = px.scatter(df, x="cumsum", y="CPI", title="Data and Regression Line")

x_range = [[i] for i in range(0, 1000)]
y_pred = disasters_model.predict(x_range)

# Regression line
fig.add_scatter(
    x=list(range(0, 1000)),
    y=y_pred,
    mode="lines",
    name="Regression Line",
    line=dict(color="red"),
)

# Vertical line showing slider value
fig.add_vline(
    x=number_of_disasters,
    line_dash="dash",
    line_color="green",
    annotation_text=f"Selected: {st.metric("Predicted CPI", f"{predicted_cpi:.2f}")}",
)

# Point showing predicted CPI
fig.add_scatter(
    x=[number_of_disasters],
    y=[predicted_cpi],
    mode="markers",
    marker=dict(size=15, color="green", symbol="circle"),
    name="Prediction",
)

fig.update_xaxes(title_text="Disasters Since 1989")

# 4.3 Render Graph
st.plotly_chart(fig, use_container_width=True)

