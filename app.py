import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
 
st.set_page_config(page_title="Predicting CPI of Vietnam", layout="wide")
st.title("Predicting CPI by number of disasters")


# 1. Load model
def load_model_year():
    return joblib.load("pred_CPI_year.pkl")

def load_model_disasters():
    return joblib.load("predCPI_disasterAmount.pkl")
 
year_model = load_model_year()
 
disasters_model = load_model_disasters()

# 2. Value
number_of_disasters = st.slider(label="Number of disasters since 1989", min_value=0, max_value=1000, step=1)

# 3. Predict
if st.button("Predict"):
    diem = disasters_model.predict([[number_of_disasters]])[0]
    st.metric("Predicted CPI", f"{diem:.2f}")

# 4. Graph
    # 4.1 DataFrame
df = pd.read_csv("data/data.csv")
df = df[df["Start Year"] > 1967]
df  = df[["Start Year", "CPI", "Total Damage, Adjusted ('000 US$)"]]
df = df.dropna(subset = "Total Damage, Adjusted ('000 US$)")
df[["Sum"]] = df.groupby(["Start Year"])[["CPI"]].transform("count")
df["Total Damage, Adjusted ('000 US$)"] = df.groupby(["Start Year"])["Total Damage, Adjusted ('000 US$)"].transform("sum")
df["cumsum"] = df["Sum"].cumsum()

    # 4.2 Graph drawing
fig = px.scatter(df, x="cumsum", y="CPI", title="Data and Regression Line")
x_range = [[i] for i in range(0, 41)]
y_pred = disasters_model.predict(x_range)
fig.add_scatter(
    x=list(range(0, 700)), 
    y=y_pred,
    mode="lines", name="Regression Line",
    line=dict(color="red")
)

fig.update_xaxes(title_text="Disasters Since 1989")

st.plotly_chart(fig, use_container_width=True)
