import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly_express as px

df = pd.read_csv("messy_data.csv", sep=', ')
df.columns = [col.strip() for col in df.columns]
df.clarity = df.clarity.str.upper()
df.color = df.color.str.title()
df.cut = df.cut.str.title()
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df = df[df.price < 100000]
df['x dimension'] = df['x dimension'].astype(float)
df['y dimension'] = df['y dimension'].astype(float)
df['z dimension'] = df['z dimension'].astype(float)
df['carat'] = df['carat'].astype(float)
df['depth'] = df['depth'].astype(float)
df['table'] = df['table'].astype(float)
df.interpolate(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
table=df

st.title("Projekt PAD Mikołaj Cieślak s32422")
st.subheader("Oto oczyszczone dane. Zakładając, że dane są posegregowane, użyłem funkcji interpolate aby uzyskać nieistniejące wartości. Elementy które pozostały jako NaN po interpolate, usunąłem.")
st.dataframe(table)

plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
x_axis= st.selectbox('Wybór wartości dla osi X', options = ['carat', 'depth', 'price', 'x dimension', 'y dimension', 'z dimension'])
y_axis= st.selectbox('Wybór wartości dla osi Y', options = ['carat', 'depth', 'price', 'x dimension', 'y dimension', 'z dimension'])
plot = px.scatter(df, df[x_axis], df[y_axis])
st.plotly_chart(plot)

st.subheader("Na podstawie niżej załączonej heatmapy można dojść do wniosku, iż najbardziej cena jest zależna od wymiaru X")

fig = plt.figure()
sns.heatmap(df[["carat", "x dimension", "y dimension", "z dimension", "depth", 'price']].corr(), annot=True, center=0.0)
plt.title("Heatmapa zależności")
st.pyplot(fig)

st.subheader("Zatem największy sens jest sprawdzić regresję liniową dla ceny w zależności od wymiaru X")
fig = plt.figure()
X = sm.add_constant(df['x dimension'])
y = df['price']
lm = sm.OLS(y, X)
lm_fit = lm.fit()
lm_fit.summary()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

df.plot(kind='scatter', y='price', x='x dimension', ax=ax1)
(b0, b1) = lm_fit.params

ax1.axline(xy1=(0,b0), slope=b1, color='r')

ax2.scatter(df.price, lm_fit.resid)
ax2.axhline(0, linestyle='--', color='r')
ax2.set_xlabel('x dimension')
ax2.set_ylabel('price')
st.pyplot(fig)