import streamlit as st
import pandas as pd
import re

st.title("Resale Fashion Price Analysis")
st.write("Exploring which brands and words are associated with higher resale prices.")

df = pd.read_csv('updated_dataset.csv')

# Brand chart
st.header("Average Price by Brand")
brand_prices = df.groupby('brand')['price'].mean().sort_values(ascending=False)
st.bar_chart(brand_prices)

# Interactive brand filter
st.header("Explore by Brand")
selected_brand = st.selectbox("Choose a brand", df['brand'].unique())
filtered = df[df['brand'] == selected_brand]
st.write(f"Average price for {selected_brand}: ${filtered['price'].mean():.2f}")
st.write(f"Number of listings: {len(filtered)}")

# Word-price chart
st.header("Top Words by Average Price")

def clean_split(title):
    title = title.lower()
    title = re.sub(r'[^a-z\s]', ' ', title)
    return title.split()

all_words = df['title'].apply(clean_split).explode()
word_counts = all_words.value_counts()
common_words = word_counts[word_counts > 50].index

word_prices = {}
for word in common_words:
    mask = df['title'].str.lower().str.contains(word, na=False)
    word_prices[word] = df.loc[mask, 'price'].mean()

word_price_series = pd.Series(word_prices).sort_values(ascending=False).head(15)
st.bar_chart(word_price_series)