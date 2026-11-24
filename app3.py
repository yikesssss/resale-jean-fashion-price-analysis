import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Resale Listing Keyword & Price Analyzer",
    page_icon="🏷️",
    layout="wide",
)

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_data(path: str = "updated_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["title_clean"] = (
        df["title"]
        .fillna("")
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
    )
    df["condition_label"] = (
        df["condition"]
        .str.replace("is_", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.title()
    )
    df["has_price_drop"] = df["price_drops"].notna() & (
        df["price_drops"] != "[nan, nan, nan, nan]"
    )
    return df


df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Dataset")
selected_brands = st.sidebar.multiselect(
    "Brands",
    options=sorted(df["brand"].unique()),
    default=sorted(df["brand"].unique()),
)
selected_conditions = st.sidebar.multiselect(
    "Item Condition",
    options=sorted(df["condition_label"].unique()),
    default=sorted(df["condition_label"].unique()),
)
max_price_cap = int(df["price"].quantile(0.99))
price_range = st.sidebar.slider(
    "Listing Price Range ($)",
    min_value=int(df["price"].min()),
    max_value=int(df["price"].max()),
    value=(int(df["price"].min()), max_price_cap),
    help="Default upper bound is set to the 99th percentile to reduce extreme outlier distortion.",
)

filtered_df = df[
    (df["brand"].isin(selected_brands))
    & (df["condition_label"].isin(selected_conditions))
    & (df["price"].between(price_range[0], price_range[1]))
]

if filtered_df.empty:
    st.warning("No listings match your current filters. Adjust the sidebar settings.")
    st.stop()

# --- HEADER & KPIs ---
st.title("🏷️ Do Listing Keywords Actually Increase Resale Prices?")
st.markdown(
    "An empirical analysis of **5,440 fashion resale listings** testing whether hype words "
    "(*rare, grail, vintage*) or concrete attributes (*leather, runway, patchwork, NWT*) "
    "drive higher asking prices once **brand** and **condition** are controlled."
)
st.info(
    "**Methodological Note:** All rows represent active listings (`ListProduct`). "
    "Because resale prices are heavily right-skewed (max: $29,950), this dashboard uses **Median Price** "
    "as its primary metric so single ultra-rare items do not distort keyword effects."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Listings", f"{len(filtered_df):,}")
col2.metric("Median Listing Price", f"${filtered_df['price'].median():,.0f}")
col3.metric("Mean Listing Price", f"${filtered_df['price'].mean():,.0f}")
col4.metric("Listings w/ Price Drops", f"{filtered_df['has_price_drop'].mean()*100:.1f}%")

st.divider()

# --- TABS FOR STRUCTURED EXPLORATION ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Hype vs. Material Keywords",
        "2. Custom Keyword Tester",
        "3. Top Words Within a Brand",
        "4. Listing Title Price Estimator",
    ]
)

# ==========================================
# TAB 1: CURATED KEYWORD CATEGORY BENCHMARK
# ==========================================
with tab1:
    st.subheader("Hype Buzzwords vs. Verifiable Material & Provenance Words")
    st.write(
        "Compare how specific keywords shift the **median price** relative to listings "
        "without that keyword inside your current brand/condition selection."
    )

    keyword_groups = {
        "Hype / Subjective": ["rare", "vintage", "grail", "archive", "limited", "custom"],
        "Condition / Tag": ["new", "nwt", "bnwt", "deadstock"],
        "Material / Craft / Provenance": [
            "leather",
            "waxed",
            "patchwork",
            "boro",
            "selvedge",
            "distressed",
            "runway",
            "japan",
            "japanese",
        ],
    }

    rows = []
    for category, words in keyword_groups.items():
        for w in words:
            mask = filtered_df["title_clean"].str.contains(rf"\b{w}\b", regex=True)
            count_with = int(mask.sum())
            if count_with >= 5:
                med_with = filtered_df.loc[mask, "price"].median()
                med_without = filtered_df.loc[~mask, "price"].median()
                lift_pct = ((med_with - med_without) / med_without) * 100
                drop_rate = filtered_df.loc[mask, "has_price_drop"].mean() * 100
                rows.append(
                    {
                        "Keyword": w,
                        "Type": category,
                        "Listings (n)": count_with,
                        "Median With ($)": med_with,
                        "Median Without ($)": med_without,
                        "Price Lift (%)": round(lift_pct, 1),
                        "Price Drop Rate (%)": round(drop_rate, 1),
                    }
                )

    kw_df = pd.DataFrame(rows).sort_values("Price Lift (%)", ascending=True)

    if not kw_df.empty:
        fig_lift = px.bar(
            kw_df,
            x="Price Lift (%)",
            y="Keyword",
            color="Type",
            orientation="h",
            text="Price Lift (%)",
            hover_data=["Listings (n)", "Median With ($)", "Median Without ($)", "Price Drop Rate (%)"],
            title="Median Price Lift (%) When Keyword Appears in Title (min. 5 listings)",
            color_discrete_map={
                "Hype / Subjective": "#EF553B",
                "Condition / Tag": "#636EFA",
                "Material / Craft / Provenance": "#00CC96",
            },
        )
        fig_lift.update_traces(texttemplate="%{text:+.1f}%", textposition="outside")
        fig_lift.update_layout(height=520, xaxis_title="Median Price Difference vs. Listings Without Word (%)")
        st.plotly_chart(fig_lift, use_container_width=True)

        st.dataframe(
            kw_df.sort_values("Price Lift (%)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Not enough keyword matches in the current filter selection (minimum 5 listings required).")

# ==========================================
# TAB 2: INTERACTIVE CUSTOM KEYWORD TESTER
# ==========================================
with tab2:
    st.subheader("Test Any Word or Phrase Across Brands")
    st.write(
        "Type any keyword (e.g., `rare`, `new`, `leather`, `selvedge`, `runway`, `exquisite`) "
        "to see if its price premium holds **within each individual brand** or is just a brand illusion."
    )

    user_kw = st.text_input("Enter a keyword to test:", value="rare").strip().lower()

    if user_kw:
        clean_kw = re.sub(r"[^a-z0-9\s]", "", user_kw)
        pattern = rf"\b{re.escape(clean_kw)}\b"
        has_kw = filtered_df["title_clean"].str.contains(pattern, regex=True)

        total_matches = int(has_kw.sum())
        if total_matches == 0:
            st.warning(f"No listings found containing the word **'{clean_kw}'** in the current filter.")
        else:
            m1, m2, m3 = st.columns(3)
            med_w = filtered_df.loc[has_kw, "price"].median()
            med_wo = filtered_df.loc[~has_kw, "price"].median()
            overall_lift = ((med_w - med_wo) / med_wo) * 100

            m1.metric("Matching Listings", f"{total_matches:,}")
            m2.metric(f"Median WITH '{clean_kw}'", f"${med_w:,.0f}", f"{overall_lift:+.1f}% vs without")
            m3.metric(f"Median WITHOUT '{clean_kw}'", f"${med_wo:,.0f}")

            # Brand-by-brand breakdown
            brand_rows = []
            for brand_name, grp in filtered_df.groupby("brand"):
                b_mask = grp["title_clean"].str.contains(pattern, regex=True)
                n_w = int(b_mask.sum())
                if n_w > 0 and (~b_mask).sum() > 0:
                    b_med_w = grp.loc[b_mask, "price"].median()
                    b_med_wo = grp.loc[~b_mask, "price"].median()
                    brand_rows.append(
                        {
                            "Brand": brand_name,
                            "Count With Keyword": n_w,
                            "With Keyword ($)": b_med_w,
                            "Without Keyword ($)": b_med_wo,
                            "Within-Brand Lift (%)": round(((b_med_w - b_med_wo) / b_med_wo) * 100, 1),
                        }
                    )

            if brand_rows:
                brand_comp_df = pd.DataFrame(brand_rows)
                melted = brand_comp_df.melt(
                    id_vars=["Brand", "Count With Keyword", "Within-Brand Lift (%)"],
                    value_vars=["With Keyword ($)", "Without Keyword ($)"],
                    var_name="Listing Group",
                    value_name="Median Price ($)",
                )
                fig_brand_kw = px.bar(
                    melted,
                    x="Brand",
                    y="Median Price ($)",
                    color="Listing Group",
                    barmode="group",
                    text="Median Price ($)",
                    hover_data=["Count With Keyword", "Within-Brand Lift (%)"],
                    title=f"Within-Brand Median Price: With vs. Without '{clean_kw}'",
                )
                fig_brand_kw.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
                st.plotly_chart(fig_brand_kw, use_container_width=True)
                st.dataframe(brand_comp_df, use_container_width=True, hide_index=True)

            with st.expander(f"Inspect sample listings containing '{clean_kw}'"):
                sample_cols = ["brand", "title", "condition_label", "price", "has_price_drop"]
                st.dataframe(
                    filtered_df.loc[has_kw, sample_cols].sort_values("price", ascending=False).head(25),
                    use_container_width=True,
                    hide_index=True,
                )

# ==========================================
# TAB 3: SINGLE-BRAND WORD DISCOVERY
# ==========================================
with tab3:
    st.subheader("Discover High-Value Words Within a Single Brand")
    st.write(
        "Isolating one brand eliminates the brand-proxy effect (e.g., `'drkshdw'` looking expensive "
        "only because Rick Owens is expensive)."
    )

    c1, c2 = st.columns([2, 1])
    target_brand = c1.selectbox("Select Brand to Analyze:", sorted(df["brand"].unique()))
    min_freq = c2.slider("Minimum Word Occurrences:", min_value=5, max_value=50, value=15)

    brand_sub = df[df["brand"] == target_brand]
    brand_baseline = brand_sub["price"].median()

    stopwords = {
        "the", "and", "for", "with", "size", "mens", "pants", "jeans", "denim",
        "black", "blue", "gray", "grey", "white", "brown", "green", "in", "of", "on", "to", "a"
    }
    brand_tokens = set(re.sub(r"[^a-z0-9\s]", " ", target_brand.lower()).split())
    ignore_words = stopwords | brand_tokens

    word_series = brand_sub["title_clean"].str.split().explode()
    word_series = word_series[(word_series.str.len() > 2) & (~word_series.isin(ignore_words))]
    frequent_words = word_series.value_counts()
    frequent_words = frequent_words[frequent_words >= min_freq].index

    discovery_rows = []
    for w in frequent_words:
        m = brand_sub["title_clean"].str.contains(rf"\b{re.escape(w)}\b", regex=True)
        if m.sum() >= min_freq:
            w_med = brand_sub.loc[m, "price"].median()
            discovery_rows.append(
                {
                    "Word": w,
                    "Listings": int(m.sum()),
                    "Median Price ($)": w_med,
                    "Brand Baseline ($)": brand_baseline,
                    "Premium vs. Brand (%)": round(((w_med - brand_baseline) / brand_baseline) * 100, 1),
                }
            )

    if discovery_rows:
        disc_df = pd.DataFrame(discovery_rows).sort_values("Median Price ($)", ascending=False)
        top_15 = disc_df.head(15).sort_values("Median Price ($)", ascending=True)

        fig_top = px.bar(
            top_15,
            x="Median Price ($)",
            y="Word",
            orientation="h",
            text="Premium vs. Brand (%)",
            hover_data=["Listings", "Brand Baseline ($)"],
            title=f"Top 15 Highest-Priced Keywords in {target_brand} (Baseline Median: ${brand_baseline:,.0f})",
        )
        fig_top.add_vline(x=brand_baseline, line_dash="dash", line_color="red", annotation_text="Brand Median")
        fig_top.update_traces(texttemplate="%{text:+.1f}%", textposition="outside")
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("Try lowering the minimum word occurrences slider.")

# ==========================================
# TAB 4: LISTING TITLE SIMULATOR
# ==========================================
with tab4:
    st.subheader("Listing Title & Pricing Simulator")
    st.write("Draft a listing title to see comparable items and how keywords shift expected pricing.")

    sim_col1, sim_col2 = st.columns(2)
    sim_brand = sim_col1.selectbox("Brand", sorted(df["brand"].unique()), key="sim_brand")
    sim_cond = sim_col2.selectbox("Condition", sorted(df["condition_label"].unique()), key="sim_cond")
    sim_title = st.text_input(
        "Draft Listing Title",
        value="Rare Vintage Selvedge Denim Jeans",
    )

    base_cohort = df[(df["brand"] == sim_brand) & (df["condition_label"] == sim_cond)]
    base_med = base_cohort["price"].median() if not base_cohort.empty else df[df["brand"] == sim_brand]["price"].median()

    tokens = [
        t for t in re.sub(r"[^a-z0-9\s]", " ", sim_title.lower()).split()
        if len(t) > 2 and t not in set(sim_brand.lower().split())
    ]

    token_impacts = []
    for t in dict.fromkeys(tokens):
        m = base_cohort["title_clean"].str.contains(rf"\b{re.escape(t)}\b", regex=True)
        if m.sum() >= 3:
            t_med = base_cohort.loc[m, "price"].median()
            token_impacts.append(
                {
                    "Detected Word": t,
                    "Comparable Listings": int(m.sum()),
                    "Cohort Median w/ Word ($)": t_med,
                    "Shift vs. Baseline (%)": round(((t_med - base_med) / base_med) * 100, 1),
                }
            )

    st.metric(
        f"Baseline Median ({sim_brand} · {sim_cond})",
        f"${base_med:,.0f}",
        help=f"Based on {len(base_cohort)} matching listings",
    )

    if token_impacts:
        st.write("**Keyword Signal Breakdown in Your Draft Title:**")
        st.dataframe(pd.DataFrame(token_impacts), use_container_width=True, hide_index=True)
    else:
        st.info("None of the non-brand words in your draft title appeared at least 3 times in this brand/condition tier.")