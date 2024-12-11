import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# Set the page configuration
st.set_page_config(
    page_title="📊 Market Trend Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("📊 Market Trend Analysis Dashboard")

# Sidebar for user inputs
st.sidebar.header("Product Analysis")

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    try:
        df = pd.read_csv("C://dataset.csv")
        # Ensure categorical columns are in string format
        categorical_columns = [
            'Product_Type', 'Price_Range', 'Geography', 'Age_Group', 'Gender',
            'Income_Level', 'Preferred_Product_Type', 'Purchase_Channel',
            'Trending_Ingredients', 'Region'
        ]
        for column in categorical_columns:
            if column in df.columns:
                df[column] = df[column].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_product_demographics(df, product_name):
    """
    Analyze and visualize the demographic groups that prefer a given product.
    """
    # Check if the product exists in the dataset
    if product_name not in df['Preferred_Product_Type'].unique():
        st.warning(f"Product '{product_name}' not found in the dataset.")
        return

    # Filter the dataset for the given product
    product_df = df[df['Preferred_Product_Type'] == product_name]

    # Total number of users who prefer the product
    total_users = len(product_df)
    st.subheader(f"Total Users who prefer '{product_name}': {total_users}")

    # Define demographic columns to analyze
    demographic_columns = ['Gender', 'Age_Group', 'Income_Level', 'Geography', 'Region']

    # Check which demographic columns exist in the dataframe
    available_demographics = [col for col in demographic_columns if col in product_df.columns]

    for column in available_demographics:
        st.markdown(f"### {column} Distribution")
        distribution = product_df[column].value_counts(normalize=True) * 100
        st.write(distribution.round(2))

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=product_df, x=column, palette='Set2', ax=ax)
        ax.set_title(f"{column} Distribution for '{product_name}'")
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Additional: Crosstab for Gender vs Age_Group if both columns exist
    if 'Gender' in available_demographics and 'Age_Group' in available_demographics:
        st.markdown(f"### Gender vs Age Group for '{product_name}'")
        crosstab = pd.crosstab(product_df['Gender'], product_df['Age_Group'], normalize='index') * 100
        st.dataframe(crosstab.round(2))

        # Heatmap for the crosstab
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(crosstab, annot=True, fmt=".2f", cmap='YlGnBu', ax=ax)
        ax.set_title(f"Gender vs Age Group Heatmap for '{product_name}'")
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Gender')
        st.pyplot(fig)

    # Trending Ingredients Analysis
    if 'Trending_Ingredients' in product_df.columns:
        st.markdown(f"### Trending Ingredients for '{product_name}'")

        # Split ingredients and clean them
        ingredients_series = product_df['Trending_Ingredients'].dropna().apply(lambda x: [ingredient.strip().lower() for ingredient in x.split(',')])
        all_ingredients = [ingredient for sublist in ingredients_series for ingredient in sublist]

        # Count frequency of each ingredient
        ingredient_counts = Counter(all_ingredients)
        most_common_ingredients = ingredient_counts.most_common(10)

        # Display top 10 trending ingredients
        st.markdown("**Top 10 Trending Ingredients:**")
        top_ingredients_df = pd.DataFrame(most_common_ingredients, columns=['Ingredient', 'Count'])
        st.dataframe(top_ingredients_df)

        # Bar Chart for Trending Ingredients
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_ingredients_df, x='Count', y='Ingredient', palette='viridis', ax=ax)
        ax.set_title(f"Top 10 Trending Ingredients for '{product_name}'")
        ax.set_xlabel('Count')
        ax.set_ylabel('Ingredient')
        st.pyplot(fig)

        # Word Cloud for Trending Ingredients
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ingredient_counts)
            fig, ax = plt.subplots(figsize=(15, 7.5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f"Word Cloud of Trending Ingredients for '{product_name}'", fontsize=20)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating word cloud: {e}")
    else:
        st.warning("The dataset does not contain a 'Trending_Ingredients' column.")

    # Optionally, allow downloading the filtered data
    csv = product_df.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f'{product_name}_demographics.csv',
        mime='text/csv',
    )

def main():
    # Define file path
    file_path = 'dataset.csv'  # Ensure this file is in the same directory as app.py

    # Load data
    df = load_data(file_path)
    if df is None:
        return

    # Ensure that 'Preferred_Product_Type' column exists
    if 'Preferred_Product_Type' not in df.columns:
        st.error("The dataset must contain a 'Preferred_Product_Type' column.")
        return

    # Dropdown for product name
    product_input = st.sidebar.selectbox(
        "Select Product Name",
        df['Preferred_Product_Type'].unique()
    )

    # Button to trigger analysis
    if st.sidebar.button("Analyze"):
        get_product_demographics(df, product_input)

if __name__ == "__main__":
    main()
