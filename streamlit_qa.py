import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load sample datasets (if not already loaded)
try:
    iris = sns.load_dataset('iris')
    penguins = sns.load_dataset('penguins')
    titanic = sns.load_dataset('titanic')
    # Add more datasets as needed for examples
except:
    iris = None
    penguins = None
    titanic = None


def get_relevant_information(query):
    query = query.lower()
    relevant_info = []

    if "seaborn" not in query:
        return "Please specify that your question is related to the Seaborn library."

    # --- Plot Types ---
    if "scatter plot" in query or "scatterplot" in query:
        relevant_info.append("""
        **Scatter plots** show the relationship between two numerical variables.  Use `sns.scatterplot()` or `sns.relplot(kind="scatter")`.  Customize with `hue`, `size`, and `style`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.scatterplot(x="sepal_length", y="sepal_width", data=iris, hue="species")
        plt.title("Scatter Plot")
        plt.show()
        ```
        """)

    if "histogram" in query or "distribution" in query:
        relevant_info.append("""
        **Histograms** and density plots visualize the distribution of a single variable. Use `sns.histplot()` for histograms and `sns.kdeplot()` for kernel density estimates. `sns.displot()` is a figure-level function combining both.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.histplot(data=iris, x="sepal_length", kde=True)
        plt.title("Histogram with KDE")
        plt.show()
        ```
        """)

    if "boxplot" in query or "box plot" in query:
        relevant_info.append("""
        **Box plots** compare the distributions of a numerical variable across categories. They display quartiles, median, and outliers.  Use `sns.boxplot()`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.boxplot(x="species", y="sepal_length", data=iris)
        plt.title("Boxplot")
        plt.show()
        ```
        """)

    if "countplot" in query or "count plot" in query:
        relevant_info.append("""
        **Count plots** show the counts of observations in each category of a categorical variable. Use `sns.countplot()`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.countplot(x="species", data=iris)
        plt.title("Count Plot")
        plt.show()
        ```
        """)

    if "heatmap" in query:
        relevant_info.append("""
        **Heatmaps** visualize the correlation between multiple numerical variables. Use `sns.heatmap()`. Requires a correlation matrix (use `.corr()`).

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        corr_matrix = iris.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Heatmap")
        plt.show()
        ```
        """)

    if "pairplot" in query:
        relevant_info.append("""
        **Pair plots** show pairwise relationships between multiple variables in a dataset.  `sns.pairplot()` creates a grid of scatter plots and histograms. Use `hue` for color-coding.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.pairplot(iris, hue="species")
        plt.suptitle("Pair Plot", y=1.02)
        plt.show()
        ```
        """)

    if "relplot" in query:
        relevant_info.append("""
        `sns.relplot` is a figure-level function that combines scatter and line plots.  Offers flexible figure layouts.  Use `kind="scatter"` or `kind="line"`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.relplot(x="sepal_length", y="sepal_width", data=iris, hue="species", col="species")
        plt.show()
        ```
        """)

    if "catplot" in query:
        relevant_info.append("""
        `sns.catplot` is a figure-level function for visualizing categorical data.  Provides a unified interface to several plot types: stripplot, swarmplot, boxplot, violinplot, countplot, barplot, and pointplot (using the `kind` parameter).

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.catplot(x="species", y="sepal_length", data=iris, kind="box")
        plt.title("Catplot (Boxplot)")
        plt.show()
        ```
        """)

    if "facetgrid" in query or "facet grid" in query:
        relevant_info.append("""
        `FacetGrid` is a powerful tool for creating grid-based plots.  It helps to visualize different subsets of your data across multiple panels. You map plot types on it.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        g = sns.FacetGrid(iris, col="species")
        g.map(sns.histplot, "sepal_length")
        plt.title("FacetGrid")
        plt.show()
        ```
        """)

    # --- Advanced Plot Types ---
    if "correlation" in query or "correlations" in query:
        relevant_info.append("""
        Visualize correlations:
        1.  Calculate correlation matrix using `df.corr()`.
        2.  Use `sns.heatmap()` to visualize the matrix.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        corr_matrix = iris.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()
        ```
        """)

    if "regression" in query or "lm plot" in query:
        relevant_info.append("""
        **Regression plots** display a scatter plot with a fitted regression line. Use `sns.regplot()` or `sns.lmplot()` (figure-level).

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.regplot(x="sepal_length", y="sepal_width", data=iris)
        plt.title("Regression Plot")
        plt.show()
        ```
        """)

    if "joint plot" in query or "jointplot" in query:
        relevant_info.append("""
        **Joint plots** show the relationship between two variables, along with their marginal distributions (histograms or KDE plots).  Use `sns.jointplot()`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="reg") # kind="reg" for regression
        plt.title("Joint Plot")
        plt.show()
        ```
        """)

    if "swarm plot" in query:
        relevant_info.append("""
        **Swarm plots** show the distribution of data points for each category, avoiding overlap.  Use `sns.swarmplot()`.  Good for showing individual data points.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.swarmplot(x="species", y="sepal_length", data=iris)
        plt.title("Swarm Plot")
        plt.show()
        ```
        """)

    if "violin plot" in query:
        relevant_info.append("""
        **Violin plots** show the distribution of data, including density estimates, for each category. Use `sns.violinplot()`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.violinplot(x="species", y="sepal_length", data=iris)
        plt.title("Violin Plot")
        plt.show()
        ```
        """)

    if "strip plot" in query:
        relevant_info.append("""
        **Strip plots** show individual data points. Can be combined with other plots like boxplots.  Use `sns.stripplot()`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.stripplot(x="species", y="sepal_length", data=iris)
        plt.title("Strip Plot")
        plt.show()
        ```
        """)

    if "point plot" in query:
        relevant_info.append("""
        **Point plots** show point estimates (e.g., means) and confidence intervals for different categories. Use `sns.pointplot()`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.pointplot(x="species", y="sepal_length", data=iris)
        plt.title("Point Plot")
        plt.show()
        ```
        """)

    if "bar plot" in query:
        relevant_info.append("""
        **Bar plots** display categorical data using bars to represent values. Use `sns.barplot()`. Useful for comparing means.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.barplot(x="species", y="sepal_length", data=iris)
        plt.title("Bar Plot")
        plt.show()
        ```
        """)

    # --- Customization ---
    if "color palettes" in query or "colors" in query:
        relevant_info.append("""
        Customize plot colors using Seaborn's palettes.  Use the `palette` parameter.  Examples: `deep`, `muted`, `bright`, `pastel`, `dark`, `colorblind`, `viridis`, `plasma`, `magma`, `cividis`.  Also, specify a palette name (e.g., `"Set2"`) or create a custom palette.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.scatterplot(x="sepal_length", y="sepal_width", data=iris, hue="species", palette="Set1")
        plt.title("Scatter Plot with Color Palette")
        plt.show()
        ```
        """)

    if "themes" in query or "style" in query:
        relevant_info.append("""
        Set the overall style of your plots using `sns.set()`: `darkgrid`, `whitegrid`, `dark`, `white`, `ticks`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.set_style("whitegrid")
        sns.scatterplot(x="sepal_length", y="sepal_width", data=iris, hue="species")
        plt.title("Scatter Plot with Theme")
        plt.show()
        ```
        """)

    if "customization" in query:
        relevant_info.append("""
        Seaborn plots are highly customizable with Matplotlib functions:
        *   Axes labels: `plt.xlabel()`, `plt.ylabel()`
        *   Titles: `plt.title()`
        *   Legends: Use the `legend` parameter in Seaborn and `plt.legend()`
        *   Axis limits: `plt.xlim()`, `plt.ylim()`
        *   Figure size: `plt.figure(figsize=(width, height))`

        Remember to import `matplotlib.pyplot as plt`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.scatterplot(x="sepal_length", y="sepal_width", data=iris, hue="species")
        plt.xlabel("Sepal Length (cm)")
        plt.ylabel("Sepal Width (cm)")
        plt.title("Customized Plot")
        plt.show()
        ```
        """)

    # --- Figure-Level vs. Axes-Level Functions ---
    if "figure-level functions" in query or "axes-level functions" in query:
        relevant_info.append("""
        Seaborn offers two function types:
        *   **Axes-level functions:** Draw plots directly on Matplotlib Axes (e.g., `sns.scatterplot()`). Simpler for basic plots.
        *   **Figure-level functions:** Create a figure and manage axes (e.g., `sns.relplot()`, `sns.catplot()`, `sns.displot()`). Offer more control and are useful for faceting and organizing multiple plots.  They often return a `FacetGrid` or similar object that lets you manipulate the whole figure.
        """)

    # --- Working with Data ---
    if "categorical data" in query or "categorical variables" in query:
        relevant_info.append("""
        Seaborn provides functions for visualizing categorical data: `countplot`, `boxplot`, `violinplot`, `catplot` (figure-level). Use `hue` to add another dimension, and `col` or `row` with figure-level plots for faceting.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        titanic = sns.load_dataset('titanic')
        sns.countplot(x="class", data=titanic, hue="survived")
        plt.title("Survival Count by Class")
        plt.show()
        ```
        """)

    if "numerical data" in query or "numerical variables" in query:
        relevant_info.append("""
        Seaborn provides functions for visualizing numerical data:  `scatterplot`, `histplot`, `kdeplot`, `boxplot`, `violinplot`, `regplot`, `heatmap`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        sns.histplot(x="sepal_length", data=iris, kde=True)
        plt.title("Sepal Length Distribution")
        plt.show()
        ```
        """)

    if "multiple plots" in query or "subplots" in query:
        relevant_info.append("""
        Create multiple plots (subplots) using:
        *   `sns.FacetGrid`:  A powerful way to create grids of plots based on different conditions.
        *   `sns.relplot`, `sns.catplot`, and `sns.displot`:  Use the `col` or `row` arguments to create separate plots for each level of a categorical variable.
        *   Matplotlib subplots: Use `plt.subplots()` to create a figure and axes objects.
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        g = sns.FacetGrid(iris, col="species")
        g.map(sns.histplot, "sepal_length")
        plt.suptitle("FacetGrid of Sepal Length", y=1.02)
        plt.show()
        ```
        """)

    # --- Advanced Topics ---
    if "grids" in query or "grid plots" in query:
        relevant_info.append("""
        Seaborn's grid-based plots help with multi-dimensional data visualization, using plots like `sns.FacetGrid`, `sns.pairplot`, and the use of the `col` and `row` arguments in `sns.relplot`, `sns.catplot`, and `sns.displot`.

        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        iris = sns.load_dataset('iris')
        g = sns.PairGrid(iris) # Create the PairGrid object
        g.map_diag(sns.histplot) # Map a histogram to the diagonal
        g.map_offdiag(sns.scatterplot) # Map a scatterplot to the off-diagonal
        plt.show()
        ```
        """)
    if "themes" in query or "style" in query:
         relevant_info.append("""
         Seaborn offers styling options to customize the overall look and feel of your plots. Use the `sns.set()` function to set global style parameters.
         Common styles include: `darkgrid`, `whitegrid`, `dark`, `white`, and `ticks`.

         ```python
         import seaborn as sns
         import matplotlib.pyplot as plt
         import pandas as pd
         iris = sns.load_dataset('iris')
         sns.set_style("whitegrid")
         sns.scatterplot(x="sepal_length", y="sepal_width", data=iris, hue="species")
         plt.title("Scatter Plot with Whitegrid Style")
         plt.show()
         ```
         """)
    if not relevant_info:
        relevant_info.append("I'm sorry, I don't have specific information on that topic. Please rephrase your question or try a different query. I am focused on basic Seaborn usage.")

    return "\n\n".join(relevant_info)


# --- Streamlit App ---
st.title("Seaborn Question Answering Bot")
st.write("Ask me questions about the Seaborn library!")

user_query = st.text_input("Ask your question here:", "")

if user_query:
    with st.spinner("Searching for an answer..."):
        answer = get_relevant_information(user_query)
    st.subheader("Answer:")
    st.markdown(answer)  # Use st.markdown to render the answer, which can include code examples