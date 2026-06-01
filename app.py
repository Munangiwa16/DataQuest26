"""
App code for DataQuest26.
Author: Munangiwa Gift Lithole
"""


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import streamlit as st

from io import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from streamlit_option_menu import option_menu


def main():
    st.title(" 🏦 Smart View", text_alignment="center")
    handler(menu(),data_handler())



def menu():

    # This menu function is responsible for the  menu  section in the app
    menu_selected = option_menu(None, ["Home", "Univariate Explorer", "Bivariate Explorer","Data Quality","Model Evaluation"],
                            icons=['house',"bar-chart-line", "graph-up","clipboard-data","goal"],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    return menu_selected

def home(data):
    st.markdown("""
        <style>
        .main-title {
            font-size: 45px;
            font-weight: bold;
            color: #1f4e79;
            text-align: center;
            margin-bottom: 10px;
        }

        .sub-title {
            font-size: 20px;
            color: #555;
            text-align: center;
            margin-bottom: 40px;
        }

        .feature-box {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        }

        .feature-title {
            font-size: 24px;
            font-weight: bold;
            color: #1f4e79;
            margin-bottom: 10px;
        }

        .feature-text {
            font-size: 16px;
            color: #444;
        }

        .footer {
            text-align: center;
            font-size: 16px;
            color: gray;
            margin-top: 40px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.balloons()

    # Hero Section
    st.markdown('<div class="main-title">📊 Smart View Dashboard</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sub-title">'
        'Making credit analysis simple, intelligent, and insightful.'
        '</div>',
        unsafe_allow_html=True
    )

    # Intro
    st.info(
        "Explore hidden patterns in your credit data using interactive visualizations, "
        "data quality assessment tools, and model evaluation dashboards."
    )

    # Features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-title">📈 Univariate Explorer</div>
                <div class="feature-text">
                    Analyze one variable at a time with interactive visualizations.
                    <br><br>
                    • Histogram <br>
                    • KDE Plot <br>
                    • Box Plot <br>
                    • Bar Plot <br>
                    • WoE (Weight of Evidence) Analysis
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-title">🔍 Bivariate Explorer</div>
                <div class="feature-text">
                    Understand relationships between variables and target outcomes.
                    Detect trends, risk patterns, and correlations easily.
                </div>
            </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-title">🛠 Data Quality</div>
                <div class="feature-text">
                    Assess missing values, duplicates, and inconsistencies in your dataset
                    to improve model reliability and decision-making.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-title">🤖 Model Evaluation</div>
                <div class="feature-text">
                    Evaluate machine learning models using performance metrics,
                    confusion matrices, ROC curves, and predictive insights.
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Sidebar Hint
    st.success(
        "💡 Each section includes a sidebar where you can customize configurations, "
        "filters, and visualization settings."
    )

    # Footer
    st.markdown(
        '<div class="footer">🚀 Smart View — Turning Data into Credit Intelligence</div>',
        unsafe_allow_html=True
    )

def display_data_quality(profile,data):

    st.header("Data Quality")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", profile["rows"])
    col2.metric("Columns", profile["cols"])
    col3.metric("Duplicate Rows", profile["duplicate_rows"])
    col4.metric("Duplicate %", f"{profile['duplicate_rows_pct']}%")

    title_col, sort_col = st.columns([3, 2])

    with title_col:
        st.subheader("Columns")

    with sort_col:
        sort_by = st.pills(
            "Sort by",
            ["Missing Values", "Unique Values"])
    display_columns(profile,data,sort_by)


def display_columns(profile,data,sort_by:None):
    columns = profile["columns"]
    col_names = list(columns.keys())

    if sort_by == "Missing Values":
        col_names = sorted(col_names, key=lambda c: columns[c]["nulls"], reverse=True)

    elif sort_by == "Unique Values":
        col_names = sorted(col_names, key=lambda c: columns[c]["distinct"], reverse=True)

    for col in col_names:

        st.markdown(f"""
        <span style="
            font-size:20px;
            font-weight:600;
            padding:4px 10px;
            border-radius:6px;
            background-color:#f0f2f6;
            display:inline-block;
        ">
        {col}
        </span>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            col1, col2, col3= st.columns(3)
            col1.metric("dtype", columns[col]["dtype"],border=True)

            missing_pct = columns[col]["null_pct"]

            if columns[col]["nulls"]== 0:
                col2.metric(
                    "Missing Values",
                    columns[col]["nulls"],
                    delta="0%",
                    delta_color="normal",border=True
                )
            else:
                col2.metric(
                    "Missing Values",
                    columns[col]["nulls"],
                    delta=f"{missing_pct:.1f}%",
                    delta_color="inverse",
                    border=True
                )
            col3.metric("Unique values", columns[col]['distinct'],border=True)

            st.markdown("""
                            <span style="
                                font-size:14px;
                                font-weight:600;
                                padding:6px 12px;
                                border-radius:8px;
                                background-color:#4CAF50;
                                color:white;
                                display:inline-block;
                            ">
                            Summary
                            </span>
                            """, unsafe_allow_html=True)

            if columns[col]["dtype"]  in ["int64", "float64","bool"]:

                dat = pd.DataFrame({
                    "min": [f"{columns[col]['min']:.2f}"],
                    "median": [f"{columns[col]['median']:.2f}"],
                    "mean": [f"{columns[col]['mean']:.2f}"],
                    "max": [f"{columns[col]['max']:.2f}"],
                    "std": [f"{columns[col]['std']:.2f}"],
                    "skew": [f"{columns[col]['skew']:.2f}"]
                })
                st.table(dat,border="horizontal")

            elif columns[col]["dtype"] == "datetime64[ns]":

                dat = pd.DataFrame({
                    "min_date": [data[col].min().strftime("%Y-%m-%d")],
                    "max_date": [data[col].max().strftime("%Y-%m-%d")],
                    "range_days": [(data[col].max() - data[col].min()).days]
                })

                st.table(dat,border="horizontal")
            else:
                props = data[col].value_counts(normalize=True) * 100
                dat = pd.DataFrame({"Values":props.index,
                                    "Proportion (%)": props.round(2).values})
                st.table(dat, border="horizontal")

def load_data():
    loan_book = pd.read_csv("loan_book.csv", index_col=0)
    return loan_book

def profile_dataframe(df):
    """Profiles a dataframe, returning a dictionary of quality metrics."""
    total = len(df)
    profile = {"rows": total, "cols": df.shape[1], "columns": {}}
    for c in df.columns:
        s = df[c]
        nulls = int(s.isna().sum())
        non_null = total - nulls
        distinct = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        uniq_pct = round((distinct / non_null * 100), 2) if non_null else 0.0
        col_info = {
            "dtype": dtype,
            "nulls": nulls,
            "null_pct": round((nulls/total*100), 2) if total else 0.0,
            "distinct": distinct,
            "uniqueness_pct": uniq_pct,
        }
        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors='coerce')
            col_info.update({
                "min": None if pd.isna(s_num.min()) else float(s_num.min()),
                "max": None if pd.isna(s_num.max()) else float(s_num.max()),
                "mean": None if pd.isna(s_num.mean()) else float(s_num.mean()),
                "std": None if pd.isna(s_num.std()) else float(s_num.std()),
                "skew": None if pd.isna(s_num.skew()) else float(s_num.skew()),
                "median": None if pd.isna(s_num.median()) else float(s_num.median()),
            })
        profile["columns"][c] = col_info

    dup_rows = int(df.duplicated().sum())
    profile["duplicate_rows"] = dup_rows
    profile["duplicate_rows_pct"] = round((dup_rows / profile["rows"] * 100), 2) if profile["rows"] else 0.0
    return profile

def univariate_explorer(data,num_var:"numeric variables names",cat_var:"categorical variable names",date_var:"date variable names"):

    # sidebar options
    with st.sidebar:

        st.header("⚙️ Visualization Settings")

        kde = None
        color = None
        stats = "count"
        scale = False
        bins = None
        fill = None
        orient = "Vertical"
        order = None
        with_target = None
        woe =False
        graph_type = st.selectbox("Graph Type",
                                  options=["Histogram","Kernel Density","Box-Plot","Bar-Plot"])

        if graph_type in ["Histogram","Kernel Density","Box-Plot"]:  # for numerical features
            if graph_type=="Box-Plot":  # orientation
                orient = st.selectbox("Orientation",options=["Horizontal","Vertical"])
                if orient=="Vertical":
                    x= st.selectbox("Y-axis",options=num_var)   # used as y-axis when plotting
                else:
                    x= st.selectbox("X-axis",options=num_var)  # X-axis
            else:
                x = st.selectbox("X-axis",options=num_var)   # X-axis

            if graph_type == "Histogram":   # graph type
                stats = st.selectbox("Aggregation Statistic",
                                    options=["count","frequency","probability","percent","density"])

            with_target = st.toggle("Compare with Target (Default Status)")    # for Target

            if graph_type == "Histogram":
                woe = st.toggle("WoE Plot")        #

            if graph_type == "Kernel Density":
                fill = st.toggle("Fill",True)
            if data[x].min()>0 and woe == False:
                scale = st.toggle(label="Log-scale")             #log scale

            if graph_type == "Histogram":
                kde = st.toggle("Kernel Density Estimation (KDE)")
                value= len(np.histogram_bin_edges(data[x],bins='doane'))-1
                max_bin = min(len(np.histogram_bin_edges(data[x],bins='sqrt'))-1,50)
                bins = st.slider("Number of Bins", 2,max_bin , value,step=1)

            if not with_target:
                color = st.color_picker("Color",value="#18A9D5")

        else:   # for categorical features
            orient = st.selectbox("Orientation",options=["Horizontal","Vertical"])
            if orient == "Vertical":
                x = st.selectbox("Y-axis", options=cat_var)  # used as y-axis when plotting
            else:
                x = st.selectbox("X-axis", options=cat_var)  # X-axis

            stats = st.selectbox("Aggregation Statistic",
                                 options=["count", "probability", "percent", "proportion"])  # stat

            if x not in ["default_flag","Loan Outcome"]:
                with_target = st.toggle("Compare with Target (Default Status)")  # Compare with Target

            if x != "default_flag":
                woe = st.toggle("WoE Plot")

            order = st.toggle("Sort by frequency (descending)")

            show_values = st.toggle("Show values on bars")

            if not with_target:
                color = st.color_picker("Color",value="#18A9D5")


    # options executions
    if graph_type in ["Histogram", "Kernel Density", "Box-Plot"]:
        plot_distribution(data,x,kind=graph_type,log_scale=scale,hue=with_target,color=color,kde=kde,stats=stats,bins=bins,
                          fill=fill,orient = orient,woe=woe)
    else:
        plot_cat_distribution(data,x,orient=orient,stat=stats,hue=with_target,color=color,order=order,show_values=show_values,
                              woe=woe)

def plot_distribution(data,x,kind ="Histogram",hue=False,log_scale=False,color=None,kde=None,stats="count",bins=None,fill=None
                      ,orient = "Vertical",woe=False):
    """
    This function which we call to display numerical visuals graphs on streamlit. for `Histogram`, `Kernel Density`
    and `Box-Plot`
    :param data: the data to be plotted
    :param x: x-axis
    :param kind: Graph type
    :param hue: take/false for comparing with target
    :param log_scale: take true/false logarithmic scale on the graphs
    :param color: for color using hex
    :param kde: take true/false values if some need kernel density estimation for histogram
    :param stats: aggregation statistics e.g `count`, `frequency`, `probability`, `percent`, `density` for histogram
    :param bins: number of bins for histogram
    :param fill: take true/false fill value for kernel density estimation
    :param orient: orientation option e.g `vertical`, `horizontal`
    :return: No return
    """
    with st.spinner("Wait for it...", show_time=True,width="stretch"):
        if kind == "Box-Plot":
            # for  plotting Box-Plot
            fig, ax = plt.subplots()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if  not hue: # title display format
                fig.suptitle("Distribution of {}".format(display_feature(x)))
            else:
                fig.suptitle("Distribution of {} by Loan Outcome".format(display_feature(x)))

            if orient=="Vertical":
                sns.boxplot(y=x,data=data,ax=ax,hue="Loan Outcome" if hue else None,log_scale=log_scale,color=color)
                ax.set_ylabel(display_feature(x))
            else:
                sns.boxplot(x=x, data=data, ax=ax, hue="Loan Outcome" if hue else None, log_scale=log_scale, color=color)
                ax.set_xlabel(display_feature(x))

            fig.tight_layout()
            st.pyplot(fig)

        elif kind == "Histogram":
            # for plotting histogram
            fig, ax = plt.subplots()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if  not hue: # title display format
                fig.suptitle("Distribution of {}".format(display_feature(x)))
            else:
                fig.suptitle("Distribution of {} by Loan Outcome".format(display_feature(x)))

            sns.histplot(data=data,x=x,ax=ax,hue= "Loan Outcome" if hue else None,
                         log_scale=log_scale,color=color,kde=kde,stat=stats,bins=bins)
            ax.set_xlabel(display_feature(x))

            # for WoE plot

            if woe:
                df = data.copy()
                df["default_flag"]=df["default_flag"].astype(int)
                used_bins = np.histogram_bin_edges(df[x],bins=bins)
                woe_result = midpoint_bin(woe_iv(bin_numeric(used_bins,df,x,target="default_flag")
                                                 ,x,"default_flag"),x)

                # Create second axis
                ax2 = ax.twinx()

                # Clean right axis properly
                ax2.spines['top'].set_visible(False)

                # WoE line plot
                sns.lineplot(
                    data=woe_result,
                    x="bin_midpoint",
                    y="WoE",
                    marker='o',
                    color='red',
                    ax=ax2
                )


                # Optional: match spine color to WoE line
                ax2.spines['right'].set_color('red')
                ax2.tick_params(axis='y', colors='red')

            fig.tight_layout()
            st.pyplot(fig)

            views = st.sidebar.pills("Show",options=["IV","Mean","Median","Min","Max"],selection_mode="multi")

            if len(views)>0:
                colms = st.columns(len(views))
                for i,v in enumerate(views):
                    with colms[i]:
                        if v == "IV":
                            if not woe:
                                st.warning("Please turn the WoE plot on for IV calculation")
                            else:
                                st.metric("Information Value",
                                          np.round(woe_result["iv_bin"].sum(),2))
                        else:
                            if v == "Mean":
                                st.metric("Mean",np.round(data[x].mean(),2))
                            elif v == "Median":
                                st.metric("Median",np.round(data[x].median(),2))
                            elif v == "Min":
                                st.metric("Min",np.round(data[x].min(),2))
                            else:
                                st.metric("Max",np.round(data[x].max(),2))

        else:
            # for plotting Kernal density
            fig, ax = plt.subplots()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if  not hue :
                fig.suptitle("Distribution of {}".format(display_feature(x)))
            else:
                fig.suptitle("Distribution of {} by Loan Outcome".format(display_feature(x)))

            sns.kdeplot(data=data,x=x,ax=ax,hue= "Loan Outcome" if hue else None,log_scale=log_scale,color=color,fill=fill)
            ax.set_xlabel(display_feature(x))
            fig.tight_layout()
            st.pyplot(fig)

        st.sidebar.download_button(
            label="Download Plot",
            data=savefig(fig),
            file_name="seaborn_plot.png",
            mime="image/png",
            icon=":material/download:"
        )

def plot_cat_distribution(data,x,hue=False,color=None,orient="Horizontal",stat="count",order=None,
                          show_values=False,woe=False):

    """
    This function used to plot categorical feature , bar-plot
    :param data:  data to be plotted
    :param x: feature to be plotted
    :param hue: color comparing variable
    :param color: color of plot
    :param orient: orientation option e.g `vertical`, `horizontal`
    :param stat: aggregation statistics e.g `count`, `frequency`, `probability`, `percent`, `density`
    :param order: for descending order by frequency
    :param show_values: for showing values on the plot
    :return: No return
    """

    if orient == "Horizontal":  # Plotting on horizontal orientation
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if  not hue: # title display format
            fig.suptitle("Distribution of {}".format(display_feature(x)))
        else:
            fig.suptitle("Distribution of {} by Loan Outcome".format(display_feature(x)))

        sns.countplot(data=data,x=x,stat=stat,hue="Loan Outcome" if hue else None,color=color,
                      order=  data[x].value_counts().index if order else None,ax=ax)
        ax.set_xlabel(display_feature(x))
        ax.tick_params(axis="x",rotation=75)

        if woe:
            df = data.copy()
            df["default_flag"] = df["default_flag"].astype(int)
            woe_result = woe_iv(df,x,"default_flag")

            # Create second axis
            ax2 = ax.twinx()

            # Clean right axis properly
            ax2.spines['top'].set_visible(False)

            # WoE line plot
            sns.lineplot(
                data=woe_result,
                x=x,
                y="WoE",
                marker='o',
                color='red',
                ax=ax2
            )

            # Optional: match spine color to WoE line
            ax2.spines['right'].set_color('red')
            ax2.tick_params(axis='y', colors='red')



        if show_values:
            # Add numbers on top of bars
            for container in ax.containers:
                ax.bar_label(container,fmt=format_stat(stat),fontsize=8)


        fig.tight_layout()
        st.pyplot(fig)
        views =[]
        if x != "default_flag":
            views = st.sidebar.pills("Show", options=["IV"], selection_mode="multi")
        if len(views) > 0:
            colms = st.columns(len(views))
            for i, v in enumerate(views):
                with colms[i]:
                    if v == "IV":
                        if not woe:
                            st.warning("Please turn the WoE plot on for IV calculation")
                        else:
                            st.metric("Information Value",
                                      np.round(woe_result["iv_bin"].sum(), 2))
    else:  # Plotting on vertical orientation

        fig, ax = plt.subplots()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if not hue:
            fig.suptitle(f"Distribution of {display_feature(x)}")
        else:
            fig.suptitle(f"Distribution of {display_feature(x)} by Loan Outcome")

        # Main countplot
        sns.countplot(
            data=data,
            y=x,
            hue="Loan Outcome" if hue else None,
            color=color,
            order=data[x].value_counts().index if order else None,
            ax=ax
        )

        ax.set_ylabel(display_feature(x))

        # ---------------- WoE Overlay ---------------- #
        if woe:
            df = data.copy()
            df["default_flag"] = df["default_flag"].astype(int)

            woe_result = woe_iv(df, x, "default_flag")

            # Match category ordering
            category_order = (
                data[x].value_counts().index
                if order
                else data[x].unique()
            )

            woe_result[x] = pd.Categorical(
                woe_result[x],
                categories=category_order,
                ordered=True
            )


            woe_result = woe_result.sort_values(x)

            # Secondary axis
            ax2 = ax.twiny()

            ax2.spines['top'].set_visible(True)
            ax2.spines['right'].set_visible(False)

            # WoE line
            sns.lineplot(
                data=woe_result,
                y=x,
                x="WoE",
                marker='o',
                color='red',
                sort=False,
                ax=ax2
            )

            ax2.set_xlabel("WoE")
            ax2.tick_params(axis='x', colors='red')
            ax2.spines['top'].set_color('red')

        fig.tight_layout()

        if show_values:
            # Add numbers on top of bars
            for container in ax.containers:
                ax.bar_label(container,fmt=format_stat(stat),fontsize=8)

        fig.tight_layout()
        st.pyplot(fig)

        views = []
        if x != "default_flag":
            views = st.sidebar.pills("Show", options=["IV"], selection_mode="multi")
        if len(views)>0:
            colms = st.columns(len(views))
            for i, v in enumerate(views):
                with colms[i]:
                    if v == "IV":
                        if not woe:
                            st.warning("Please turn the WoE plot on for IV calculation")
                        else:
                            st.metric("Information Value",
                                      np.round(woe_result["iv_bin"].sum(), 2))

    st.sidebar.download_button(
        label="Download Plot",
        data=savefig(fig),
        file_name="seaborn_plot.png",
        mime="image/png",
        icon=":material/download:"
    )

def bivariate_explorer(data,num_var:"numeric variables names",cat_var:"categorical variable names",date_var:"date variable names"):

    with st.sidebar:
        st.header("⚙️ Visualization Settings")

        color=None
        x_scale=False
        y_scale=False
        show_values=False
        order=None
        stats="count"
        x=None
        hue=None
        orient="Horizontal"

        graph_type = st.selectbox("Graph Type",
                                  options=["Scatter Plot", "heatmap","Box-Plot","Bar-Plot"])
        if graph_type in ["Scatter Plot"]:
            x = st.selectbox("X-axis", options=num_var)  # x-xis
            y = st.selectbox("Y-axis", options=num_var[num_var !=x])  # y-axis
            hue = st.selectbox("Group by Color", options=[None]+cat_var.tolist())

            if data[x].min() > 0:
                x_scale = st.toggle("X-axis Log-scale")
            if data[y].max() > 0:
                y_scale = st.toggle("Y-axis Log-scale" )
            if hue is None:
                color = color = st.color_picker("Color",value="#18A9D5")

        elif  graph_type == "heatmap" :

            choices = st.multiselect("Select numeric variables", options=num_var.tolist()+["default_flag"])

            if len(choices) < 2:
                st.warning("Please select at least 2 variables")

            method = st.selectbox(
                "Correlation Method",
                ["pearson", "spearman", "kendall"]
            )

            show_values = st.toggle("Show values")

        else:
            orient = st.selectbox("Orientation",options=["Horizontal","Vertical"])
            if graph_type == "Box-Plot":
                x= st.selectbox("Select a numerical variable", options=num_var)
                hue = st.selectbox("Select a categorical variable", options=cat_var)
                if data[x].min() > 0:
                    x_scale = st.toggle("Log scale for numerical variable")
            else:
                x = st.selectbox("First Category Features", options=cat_var)
                hue = st.selectbox("Second Category Features", options=cat_var[cat_var!=x])
                stats = st.selectbox("Aggregation Statistic",
                                     options=["count", "probability", "percent", "proportion"])
                order = st.toggle("Sort by frequency (descending)")
                show_values = st.toggle("Show values on bars")




    # excution
    if graph_type == "Scatter Plot":
        plot_relationship(data, x, y,hue ,kind=graph_type,color=color,x_scale=x_scale,y_scale=y_scale)

    elif graph_type == "heatmap" and len(choices) >= 2:
        plot_heatmap(data,choices,show_values=show_values,method=method)
    elif graph_type in ["Box-Plot","Bar-Plot"]:
        plot_dist_with_options(data,x,hue,graph_type,orient,stats ,order, show_values,x_scale)


def plot_dist_with_options(data,x,hue,kind=None,orient="Horizontal",stat="count",order=None,show_values=False,log_scale=False):

    if kind == "Box-Plot":
        # for  plotting Box-Plot
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

         # title display format
        fig.suptitle("{} vs {}".format(display_feature(x),display_feature(hue)))

        if orient == "Vertical":
            sns.boxplot(y=x, data=data, ax=ax, hue=hue,log_scale=log_scale)
            ax.set_ylabel(display_feature(x))
        else:
            sns.boxplot(x=x, data=data, ax=ax, hue=hue,log_scale=log_scale)
            ax.set_xlabel(display_feature(x))

        fig.tight_layout()
        st.pyplot(fig)

    else:

        if orient == "Horizontal":  # Plotting on horizontal orientation
            fig, ax = plt.subplots()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # title display format
            fig.suptitle("{} vs {}".format(display_feature(x), display_feature(hue)))

            sns.countplot(data=data, x=x, stat=stat, hue=hue,
                          order=data[x].value_counts().index if order else None, ax=ax)
            ax.set_xlabel(display_feature(x))
            ax.tick_params(axis="x", rotation=75)

            if show_values:
                # Add numbers on top of bars
                for container in ax.containers:

                    if data[hue].nunique() > 4:
                        labels = ax.bar_label(container, fmt=format_stat(stat), fontsize=8,padding=3)
                        for label in labels:
                            label.set_rotation(90)
                    else:
                        ax.bar_label(container, fmt=format_stat(stat), fontsize=8,padding=3)

            fig.tight_layout()
            st.pyplot(fig)

        else:  # Plotting on vertical orientation
            fig, ax = plt.subplots()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # title display format
            fig.suptitle("{} vs {}".format(display_feature(x), display_feature(hue)))

            sns.countplot(data=data, y=x, stat=stat, hue=hue,
                          order=data[x].value_counts().index if order else None, ax=ax)
            ax.set_ylabel(display_feature(x))

            if show_values:
                # Add numbers on top of bars
                for container in ax.containers:
                    ax.bar_label(container, fmt=format_stat(stat), fontsize=8,padding=3)

            fig.tight_layout()
            st.pyplot(fig)

    st.sidebar.download_button(
        label="Download Plot",
        data=savefig(fig),
        file_name="seaborn_plot.png",
        mime="image/png",
        icon=":material/download:"
    )


def plot_heatmap(data,choices,show_values=False,method=None):

    corr = data[choices].corr(method=method)

    fig, ax = plt.subplots(figsize=(12,10))
    ax.set_title("Correlation Heatmap")
    sns.heatmap(corr,cmap="coolwarm", ax=ax,annot= show_values)
    fig.tight_layout()
    st.pyplot(fig)
    st.sidebar.download_button(
        label="Download Plot",
        data=savefig(fig),
        file_name="seaborn_plot.png",
        mime="image/png",
        icon=":material/download:"
    )


def plot_relationship(data,x,y,hue=None,kind="Scatter Plot",color=None,x_scale=False,y_scale=False):

    if kind == "Scatter Plot":
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title("{} vs {} {}".format(display_feature(x),display_feature(y), "by "+display_feature(hue)
        if hue is not None else "" ))
        sns.scatterplot(x=x,y=y,data=data,hue=hue,ax=ax,color=color)
        if data[x].min()> 0 and x_scale:
            ax.set_xscale("log")
        if data[y].min()> 0 and y_scale:
            ax.set_yscale("log")

        ax.set(xlabel=display_feature(x),ylabel=display_feature(y))
        fig.tight_layout()
        st.pyplot(fig)

    st.sidebar.download_button(
        label="Download Plot",
        data=savefig(fig),
        file_name="seaborn_plot.png",
        mime="image/png",
        icon=":material/download:"
    )


def handler(menu_selected,data_spec):
    data = data_spec[0]
    num_names = data_spec[1]
    cat_names = data_spec[2]
    date_names = data_spec[3]
    if menu_selected == "Home":
        home(data)
    elif menu_selected == "Univariate Explorer":
        univariate_explorer(data,num_names,cat_names,date_names)
    elif menu_selected ==  "Bivariate Explorer":
        bivariate_explorer(data,num_names,cat_names,date_names)
    elif menu_selected == "Data Quality":
        d = data_handler()
        prof = profile_dataframe(d[0])
        display_data_quality(prof,d[0])
    else:
        dat = data_spec[0]
        dat['default_flag'] = dat['default_flag'].astype(int)
        X_train,X_test,y_train,y_test = process(dat)
        final_model = LogisticRegression(max_iter=10000, class_weight="balanced", random_state=42, verbose=1,
                                         solver="sag")
        final_model.fit(X_train, y_train)
        model_evaluation(final_model, X_test, y_test)

@st.cache_data
def process(data):

    train = data.loc[data["set"] == "train"]
    test = data.loc[data["set"] == "test"]

    rev = ["email_domain_type", "branch_code_id", "application_date", "application_dow", "region",
           "months_at_current_address", "phone_verified", "months_since_oldest_account","Loan Outcome"]
    train.drop(columns=rev, inplace=True)
    test.drop(columns=rev, inplace=True)

    y_train = train["default_flag"]
    y_test = test["default_flag"]
    X_train = train.drop(columns=["default_flag", "set", "home_ownership", "loan_purpose"])
    X_test = test.drop(columns=["default_flag", "set", "home_ownership", "loan_purpose"])

    eges = [25, 31, 43]  # split points
    age_bins = bin_numeric(eges, train, "age", "default_flag")  # making bins
    woe_result = midpoint_bin(woe_iv(age_bins, "age", "default_flag"), "age")
    # Create dictionary for encoding
    bin_to_woe = dict(zip(woe_result["age"], woe_result['WoE']))  # creating a map used in mapping
    X_train_2 = apply_woe(X_train, "age", bin_to_woe, eges)  # new value are introduce instead of the existing ones
    X_test_2 = apply_woe(X_test, "age", bin_to_woe, eges)

    X_train_3, X_test_3 = capping_var(X_train_2, X_test_2, "annual_income")  # capping
    X_train_3["log_annual_income_capped"] = X_train_3["annual_income_capped"].transform(lambda x: np.log(x))  # loging
    X_train_3.drop(columns=["annual_income", "annual_income_capped"], inplace=True)

    X_test_3["log_annual_income_capped"] = X_test_3["annual_income_capped"].transform(lambda x: np.log(x))
    X_test_3.drop(columns=["annual_income", "annual_income_capped"], inplace=True)

    X_train_4, X_test_4 = X_train_3.copy(), X_test_3.copy(),
    # removing a new feature
    X_train_4.drop(columns=["total_revolving_balance"], inplace=True)
    X_test_4.drop(columns=["total_revolving_balance"], inplace=True)

    woe_result_hom = woe_iv(data=train, feature="home_ownership", target="default_flag")
    bin_to_woe1 = dict(zip(woe_result_hom["home_ownership"], woe_result_hom['WoE']))

    X_train_5, X_test_5 = X_train_4.copy(), X_test_4.copy()
    X_train_5["home_WoE"], X_test_5["home_WoE"] = train["home_ownership"].replace(bin_to_woe1), test[
        "home_ownership"].replace(bin_to_woe1)

    X_train_6, X_test_6 = X_train_5.copy(), X_test_5.copy()
    accts_eges = np.histogram_bin_edges(train["num_open_accounts"], bins=5)[
        :-2]  # tried finding the optimal using the EDA app
    accts_bins = bin_numeric(accts_eges, train, "num_open_accounts", "default_flag")

    woe_result_accts = woe_iv(accts_bins, "num_open_accounts", "default_flag")
    bin_to_woe_accts = dict(zip(woe_result_accts["num_open_accounts"], woe_result_accts['WoE']))

    X_train_6 = apply_woe(X_train_6, "num_open_accounts", bin_to_woe_accts, accts_eges)
    X_test_6 = apply_woe(X_test_6, "num_open_accounts", bin_to_woe_accts, accts_eges)

    X_train_7, X_test_7 = X_train_6.copy(), X_test_6.copy()
    crd_util_eges = np.histogram_bin_edges(train["credit_utilisation_pct"], bins=8)[:-2]
    crd_util_bins = bin_numeric(crd_util_eges, train, "credit_utilisation_pct", "default_flag")

    woe_result_crd_util = woe_iv(crd_util_bins, "credit_utilisation_pct", "default_flag")
    bin_to_woe_crd_util = dict(zip(woe_result_crd_util["credit_utilisation_pct"], woe_result_crd_util['WoE']))

    X_train_7 = apply_woe(X_train_7, "credit_utilisation_pct", bin_to_woe_crd_util, crd_util_eges)
    X_test_7 = apply_woe(X_test_7, "credit_utilisation_pct", bin_to_woe_crd_util, crd_util_eges)

    X_train_8, X_test_8 = X_train_7.copy(), X_test_7.copy()
    employ_eges = [2, 5, 8, 12]
    employ_bins = bin_numeric(employ_eges, train, "employment_length_years", "default_flag")

    woe_result_employ = woe_iv(employ_bins, "employment_length_years", "default_flag")
    bin_to_woe_employ = dict(zip(woe_result_employ["employment_length_years"], woe_result_employ['WoE']))

    X_train_8 = apply_woe(X_train_8, "employment_length_years", bin_to_woe_employ, employ_eges)
    X_test_8 = apply_woe(X_test_8, "employment_length_years", bin_to_woe_employ, employ_eges)

    X_train_final, X_test_final = X_train_8.copy(), X_test_8.copy()

    scaler = StandardScaler()
    X_scaled_train_final = scaler.fit_transform(X_train_final)
    X_scaled_test_final = scaler.transform(X_test_final)

    return  X_scaled_train_final, X_scaled_test_final,y_train,y_test


def model_evaluation(model, X_test, y_test):

    st.header("🤖 Model Evaluation Dashboard")

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities for ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    st.divider()

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5, 4))

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)

    st.pyplot(fig)

    st.divider()

    # ROC Curve
    st.subheader("📈 ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots(figsize=(6, 5))

    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle="--")

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    st.pyplot(fig2)

    st.divider()

    # Interpretation
    st.info(
        "Higher AUC values indicate better model performance in distinguishing "
        "between risky and non-risky customers."
    )

@st.cache_data
def data_handler():
    """
    This function used to load data from csv file. and process all the cleaning of data
    :param filename: name of file of data
    :return: cleaned data,names of numerical variables,names of categorical variables,names of date variables
    """

    # Loading loan application data and having a view of data
    loan_book = load_data()

    # Dropping duplicates
    loan_book.drop_duplicates(inplace=True, ignore_index=True)

    # filling using by grouping by region and use median of region
    loan_book["annual_income"] = loan_book.groupby("region")["annual_income"].transform(lambda x: x.fillna(x.median()))

    # filling missing values using median
    loan_book["employment_length_years"] = loan_book["employment_length_years"].fillna(
        loan_book["employment_length_years"].median()
    )

    loan_book["num_open_accounts"] = loan_book["num_open_accounts"].fillna(
        loan_book["num_open_accounts"].median()
    )

    loan_book.drop(columns="months_since_last_delinquency", inplace=True)

    # fixing data types
    # float to int
    v = ["age", "employment_length_years", "num_open_accounts", "months_since_oldest_account"]
    for var in v:
        loan_book[var] = loan_book[var].astype("int64")

    # str(object) to date data type
    loan_book["application_date"] = loan_book["application_date"].astype("datetime64[ns]")

    # fixing labels in home_ownership
    loan_book["home_ownership"] = loan_book["home_ownership"].str.capitalize().map(
        lambda x: x if x not in ["Rent", "Own"]
        else ("Renting" if x == "Rent" else "Owner"))

    # fixing loan_purpose
    loan_book["loan_purpose"] = loan_book["loan_purpose"].str.capitalize().map(lambda x: x.replace("_", " "))

    loan_book["default_flag"] = loan_book["default_flag"].astype("category")

    return loan_book

def savefig(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer

@st.cache_data
def display_feature(name):
    """
    This is used when formating variables on the data set
    :param name: name of the variable
    :return: formated feature name
    """
    display_names = {
        "age": "Age",
        "annual_income": "Annual Income",
        "employment_length_years": "Employment Duration (Years)",
        "home_ownership": "Home Ownership",
        "region": "Region",

        "num_open_accounts": "Number of Opened Accounts",
        "num_delinquencies_2yr": "Number of Delinquencies in Last 2 Years",
        "total_revolving_balance": "Total Revolving Balance",
        "credit_utilisation_pct": "Credit Utilization Percentage",

        "months_since_oldest_account": "Age of Oldest Account (Months)",
        "num_hard_inquiries_6mo": "Number of Hard Inquiries (Last 6 Months)",

        "loan_amount": "Loan Amount",
        "interest_rate": "Interest Rate",
        "loan_purpose": "Loan Purpose",

        "dti_ratio": "Debt-to-Income Ratio",
        "months_since_last_delinquency": "Months Since Last Delinquency",
        "pct_accounts_current": "Percentage of Accounts in Good Standing",

        "application_date": "Loan Application Date",
        "application_dow": "Application Day of Week",

        "branch_code_id": "Branch ID",
        "months_at_current_address": "Time at Current Address (Months)",

        "email_domain_type": "Email Domain Category",
        "phone_verified": "Phone Verification Status",

        "default_flag": "Loan Default Status",
        "set": "Dataset Partition",
        "Loan Outcome": "Loan Default Status",

    }

    return display_names[name]

def format_stat(stat):
    """
    This is used when formatting using statistics aggregation
    :param stat: name of the statistic aggregation
    :return: format for statistic aggregation
    """
    if stat == "percent":
        fmt = "%.2f%%"

    elif stat == "proportion":
        fmt = "%.2f"

    elif stat == "probability":
        fmt = "%.3f"
    else:
        fmt = "%d"
    return fmt

def woe_iv(data, feature, target):
    """for WoE and Iv calculations"""
    df = data[[feature, target]].dropna().copy()

    grouped = df.groupby(feature,observed=False)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']

    grouped['good'] = grouped['total'] - grouped['bad']

    grouped['dist_good'] = grouped['good'] / grouped['good'].sum()
    grouped['dist_bad'] = grouped['bad'] / grouped['bad'].sum()

    grouped['WoE'] = np.log(
        (grouped['dist_good'] + 1e-6) /
        (grouped['dist_bad'] + 1e-6)
    )

    # IV contribution per bin
    grouped['iv_bin'] = (
        grouped['dist_good'] - grouped['dist_bad']
    ) * grouped['WoE']

    return grouped.reset_index()

def bin_numeric(bin_edges: pd.Series | list | np.ndarray,
                data: pd.DataFrame,
                feature: str,target: str):

    bin_edges = np.sort(np.array(bin_edges))

    bins = [data[feature].min()] + list(bin_edges) + [data[feature].max()]

    new_df = pd.DataFrame()

    new_df[feature] = pd.cut(
        data[feature],
        bins=bins,
        include_lowest=True,
        duplicates="drop"
    )
    new_df[target] = data[target]

    return new_df

def midpoint_bin(data: pd.DataFrame,feature: str):
    new_df = data.copy()
    # midpoint calculation
    new_df["bin_midpoint"] = new_df[feature].apply(
        lambda x: (x.left + x.right) / 2 if pd.notnull(x) else np.nan
    )

    return new_df

def capping_var(df_train:pd.DataFrame,df_test:pd.DataFrame,var:str):
    """
    Used to cap  variable with outliers
    :param df_train: Dataframe of the data with the variable of interest (train set)
    :param df_test: Dataframe of the data with the variable of interest (test set)
    :param var: name of the variable of interest
    :return: new dataframe with the variable of interest capped
    """
    df_train,df_test = df_train.copy(),df_test.copy()
    Q1 = df_train[var].quantile(0.25)
    Q3 = df_train[var].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_train[f"{var}_capped"] = df_train[var].clip(lower, upper)
    df_test[f"{var}_capped"] = df_test[var].clip(lower, upper)
    return df_train,df_test

def apply_woe(df, original_var, bin_to_woe_dict, bins=None):
    """
    Replace original variable with its WoE value
    """

    df = df.copy()
    bins = [df[original_var].min()] + list(bins) + [df[original_var].max()]
    # Create binned version and map WoE
    df[f'{original_var}_woe'] = pd.cut(df[original_var],
                                       bins=bins,
                                       labels=list(bin_to_woe_dict.keys()),
                                       right=True,include_lowest=True,duplicates="drop").map(bin_to_woe_dict)
    df.drop(columns = original_var, inplace = True)

    return df

if __name__ == "__main__":
    main()
