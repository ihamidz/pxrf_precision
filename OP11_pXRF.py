import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="pXRF Precision App", layout="wide")

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "pxrf_plot",
        "height": 700,
        "width": 1200,
        "scale": 2,
    },
}

st.title("pXRF Precision and Repeatability App")
st.caption(
    "Upload pXRF replicate data where repeated readings come from the same pulp sample. "
    "The app calculates precision metrics by sample and element, and provides interactive plots for review."
)


# -----------------------------
# Helpers
# -----------------------------
def infer_sample_and_replicate(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    s = series.astype(str).str.strip()

    # Split on the last hyphen:
    # CT01-T1 -> sample_id=CT01, replicate=T1
    # CT01-B2 -> sample_id=CT01, replicate=B2
    extracted = s.str.extract(r"^(.*)-([^-]+)$")
    sample = extracted[0].fillna(s)
    replicate = extracted[1]

    return sample, replicate


def identify_element_columns(df: pd.DataFrame, id_col: str, sample_col: str, rep_col: str) -> List[str]:
    excluded = {id_col, sample_col, rep_col}
    numeric_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            numeric_cols.append(col)
    return numeric_cols


def tidy_pxrf(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()
    out["sample_id"], out["replicate"] = infer_sample_and_replicate(out[id_col])

    element_cols = identify_element_columns(out, id_col, "sample_id", "replicate")
    for col in element_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    long_df = out.melt(
        id_vars=[id_col, "sample_id", "replicate"],
        value_vars=element_cols,
        var_name="element",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"])
    return long_df


def group_precision_stats(group: pd.DataFrame) -> pd.Series:
    vals = group["value"].dropna().astype(float)
    n = len(vals)
    mean = vals.mean() if n else np.nan
    std = vals.std(ddof=1) if n > 1 else np.nan
    median = vals.median() if n else np.nan
    min_v = vals.min() if n else np.nan
    max_v = vals.max() if n else np.nan
    value_range = max_v - min_v if n else np.nan
    rsd = (std / mean * 100) if (n > 1 and pd.notna(mean) and mean != 0) else np.nan
    mad = np.median(np.abs(vals - median)) if n else np.nan

    pairwise_abs = []
    pairwise_rel = []
    arr = vals.to_numpy(dtype=float)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            diff = abs(arr[i] - arr[j])
            pairwise_abs.append(diff)
            denom = np.mean([arr[i], arr[j]])
            pairwise_rel.append((diff / denom * 100) if denom != 0 else np.nan)

    mpad = np.nanmean(pairwise_abs) if pairwise_abs else np.nan
    mprd = np.nanmean(pairwise_rel) if pairwise_rel else np.nan

    return pd.Series(
        {
            "n_reps": n,
            "mean": mean,
            "std_dev": std,
            "rsd_percent": rsd,
            "median": median,
            "min": min_v,
            "max": max_v,
            "range": value_range,
            "mad": mad,
            "mean_pairwise_abs_diff": mpad,
            "mean_pairwise_rel_diff_percent": mprd,
        }
    )


def precision_table(long_df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        long_df.groupby(["sample_id", "element"], dropna=False)
        .apply(group_precision_stats)
        .reset_index()
    )
    return stats


def element_summary(stats_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        stats_df.groupby("element", dropna=False)
        .agg(
            samples=("sample_id", "nunique"),
            median_rsd_percent=("rsd_percent", "median"),
            p90_rsd_percent=(
                "rsd_percent",
                lambda x: np.nanpercentile(x.dropna(), 90) if x.dropna().any() else np.nan,
            ),
            worst_rsd_percent=("rsd_percent", "max"),
            median_range=("range", "median"),
        )
        .reset_index()
        .sort_values(["median_rsd_percent", "worst_rsd_percent"], ascending=[True, True])
    )
    return summary


def make_download_file(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="precision_results", index=False)
    return output.getvalue()


def make_template_file() -> bytes:
    template_df = pd.DataFrame(
        {
            "info": ["S1-0-A", "S1-0-B", "S1-0-C", "S2-0-A", "S2-0-B", "S2-0-C"],
            "Ag (ppm)": [2.5, 2.5, 2.5, 2.6, 2.4, 2.5],
            "Al (ppm)": [4823, 4713, 5031, 4890, 4760, 4985],
            "As (ppm)": [5, 5, 4, 5, 4, 5],
            "Au (ppm)": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
            "Ba (ppm)": [265, 264, 284, 272, 269, 281],
        }
    )
    return template_df.to_csv(index=False).encode("utf-8")


def duplicate_analysis_table(long_df: pd.DataFrame, sample_1: str, sample_2: str) -> pd.DataFrame:
    dup_df = long_df[long_df["sample_id"].isin([sample_1, sample_2])].copy()

    summary = (
        dup_df.groupby(["sample_id", "element"], dropna=False)["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"std": "std_dev", "count": "n_reps"})
    )

    pivot_mean = summary.pivot(index="element", columns="sample_id", values="mean")
    pivot_std = summary.pivot(index="element", columns="sample_id", values="std_dev")
    pivot_n = summary.pivot(index="element", columns="sample_id", values="n_reps")

    result = pd.DataFrame(index=sorted(dup_df["element"].dropna().unique()))
    result.index.name = "element"
    result = result.reset_index()

    result[f"{sample_1}_mean"] = result["element"].map(pivot_mean[sample_1]) if sample_1 in pivot_mean.columns else np.nan
    result[f"{sample_2}_mean"] = result["element"].map(pivot_mean[sample_2]) if sample_2 in pivot_mean.columns else np.nan
    result[f"{sample_1}_std_dev"] = result["element"].map(pivot_std[sample_1]) if sample_1 in pivot_std.columns else np.nan
    result[f"{sample_2}_std_dev"] = result["element"].map(pivot_std[sample_2]) if sample_2 in pivot_std.columns else np.nan
    result[f"{sample_1}_n_reps"] = result["element"].map(pivot_n[sample_1]) if sample_1 in pivot_n.columns else np.nan
    result[f"{sample_2}_n_reps"] = result["element"].map(pivot_n[sample_2]) if sample_2 in pivot_n.columns else np.nan

    result["mean_difference"] = result[f"{sample_1}_mean"] - result[f"{sample_2}_mean"]
    result["absolute_difference"] = result["mean_difference"].abs()
    result["average_of_means"] = result[[f"{sample_1}_mean", f"{sample_2}_mean"]].mean(axis=1)
    result["relative_difference_percent"] = np.where(
        result["average_of_means"].ne(0),
        result["absolute_difference"] / result["average_of_means"] * 100,
        np.nan,
    )
    result["pooled_std_dev"] = np.sqrt(
        (
            result[f"{sample_1}_std_dev"].fillna(0) ** 2
            + result[f"{sample_2}_std_dev"].fillna(0) ** 2
        ) / 2
    )
    result["z_score_like"] = np.where(
        result["pooled_std_dev"].ne(0),
        result["absolute_difference"] / result["pooled_std_dev"],
        np.nan,
    )

    return result


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Inputs")
st.sidebar.info(
    "Expected format: one row per reading, with one column containing IDs such as "
    "S1-0-A, S1-0-B, S1-0-C, or CT01-T1, CT01-T2, CT01-B1, CT01-B2, "
    "and numeric columns for each element such as Ag (ppm), Al (ppm), As (ppm). "
    "After upload, choose the correct Sample ID column."
)

st.sidebar.download_button(
    label="Download CSV template",
    data=make_template_file(),
    file_name="pxrf_template.csv",
    mime="text/csv",
)

uploaded = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

st.sidebar.subheader("Precision flags")
rsd_warn = st.sidebar.number_input("Warn if RSD (%) is above", min_value=0.0, value=10.0, step=0.5)


# -----------------------------
# Load data
# -----------------------------
if uploaded is None:
    st.info("Please upload a CSV or Excel file to begin.")
    st.stop()

if uploaded.name.lower().endswith(".csv"):
    raw_df = pd.read_csv(uploaded)
else:
    raw_df = pd.read_excel(uploaded)

if raw_df.empty:
    st.warning("The uploaded file is empty.")
    st.stop()

st.sidebar.subheader("Column selection")

id_column_options = raw_df.columns.tolist()
default_id_index = 0 if id_column_options else None

id_col = st.sidebar.selectbox(
    "Choose the Sample ID column",
    id_column_options,
    index=default_id_index,
    help="This column should contain values such as S1-0-A, S1-0-B, S1-0-C, or CT01-T1, CT01-T2, CT01-B1, CT01-B2.",
)

long_df = tidy_pxrf(raw_df, id_col=id_col)

if long_df.empty:
    st.error("No numeric element columns were found in the uploaded file after selecting the Sample ID column.")
    st.stop()

# -----------------------------
# Global filters
# -----------------------------
st.subheader("Analysis filters")

all_samples = sorted(long_df["sample_id"].dropna().unique().tolist())

if "selected_samples" not in st.session_state:
    st.session_state["selected_samples"] = all_samples.copy()
else:
    st.session_state["selected_samples"] = [
        s for s in st.session_state["selected_samples"] if s in all_samples
    ]

sample_col1, sample_col2, sample_col3 = st.columns([6, 1.5, 1.5])

with sample_col2:
    if st.button("Select all samples", key="select_all_samples"):
        st.session_state["selected_samples"] = all_samples.copy()

with sample_col3:
    if st.button("Clear samples", key="clear_all_samples"):
        st.session_state["selected_samples"] = []

with sample_col1:
    selected_samples = st.multiselect(
        "Choose samples to analyse",
        all_samples,
        default=st.session_state["selected_samples"],
        key="selected_samples",
    )

if not selected_samples:
    st.warning("Please select at least one sample to analyse.")
    st.stop()

filtered_long_df = long_df[long_df["sample_id"].isin(selected_samples)].copy()

available_elements = sorted(filtered_long_df["element"].dropna().unique().tolist())

if "selected_elements" not in st.session_state:
    st.session_state["selected_elements"] = available_elements.copy()
else:
    st.session_state["selected_elements"] = [
        e for e in st.session_state["selected_elements"] if e in available_elements
    ]

element_col1, element_col2, element_col3 = st.columns([6, 1.5, 1.5])

with element_col2:
    if st.button("Select all elements", key="select_all_elements"):
        st.session_state["selected_elements"] = available_elements.copy()

with element_col3:
    if st.button("Clear elements", key="clear_all_elements"):
        st.session_state["selected_elements"] = []

with element_col1:
    selected_elements = st.multiselect(
        "Choose variables to analyse",
        available_elements,
        default=st.session_state["selected_elements"],
        key="selected_elements",
    )

if not selected_elements:
    st.warning("Please select at least one variable to analyse.")
    st.stop()

filtered_long_df = filtered_long_df[filtered_long_df["element"].isin(selected_elements)].copy()

if filtered_long_df.empty:
    st.warning("No data remains after applying the selected filters.")
    st.stop()

stats_df = precision_table(filtered_long_df)
summary_df = element_summary(stats_df)

stats_df["precision_flag"] = np.where(
    stats_df["rsd_percent"] > rsd_warn,
    "Review",
    "OK",
)


# -----------------------------
# Main tables
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Raw data",
    "Per-sample precision",
    "Element summary",
    "Plots",
    "Duplicate analysis",
])

with tab1:
    st.subheader("Raw uploaded data")
    filtered_raw_df = raw_df[
        raw_df[id_col]
        .astype(str)
        .str.strip()
        .str.replace(r"-([A-Za-z0-9]+)$", "", regex=True)
        .isin(selected_samples)
    ].copy()
    st.dataframe(filtered_raw_df, use_container_width=True)

    st.subheader(f"Filtered long-format data used for calculations (Sample ID column: {id_col})")
    st.dataframe(filtered_long_df, use_container_width=True)

with tab2:
    st.subheader("Precision by sample and element")
    st.dataframe(stats_df, use_container_width=True)

    st.download_button(
        label="Download precision results as Excel",
        data=make_download_file(stats_df),
        file_name="pxrf_precision_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    flagged = stats_df[stats_df["precision_flag"] == "Review"]
    st.subheader("Flagged results")
    if flagged.empty:
        st.success("No results exceeded the current RSD (%) warning threshold.")
    else:
        st.dataframe(flagged, use_container_width=True)

with tab3:
    st.subheader("Precision summary by element")
    st.dataframe(summary_df, use_container_width=True)

    fig_summary = px.bar(
        summary_df,
        x="element",
        y="median_rsd_percent",
        hover_data={
            "element": True,
            "samples": True,
            "median_rsd_percent": ":.3f",
            "p90_rsd_percent": ":.3f",
            "worst_rsd_percent": ":.3f",
            "median_range": ":.3f",
        },
        category_orders={"element": summary_df.sort_values("median_rsd_percent", ascending=False)["element"].tolist()},
        labels={
            "element": "Element",
            "median_rsd_percent": "Median RSD (%)",
        },
        height=420,
    )
    fig_summary.update_layout(
        dragmode="zoom",
        xaxis_title="Element",
        yaxis_title="Median RSD (%)",
    )
    st.plotly_chart(fig_summary, use_container_width=True, config=PLOTLY_CONFIG)

with tab4:
    st.subheader("Plots")

    elements_all = sorted(filtered_long_df["element"].dropna().unique().tolist())
    chosen_element = st.selectbox("Choose an element", elements_all)

    plot_long = filtered_long_df[filtered_long_df["element"] == chosen_element].copy()
    plot_stats = stats_df[stats_df["element"] == chosen_element].copy()

    mean_plot_df = (
        plot_long.groupby("sample_id", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean_value"})
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Replicate readings by sample**")
        fig_line = px.line(
            plot_long,
            x="sample_id",
            y="value",
            color="replicate",
            markers=True,
            category_orders={"sample_id": selected_samples},
            hover_data={
                "sample_id": True,
                "replicate": True,
                "value": ":.6g",
            },
            labels={
                "sample_id": "Sample",
                "value": chosen_element,
                "replicate": "Replicate",
            },
            height=420,
        )
        fig_line.update_layout(
            dragmode="zoom",
            xaxis_title="Sample",
            yaxis_title=chosen_element,
            hovermode="x unified",
        )
        st.plotly_chart(fig_line, use_container_width=True, config=PLOTLY_CONFIG)

    with col2:
        st.markdown("**Average replicate amount by sample**")
        fig_mean = px.line(
            mean_plot_df,
            x="sample_id",
            y="mean_value",
            markers=True,
            category_orders={"sample_id": selected_samples},
            hover_data={
                "sample_id": True,
                "mean_value": ":.6g",
            },
            labels={
                "sample_id": "Sample",
                "mean_value": f"{chosen_element} mean",
            },
            height=420,
        )
        fig_mean.update_layout(
            dragmode="zoom",
            xaxis_title="Sample",
            yaxis_title=f"{chosen_element} mean",
        )
        st.plotly_chart(fig_mean, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("**RSD (%) by sample**")
    fig_rsd = px.bar(
        plot_stats,
        x="sample_id",
        y="rsd_percent",
        category_orders={"sample_id": selected_samples},
        hover_data={
            "sample_id": True,
            "mean": ":.6g",
            "std_dev": ":.6g",
            "rsd_percent": ":.3f",
            "range": ":.6g",
            "n_reps": True,
        },
        labels={
            "sample_id": "Sample",
            "rsd_percent": "RSD (%)",
        },
        height=420,
    )
    fig_rsd.update_layout(
        dragmode="zoom",
        xaxis_title="Sample",
        yaxis_title="RSD (%)",
    )
    st.plotly_chart(fig_rsd, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("**Mean vs standard deviation**")
    fig_scatter = px.scatter(
        stats_df,
        x="mean",
        y="std_dev",
        color="element",
        hover_data={
            "sample_id": True,
            "element": True,
            "mean": ":.6g",
            "std_dev": ":.6g",
            "rsd_percent": ":.3f",
            "range": ":.6g",
        },
        labels={
            "mean": "Mean",
            "std_dev": "Standard deviation",
            "element": "Element",
        },
        height=450,
    )
    fig_scatter.update_traces(marker={"size": 10})
    fig_scatter.update_layout(
        dragmode="zoom",
        xaxis_title="Mean",
        yaxis_title="Standard deviation",
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CONFIG)

with tab5:
    st.subheader("Duplicate analysis")
    st.caption(
        "Choose two samples to compare as duplicates. The analysis uses all readings from both selected samples."
    )

    sample_options = sorted(filtered_long_df["sample_id"].dropna().unique().tolist())

    if len(sample_options) < 2:
        st.warning("Please select at least two samples in the analysis filters to run duplicate analysis.")
    else:
        dup_col1, dup_col2 = st.columns(2)
        with dup_col1:
            duplicate_sample_1 = st.selectbox("Duplicate sample 1", sample_options, key="duplicate_sample_1")
        with dup_col2:
            default_index_2 = 1 if len(sample_options) > 1 else 0
            duplicate_sample_2 = st.selectbox(
                "Duplicate sample 2",
                sample_options,
                index=default_index_2,
                key="duplicate_sample_2",
            )

        if duplicate_sample_1 == duplicate_sample_2:
            st.warning("Please choose two different samples for duplicate analysis.")
        else:
            selected_duplicate_df = filtered_long_df[
                filtered_long_df["sample_id"].isin([duplicate_sample_1, duplicate_sample_2])
            ].copy()

            st.markdown("**Selected duplicate readings**")
            st.dataframe(selected_duplicate_df, use_container_width=True)

            duplicate_results_df = duplicate_analysis_table(
                filtered_long_df, duplicate_sample_1, duplicate_sample_2
            )

            st.markdown("**Duplicate comparison by element**")
            st.dataframe(duplicate_results_df, use_container_width=True)

            st.download_button(
                label="Download duplicate analysis as Excel",
                data=make_download_file(duplicate_results_df),
                file_name=f"duplicate_analysis_{duplicate_sample_1}_vs_{duplicate_sample_2}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            duplicate_elements = sorted(duplicate_results_df["element"].dropna().unique().tolist())

            if duplicate_elements:
                element_for_duplicate_plot = st.selectbox(
                    "Choose an element for duplicate plot",
                    duplicate_elements,
                    key="duplicate_plot_element",
                )

                duplicate_plot_df = selected_duplicate_df[
                    selected_duplicate_df["element"] == element_for_duplicate_plot
                ].copy()

                st.markdown("**Duplicate readings plot**")
                fig_dup_points = px.scatter(
                    duplicate_plot_df,
                    x="sample_id",
                    y="value",
                    color="replicate",
                    category_orders={"sample_id": [duplicate_sample_1, duplicate_sample_2]},
                    hover_data={
                        "sample_id": True,
                        "replicate": True,
                        "value": ":.6g",
                        "element": True,
                    },
                    labels={
                        "sample_id": "Duplicate sample",
                        "value": element_for_duplicate_plot,
                        "replicate": "Replicate",
                    },
                    height=420,
                )
                fig_dup_points.update_traces(marker={"size": 11})
                fig_dup_points.update_layout(
                    dragmode="zoom",
                    xaxis_title="Duplicate sample",
                    yaxis_title=element_for_duplicate_plot,
                )
                st.plotly_chart(fig_dup_points, use_container_width=True, config=PLOTLY_CONFIG)

                st.markdown("**Duplicate relative difference by element**")
                fig_rel_diff = px.bar(
                    duplicate_results_df,
                    x="element",
                    y="relative_difference_percent",
                    hover_data={
                        "element": True,
                        f"{duplicate_sample_1}_mean": ":.6g",
                        f"{duplicate_sample_2}_mean": ":.6g",
                        "mean_difference": ":.6g",
                        "absolute_difference": ":.6g",
                        "relative_difference_percent": ":.3f",
                        "z_score_like": ":.3f",
                    },
                    labels={
                        "element": "Element",
                        "relative_difference_percent": "Relative difference (%)",
                    },
                    height=420,
                )
                fig_rel_diff.update_layout(
                    dragmode="zoom",
                    xaxis_title="Element",
                    yaxis_title="Relative difference (%)",
                )
                st.plotly_chart(fig_rel_diff, use_container_width=True, config=PLOTLY_CONFIG)

# -----------------------------
# Notes section
# -----------------------------
with st.expander("How the precision metrics are calculated"):
    st.markdown(
        """
        - **Mean**: average of repeated readings for a sample.
        - **Standard deviation**: spread of the repeated readings.
        - **RSD (%)**: standard deviation divided by the mean, multiplied by 100.
        - **Range**: maximum minus minimum reading.
        - **MAD**: median absolute deviation, a robust spread metric.
        - **Mean pairwise absolute difference**: average absolute difference across all replicate pairs.
        - **Mean pairwise relative difference (%)**: average relative difference across all replicate pairs.

        In practice, **RSD (%)** is often the simplest precision measure to compare across elements,
        while **range** is useful as a descriptive metric.
        """
    )

st.markdown("---")
st.caption("Upload-first app for pXRF repeatability checks across replicate readings.")
