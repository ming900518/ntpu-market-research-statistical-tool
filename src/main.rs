#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use mimalloc::MiMalloc;
use polars::{
    export::rayon::prelude::{IntoParallelRefIterator, ParallelIterator},
    prelude::*,
};
use pyo3::{
    types::{PyDict, PyModule},
    IntoPy, Python,
};
use pyo3_polars::PyDataFrame;
use std::{env, fmt::Display, fs, process::exit};

enum CorrelationValue {
    Valid(CorrelationResult),
    NotValid,
}

impl Display for CorrelationValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotValid => write!(f, "不適用"),
            Self::Valid(result) => write!(f, "{result}"),
        }
    }
}

struct CorrelationResult {
    r: f64,
    p_value: f64,
}

impl Display for CorrelationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = 5;
        write!(
            f,
            "r: {:.precision$}<br>p value: {:.precision$}",
            self.r, self.p_value
        )
    }
}

impl From<(f64, f64)> for CorrelationResult {
    fn from(value: (f64, f64)) -> Self {
        Self {
            r: value.0,
            p_value: value.1,
        }
    }
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    let args: Vec<String> = env::args().collect();

    let Some(file_name) = args.get(1) else { 
        eprintln!("No file specified.");
        exit(1)
    };

    let mut result = Vec::new();
    set_env();
    let Ok(Ok(orig_dataframe)) = CsvReader::from_path(file_name).map(|csv| csv.infer_schema(None).has_header(true).finish()) else {
        eprintln!("Unable to open CSV file.");
        exit(1)
    };
    result.push(format!(
        "## 敘述統計\n\n{}\n\n",
        orig_dataframe
            .describe(Some(&[0.05, 0.25, 0.5, 0.75, 0.95]))
            .unwrap()
    ));
    let column_names = orig_dataframe.get_column_names();
    let processed_data = column_names
        .par_iter()
        .map(|column| {
            orig_dataframe
                .column(column)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .map(|data| data.unwrap_or(0.0))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();
    let mut pearson_series_vec = Vec::new();
    let first_column = Series::new("", column_names.as_slice());
    pearson_series_vec.push(first_column);
    processed_data.iter().enumerate().for_each(|(index, x)| {
        let series_name = column_names.get(index).unwrap_or(&"未知");
        let pearson_series = processed_data
            .iter()
            .map(|y| {
                if x == y {
                    format!("{}", CorrelationValue::NotValid)
                } else {
                    format!(
                        "{}",
                        CorrelationValue::Valid(CorrelationResult::from(pearson(
                            x.clone(),
                            y.clone()
                        )))
                    )
                }
            })
            .collect::<Vec<String>>();
        pearson_series_vec.push(Series::new(series_name, pearson_series.as_slice()));
    });
    result.push(format!(
        "## Pearson \n\n{}\n\n",
        DataFrame::new(pearson_series_vec).unwrap()
    ));

    let factor_analysis_dataframe = DataFrame::new(
        column_names
            .par_iter()
            .filter(|&&name| {
                name == "請問您一次願意花多少新台幣購買手機充電設備 (例如：充電線、豆腐頭) ?"
                    || name == "您一個月的平均花費為多少新台幣?"
            })
            .map(|column| {
                let data = orig_dataframe
                    .column(column)
                    .unwrap()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|data| data.unwrap_or(0.0))
                    .collect::<Vec<f64>>();
                Series::new(column, data)
            })
            .collect::<Vec<Series>>(),
    )
    .unwrap();

    result.push(format!(
        "## 因子分析 \n\n{}\n\n",
        factor_analysis(factor_analysis_dataframe)
    ));

    if fs::write(format!("{file_name}.md"), result.join("")).is_err() {
        eprintln!("Unable to write result.");
        exit(1)
    }
}

fn set_env() {
    env::set_var("POLARS_FMT_MAX_ROWS", u16::MAX.to_string());
    env::set_var("POLARS_FMT_MAX_COLS", u16::MAX.to_string());
    env::set_var("POLARS_FMT_STR_LEN", u16::MAX.to_string());
    env::set_var("POLARS_FMT_TABLE_FORMATTING", "ASCII_MARKDOWN");
    env::set_var("POLARS_TABLE_WIDTH", u16::MAX.to_string());
    env::set_var("POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES", 1.to_string());
    env::set_var("POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR", 1.to_string());
    env::set_var(
        "POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION",
        1.to_string(),
    );
}

fn factor_analysis(dataframe: DataFrame) -> DataFrame {
    Python::with_gil(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("dataframe", PyDataFrame(dataframe).into_py(py))
            .unwrap();
        py.run(
            r#"
from factor_analyzer import FactorAnalyzer
import polars
fa = FactorAnalyzer(rotation="promax")
converted = dataframe.to_pandas(use_pyarrow_extension_array=True)
fa.fit(converted)
result = polars.DataFrame(data=fa.loadings_,schema=converted.columns.tolist())
        "#,
            None,
            Some(locals),
        )
        .unwrap();
        locals
            .get_item("result")
            .unwrap()
            .extract::<PyDataFrame>()
            .unwrap()
            .into()
    })
}

fn pearson(x: Vec<f64>, y: Vec<f64>) -> (f64, f64) {
    Python::with_gil(|py| {
        let stats = PyModule::import(py, "scipy.stats").unwrap();
        let pearson: (f64, f64) = stats
            .getattr("pearsonr")
            .unwrap()
            .call1((x, y))
            .unwrap()
            .extract()
            .unwrap();
        pearson
    })
}
