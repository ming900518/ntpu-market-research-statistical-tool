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
use serde::Deserialize;
use serde_json::from_reader;
use std::{
    env,
    fmt::Display,
    fs::{write, File},
    io::BufReader,
    process::exit,
};

#[derive(Deserialize, Debug)]
struct Field {
    name: String,
    scale: Scale,
}

#[derive(Deserialize, Debug)]
enum Scale {
    Nominal,
    Ordinal,
}

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
        if self.r > 0.0 && self.p_value < 0.05 {
            write!(
                f,
                "**r: {:.precision$}** <br> **p value: {:.precision$}**",
                self.r, self.p_value
            )
        } else {
            write!(
                f,
                "r: {:.precision$}<br>p value: {:.precision$}",
                self.r, self.p_value
            )
        }
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

    let Some(source_file_name) = args.get(1) else {
        eprintln!("No source file specified.");
        exit(1)
    };

    let Some(field_file_name) = args.get(2) else {
        eprintln!("No field description file specified.");
        exit(1)
    };

    let parse = from_reader(BufReader::new(File::open(field_file_name).unwrap()));

    let Ok(fields): Result<Vec<Field>, serde_json::Error> = parse else {
        eprintln!("Unable to parse fields from JSON file you specified. {}", parse.unwrap_err());
        exit(1)
    };

    let mut result = Vec::new();
    set_env();
    let Ok(Ok(orig_dataframe)) = CsvReader::from_path(source_file_name).map(|csv| csv.infer_schema(None).has_header(true).finish()) else {
        eprintln!("Unable to open CSV file.");
        exit(1)
    };
    result.push(format!(
        "## 敘述統計\n\n{}\n\n",
        orig_dataframe
            .describe(Some(&[0.05, 0.25, 0.5, 0.75, 0.95]))
            .unwrap()
    ));
    let processed_data = fields
        .par_iter()
        .map(|field| {
            orig_dataframe
                .column(&field.name)
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
    let (pearson_series_vec, kendall_series_vec) = correlation(&processed_data, &fields);
    result.push(format!(
        "## Pearson \n\n{}\n\n",
        DataFrame::new(pearson_series_vec).unwrap()
    ));

    result.push(format!(
        "## Kendall \n\n{}\n\n",
        DataFrame::new(kendall_series_vec).unwrap()
    ));

    let factor_analysis_dataframe = DataFrame::new(
        fields
            .par_iter()
            .filter(|field| {
                field
                    .name
                    .contains("請問您一次願意花多少新台幣購買手機充電設備 (例如：充電線、豆腐頭) ?")
                    || field.name.contains("您一個月的平均花費為多少新台幣?")
            })
            .map(|field| {
                let data = orig_dataframe
                    .column(&field.name)
                    .unwrap()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|data| data.unwrap_or(0.0))
                    .collect::<Vec<f64>>();
                Series::new(&field.name, data)
            })
            .collect::<Vec<Series>>(),
    )
    .unwrap();

    result.push(format!(
        "## 因子分析 \n\n{}\n\n",
        factor_analysis(factor_analysis_dataframe)
    ));

    if write(format!("{source_file_name}.md"), result.join("")).is_err() {
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

fn correlation(processed_data: &[Vec<f64>], fields: &Vec<Field>) -> (Vec<Series>, Vec<Series>) {
    let column_names = fields
        .iter()
        .map(|field| field.name.clone())
        .collect::<Vec<String>>();
    let pearson_column_names = Series::new(
        "",
        fields
            .iter()
            .filter(|field| matches!(field.scale, Scale::Ordinal))
            .map(|field| field.name.clone())
            .collect::<Vec<String>>()
            .as_slice(),
    );
    let kendall_column_names = Series::new(
        "",
        fields
            .iter()
            .filter(|field| matches!(field.scale, Scale::Nominal))
            .map(|field| field.name.clone())
            .collect::<Vec<String>>()
            .as_slice(),
    );
    let mut pearson_series_vec = Vec::new();
    let mut kendall_series_vec = Vec::new();
    pearson_series_vec.push(pearson_column_names);
    kendall_series_vec.push(kendall_column_names);
    processed_data
        .iter()
        .enumerate()
        .zip(fields)
        .for_each(|((index, x), field)| {
            let unknown_value = String::from("未知");
            let series_name = column_names.get(index).unwrap_or(&unknown_value);
            match field.scale {
                Scale::Nominal => {
                    let kendall_series = processed_data
                        .iter()
                        .zip(fields)
                        .filter(|(_, field)| matches!(field.scale, Scale::Nominal))
                        .map(|(y, _)| {
                            if x == y {
                                format!("{}", CorrelationValue::NotValid)
                            } else {
                                format!(
                                    "{}",
                                    CorrelationValue::Valid(CorrelationResult::from(kendall(
                                        x.clone(),
                                        y.clone()
                                    )))
                                )
                            }
                        })
                        .collect::<Vec<String>>();
                    kendall_series_vec.push(Series::new(series_name, kendall_series.as_slice()));
                }
                Scale::Ordinal => {
                    let pearson_series = processed_data
                        .iter()
                        .zip(fields)
                        .filter(|(_, field)| matches!(field.scale, Scale::Ordinal))
                        .map(|(y, _)| {
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
                }
            }
        });
    (pearson_series_vec, kendall_series_vec)
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

fn kendall(x: Vec<f64>, y: Vec<f64>) -> (f64, f64) {
    Python::with_gil(|py| {
        let stats = PyModule::import(py, "scipy.stats").unwrap();
        let pearson: (f64, f64) = stats
            .getattr("kendalltau")
            .unwrap()
            .call1((x, y))
            .unwrap()
            .extract()
            .unwrap();
        pearson
    })
}
