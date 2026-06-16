from __future__ import annotations

import pandas as pd


def downcast_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将面板中的 float64 列就地降为 float32，约把数值内存减半。

    因子与 IC/分层指标的计算本就以 float32 进行（见 MatrixPanel 与
    _calculate_factor_metrics_bundle_matrix），价格/收益/特征在 float32 下精度足够
    （~7 位有效数字）。这里只转 float64→float32，不动整型与其它 dtype（避免整型
    溢出），也不动索引。原地修改并返回同一对象；空表或 None 原样返回。

    用于在加载大面板时降低常驻内存——2.7GB 级输入下，面板及其派生的稠密 (T,N)
    矩阵内存可直接砍半。放在 factor_runtime（叶子依赖）以避免 infra/agents 间的
    循环导入。
    """
    if df is None or getattr(df, "empty", True):
        return df
    for column in df.columns:
        if str(df[column].dtype) == "float64":
            try:
                df[column] = df[column].astype("float32")
            except Exception:
                pass
    return df
