import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
import psutil
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_model.pkl")
BASELINE_METRICS_PATH = os.path.join(MODEL_DIR, "baseline_metrics.json")
TEST_RESULT_PATH = os.path.join(os.path.dirname(__file__), "test_result.txt")


def log_test_result(test_name, status, details=None, metrics=None):
    """テスト結果をファイルに記録する"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(TEST_RESULT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"テスト名: {test_name}\n")
        f.write(f"実行時刻: {timestamp}\n")
        f.write(f"結果: {status}\n")
        
        if details:
            f.write(f"詳細: {details}\n")
        
        if metrics:
            f.write("メトリクス:\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        
        f.write(f"{'='*80}\n")


@pytest.fixture(scope="session", autouse=True)
def setup_test_log():
    """テスト開始時にログファイルを初期化"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TEST_RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(f"MLモデルテスト実行結果レポート\n")
        f.write(f"実行開始時刻: {timestamp}\n")
        f.write(f"{'='*80}\n")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    try:
        if not os.path.exists(MODEL_PATH):
            log_test_result("test_model_exists", "SKIPPED", "モデルファイルが存在しないためスキップ")
            pytest.skip("モデルファイルが存在しないためスキップします")
        
        assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"
        log_test_result("test_model_exists", "PASSED", "モデルファイルの存在を確認")
    except Exception as e:
        log_test_result("test_model_exists", "FAILED", f"エラー: {str(e)}")
        raise


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    try:
        model, X_test, y_test = train_model

        # 予測と精度計算
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
        assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"
        
        log_test_result("test_model_accuracy", "PASSED", 
                       f"精度検証成功 (閾値: 0.75以上)", 
                       {"accuracy": accuracy, "threshold": 0.75})
    except Exception as e:
        log_test_result("test_model_accuracy", "FAILED", f"エラー: {str(e)}")
        raise


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"


def test_create_baseline_model(sample_data, preprocessor):
    """ベースラインモデルを作成・保存（初回実行時のみ）"""
    if not os.path.exists(BASELINE_MODEL_PATH):
        # データの分割
        X = sample_data.drop("Survived", axis=1)
        y = sample_data["Survived"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ベースラインモデル（シンプルなRandomForest）
        baseline_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=50, random_state=42)),
            ]
        )

        # 学習
        baseline_model.fit(X_train, y_train)
        
        # 評価
        y_pred = baseline_model.predict(X_test)
        baseline_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }

        # ベースラインモデルとメトリクスを保存
        with open(BASELINE_MODEL_PATH, "wb") as f:
            pickle.dump(baseline_model, f)
        
        with open(BASELINE_METRICS_PATH, "w") as f:
            json.dump(baseline_metrics, f)
        
        print(f"ベースラインモデルを作成しました: accuracy={baseline_metrics['accuracy']:.4f}")


def test_model_performance_vs_baseline(train_model):
    """現在のモデルがベースラインより性能が劣化していないか検証"""
    try:
        model, X_test, y_test = train_model
        
        # ベースラインメトリクスが存在しない場合はスキップ
        if not os.path.exists(BASELINE_METRICS_PATH):
            log_test_result("test_model_performance_vs_baseline", "SKIPPED", 
                           "ベースラインメトリクスが存在しないためスキップ")
            pytest.skip("ベースラインメトリクスが存在しないためスキップします")
        
        # ベースラインメトリクスを読み込み
        with open(BASELINE_METRICS_PATH, "r") as f:
            baseline_metrics = json.load(f)
        
        # 現在のモデルの性能を評価
        y_pred = model.predict(X_test)
        current_accuracy = accuracy_score(y_test, y_pred)
        current_precision = precision_score(y_test, y_pred, average='weighted')
        current_recall = recall_score(y_test, y_pred, average='weighted')
        current_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ベースラインとの比較（5%の劣化まで許容）
        tolerance = 0.05
        
        assert current_accuracy >= (baseline_metrics["accuracy"] - tolerance), \
            f"精度が劣化しています: current={current_accuracy:.4f}, baseline={baseline_metrics['accuracy']:.4f}"
        
        assert current_precision >= (baseline_metrics["precision"] - tolerance), \
            f"適合率が劣化しています: current={current_precision:.4f}, baseline={baseline_metrics['precision']:.4f}"
        
        assert current_recall >= (baseline_metrics["recall"] - tolerance), \
            f"再現率が劣化しています: current={current_recall:.4f}, baseline={baseline_metrics['recall']:.4f}"
        
        assert current_f1 >= (baseline_metrics["f1"] - tolerance), \
            f"F1スコアが劣化しています: current={current_f1:.4f}, baseline={baseline_metrics['f1']:.4f}"
        
        # 結果をログに記録
        comparison_metrics = {
            "current_accuracy": current_accuracy,
            "baseline_accuracy": baseline_metrics["accuracy"],
            "current_precision": current_precision,
            "baseline_precision": baseline_metrics["precision"],
            "current_recall": current_recall,
            "baseline_recall": baseline_metrics["recall"],
            "current_f1": current_f1,
            "baseline_f1": baseline_metrics["f1"],
            "tolerance": tolerance
        }
        
        log_test_result("test_model_performance_vs_baseline", "PASSED", 
                       "ベースラインとの比較で性能劣化なし", comparison_metrics)
    except Exception as e:
        log_test_result("test_model_performance_vs_baseline", "FAILED", f"エラー: {str(e)}")
        raise


def test_model_memory_usage(train_model):
    """モデルの推論時のメモリ使用量を検証"""
    try:
        model, X_test, _ = train_model
        
        # 推論前のメモリ使用量
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 推論実行
        _ = model.predict(X_test)
        
        # 推論後のメモリ使用量
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # メモリ使用量が100MB未満であることを確認
        assert memory_increase < 100, f"推論時のメモリ使用量が多すぎます: {memory_increase:.2f}MB"
        
        memory_metrics = {
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_increase_mb": memory_increase,
            "threshold_mb": 100
        }
        
        log_test_result("test_model_memory_usage", "PASSED", 
                       f"メモリ使用量検証成功 (増加量: {memory_increase:.2f}MB)", 
                       memory_metrics)
    except Exception as e:
        log_test_result("test_model_memory_usage", "FAILED", f"エラー: {str(e)}")
        raise


def test_model_inference_time_benchmark(train_model):
    """推論時間のベンチマークテスト（複数回実行して平均を取る）"""
    try:
        model, X_test, _ = train_model
        
        # 複数回実行して推論時間を計測
        inference_times = []
        num_runs = 10
        
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.predict(X_test)
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        # 統計情報を計算
        avg_time = np.mean(inference_times)
        max_time = np.max(inference_times)
        min_time = np.min(inference_times)
        std_time = np.std(inference_times)
        
        # 平均推論時間が0.5秒未満、最大時間が1秒未満であることを確認
        assert avg_time < 0.5, f"平均推論時間が長すぎます: {avg_time:.4f}秒"
        assert max_time < 1.0, f"最大推論時間が長すぎます: {max_time:.4f}秒"
        assert std_time < 0.1, f"推論時間のばらつきが大きすぎます: {std_time:.4f}秒"
        
        timing_metrics = {
            "num_runs": num_runs,
            "avg_time_sec": avg_time,
            "max_time_sec": max_time,
            "min_time_sec": min_time,
            "std_time_sec": std_time,
            "avg_threshold_sec": 0.5,
            "max_threshold_sec": 1.0,
            "std_threshold_sec": 0.1
        }
        
        log_test_result("test_model_inference_time_benchmark", "PASSED", 
                       f"推論時間ベンチマーク成功 (平均: {avg_time:.4f}秒)", 
                       timing_metrics)
    except Exception as e:
        log_test_result("test_model_inference_time_benchmark", "FAILED", f"エラー: {str(e)}")
        raise


def test_model_accuracy_strict(train_model):
    """より厳密な精度検証（複数のメトリクスで評価）"""
    try:
        model, X_test, y_test = train_model
        
        # 予測実行
        y_pred = model.predict(X_test)
        
        # 複数のメトリクスで評価
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # より厳密な閾値で検証
        assert accuracy >= 0.80, f"精度が基準を下回っています: {accuracy:.4f} < 0.80"
        assert precision >= 0.80, f"適合率が基準を下回っています: {precision:.4f} < 0.80"
        assert recall >= 0.80, f"再現率が基準を下回っています: {recall:.4f} < 0.80"
        assert f1 >= 0.80, f"F1スコアが基準を下回っています: {f1:.4f} < 0.80"
        
        strict_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "threshold": 0.80
        }
        
        log_test_result("test_model_accuracy_strict", "PASSED", 
                       "厳密な精度検証成功 (全メトリクス0.80以上)", 
                       strict_metrics)
    except Exception as e:
        log_test_result("test_model_accuracy_strict", "FAILED", f"エラー: {str(e)}")
        raise


def test_model_prediction_consistency(train_model):
    """同じ入力に対する予測の一貫性を検証"""
    try:
        model, X_test, _ = train_model
        
        # 同じデータで複数回予測
        predictions_1 = model.predict(X_test)
        predictions_2 = model.predict(X_test)
        predictions_3 = model.predict(X_test)
        
        # 全ての予測が同じであることを確認
        assert np.array_equal(predictions_1, predictions_2), "予測結果に一貫性がありません（1回目と2回目）"
        assert np.array_equal(predictions_2, predictions_3), "予測結果に一貫性がありません（2回目と3回目）"
        
        consistency_metrics = {
            "num_predictions": len(predictions_1),
            "prediction_1_sum": int(np.sum(predictions_1)),
            "prediction_2_sum": int(np.sum(predictions_2)),
            "prediction_3_sum": int(np.sum(predictions_3)),
            "all_equal": True
        }
        
        log_test_result("test_model_prediction_consistency", "PASSED", 
                       "予測一貫性検証成功 (3回の予測が全て同一)", 
                       consistency_metrics)
    except Exception as e:
        log_test_result("test_model_prediction_consistency", "FAILED", f"エラー: {str(e)}")
        raise
