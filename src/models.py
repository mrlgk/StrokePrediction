"""Danh sách mô hình – người phụ trách: KHÁNH LINH."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_models(random_state: int = 42) -> dict:
    """Trả về 6 mô hình dùng trong đồ án, chưa fit."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),

        "KNN": KNeighborsClassifier(),

        "SVM": SVC(
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=random_state
        ),
    }