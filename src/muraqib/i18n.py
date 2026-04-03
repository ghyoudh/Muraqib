import json
import os

_TRANSLATIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "translations.json"
)

_cache = {}


def _load():
    global _cache
    if not _cache:
        with open(_TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def get_text(lang: str) -> dict:
    """Return the full translation dict for a language ('en' or 'ar')."""
    data = _load()
    return data.get(lang, data["en"])


def translate_activity(name: str, lang: str) -> str:
    """Translate an activity name from Arabic→EN or EN→AR."""
    data = _load()
    return data.get(lang, {}).get("activities", {}).get(name, name)


def translate_contractor(name: str, lang: str) -> str:
    """Translate a contractor name from Arabic→EN or EN→AR."""
    data = _load()
    return data.get(lang, {}).get("contractors", {}).get(name, name)


def translate_complexity(value: str, lang: str) -> str:
    """Translate complexity level label."""
    data = _load()
    return data.get(lang, {}).get("complexity", {}).get(value, value)


# UI string dictionary (not in translations.json — app-level strings)
UI_STRINGS = {
    "en": {
        "app_title": "Muraqib — Project Delay Predictor",
        "app_subtitle": "AI-powered risk analysis for Saudi construction projects",
        "lang_toggle": "العربية 🌐",
        "tab_overview": "📊 Data Overview",
        "tab_predict": "🤖 Risk Prediction",
        "tab_analytics": "📈 Analytics",
        "table_title": "Historical Project Activities",
        "total_activities": "Total Activities",
        "high_risk": "High Complexity",
        "delay_rate": "Predicted Delay Rate",
        "unique_contractors": "Contractors",
        "predict_title": "Predict Delay Risk for a New Activity",
        "select_activity": "Activity",
        "select_contractor": "Contractor",
        "select_complexity": "Complexity Level",
        "supply_delay": "Supply Delay (days)",
        "subcontractor_perf": "Subcontractor Performance Score (1–10)",
        "weather_risk": "Weather Risk",
        "labor_availability": "Labor Availability (%)",
        "predict_btn": "🔍 Predict Risk",
        "result_low": "✅ Low Risk",
        "result_high": "⚠️ High Risk of Delay",
        "probability": "Delay Probability",
        "feature_importance": "Key Risk Factors",
        "analytics_title": "Risk Analytics Dashboard",
        "delay_by_complexity": "Delay Rate by Complexity Level",
        "delay_by_contractor": "Delay Rate by Contractor",
        "activity_distribution": "Activity Distribution by Complexity",
        "monthly_heatmap": "Monthly Activity Start Distribution",
        "risk_gauge": "Risk Gauge",
        "weather_low": "Low",
        "weather_medium": "Medium",
        "weather_high": "High",
        "complexity_low": "Low",
        "complexity_medium": "Medium",
        "complexity_high": "High",
    },
    "ar": {
        "app_title": "مراقب — نظام توقع تأخير المشاريع",
        "app_subtitle": "تحليل المخاطر بالذكاء الاصطناعي لمشاريع البناء السعودية",
        "lang_toggle": "English 🌐",
        "tab_overview": "📊 نظرة عامة على البيانات",
        "tab_predict": "🤖 التنبؤ بالمخاطر",
        "tab_analytics": "📈 التحليلات",
        "table_title": "أنشطة المشاريع التاريخية",
        "total_activities": "إجمالي الأنشطة",
        "high_risk": "تعقيد عالٍ",
        "delay_rate": "معدل التأخير المتوقع",
        "unique_contractors": "المقاولون",
        "predict_title": "توقع مخاطر التأخير لنشاط جديد",
        "select_activity": "النشاط",
        "select_contractor": "المقاول",
        "select_complexity": "مستوى التعقيد",
        "supply_delay": "تأخير التوريد (أيام)",
        "subcontractor_perf": "أداء المقاول الباطن (1–10)",
        "weather_risk": "مخاطر الطقس",
        "labor_availability": "توافر العمالة (%)",
        "predict_btn": "🔍 توقع المخاطر",
        "result_low": "✅ مخاطر منخفضة",
        "result_high": "⚠️ مخاطر تأخير عالية",
        "probability": "احتمالية التأخير",
        "feature_importance": "أهم عوامل الخطر",
        "analytics_title": "لوحة تحليلات المخاطر",
        "delay_by_complexity": "معدل التأخير حسب مستوى التعقيد",
        "delay_by_contractor": "معدل التأخير حسب المقاول",
        "activity_distribution": "توزيع الأنشطة حسب التعقيد",
        "monthly_heatmap": "توزيع تواريخ بدء الأنشطة الشهرية",
        "risk_gauge": "مقياس الخطر",
        "weather_low": "منخفض",
        "weather_medium": "متوسط",
        "weather_high": "عالٍ",
        "complexity_low": "منخفض",
        "complexity_medium": "متوسط",
        "complexity_high": "عالٍ",
    },
}


def ui(key: str, lang: str) -> str:
    """Return a UI string for the given language."""
    return UI_STRINGS.get(lang, UI_STRINGS["en"]).get(key, key)
