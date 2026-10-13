"""
Cybong — AI Chatbot tư vấn build PC & chọn Laptop theo nhu cầu và ngân sách
============================================================================
Kết hợp CẢ HAI:
1. Một model Machine Learning TỰ HUẤN LUYỆN (TF-IDF + Logistic Regression,
   xem train_model.py) để nhận diện nhu cầu sử dụng (Gaming, Đồ họa, Dựng
   phim, Lập trình, Văn phòng, Học tập, AI/Học máy) từ câu tiếng Việt tự do,
   và một bộ luật ghép cấu hình (rule-based) để phân bổ ngân sách + chọn sản
   phẩm phù hợp từ infor.csv. Đây là phần "trí tuệ" chính, đo được bằng
   accuracy (xem model/metrics.json) — không phụ thuộc API ngoài.
2. API Gemini (Google) để sinh câu trả lời tự nhiên, thân thiện hơn thay vì
   template cứng. Model ML và bộ luật vẫn quyết định NỘI DUNG cấu hình đề
   xuất; Gemini chỉ đảm nhiệm phần VĂN PHONG trả lời. Nếu API lỗi/quá tải,
   app tự động rơi về câu trả lời mẫu có sẵn (template) để không bị gián đoạn.

Cấu trúc file:
    app.py                   -> Logic ứng dụng + giao diện Streamlit
    config.json               -> Cấu hình phân bổ ngân sách, nhãn, prompt, giao diện
    .env                       -> Chứa GEMINI_API_KEY (tuỳ chọn, không bắt buộc để app chạy)
    infor.csv                 -> Dữ liệu giá linh kiện PC và laptop
    training_data.csv         -> Dữ liệu đã dùng để huấn luyện model
    train_model.py             -> Script huấn luyện (chạy TRƯỚC khi chạy app)
    model/intent_pipeline.joblib -> Model đã huấn luyện (do train_model.py tạo ra)
    model/metrics.json         -> Accuracy train/test của model
"""

import json
import os
import random
import re
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ClientError
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

BASE_DIR = Path(__file__).parent
APP_VERSION = "2.0-ML"


# --------------------------------------------------------------------------- #
# 1. Cấu hình
# --------------------------------------------------------------------------- #
class AppConfig:
    def __init__(self, config_path: Path = BASE_DIR / "config.json"):
        load_dotenv(BASE_DIR / ".env")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.app_name: str = data.get("app_name", "Cybong")
        self.app_icon: str = data.get("app_icon", "🖥️")
        self.currency: str = data.get("currency", "VND")
        self.reference_data_file: str = data.get("reference_data_file", "infor.csv")
        self.budget_allocation: dict = data.get("budget_allocation", {})
        self.budget_tolerance: float = data.get("budget_tolerance", 0.1)
        self.ui: dict = data.get("ui", {})

        # Cấu hình Gemini API (dùng để sinh văn phong trả lời, KHÔNG dùng để quyết định cấu hình)
        self.gemini_model: str = data.get("gemini_model", "gemini-3.6-flash")
        self.gemini_max_tokens: int = data.get("gemini_max_tokens", 400)
        self.gemini_temperature: float = data.get("gemini_temperature", 0.7)
        self.gemini_max_retries: int = data.get("gemini_max_retries", 2)
        self.gemini_retry_base_delay: float = data.get("gemini_retry_base_delay_seconds", 1.5)
        self.gemini_system_prompt: str = data.get("gemini_system_prompt", "")
        self.api_key: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------------------------------- #
# 2. Nạp model đã huấn luyện
# --------------------------------------------------------------------------- #
class IntentClassifier:
    """Bọc model TF-IDF + Logistic Regression đã huấn luyện ở train_model.py."""

    def __init__(self, model_path: Path = BASE_DIR / "model" / "intent_pipeline.joblib",
                 metrics_path: Path = BASE_DIR / "model" / "metrics.json"):
        self.pipeline = None
        self.metrics = {}
        if model_path.exists():
            self.pipeline = joblib.load(model_path)
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    def predict(self, text: str) -> tuple[str, float]:
        """Trả về (nhãn dự đoán, độ tin cậy 0-1)."""
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best_idx = proba.argmax()
        return classes[best_idx], float(proba[best_idx])


class GeminiExplainer:
    """Gọi Gemini API để sinh câu trả lời tự nhiên, thân thiện cho Cybong.

    QUAN TRỌNG: lớp này KHÔNG quyết định cấu hình nào được chọn — việc đó do
    IntentClassifier (model ML đã train) + bộ luật ghép cấu hình đảm nhiệm.
    GeminiExplainer chỉ nhận một bản tóm tắt (use_case, cấu hình, giá...) rồi
    viết lại thành câu văn tự nhiên. Nếu API lỗi, hàm trả về None và nơi gọi
    sẽ tự dùng câu trả lời mẫu (template) thay thế — app không bị gián đoạn.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.client = None
        if GENAI_AVAILABLE and config.api_key and config.api_key != "your_api_key_here":
            try:
                self.client = genai.Client(api_key=config.api_key)
            except Exception:
                self.client = None

    @property
    def is_ready(self) -> bool:
        return self.client is not None

    @staticmethod
    def _is_retryable_error(e: Exception) -> bool:
        text = str(e)
        return any(m in text for m in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"])

    def explain(self, situation: dict) -> str | None:
        """situation: dict mô tả ngữ cảnh (use_case, confidence, build, total,
        budget, build_type, is_finalized, action). Trả về câu trả lời (str)
        hoặc None nếu không gọi được API (sẽ fallback sang template)."""
        if not self.is_ready:
            return None

        summary_text = json.dumps(situation, ensure_ascii=False)
        contents = [types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"Thông tin tình huống hiện tại (JSON): {summary_text}\n\n"
                                              f"Hãy viết câu trả lời cho người dùng dựa trên thông tin này.")],
        )]

        for attempt in range(self.config.gemini_max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.config.gemini_system_prompt,
                        temperature=self.config.gemini_temperature,
                        max_output_tokens=self.config.gemini_max_tokens,
                    ),
                )
                text = (response.text or "").strip()
                return text if text else None
            except (ClientError, APIError) as e:
                if self._is_retryable_error(e) and attempt < self.config.gemini_max_retries:
                    time.sleep(self.config.gemini_retry_base_delay * (2 ** attempt))
                    continue
                return None
            except Exception:
                return None
        return None


# --------------------------------------------------------------------------- #
# 3. Dữ liệu linh kiện + trích xuất ngân sách
# --------------------------------------------------------------------------- #
class PartsDatabase:
    def __init__(self, csv_path: Path):
        self.df = pd.DataFrame()
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)

    def filter_by(self, category: str, use_case: str | None = None) -> pd.DataFrame:
        subset = self.df[self.df["category"] == category]
        if use_case:
            matched = subset[subset["use_case"].fillna("").str.contains(use_case, case=False)]
            if not matched.empty:
                return matched.sort_values("price_vnd")
        return subset.sort_values("price_vnd")


BUDGET_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(tri[ệe]u|tr\b|củ)", re.IGNORECASE
)
BUDGET_PATTERN_RAW = re.compile(r"\b(\d{1,3}(?:[.,]\d{3}){2,})\b")  # vd 20.000.000


def extract_budget_vnd(text: str) -> int | None:
    """Trích xuất ngân sách (VNĐ) từ câu tự do, vd '20 triệu', '20tr', '20.000.000'."""
    m = BUDGET_PATTERN.search(text)
    if m:
        number = float(m.group(1).replace(",", "."))
        return int(number * 1_000_000)
    m2 = BUDGET_PATTERN_RAW.search(text)
    if m2:
        number_str = m2.group(1).replace(".", "").replace(",", "")
        return int(number_str)
    return None


LAPTOP_KEYWORDS = ["laptop", "máy xách tay", "máy tính xách tay"]
PC_KEYWORDS = ["pc", "máy tính bàn", "máy bàn", "dàn máy", "build pc"]


def detect_build_type(text: str) -> str:
    low = text.lower()
    if any(k in low for k in LAPTOP_KEYWORDS):
        return "laptop"
    if any(k in low for k in PC_KEYWORDS):
        return "pc"
    return "pc"  # mặc định


CATEGORY_SYNONYMS = {
    "CPU": ["cpu", "chip", "vi xử lý", "bộ xử lý"],
    "Mainboard": ["mainboard", "bo mạch chủ", "main"],
    "RAM": ["ram", "bộ nhớ trong"],
    "VGA": ["vga", "card đồ họa", "card màn hình", "gpu"],
    "SSD": ["ssd", "ổ cứng", "ổ lưu trữ"],
    "PSU": ["psu", "nguồn", "bộ nguồn"],
    "Case": ["case", "vỏ máy", "thùng máy"],
    "Màn hình": ["màn hình", "monitor", "màn"],
    "Laptop": ["laptop"],
}

UPGRADE_WORDS = ["mạnh hơn", "cao hơn", "tốt hơn", "cao cấp", "xịn hơn", "nâng cấp", "khủng hơn"]
DOWNGRADE_WORDS = ["rẻ hơn", "thấp hơn", "tiết kiệm", "giảm xuống", "yếu hơn", "bớt tiền"]
FINALIZE_WORDS = ["chốt", "ok vậy", "lấy cấu hình này", "được rồi", "đồng ý", "vậy đi", "quyết định vậy"]


def detect_swap_category(text: str) -> str | None:
    low = text.lower()
    for category, synonyms in CATEGORY_SYNONYMS.items():
        if any(s in low for s in synonyms):
            return category
    return None


def detect_direction(text: str) -> str:
    low = text.lower()
    if any(w in low for w in UPGRADE_WORDS):
        return "up"
    if any(w in low for w in DOWNGRADE_WORDS):
        return "down"
    return "unknown"


def detect_finalize(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in FINALIZE_WORDS)


# --------------------------------------------------------------------------- #
# 4. Bộ luật ghép cấu hình (recommender)
# --------------------------------------------------------------------------- #
PC_CATEGORIES = ["CPU", "Mainboard", "RAM", "VGA", "SSD", "PSU", "Case", "Màn hình"]


def pick_closest_within_budget(items: pd.DataFrame, sub_budget: float) -> pd.Series:
    """Chọn sản phẩm có giá gần sát sub_budget nhất mà KHÔNG vượt quá; nếu không có
    sản phẩm nào trong ngân sách, chọn sản phẩm rẻ nhất của danh mục đó."""
    within = items[items["price_vnd"] <= sub_budget]
    if not within.empty:
        return within.iloc[-1]  # giá cao nhất trong số các lựa chọn vẫn vừa túi tiền
    return items.iloc[0]  # rẻ nhất trong toàn danh mục nếu không món nào vừa ngân sách


def build_pc(db: PartsDatabase, use_case: str, budget: int, allocation: dict) -> list[dict]:
    build = []
    for category in PC_CATEGORIES:
        pct = allocation.get(category, 1 / len(PC_CATEGORIES))
        sub_budget = budget * pct
        items = db.filter_by(category, use_case)
        if items.empty:
            continue
        chosen = pick_closest_within_budget(items, sub_budget)
        build.append({
            "category": category,
            "name": chosen["name"],
            "price_vnd": int(chosen["price_vnd"]),
            "reason": str(chosen.get("notes", "")),
        })
    return build


def build_laptop(db: PartsDatabase, use_case: str, budget: int) -> list[dict]:
    items = db.filter_by("Laptop", use_case)
    if items.empty:
        items = db.filter_by("Laptop", None)
    chosen = pick_closest_within_budget(items, budget)
    return [{
        "category": "Laptop",
        "name": chosen["name"],
        "price_vnd": int(chosen["price_vnd"]),
        "reason": str(chosen.get("notes", "")),
    }]


def swap_component(db: PartsDatabase, current_build: list[dict], category: str,
                    direction: str, use_case: str) -> list[dict]:
    """Đổi đúng 1 linh kiện trong build hiện tại, giữ nguyên phần còn lại."""
    items = db.filter_by(category, use_case)
    if items.empty:
        return current_build

    items = items.reset_index(drop=True)
    current_item = next((it for it in current_build if it["category"] == category), None)
    if current_item is None:
        return current_build

    # Tìm vị trí món hiện tại trong danh sách đã sắp xếp theo giá
    match_idx = items.index[items["name"] == current_item["name"]]
    idx = int(match_idx[0]) if len(match_idx) else len(items) // 2

    if direction == "up":
        new_idx = min(idx + 1, len(items) - 1)
    elif direction == "down":
        new_idx = max(idx - 1, 0)
    else:
        new_idx = min(idx + 1, len(items) - 1)  # mặc định nâng cấp nếu không rõ hướng

    new_item = items.iloc[new_idx]
    new_build = []
    for it in current_build:
        if it["category"] == category:
            new_build.append({
                "category": category,
                "name": new_item["name"],
                "price_vnd": int(new_item["price_vnd"]),
                "reason": str(new_item.get("notes", "")),
            })
        else:
            new_build.append(it)
    return new_build


# --------------------------------------------------------------------------- #
# 5. Sinh câu trả lời tự nhiên (template-based, không cần LLM)
# --------------------------------------------------------------------------- #
GREETINGS = [
    "Dựa trên nhu cầu của bạn, mình đề xuất cấu hình sau nhé:",
    "Đây là lựa chọn mình thấy hợp lý nhất cho bạn:",
    "Cybong tính toán ra cấu hình này cho bạn:",
]

FOLLOW_UPS = [
    "Bạn có muốn đổi linh kiện nào không, hay chốt luôn cấu hình này?",
    "Ưng ý chưa? Muốn đổi món nào thì cứ nói, không thì mình chốt nhé!",
    "Bạn xem có cần điều chỉnh gì không, hay vậy là chốt được rồi?",
]

SWAP_MESSAGES = [
    "Mình đã đổi {category} cho bạn rồi nè, các phần còn lại giữ nguyên:",
    "Ok, cập nhật lại {category} như sau, phần khác mình giữ nguyên nhé:",
]


def render_message(use_case: str, confidence: float, build_type: str, is_finalized: bool) -> str:
    if is_finalized:
        return "🎉 Chốt cấu hình này cho bạn luôn nha! Cảm ơn bạn đã tin tưởng Cybong."
    base = random.choice(GREETINGS)
    device_word = "laptop" if build_type == "laptop" else "cấu hình PC"
    return f"Mình nhận diện nhu cầu của bạn là **{use_case}** (độ tin cậy {confidence:.0%}). {base.replace('cấu hình', device_word)}"


# --------------------------------------------------------------------------- #
# 6. Hiển thị bảng build
# --------------------------------------------------------------------------- #
def format_vnd(value) -> str:
    try:
        return f"{int(value):,}".replace(",", ".") + "đ"
    except (TypeError, ValueError):
        return "?"


def render_build_table(build: list[dict], total_price: int, budget: int | None,
                        fits_budget: bool, is_finalized: bool):
    if not build:
        return
    df = pd.DataFrame(build)
    df_display = df.rename(columns={
        "category": "Loại", "name": "Sản phẩm", "price_vnd": "Giá", "reason": "Ghi chú"
    })
    df_display["Giá"] = df_display["Giá"].apply(format_vnd)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("💰 Tổng giá", format_vnd(total_price))
    with col2:
        if budget:
            st.metric("🎯 Ngân sách", format_vnd(budget))

    if budget:
        st.success("✅ Cấu hình nằm trong ngân sách.") if fits_budget else \
            st.warning("⚠️ Cấu hình đang lệch so với ngân sách bạn đưa ra.")

    if is_finalized:
        st.info("🎉 Đây là cấu hình đã CHỐT. Bạn vẫn có thể yêu cầu đổi lại bất cứ lúc nào.")


# --------------------------------------------------------------------------- #
# 7. Giao diện Streamlit
# --------------------------------------------------------------------------- #
def init_session_state(config: AppConfig):
    if "messages" not in st.session_state:
        st.session_state.messages = []
        if config.ui.get("welcome_message"):
            st.session_state.messages.append({
                "role": "assistant", "content": config.ui["welcome_message"], "build_data": None,
            })
    if "current_build" not in st.session_state:
        st.session_state.current_build = None
    if "current_use_case" not in st.session_state:
        st.session_state.current_use_case = None
    if "current_budget" not in st.session_state:
        st.session_state.current_budget = None
    if "current_build_type" not in st.session_state:
        st.session_state.current_build_type = None


def render_sidebar(config: AppConfig, db: PartsDatabase, clf: IntentClassifier, explainer: "GeminiExplainer"):
    with st.sidebar:
        st.header(config.ui.get("sidebar_title", "⚙️ Cài đặt"))
        st.caption(f"App version: `{APP_VERSION}`")
        st.divider()

        st.subheader("🧠 Model đã huấn luyện (ML)")
        if clf.is_ready and clf.metrics:
            m = clf.metrics
            col1, col2 = st.columns(2)
            col1.metric("Train accuracy", f"{m.get('train_accuracy', 0):.1%}")
            col2.metric("Test accuracy", f"{m.get('test_accuracy', 0):.1%}")
            st.caption(f"{m.get('n_train_samples', '?')} mẫu train, "
                       f"{m.get('n_test_samples', '?')} mẫu test, "
                       f"{m.get('n_labels', '?')} nhãn")
            with st.expander("Xem chi tiết các nhãn"):
                st.write(", ".join(m.get("labels", [])))
        elif not clf.is_ready:
            st.error("⚠️ Chưa có model. Chạy `python train_model.py` trước.")

        st.divider()
        st.subheader("✨ Gemini API (văn phong trả lời)")
        if explainer.is_ready:
            st.success(f"✅ Đã kết nối — model `{config.gemini_model}`")
            st.caption("Nội dung cấu hình vẫn do model ML + dữ liệu quyết định. "
                       "Gemini chỉ viết lại câu trả lời cho tự nhiên hơn.")
        else:
            st.warning("⚠️ Chưa cấu hình GEMINI_API_KEY (không bắt buộc). "
                       "App vẫn chạy đầy đủ chức năng bằng câu trả lời mẫu có sẵn.")
            st.caption("Muốn bật: điền key vào file `.env` rồi khởi động lại app.")

        st.divider()
        st.subheader("📦 Dữ liệu linh kiện")
        if not db.df.empty:
            st.caption(f"{len(db.df)} sản phẩm, {db.df['category'].nunique()} danh mục")
        st.divider()
        if st.button("🗑️ Bắt đầu tư vấn lại từ đầu", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_build = None
            st.session_state.current_use_case = None
            st.session_state.current_budget = None
            st.session_state.current_build_type = None
            st.rerun()
        st.caption("Made with Streamlit + scikit-learn (model tự huấn luyện)")


def render_chat_history():
    for msg in st.session_state.messages:
        avatar = "🖥️" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            bd = msg.get("build_data")
            if bd and bd.get("build"):
                render_build_table(bd["build"], bd.get("total_price_vnd", 0),
                                    bd.get("budget_vnd"), bd.get("fits_budget", True),
                                    bd.get("is_finalized", False))


def process_user_message(text: str, db: PartsDatabase, clf: IntentClassifier, config: AppConfig,
                          explainer: "GeminiExplainer | None" = None) -> dict:
    """Xử lý 1 tin nhắn người dùng. QUYẾT ĐỊNH cấu hình luôn do model ML (IntentClassifier)
    + bộ luật ghép cấu hình đảm nhiệm (không đổi). Nếu có GeminiExplainer sẵn sàng, câu trả
    lời văn bản sẽ được Gemini viết lại cho tự nhiên hơn; nếu không, dùng template có sẵn."""

    def finalize_message(situation_action: str, template_message: str, extra: dict) -> tuple[str, bool]:
        """Gọi Gemini để viết lại câu trả lời (nếu có), fallback về template nếu lỗi/không có key.
        Trả về (nội_dung, đã_dùng_gemini)."""
        if explainer is None or not explainer.is_ready:
            return template_message, False
        situation = {"action": situation_action, **extra}
        gemini_text = explainer.explain(situation)
        if gemini_text:
            return gemini_text, True
        return template_message, False

    # Trường hợp 1: người dùng xác nhận CHỐT
    if detect_finalize(text) and st.session_state.current_build:
        build = st.session_state.current_build
        total = sum(i["price_vnd"] for i in build)
        template_msg = render_message("", 0, st.session_state.current_build_type or "pc", True)
        message, used_gemini = finalize_message("finalize", template_msg, {
            "build": build, "total_price_vnd": total,
        })
        return {
            "message": message, "build": build, "total_price_vnd": total,
            "budget_vnd": st.session_state.current_budget, "fits_budget": True,
            "is_finalized": True, "needs_clarification": False, "used_gemini": used_gemini,
        }

    # Trường hợp 2: người dùng yêu cầu ĐỔI 1 linh kiện cụ thể, đã có build trước đó
    swap_category = detect_swap_category(text)
    if swap_category and st.session_state.current_build:
        has_category = any(i["category"] == swap_category for i in st.session_state.current_build)
        if has_category:
            direction = detect_direction(text)
            new_build = swap_component(
                db, st.session_state.current_build, swap_category, direction,
                st.session_state.current_use_case,
            )
            total = sum(i["price_vnd"] for i in new_build)
            budget = st.session_state.current_budget
            fits = (total <= budget * (1 + config.budget_tolerance)) if budget else True
            template_msg = random.choice(SWAP_MESSAGES).format(category=swap_category)
            message, used_gemini = finalize_message("swap_component", template_msg, {
                "swapped_category": swap_category, "direction": direction,
                "build": new_build, "total_price_vnd": total, "budget_vnd": budget, "fits_budget": fits,
            })
            st.session_state.current_build = new_build
            return {
                "message": message, "build": new_build, "total_price_vnd": total,
                "budget_vnd": budget, "fits_budget": fits,
                "is_finalized": False, "needs_clarification": False, "used_gemini": used_gemini,
            }

    # Trường hợp 3: yêu cầu build mới (hoặc không nhận diện được là chốt/đổi)
    use_case, confidence = clf.predict(text)  # <-- model ML đã train quyết định nhu cầu
    budget = extract_budget_vnd(text)

    if budget is None:
        return {
            "message": "Mình chưa thấy bạn nói rõ ngân sách. Bạn cho mình biết khoảng bao nhiêu tiền nhé "
                       "(vd: 'khoảng 20 triệu')?",
            "build": [], "total_price_vnd": 0, "budget_vnd": None, "fits_budget": True,
            "is_finalized": False, "needs_clarification": True,
        }

    build_type = detect_build_type(text)
    if build_type == "laptop":
        build = build_laptop(db, use_case, budget)  # <-- chọn theo dữ liệu infor.csv (rule-based)
    else:
        build = build_pc(db, use_case, budget, config.budget_allocation)

    total = sum(i["price_vnd"] for i in build)
    fits = total <= budget * (1 + config.budget_tolerance)

    st.session_state.current_build = build
    st.session_state.current_use_case = use_case
    st.session_state.current_budget = budget
    st.session_state.current_build_type = build_type

    template_msg = render_message(use_case, confidence, build_type, False)
    message, used_gemini = finalize_message("new_build", template_msg, {
        "use_case": use_case, "confidence": round(confidence, 2), "build_type": build_type,
        "build": build, "total_price_vnd": total, "budget_vnd": budget, "fits_budget": fits,
    })
    return {
        "message": message, "build": build, "total_price_vnd": total,
        "budget_vnd": budget, "fits_budget": fits,
        "is_finalized": False, "needs_clarification": False, "used_gemini": used_gemini,
    }


def main():
    config = AppConfig()
    st.set_page_config(page_title=config.app_name, page_icon=config.app_icon, layout="centered")

    db = PartsDatabase(BASE_DIR / config.reference_data_file)
    clf = IntentClassifier()
    explainer = GeminiExplainer(config)

    init_session_state(config)

    st.title(f"{config.app_icon} {config.app_name}")
    st.caption("Nói cho Cybong biết bạn cần máy để làm gì và ngân sách bao nhiêu — "
               "model ML tự huấn luyện sẽ nhận diện nhu cầu và đề xuất cấu hình tối ưu.")

    render_sidebar(config, db, clf, explainer)
    render_chat_history()

    if not clf.is_ready:
        st.error("⚠️ Chưa tìm thấy model đã huấn luyện. Vui lòng chạy `python train_model.py` "
                 "trong thư mục dự án trước, sau đó khởi động lại ứng dụng này.")
        return

    user_text = st.chat_input("Vd: 'Mình cần PC chơi game tầm 20 triệu' hoặc 'đổi VGA sang loại mạnh hơn'...")

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text, "build_data": None})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_text)

        with st.chat_message("assistant", avatar="🖥️"):
            with st.spinner("Cybong đang suy nghĩ..."):
                result = process_user_message(user_text, db, clf, config, explainer)

            display_text = result["message"]
            if result.get("needs_clarification"):
                pass  # câu hỏi làm rõ đã nằm trong message
            elif not result.get("is_finalized") and not result.get("used_gemini"):
                # Nếu Gemini đã viết câu trả lời tự nhiên (bao gồm cả câu hỏi tiếp theo),
                # không cần ghép thêm template follow-up nữa.
                display_text += "\n\n" + random.choice(FOLLOW_UPS)

            st.markdown(display_text)
            if result.get("build"):
                render_build_table(result["build"], result["total_price_vnd"],
                                    result.get("budget_vnd"), result["fits_budget"],
                                    result["is_finalized"])

            st.session_state.messages.append({
                "role": "assistant", "content": display_text, "build_data": result,
            })

        st.rerun()


if __name__ == "__main__":
    main()
