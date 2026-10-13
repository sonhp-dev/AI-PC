"""
Sinh dữ liệu huấn luyện cho model phân loại nhu cầu sử dụng (use_case) từ câu
tiếng Việt tự do mà người dùng nhập. Dữ liệu được ghép từ nhiều mẫu câu +
nhiều hoạt động/ngữ cảnh khác nhau để tăng độ đa dạng, tránh model chỉ học
thuộc từ khoá đơn lẻ.
"""
import csv
import random

random.seed(42)

LABELS = {
    "Gaming": {
        "activities": [
            "chơi game", "chơi liên minh huyền thoại", "chơi valorant", "chơi pubg",
            "chơi liên quân mobile", "cày game liên tục", "chơi game AAA đồ hoạ nặng",
            "chơi free fire", "chơi đột kích", "chơi cs go", "chiến game fps",
            "livestream chơi game", "chơi game 4K", "test benchmark game",
        ],
        "subjects": ["mình", "tôi", "em trai mình", "con mình", "bạn mình", ""],
    },
    "Đồ họa": {
        "activities": [
            "thiết kế đồ họa", "vẽ kỹ thuật số", "chỉnh sửa ảnh chuyên nghiệp",
            "dùng photoshop illustrator", "thiết kế banner quảng cáo", "làm đồ họa 3D",
            "render hình ảnh sản phẩm", "thiết kế logo nhận diện thương hiệu",
            "dựng mô hình 3D render", "làm việc với file psd ai nặng",
        ],
        "subjects": ["mình", "tôi", "designer", ""],
    },
    "Dựng phim": {
        "activities": [
            "dựng phim", "edit video youtube", "làm video premiere",
            "chỉnh sửa video after effects", "biên tập phim ngắn", "render video 4K",
            "làm clip tiktok chuyên nghiệp", "dựng phim quảng cáo",
            "xử lý hậu kỳ video", "cắt ghép video dài",
        ],
        "subjects": ["mình", "tôi", "editor", "nhà sản xuất nội dung", ""],
    },
    "Lập trình": {
        "activities": [
            "lập trình phần mềm", "code web", "học lập trình", "viết ứng dụng mobile",
            "chạy docker và các container", "lập trình backend", "làm dự án machine learning nhẹ",
            "compile code nhiều dự án", "dev full-stack", "chạy nhiều IDE cùng lúc",
            "lập trình game", "code python và java",
        ],
        "subjects": ["mình", "tôi", "sinh viên CNTT", "lập trình viên", ""],
    },
    "Văn phòng": {
        "activities": [
            "làm việc văn phòng", "soạn thảo văn bản word excel", "làm báo cáo powerpoint",
            "gửi email họp online", "nhập liệu kế toán", "làm việc hành chính văn phòng",
            "quản lý giấy tờ sổ sách", "họp zoom thường xuyên",
        ],
        "subjects": ["mình", "tôi", "nhân viên văn phòng", "kế toán", ""],
    },
    "Học tập": {
        "activities": [
            "học tập online", "học trực tuyến mùa dịch", "làm bài tập về nhà",
            "nghe giảng qua zoom", "ôn thi đại học", "làm slide thuyết trình",
            "tra cứu tài liệu học tập", "học sinh cấp 3 dùng học bài",
        ],
        "subjects": ["mình", "tôi", "con mình", "em mình", "sinh viên", ""],
    },
    "AI/Học máy": {
        "activities": [
            "học máy machine learning", "deep learning", "train model AI",
            "xử lý dữ liệu lớn data science", "huấn luyện mạng neural network",
            "nghiên cứu AI", "chạy mô hình học sâu", "phân tích dữ liệu big data",
        ],
        "subjects": ["mình", "tôi", "nhà nghiên cứu", "sinh viên khoa học dữ liệu", ""],
    },
}

TEMPLATES = [
    "{subj} cần {device} để {activity}",
    "Tư vấn giúp {subj} {device} dùng {activity}, ngân sách khoảng {budget}",
    "{device} nào phù hợp để {activity} vậy",
    "{subj} muốn mua {device} chủ yếu {activity}",
    "Cho {subj} hỏi {device} tầm {budget} để {activity} được không",
    "{subj} đang tìm {device} phục vụ nhu cầu {activity}",
    "Mình cần tư vấn {device} để {activity}, budget {budget}",
    "{device} giá {budget} có đủ để {activity} không",
    "{subj} sắp mua {device} mới, mục đích chính là {activity}",
    "Nhờ tư vấn {device} cấu hình để {activity} mượt",
]

DEVICES = ["PC", "máy tính bàn", "laptop", "máy tính", "dàn máy", "bộ PC"]
BUDGETS = ["10 triệu", "15 triệu", "20 triệu", "25 triệu", "30 triệu", "12tr", "18tr", "35 triệu", "8 triệu", "50 triệu"]


def generate_rows(per_label: int = 35):
    rows = []
    for label, cfg in LABELS.items():
        combos = []
        for activity in cfg["activities"]:
            for subj in cfg["subjects"]:
                for template in TEMPLATES:
                    combos.append((template, subj, activity))
        random.shuffle(combos)

        seen_texts = set()
        count = 0
        for template, subj, activity in combos:
            if count >= per_label:
                break
            device = random.choice(DEVICES)
            budget = random.choice(BUDGETS)
            text = template.format(subj=subj, device=device, activity=activity, budget=budget)
            text = " ".join(text.split())  # dọn khoảng trắng thừa khi subj rỗng
            text = text[0].upper() + text[1:] if text else text
            if text in seen_texts:
                continue
            seen_texts.add(text)
            rows.append({"text": text, "label": label})
            count += 1
    return rows


if __name__ == "__main__":
    rows = generate_rows(per_label=35)
    random.shuffle(rows)

    with open("training_data.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã sinh {len(rows)} dòng dữ liệu huấn luyện vào training_data.csv")
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    for label, c in counts.items():
        print(f"  {label}: {c} câu")
