# 🖥️ Cybong (bản Hybrid: ML đã train + API Gemini) — Trợ Lý Tư Vấn PC & Laptop

Phiên bản này kết hợp **cả hai**, đúng yêu cầu đề bài:
1. Một **model Machine Learning tự huấn luyện** (có accuracy đo được) để quyết
   định cấu hình đề xuất — đây là phần "trí tuệ" chính, không phụ thuộc API.
2. **API Gemini (Google)** để viết lại câu trả lời cho tự nhiên, thân thiện
   hơn — chỉ phụ trách VĂN PHONG, không quyết định linh kiện/giá nào được chọn.

⚠️ **API key KHÔNG bắt buộc để app chạy.** Nếu không có key hoặc Gemini bị lỗi
(quá tải, hết quota...), app tự động dùng câu trả lời mẫu có sẵn — chức năng
tư vấn cấu hình vẫn đầy đủ 100%, chỉ là văn phong kém tự nhiên hơn một chút.

## Cách hoạt động (kiến trúc)
1. **`train_model.py`** huấn luyện một model phân loại văn bản:
   - Input: câu tiếng Việt tự do (vd: "mình cần PC chơi game tầm 20 triệu")
   - Output: 1 trong 7 nhãn nhu cầu sử dụng (Gaming, Đồ họa, Dựng phim, Lập
     trình, Văn phòng, Học tập, AI/Học máy)
   - Thuật toán: TF-IDF (vector hoá văn bản) + Logistic Regression
   - Dữ liệu: `training_data.csv` (245 câu, 7 nhãn, chia train/test 80/20)
   - Kết quả: **Train accuracy ~100%, Test accuracy ~90%** (xem `model/metrics.json`)
2. **`app.py`** nạp model đã huấn luyện, dùng nó để dự đoán nhu cầu từ câu
   nhập, trích xuất ngân sách bằng regex, rồi ghép cấu hình từ `infor.csv`
   theo bảng phân bổ ngân sách trong `config.json` (CPU 18%, VGA 30%, RAM
   10%... — có thể chỉnh). **Toàn bộ bước này KHÔNG dùng API.**
3. Sau khi đã chốt xong linh kiện nào, app gửi một bản TÓM TẮT (không phải
   toàn quyền quyết định) sang Gemini API, nhờ viết lại thành câu trả lời tự
   nhiên. Có retry tự động (2 lần) khi gặp lỗi quá tải tạm thời; nếu vẫn lỗi,
   dùng câu trả lời mẫu.
4. Khi người dùng yêu cầu **đổi linh kiện**, app dùng luật nhận diện từ khoá
   (CPU/VGA/RAM...) + hướng (mạnh hơn/rẻ hơn) để chỉ đổi đúng món đó, rồi cũng
   nhờ Gemini viết lại câu xác nhận.

## Cấu hình API key (tuỳ chọn)
Mở file `.env`, điền key lấy tại https://aistudio.google.com/apikey:
```
GEMINI_API_KEY=AIzaSy...
```
Không điền hoặc để nguyên `your_api_key_here` thì app vẫn chạy bình thường.

## Cấu trúc dự án
```
app.py                      # Giao diện + logic chính (không gọi API)
train_model.py                # Script huấn luyện model (CHẠY TRƯỚC app.py)
generate_training_data.py     # Script sinh dữ liệu huấn luyện (đã chạy sẵn)
training_data.csv             # 245 câu dữ liệu huấn luyện đã gắn nhãn
config.json                  # Phân bổ ngân sách, giao diện
infor.csv                     # Dữ liệu giá linh kiện PC và laptop
requirements.txt
model/
  intent_pipeline.joblib      # Model đã huấn luyện (TF-IDF + LogisticRegression)
  metrics.json                 # Accuracy train/test, classification report
```

## Cài đặt
```bash
pip install -r requirements.txt
```

## Bước 1: Huấn luyện model (chỉ cần chạy 1 lần, hoặc khi sửa training_data.csv)
```bash
python train_model.py
```
Lệnh này sẽ in ra accuracy trên cả tập train và test, rồi lưu model vào
`model/intent_pipeline.joblib`.

## Bước 2: Chạy ứng dụng
```bash
streamlit run app.py
```
(Không cần `.env`, không cần API key — chạy offline hoàn toàn sau khi đã có model.)

## Cách dùng
1. Nhắn: *"Mình cần PC chơi game tầm 20 triệu"* hoặc *"laptop lập trình 16GB, dưới 20 triệu"*
2. Cybong dự đoán nhu cầu (hiện kèm % độ tin cậy) và đề xuất cấu hình
3. Nhắn tiếp: *"đổi VGA sang loại mạnh hơn"* / *"đổi RAM rẻ hơn"* để điều chỉnh
4. Nhắn *"chốt vậy đi"* để chốt cấu hình cuối cùng

## Cải thiện thêm (nếu muốn)
- Thêm câu mẫu vào `training_data.csv` rồi chạy lại `train_model.py` để tăng
  độ chính xác hoặc thêm nhãn mới
- Chỉnh tỉ lệ phân bổ ngân sách trong `config.json` (budget_allocation)
- Thêm sản phẩm mới vào `infor.csv`
