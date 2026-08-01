<a id="top"></a>
# 🗺️ Phase 2. Toán & Thống kê trong AI

> **Mục tiêu**: Hiểu các thuật toán phía sau những mô hình - không cần toàn bộ nhưng phỉa hiểu chúng hoạt động như thế nào?.

---

## 📑 Mục lục

- [2.1 Đại số tuyến tính](#21-dai-so-tuyen-tinh)
  - [Vector](#vector)
  - [Ma trận (Matrices)](#ma-tran)
  - [Phép toán Ma trận trong Học máy](#phep-toan-ma-tran)
  - [Giảm chiều](#giam-chieu)
- [2.2 Giải tích](#22-giai-tich)
- [2.3 Xác suất & Thống kê](#23-xac-suat-thong-ke)
- [2.4 Tối ưu hóa](#24-toi-uu-hoa)
- [2.5 Các định lý](#25-cac-dinh-ly)


---

<a id="21-dai-so-tuyen-tinh"></a>
## 2.1 Đại số tuyến tính

<a id="kieu-du-lieu--bien"></a>
### 🔹 Kiểu dữ liệu & Biến

| Loại | Mô tả | Ép kiểu |
|---|---|---|
| `int` | Số nguyên | `int()` |
| `float` | Số thực | `float()` |
| `str` | Chuỗi | `str()` |
| `bool` | Đúng/Sai (`True`/`False`) | `bool()` |

- `type()`: kiểm tra kiểu dữ liệu của biến.
- `isinstance(object, classinfo)`: kiểm tra một đối tượng có thuộc một kiểu/lớp nhất định hay không.

**Mutable vs Immutable**

- **Mutable**: sau khi khởi tạo có thể chỉnh sửa nội dung bên trong vùng nhớ **mà không đổi địa chỉ `id`**.
  - Ví dụ: `list`, `dict`, `set`, `torch.Tensor`
- **Immutable**: sau khi tạo, giá trị bên trong **không bao giờ thay đổi**. Mọi thao tác đều tạo ra đối tượng mới với vùng nhớ mới.
  - Ví dụ: `int`, `float`, `str`, ...

**Ứng dụng trong AI/MLOps Pipeline**

| Thành phần Pipeline | Nên dùng kiểu | Thư viện / Data Type khuyên dùng | Rủi ro nếu làm sai | Best Practice |
|---|---|---|---|---|
| **Model & Pipeline Config** | Immutable | `dataclasses(frozen=True)`, `NamedTuple`, `Hydra DictConfig` | **Data Leakage / Non-reproducible Runs**: config bị sửa ngầm giữa Train/Val/Test làm mất tính đồng nhất thực nghiệm | Khóa hoàn toàn config sau khi parse. Không truyền `dict` thuần làm config cho hàm xử lý dữ liệu |
| **Category Mapping & Class Labels** | Immutable | `tuple`, `frozenset`, `enum.Enum` | **Index Corruption**: thứ tự class ID lệch giữa Training và Inference | Khai báo hằng số mapping dạng `Tuple[str, ...]` hoặc `Enum` để cố định index |
| **Feature Caching & Hash Keys** | Immutable | `str`, `tuple`, `bytes` | **TypeError (Unhashable)**: `@lru_cache`/Redis key cần input hashable | Ép kiểu tham số sang `tuple`/`str` trước khi truyền vào hàm cache |
| **Dataset In-Memory Buffers** | Mutable | `torch.Tensor`, `np.ndarray`, `pandas.DataFrame` | **OOM**: lạm dụng Immutable (copy liên tục) làm tràn RAM/VRAM | Dùng Mutable nhưng **hạn chế in-place operation** (`inplace=True`, `tensor.add_()`) |
| **Data Augmentation & Preprocessing** | Immutable (Input) / Mutable (Output) | PyTorch `Transforms`, `albumentations` | **Dataset Corruption**: biến đổi trực tiếp trên mảng gốc làm dữ liệu epoch này đè epoch khác | Luôn trả về tensor/array mới (`copy()`) trong `__getitem__` của DataLoader |
| **Experiment Metrics & State Logging** | Mutable | `MLflow`, `W&B`, `list`/`dict` tự quản lý | **Missing Logs / Race Conditions**: dùng hằng số/tuple khiến append metric cồng kềnh | Gom metric vào `dict` tạm, ghi sang tracking system rồi giải phóng bộ nhớ |

---

<a id="chuoi-string"></a>
### 🔹 Chuỗi (String)

- **Cắt chuỗi**: `s[start:stop:step]`
  - `start`: vị trí bắt đầu cắt
  - `stop`: vị trí kết thúc cắt
  - `step`: bước nhảy cắt

| Hàm | Chức năng |
|---|---|
| `split()` | Chia chuỗi theo ký tự (mặc định `" "`) |
| `join()` | Gộp chuỗi theo ký tự chèn vào |
| `strip()` | Loại bỏ khoảng trắng đầu/cuối chuỗi |
| `replace()` | Thay thế chuỗi |
| `find()` | Tìm kiếm ký tự trong chuỗi |
| `startswith(obj)` | Kiểm tra bắt đầu bằng `obj` |
| `endswith(obj)` | Kiểm tra kết thúc bằng `obj` |

```python
# f-string: {} chứa biến cần in ra màn hình
value = f"value is {x:.2f}"

# Chuỗi dài dùng dấu ba nháy
text = """Đây là một
chuỗi nhiều dòng"""
```

---

<a id="collection-tap-hop"></a>
### 🔹 Collection (Tập hợp)

Dùng để lưu trữ nhiều biến trong một chương trình.

#### List (Danh sách)

- Lưu trữ nhiều dữ liệu, **cho phép trùng lặp**, đặt trong dấu `[]`.
- Có thể lồng nhau, có thể thay đổi được (mutable). Thường dùng cho danh sách có thứ tự.

```python
my_list = [1, 2, 3, 4, 5]
```

| Thao tác | Cú pháp |
|---|---|
| Truy cập | `list[0]` (đầu), `list[-1]` (cuối) |
| Cắt lát (Slicing) | `list[1:3]` (từ index 1 đến 2) |
| Thêm | `append(x)`, `insert(index, x)`, `extend(iterable)` |
| Xóa | `pop()` (xóa cuối, trả về giá trị), `remove(x)` (xóa phần tử đầu tiên có giá trị x) |
| Sắp xếp | `sort()` (thay đổi trực tiếp) hoặc `sorted(my_list)` (trả về list mới) |

#### Tuple (Bộ)

- Giống List nhưng **không thể thay đổi** sau khi tạo, đặt trong dấu `()`.
- Dùng lưu trữ dữ liệu cố định, bảo vệ dữ liệu không bị sửa đổi.

```python
my_tuple = (1, 2, 3, 5)
```

| Thao tác | Cú pháp |
|---|---|
| Truy cập | Giống List: `my_tuple[0]`, `my_tuple[:2]` |
| Chỉnh sửa | Không có |
| Tìm kiếm | `count(x)` (đếm số lần xuất hiện), `index(x)` (tìm vị trí của x) |
| Unpack dữ liệu | `x, y = (10, 20)` |

#### Sets (Tập hợp)

- **Không cho phép trùng lặp**, không quan tâm thứ tự, đặt trong dấu `{}`. Có thể thay đổi được.
- Dùng cho các phép toán tập hợp, tìm phần tử duy nhất.

```python
my_set = {1, 2, 3, 5}
# Khởi tạo set rỗng: set() (KHÔNG dùng {} vì đó là dict rỗng)
```

| Thao tác | Cú pháp |
|---|---|
| Truy cập | Chỉ có thể duyệt qua bằng vòng lặp `for` |
| Thêm/Xóa | `add(x)`, `remove(x)` (lỗi nếu không có x), `discard(x)` (không lỗi nếu không có x) |
| Hợp (Union) | `set1 \| set2` hoặc `set1.union(set2)` |
| Giao (Intersection) | `set1 & set2` hoặc `set1.intersection(set2)` |
| Hiệu (Difference) | `set1 - set2` |

#### Dictionary (Từ điển)

- Tổ chức theo `{key: value}`. **Key phải duy nhất**, Value cho phép trùng lặp.
- Có thể thay đổi được. Dùng để lưu trữ thông tin (user, giao dịch, ...).

```python
my_dict = {'name': 'An', 'age': 20}
```

| Thao tác | Cú pháp |
|---|---|
| Truy cập | `my_dict['name']` (lỗi nếu key không tồn tại) hoặc an toàn hơn: `my_dict.get('name', 'Không thấy')` |
| Thêm/Sửa | `my_dict['job'] = 'Dev'` (thêm mới nếu chưa có key, ghi đè nếu đã có) |
| Xóa | `del my_dict['age']` hoặc `my_dict.pop('age')` |
| Lấy Key | `my_dict.keys()` |
| Lấy Value | `my_dict.values()` |
| Lấy cặp (Key, Value) | `my_dict.items()` (dạng Tuple) |

---

<a id="control-flow"></a>
### 🔹 Control Flow

| Từ khóa | Ý nghĩa |
|---|---|
| `if` / `elif` / `else` | Rẽ nhánh điều kiện |
| `for` | Vòng lặp — `for i in list`, `for i, j in dict.items()`, `for i in range(10)` |
| `while` | Vòng lặp theo điều kiện |
| `break` | Dừng vòng lặp khi thỏa điều kiện |
| `continue` | Bỏ qua vòng lặp hiện tại, tiếp tục vòng kế |
| `range()` | Chỉ định vùng dữ liệu |
| `enumerate(iterable, start=0)` | Duyệt phần tử kèm chỉ số |
| `zip()` | Ghép song song nhiều danh sách |

```python
feature_names = ["age", "income", "credit_score"]
feature_importance = [0.15, 0.65, 0.20]

# Ghép song song 2 list
for name, importance in zip(feature_names, feature_importance):
    print(f"Feature '{name}': {importance}")

# Kết quả:
# Feature 'age': 0.15
# Feature 'income': 0.65
# Feature 'credit_score': 0.20
```

[⬆ Về mục lục](#top)

---

<a id="12-function-ham"></a>
## 1.2 Function (Hàm)

Hầu hết chương trình được viết dựa trên hàm — tập hợp các bước thao tác của một tác vụ. Định nghĩa hàm bằng từ khóa `def`.

```python
def test():
    return True  # def là từ khóa | test là tên hàm
```

**Hàm có tham số**

Tham số (parameter) dùng để biểu diễn giá trị được truyền vào hàm.

```python
def get(name):
    print(f"{name}")  # name là tham số truyền vào hàm
```

**Hàm trả về giá trị**

Kết thúc bằng `return`, có thể là số, chữ, danh sách, phép tính, ...

```python
def test():
    return True
```

**Parameter mặc định**

Giá trị mặc định được dùng khi gọi hàm mà không truyền tham số. Khi truyền giá trị khác, nó được xem là Keyword Argument.

```python
def greet(name="Guest"):
    print(name)  # Không truyền giá trị khác thì mặc định name = "Guest"
```

**Scope (Phạm vi biến)**

- **Local**: biến khai báo trong hàm, chỉ dùng được trong hàm đó.
- **Global**: biến có thể dùng ở bất cứ đâu trong chương trình.
- `*args`: gom các tham số vị trí không cố định thành một `tuple`.
- `**kwargs`: gom các tham số vị trí không cố định thành một `dict`.
  - `*args` và `**kwargs` giúp tùy nghi truyền thêm tham số mà không làm vỡ code — mở rộng, sửa code dễ dàng.

**Lambda function**

Hàm vô danh, chỉ chứa một biểu thức duy nhất, không có tên khi khai báo.

```python
lambda arguments: expression

# Hàm thông thường
def add_one(x):
    return x + 1

# Viết tương đương bằng Lambda
add_one_lambda = lambda x: x + 1

print(add_one_lambda(5))  # Kết quả: 6
```

**Đệ quy (Recursion)**

Kỹ thuật một hàm tự gọi lại chính nó, chia nhỏ bài toán lớn thành các bài toán nhỏ hơn.

- **Điều kiện dừng**: để hàm ngừng gọi lại chính nó.
- **Bước đệ quy**: nơi hàm gọi lại chính nó với bài toán quy mô nhỏ hơn.

**Docstrings**

Chuỗi văn bản đặt ở đầu hàm/class/module giải thích chức năng, tham số, kết quả trả về. Giúp IDE hiển thị gợi ý.

```python
def train_epoch(model: object, dataloader: object, optimizer: object) -> float:
    """Thực hiện huấn luyện mô hình qua một epoch dữ liệu.

    Args:
        model (object): Mô hình PyTorch/TensorFlow cần huấn luyện.
        dataloader (object): DataLoader chứa dữ liệu theo từng batch.
        optimizer (object): Thuật toán tối ưu hóa (vd: Adam, SGD).

    Returns:
        float: Giá trị Loss trung bình của cả Epoch.

    Raises:
        ValueError: Nếu DataLoader rỗng.
    """
    if len(dataloader) == 0:
        raise ValueError("DataLoader không chứa dữ liệu!")

    # Giả lập logic tính loss
    total_loss = 0.5
    return total_loss
```

[⬆ Về mục lục](#top)

---

<a id="13-oop-huong-doi-tuong"></a>
## 1.3 OOP (Hướng đối tượng)

**Class (Lớp)**

Đối tượng lớn chứa tất cả thuộc tính chung; mọi object mới đều xuất phát từ class.

```python
class Employee:  # Lớp nhân viên
    pass
```

**Object (Đối tượng)**

Là thực thể cụ thể được tạo ra từ class, mang thuộc tính và hành vi riêng.

```python
class Meo:
    pass

meo_vang = Meo()  # meo_vang là một object
```

**Constructor (`__init__`)**

Hàm khởi tạo tự động chạy ngay khi một đối tượng được tạo ra, dùng gắn giá trị ban đầu cho thuộc tính.

```python
class Meo:
    def __init__(self, ten):
        self.ten = ten  # Gán tên ngay khi tạo mèo

meo = Meo("Miu")  # Lúc này meo.ten sẽ là "Miu"
```

**Instance Method**

Hàm thông thường trong class, hoạt động trực tiếp trên từng Object cụ thể. Luôn có tham số `self` ở đầu, đại diện cho chính Object đó.

```python
class Meo:
    def __init__(self, ten):
        self.ten = ten

    def keu(self):  # Instance Method
        return f"{self.ten} kêu meo meo"
```

**Class Method**

Quản lý vấn đề chung của cả Class chứ không riêng một Object. Dùng decorator `@classmethod`, tham số đầu tiên là `cls`.

```python
class Meo:
    so_luong = 0

    def __init__(self):
        Meo.so_luong += 1

    @classmethod
    def lay_so_luong(cls):  # Class Method
        return f"Tổng số mèo: {cls.so_luong}"
```

**Static Method**

Hàm tiện ích nằm trong Class nhưng độc lập, không dùng dữ liệu của Object (`self`) hay Class (`cls`). Dùng decorator `@staticmethod`.

```python
class CungCu:
    @staticmethod
    def doi_tieng_meo(chuoi):  # Static Method
        return chuoi.replace("người", "meo")
```

### Các tính chất hướng đối tượng

**1. Kế thừa (Inheritance)**

Class con dùng lại thuộc tính/hàm của Class cha, tránh lặp code, dùng từ khóa `super()`.

```python
class DongVat:  # Cha
    def an(self):
        return "Đang ăn..."

class Cho(DongVat):  # Con kế thừa từ Cha
    def sua(self):
        return "Gâu gâu!"

milu = Cho()
print(milu.an())   # Kế thừa từ cha: "Đang ăn..."
print(milu.sua())  # Của riêng con: "Gâu gâu!"
```

**2. Đóng gói (Encapsulation)**

Che giấu dữ liệu bên trong Object để tránh bị sửa đổi tùy tiện từ bên ngoài. Dùng dấu gạch dưới `_` (protected) hoặc `__` (private).

```python
class TaiKhoan:
    def __init__(self, so_du):
        self.__so_du = so_du  # Private, bên ngoài không gọi trực tiếp được

    def xem_so_du(self):  # Hàm gián tiếp để truy cập an toàn
        return self.__so_du

tk = TaiKhoan(1000)
# print(tk.__so_du)  # LỖI! Không truy cập được.
print(tk.xem_so_du())  # Đúng: 1000
```

**3. Đa hình (Polymorphism)**

Cùng một tên hàm nhưng các Class khác nhau thực hiện theo cách khác nhau.

```python
class Cho:
    def keu(self):
        return "Gâu gâu"

class Meo:
    def keu(self):
        return "Meo meo"

# Hàm nhận vào bất kỳ con gì và bắt nó kêu
def lam_cho_keu(dong_vat):
    print(dong_vat.keu())

lam_cho_keu(Cho())  # In ra: Gâu gâu
```

**4. Abstract Class (Lớp trừu tượng)**

"Khung thiết kế" chung. Không thể tạo Object trực tiếp từ Abstract Class; các Class con bắt buộc phải kế thừa và định nghĩa cụ thể các hàm. Dùng thư viện `abc`.

```python
from abc import ABC, abstractmethod

class Hinh(ABC):  # Abstract Class
    @abstractmethod
    def tinh_dien_tich(self):
        pass

class HinhVuong(Hinh):
    def __init__(self, canh):
        self.canh = canh

    def tinh_dien_tich(self):  # Bắt buộc phải viết, nếu không sẽ lỗi
        return self.canh * self.canh
```

### Các phương thức đặc biệt (dunder methods)

**`__str__` vs `__repr__`** — chuyển đổi đối tượng thành String

- `__str__`: hiển thị cho Người dùng cuối (End-user), ưu tiên dễ đọc, thân thiện. Gọi khi dùng `print(obj)` hoặc `str(obj)`.
- `__repr__`: dùng cho Developer/Debugging, yêu cầu chính xác, ngắn gọn, hiển thị rõ kiểu Class và thông số chính để tái tạo Object. Gọi khi dùng `repr(obj)` hoặc gõ tên biến trong console/Jupyter.

```python
class LLMModel:
    def __init__(self, name: str, temperature: float):
        self.name = name
        self.temperature = temperature

    def __str__(self) -> str:
        # Thân thiện với người dùng
        return f"Mô hình LLM: {self.name} (độ sáng tạo: {self.temperature})"

    def __repr__(self) -> str:
        # Định danh kỹ thuật cho Developer
        return f"LLMModel(name='{self.name}', temperature={self.temperature})"

model = LLMModel("GPT-4o", 0.7)
print(str(model))   # Output: Mô hình LLM: GPT-4o (độ sáng tạo: 0.7)
print(repr(model))  # Output: LLMModel(name='GPT-4o', temperature=0.7)
```

**`__len__`, `__getitem__`** — cho phép Class hoạt động như List/Dict (truy cập bằng chỉ số)

Dataset trong PyTorch bắt buộc phải implement 2 hàm này để DataLoader có thể chia batch và iterate qua dữ liệu.

```python
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        # Trả về tổng số lượng sample trong dataset
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        # Định nghĩa cách lấy ra 1 sample tại chỉ số idx
        return {
            "text": self.texts[idx],
            "label": torch.tensor(self.labels[idx])
        }

dataset = TextDataset(["Học Python AI", "MLOps chuyên nghiệp"], [1, 1])
print(len(dataset))  # Output: 2 (thông qua __len__)
print(dataset[0])    # Output: {'text': 'Học Python AI', 'label': tensor(1)} (thông qua __getitem__)
```

### Bảng tổng hợp Decorator với lớp tĩnh

| Decorator | Cú pháp phương thức | Tham số bắt buộc | Bản chất & Mục đích chính | Trường hợp sử dụng trong AI/MLOps |
|---|---|---|---|---|
| **`@property`** | `def my_attribute(self):` | `self` | Biến Method thành Attribute (chỉ đọc hoặc có validation). Truy cập dạng `obj.my_attribute` thay vì `obj.my_attribute()` | Kiểm tra trạng thái GPU/RAM khả dụng; tính dynamic attribute (`is_v2`, `embedding_dim`); bảo vệ biến nội bộ |
| **`@classmethod`** | `def my_factory(cls, ...):` | `cls` (đại diện Class) | Tác động lên cấp Class thay vì Instance. Dùng làm **Factory Methods** khởi tạo Object từ nhiều nguồn | `Model.from_pretrained("bert-base")`, `Config.from_json(json_str)`, `Dataset.from_pandas(df)` |
| **`@staticmethod`** | `def my_utility(...):` | Không nhận `self`/`cls` | Hàm tiện ích độc lập, nhóm vào Class cho gọn scope/namespace | Preprocessing/clean văn bản (`clean_text()`); validate đường dẫn config; helper math độc lập |
| **`@abstractmethod`** | `def predict(self, x):` | `self` (hoặc `cls`) | Đi kèm `ABC` để tạo khung thiết kế; **bắt buộc** Class con phải triển khai | Base Provider chuẩn LangChain/LlamaIndex (`BaseLLM`, `BaseVectorStore`); interface chung cho Custom PyTorch Pipeline |

```python
import json

class PromptTemplate:
    def __init__(self, template: str, version: float):
        self.template = template
        self.version = version

    # 1. @property: Truy cập như thuộc tính, ngăn sửa đổi trực tiếp nếu muốn
    @property
    def is_v2(self) -> bool:
        return self.version >= 2.0

    # 2. @classmethod: Factory method tạo instance từ nguồn khác (Config/JSON)
    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(template=data["template"], version=data["version"])

    # 3. @staticmethod: Utility function xử lý chuỗi đơn thuần
    @staticmethod
    def sanitize_input(user_input: str) -> str:
        return user_input.strip().replace("\n", " ")


# Sử dụng:
# Classmethod (Factory)
json_data = '{"template": "Dịch văn bản: {text}", "version": 2.1}'
prompt_obj = PromptTemplate.from_json(json_data)

# Property
print(prompt_obj.is_v2)  # Output: True (không cần gọi prompt_obj.is_v2())

# Staticmethod
clean_str = PromptTemplate.sanitize_input("   Prompt có newline \n  ")
print(clean_str)  # Output: "Prompt có newline  "
```

[⬆ Về mục lục](#top)

---

<a id="14-pythonic-code"></a>
## 1.4 Pythonic Code

**Comprehension** — tạo list, dict, set bằng cú pháp 1 dòng

```python
# List comprehension: Lấy danh sách độ dài các từ
words = ["python", "ai", "engineering"]
word_lens = [len(w) for w in words]  # [6, 2, 11]

# Dict comprehension: Map tên feature với index (dùng tạo Vocabulary)
vocab = ["<pad>", "<unk>", "cat", "dog"]
token2id = {token: idx for idx, token in enumerate(vocab)}
# {'<pad>': 0, '<unk>': 1, 'cat': 2, 'dog': 3}

# Set comprehension: Lấy tập hợp các class duy nhất từ dự đoán
predictions = ["cat", "dog", "cat", "bird", "dog"]
unique_classes = {p for p in predictions}  # {'cat', 'dog', 'bird'}
```

**Generator Expressions** — xử lý dữ liệu lớn mà không làm tràn RAM. Dùng ngoặc `()`, chỉ tính toán từng phần tử khi được yêu cầu.

```python
# GIẢ SỬ: File chứa 10 triệu dòng text

# BAD: Nạp cả 10 triệu dòng vào RAM dưới dạng List
# memory_heavy = [clean_text(line) for line in open("huge_corpus.txt")]

# GOOD: Dùng Generator Expression, RAM dùng chỉ ~0 MB
memory_efficient = (line.strip().lower() for line in open("huge_corpus.txt"))

# Duyệt từng dòng một để xử lý / đưa vào pipeline
first_sample = next(memory_efficient)
```

| Hàm | Chức năng |
|---|---|
| `map()` | Áp dụng một hàm lên toàn bộ phần tử của chuỗi dữ liệu |
| `filter()` | Lọc các phần tử thỏa mãn điều kiện trả về `True` |
| `reduce()` | Tích lũy hàm 2 tham số lên lần lượt các phần tử (trái sang phải) để rút gọn thành 1 giá trị |
| `any()` | Trả về `True` nếu có ít nhất 1 phần tử là `True` |
| `all()` | Trả về `True` nếu tất cả phần tử là `True` |
| `sorted()` | Sắp xếp theo chỉ định |

**Unpacking**: `a, b, *rest = lst` — trích xuất các phần tử trong danh sách thành các biến riêng lẻ.

```python
# Lấy phần tử đầu, phần tử cuối và gom tất cả phần còn lại
batch_data = ["sample_0", "sample_1", "sample_2", "sample_3", "target"]

first_sample, *middle_samples, label = batch_data

print(first_sample)      # 'sample_0'
print(middle_samples)    # ['sample_1', 'sample_2', 'sample_3']
print(label)              # 'target'
```

**Module `collections`**

- `Counter`: đếm tần suất phần tử.
- `defaultdict`: tự động khởi tạo giá trị mặc định cho key chưa tồn tại khi truy cập/gán.
- `deque`: cấu trúc dữ liệu hàng chờ (queue) hiệu năng cao.

```python
from collections import defaultdict

# Khai báo value mặc định là một list
category_to_files = defaultdict(list)

# Không cần kiểm tra 'if category in dict:' nữa
category_to_files["images"].append("img1.png")
category_to_files["images"].append("img2.png")
category_to_files["labels"].append("label1.txt")

print(dict(category_to_files))
# {'images': ['img1.png', 'img2.png'], 'labels': ['label1.txt']}
```

[⬆ Về mục lục](#top)

---

<a id="15-file-handling"></a>
## 1.5 File Handling

**Đọc file**

```python
with open("data.txt", "r") as file:
    content = file.read()
```

- `data.txt`: tên file muốn đọc
- `as file`: biến lưu để thao tác trong chương trình
- `"r"`: chế độ đọc (read)
- `content = file.read()`: đọc toàn bộ nội dung file

**Đọc từng dòng trong file**

```python
with open("data.txt") as file:
    for line in file:
        print(line.strip())
```

Duyệt từng dòng trong file và xóa khoảng trắng đầu/cuối mỗi dòng.

**Ghi file**

```python
with open("output.txt", "w") as file:
    file.write("Hello")
```

- `output.txt`: tên file muốn ghi
- `"w"`: chế độ ghi (write) — sẽ **ghi đè** nội dung cũ
- `file.write()`: ghi nội dung mới vào file

**Append (Ghi thêm)**

```python
with open("log.txt", "a") as file:
    file.write("New log\n")
```

- `"a"`: chế độ ghi thêm (append), không ghi đè nội dung cũ
- Lưu ý thêm `\n` để xuống dòng

**Thao tác với file CSV**

```python
import csv

# Đọc
with open("students.csv") as file:
    reader = csv.DictReader(file)  # Xác định đọc file dạng dict
    for row in reader:
        print(row)

# Ghi
with open("students.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "name", "score"])  # Thêm dòng tiêu đề (header)
    writer.writerow([1, "An", 8.5])            # Thêm dòng dữ liệu mới
```

**Thư viện xử lý đường dẫn & file**

| Thư viện | Đặc điểm |
|---|---|
| `os` | Thao tác hệ thống kiểu cũ, làm việc với đường dẫn dạng chuỗi (string) |
| `pathlib.Path` | Thao tác đường dẫn theo hướng đối tượng (**khuyên dùng**) |
| `glob` | Tìm kiếm file theo pattern (ví dụ `*.jpg`) |

[⬆ Về mục lục](#top)

---

<a id="16-sua-loi--debugging"></a>
## 1.6 Sửa lỗi & Debugging

Trong quá trình xây dựng chương trình sẽ gặp các lỗi như kiểu dữ liệu không đúng, file không tồn tại, ... Exception ra đời nhằm giám sát chất lượng dữ liệu, xây dựng hệ thống có thể tự phục hồi và phát hiện lỗi tự động, giảm tải nhân công và thời gian.

**Try / Except**

Đóng gói đoạn code có nguy cơ lỗi. Nếu lỗi xảy ra, chương trình không sập mà nhảy vào khối `except` để xử lý.

```python
try:
    # Khối lệnh có nguy cơ xảy ra lỗi (ví dụ: ép kiểu dữ liệu lỗi)
    age = int("không phải số")
except ValueError:
    # Khối lệnh xử lý khi bắt đúng loại lỗi ValueError
    print("Lỗi: Vui lòng nhập vào một số nguyên hợp lệ!")
```

**Multiple Except**

Bắt riêng biệt từng loại lỗi từ trên xuống dưới. Thêm `Exception` ở cuối làm lưới bảo hiểm cho các lỗi chưa lường trước.

```python
try:
    # Khối lệnh có thể sinh ra nhiều loại lỗi khác nhau
    with open("data.csv", "r") as f:
        content = f.read()
    result = 10 / len(content)
except FileNotFoundError:
    # Xử lý riêng khi không tìm thấy file
    print("Lỗi: Không tìm thấy file dữ liệu đầu vào.")
except ZeroDivisionError:
    # Xử lý riêng khi file rỗng dẫn đến phép chia cho 0
    print("Lỗi: File rỗng, không thể thực hiện phép chia.")
except Exception as e:
    # Bắt tất cả các loại lỗi còn lại (lưới bọc an toàn)
    print(f"Đã xảy ra lỗi hệ thống không xác định: {e}")
```

**Else**

Chỉ chạy khi khối `try` thực thi thành công hoàn toàn, không xảy ra bất kỳ lỗi nào.

```python
try:
    # Thực hiện một tác vụ đọc dữ liệu từ API hoặc File
    score = int("95")
except ValueError:
    print("Lỗi định dạng số.")
else:
    # Khối này CHỈ chạy khi khối try hoàn thành trơn tru và KHÔNG có exception
    print(f"Xử lý dữ liệu thành công. Điểm số của user là: {score}")
```

**Finally**

Luôn chạy dù khối `try` có lỗi hay không, có bị `return` thoát hàm hay không. Dùng để dọn dẹp hệ thống (đóng file, ngắt kết nối DB).

```python
try:
    file = open("report.txt", "w")
    file.write("Data Engineering Report")
    # Giả sử có lỗi xảy ra ở đây...
    result = 1 / 0
except ZeroDivisionError:
    print("Lỗi chia cho 0.")
finally:
    # Khối này LUÔN LUÔN được thực thi dù 'try' có lỗi hay không.
    # Thường dùng để giải phóng tài nguyên (đóng file, đóng kết nối DB).
    file.close()
    print("Đã đóng file an toàn và giải phóng bộ nhớ.")
```

**Raise**

Chủ động ném (trigger) một lỗi khi dữ liệu vi phạm logic nghiệp vụ (dù code không sai cú pháp).

```python
def check_replicate_count(count):
    if count < 0:
        # Chủ động ném ra một ngoại lệ khi dữ liệu vi phạm logic nghiệp vụ
        raise ValueError("Số lượng bản sao (replicate) không được là số âm!")
    return True

try:
    check_replicate_count(-5)
except ValueError as e:
    print(f"Hệ thống phát hiện dữ liệu bất thường: {e}")
```

**Custom Exception**

Tự tạo một loại lỗi mới bằng cách kế thừa lớp `Exception`, giúp định nghĩa tường minh tên lỗi theo đúng nghiệp vụ dự án.

```python
# Tự định nghĩa một Class kế thừa từ lớp Exception gốc của Python
class EmptyPipelineError(Exception):
    """Ngoại lệ được ném ra khi Data Pipeline chạy nhưng không nhận được dữ liệu đầu vào."""
    pass

def run_pipeline(records):
    if len(records) == 0:
        raise EmptyPipelineError("Pipeline dừng hoạt động: Không có bản ghi nào để xử lý!")

try:
    run_pipeline([])
except EmptyPipelineError as e:
    print(f"Cảnh báo hệ thống: {e}")
```

**Logging**

Thay thế hoàn toàn cho `print()` trong code nghiệp vụ. Cung cấp các mức độ `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

- `breakpoint()` / `pdb`: dừng chương trình lại để xem biến trực tiếp từ Terminal.

```python
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Bắt đầu huấn luyện mô hình...")

# Đặt điểm dừng Debug
x = 10
# breakpoint()  # Chương trình sẽ dừng lại ở đây để tương tác CLI
```

[⬆ Về mục lục](#top)

---

<a id="17-hieu-suat-va-bo-nho"></a>
## 1.7 Hiệu suất và Bộ nhớ

Chiến lược giúp tối ưu hiệu năng tính toán và quản lý dung lượng RAM/VRAM.

**Generators & `yield`**

Cho phép hàm trả về dữ liệu từng phần dưới dạng Stream mà không nạp toàn bộ vào RAM cùng lúc. Rất quan trọng khi xử lý streaming token từ LLM.

```python
def stream_llm_response():
    tokens = ["Xin", "chào", "bạn", "tôi", "là", "AI"]
    for token in tokens:
        yield token

for chunk in stream_llm_response():
    print(chunk, end=" ")
```

**Thư viện `itertools`** — tạo các con lặp hiệu năng cao (tốc độ C) phục vụ tính toán tổ hợp

| Hàm | Chức năng |
|---|---|
| `itertools.chain()` | Nối nhiều iterable |
| `itertools.islice()` | Cắt lát generator |
| `itertools.product()` | Tích Cartesian (dùng làm Grid Search hyperparameter đơn giản) |

**Benchmarking**

| Công cụ | Chức năng |
|---|---|
| `timeit` | Đo thời gian thực thi của một đoạn code nhỏ |
| `cProfile` | Phân tích hiệu năng (profiling), xem hàm nào tốn nhiều thời gian nhất |

**Shallow vs Deep Copy**

| Loại | Mô tả |
|---|---|
| `copy.copy()` (Shallow) | Tạo đối tượng mới nhưng giữ nguyên tham chiếu đến các đối tượng con mutable bên trong |
| `copy.deepcopy()` (Deep) | Sao chép toàn bộ đối tượng và tất cả đối tượng con một cách độc lập hoàn toàn |

**Vectorization (Vector hóa)**

Thay thế vòng lặp Python thuần (`for`) bằng thao tác vector hóa của NumPy/PyTorch. Các phép toán này viết bằng C/CUDA, nhanh hơn từ **10x đến 100x**.

[⬆ Về mục lục](#top)

---

<a id="18-numpy-nen-tang-tinh-toan-mang"></a>
## 1.8 NumPy (Nền tảng tính toán mảng)

Thư viện lõi cho đại số tuyến tính và xử lý mảng đa chiều trong Python.

**Tạo mảng & Thuộc tính cơ bản**

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # Dimensions: (2, 3)
print(arr.ndim)   # Số chiều: 2
print(arr.dtype)  # Kiểu dữ liệu: int64

zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
eye = np.eye(3)   # Ma trận đơn vị
```

**Thay đổi hình dạng (Reshaping) & Nối mảng (Stacking)**

| Hàm | Chức năng |
|---|---|
| `reshape()` | Thay đổi kích thước mảng mà không đổi dữ liệu |
| `flatten()` | Chuyển mảng thành 1D (tạo ra copy mới) |
| `ravel()` | Chuyển mảng thành 1D (trả về view nếu có thể, tiết kiệm bộ nhớ) |
| `np.vstack()`, `np.hstack()`, `np.stack()` | Ghép các mảng theo chiều dọc/ngang/chiều mới |

**Indexing nâng cao & Broadcasting**

- **Boolean Indexing**: lọc dữ liệu theo điều kiện logic.
- `np.where(condition, x, y)`: chọn `x` nếu thỏa điều kiện, ngược lại chọn `y`.
- **Broadcasting Rules**: cho phép thực hiện phép toán giữa các mảng khác kích thước nếu thỏa mãn điều kiện tương thích về shape (ví dụ: cộng ma trận với một vector).

```python
data = np.array([10, 25, 30, 5])
filtered = data[data > 15]           # [25, 30]
replaced = np.where(data > 15, 1, 0)  # [0, 1, 1, 0]
```

**Phép toán Ma trận & Aggregations**

| Hàm | Chức năng |
|---|---|
| `np.dot(A, B)` hoặc `A @ B` | Phép nhân ma trận |
| `np.linalg.inv()` | Tìm ma trận nghịch đảo |
| `np.linalg.eig()` | Tìm trị riêng và vector riêng |
| `np.mean()`, `np.sum()`, `np.std()`, `np.argmax()` | Aggregations, kèm tham số `axis` (`axis=0` cho cột, `axis=1` cho dòng) |

```python
matrix = np.array([[1, 2], [3, 4]])

# Nhân ma trận
result = matrix @ matrix

# Tìm chỉ số max theo dòng
max_idx = np.argmax(matrix, axis=1)
```

[⬆ Về mục lục](#top)

---

<a id="19-pandas-xu-ly-du-lieu-bang"></a>
## 1.9 Pandas (Xử lý và phân tích dữ liệu bảng)

Thư viện chuẩn để thao tác với Tabular Data (`DataFrame` & `Series`).

**Khởi tạo & Khảo sát dữ liệu**

```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head(5))     # 5 dòng đầu
print(df.info())      # Thông tin kiểu dữ liệu và giá trị Null
print(df.describe())  # Thống kê mô tả (Mean, Std, Min, Max, Quantiles)
print(df.shape)       # Kích thước (rows, cols)
```

**Truy xuất & Lọc dữ liệu: `loc` vs `iloc`**

| Thuộc tính | Cách truy xuất |
|---|---|
| `loc` | Bằng Nhãn (Label/Name) của dòng và cột |
| `iloc` | Bằng Chỉ số nguyên (Integer Index) |

```python
# Lọc bằng loc (Boolean Mask)
high_income = df.loc[df['income'] > 50000, ['name', 'income']]

# Lọc bằng iloc (hàng từ 0 đến 5, cột từ 0 đến 2)
subset = df.iloc[0:5, 0:2]
```

**Xử lý Missing Data**

| Hàm | Chức năng |
|---|---|
| `isna()` / `isnull()` | Kiểm tra giá trị thiếu |
| `dropna()` | Bỏ bớt các dòng/cột chứa giá trị Null |
| `fillna()` | Điền giá trị thay thế (Mean, Median, Mode hoặc gán cố định) |

**Gom nhóm & Biến đổi**

```python
# GroupBy & Aggregation
df.groupby('category').agg({'price': 'mean', 'id': 'count'})
```

- **Merge (Join)**: kết nối các DataFrame tương tự SQL (`inner`, `left`, `right`, `outer`).
- **Concat**: nối theo chiều dọc hoặc chiều ngang.
- **Pivot Table & Melt**: xoay và tái cấu trúc dạng bảng (Wide vs Long format).
- **Datetime Parsing**: `pd.to_datetime()` — chuyển cột chuỗi sang kiểu thời gian để trích xuất `year`, `month`, `day`.

[⬆ Về mục lục](#top)

---

<a id="110-code-quality--cau-truc-du-an"></a>
## 1.10 Code Quality & Cấu trúc Dự án

Biến các đoạn code dạng script thành một dự án phần mềm chuẩn mực, sẵn sàng bảo trì.

**Môi trường ảo & Quản lý phụ thuộc**

Tạo môi trường cách ly dependency để tránh xung đột phiên bản:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# Hoặc dùng conda: conda create -n ai_env python=3.10
```

Lưu và cài đặt danh sách thư viện:

```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

**Cấu trúc Module & Package**

- Thêm file `__init__.py` vào thư mục để biến thư mục đó thành một Python Package.
- Tách code thành các submodule rõ ràng (`data/`, `models/`, `utils/`).

**Type Hints & Dataclasses**

- **Type Hints**: khai báo kiểu dữ liệu cho tham số và giá trị trả về, giúp IDE hiển thị gợi ý và bắt lỗi tĩnh.
- **`dataclasses`**: tạo lớp chứa dữ liệu thuần túy nhanh chóng mà không cần viết boilerplate `__init__`.

```python
from dataclasses import dataclass

@dataclass(frozen=True)  # frozen=True biến instance thành Immutable
class ModelConfig:
    model_name: str
    learning_rate: float = 0.001
    batch_size: int = 32

config = ModelConfig(model_name="ResNet18")
```

**Testing, Linting & Formatting**

| Công cụ | Chức năng |
|---|---|
| `pytest` | Unit Testing — viết test tự động cho các hàm quan trọng |
| `ruff` / `flake8` | Linters — kiểm tra lỗi cú pháp và chuẩn PEP8 |
| `black` | Formatter — tự động format code theo chuẩn thống nhất |

[⬆ Về mục lục](#top)

---

<a id="111-python-cho-ai-workflows"></a>
## 1.11 Python cho AI Workflows

Các công cụ trợ lực cho chu trình nghiên cứu, phát triển và thử nghiệm mô hình AI.

**Môi trường Interactive & Tiến trình**

- **Jupyter Notebooks / Google Colab**: môi trường thử nghiệm nhanh, hỗ trợ GPU/TPU miễn phí. Sử dụng Magic commands như `%timeit`, `%matplotlib inline`.
- **`tqdm`**: tạo thanh tiến trình (progress bar) cho các vòng lặp huấn luyện dài.

```python
from tqdm import tqdm
import time

for epoch in tqdm(range(100), desc="Training Model"):
    time.sleep(0.01)
```

**Quản lý tham số CLI & Config**

| Công cụ | Chức năng |
|---|---|
| `argparse` | Truyền tham số trực tiếp từ dòng lệnh khi thực thi script |
| YAML / Hydra Config | Quản lý cấu hình mô hình tập trung từ file `.yaml` ngoại vi |
| `python-dotenv` (`.env`) | Lưu trữ an toàn các hằng số nhạy cảm như API Keys, không lo dính vào Git |

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Nạp các biến từ file .env
api_key = os.getenv("OPENAI_API_KEY")
```

**Tái lập thực nghiệm (Reproducibility) & Lưu trữ**

- **Seeding**: cố định seed cho tất cả thư viện sinh số ngẫu nhiên để đảm bảo thực nghiệm tái lập 100%.

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- **Lưu trữ mô hình**: `joblib.dump()`, `torch.save()`.

[⬆ Về mục lục](#top)

---

<a id="112-async-python"></a>
## 1.12 Async Python (Lập trình bất đồng bộ cho AI APIs)

Kiến thức bắt buộc để xây dựng ứng dụng tương tác với LLM APIs (OpenAI, Claude, vLLM) nhằm tối ưu latency và throughput.

**Cơ chế `async` / `await` & Event Loop**

Lập trình bất đồng bộ cho phép chương trình thực hiện công việc khác trong thời gian chờ phản hồi I/O (ví dụ: chờ API trả về token) thay vì chặn toàn bộ hệ thống (blocking).

```python
import asyncio

async def fetch_llm_response(prompt: str) -> str:
    print(f"Đang gửi prompt: {prompt}")
    await asyncio.sleep(2)  # Giả lập lệnh chờ I/O từ Network
    return f"Response cho: {prompt}"

async def main():
    result = await fetch_llm_response("Hello AI")
    print(result)

# Chạy Event Loop
asyncio.run(main())
```

**Thực thi đồng thời với `asyncio.gather()`**

Gửi nhiều request API cùng một lúc thay vì gọi tuần tự từng request.

```python
async def process_batch_prompts(prompts: list[str]):
    # Gửi song song tất cả các prompts sang API
    tasks = [fetch_llm_response(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

# Tốc độ phản hồi sẽ bằng thời gian request chậm nhất,
# thay vì tổng thời gian của tất cả các request cộng lại
```

**Async HTTP Clients (`httpx`, `aiohttp`)**

Sử dụng các thư viện hỗ trợ Async Native để làm việc với REST APIs hoặc Server-Sent Events (SSE) khi streaming LLM responses.

| Thư viện | Đặc điểm | Trường hợp sử dụng |
|---|---|---|
| `requests` | Synchronous (Blocking) | Script đơn giản, không yêu cầu concurrency cao |
| `httpx` | Hỗ trợ cả Sync & Async | Chuẩn Production mới, được OpenAI SDK và FastAPI tích hợp sẵn |
| `aiohttp` | Async thuần túy | Hệ thống Async lớn, giao tiếp WebSocket hoặc Streaming pipeline nặng |

**Tại sao Streaming LLM lại cần Async?**

Các mô hình Generative AI sinh dữ liệu dưới dạng chuỗi token theo thời gian. Sử dụng Async kết hợp Async Generators (`async for ... in ...`) cho phép client nhận và render từng token lên giao diện ngay khi vừa được tạo ra, mà không làm đứng (freeze) luồng xử lý chính của ứng dụng.

[⬆ Về mục lục](#top)