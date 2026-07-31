# 🗺️ Phase 1. Nền tảng lập trình 
> **Mục tiêu**: Code sạch, chuẩn kỹ thuật, Có thể mở rộng
---
## 1.1 Nền tảng Python 

### **Kiểu dữ liệu & Biến**

- `integer:` Là các biến số nguyên - Ép kiểu về dạng này sử dụng `int()`
- `floats:` Là các biến số thực - Ép kiểu về dạng này sử dụng `float()`
- `str():` Là các biến dạng chuỗi - Ép kiểu về dạng này sử dụng `str()`
- `bool()` Là các biến đúng sai dạng _True/False_ - Ép kiểu về dạng này sử dụng `bool()`
- `type():` Dùng để kiểm tra kiểu dữ liệu của biến
- `isinstance(object, classinfo):` Dùng để kiểm tra một đối tượng có thuộc về một kiểu dữ liệu hoặc một lớp nhất định hay không
- **Mutable:** Sau khỉ khởi tạo ta có thể chỉnh sửa nội dung bên trong vùng nhớ của nó **mà không làm thay đổi địa chỉ id** của đối tượng\
  - _Ví dụ:_ list, dict, set, torch.Tensor
- **Immutable:** Sau khi dã tạo, giá trị bên trong vùng nhớ **không bao giờ được thay đổi**. Bất kỳ thao tác nào thực hiện đều tạo ra một đối tượng mới với vùng nhớ mới
  - _Ví dụ:_ int, float, str, ..  

| Thành phần Pipeline | Nên dùng kiểu | Thư viện / Data Type khuyên dùng | Rủi ro nếu làm sai (Why it matters) | Best Practice cụ thể |
| :--- | :--- | :--- | :--- | :--- |
| **Model & Pipeline Config** | Immutable | `dataclasses` (`frozen=True`), `NamedTuple`, `Hydra DictConfig` | **Data Leakage / Non-reproducible Runs:** Config bị sửa ngầm (mutation side-effects) giữa các bước Train/Val/Test làm mất tính đồng nhất của thực nghiệm. | Khóa hoàn toàn config sau khi parse. Không bao giờ truyền `dict` thuần làm config cho các hàm xử lý dữ liệu. |
| **Category Mapping & Class Labels** | Immutable | `tuple`, `frozenset`, `enum.Enum` | **Index Corruption:** Thứ tự class ID bị lệch giữa quá trình Training và Inference (ví dụ: `0: "cat"` bị đổi thành `0: "dog"` do append mảng). | Khai báo hằng số mapping dạng `Tuple[str, ...]` hoặc `Enum` để cố định index và ngăn chặn hàm khác `append()`/`insert()`. |
| **Feature Caching & Hash Keys** | Immutable | `str`, `tuple`, `bytes` | **TypeError (Unhashable):** Các decorator như `@lru_cache` hoặc Redis Key đòi hỏi input phải hashable. `list`/`dict` sẽ làm crash pipeline. | Ép kiểu các tham số feature/path sang `tuple` hoặc `str` trước khi truyền vào hàm cache. |
| **Dataset In-Memory Buffers** | Mutable | `torch.Tensor`, `np.ndarray`, `pandas.DataFrame` | **Out Of Memory (OOM):** Nếu lạm dụng Immutable (tạo copy liên tục) với mảng dữ liệu lớn, RAM/VRAM sẽ bị tràn nhanh chóng. | Sử dụng Mutable data structures nhưng **hạn chế in-place operation** (`inplace=True`, `tensor.add_()`) trong graph tính toán của PyTorch/TensorFlow. |
| **Data Augmentation & Preprocessing** | Immutable (Input)<br>Mutable (Output) | PyTorch `Transforms`, `albumentations` | **Dataset Corruption:** Biến đổi trực tiếp trên mảng gốc khiến dữ liệu của Epoch 2 bị đè bởi biến đổi của Epoch 1. | Luôn trả về một tensor/array mới (`copy()`) trong hàm `__getitem__` của DataLoader, giữ nguyên raw sample gốc. |
| **Experiment Metrics & State Logging** | Mutable | `MLflow`, `W&B`, custom accumulator (`list`/`dict`) | **Missing Logs / Race Conditions:** Cố tình xài hằng số hoặc tuple sẽ khiến việc append metric theo epoch trở nên cồng kềnh. | Gom metric theo từng step vào `dict` tạm thời, sau đó ghi sang tracking system (MLflow/W&B) và giải phóng bộ nhớ. |

### **Chuỗi**

- Cắt chuổi dùng: `s[start:stop:step]`
  - **start:** Ví trí bắt đầu cắt 
  - **stop:** Ví trí kết thúc cắt 
  - **step:** Bước nhảy cắt
- `split()`: Chia chuổi theo ký tự `" "` - theo khoảng trắng 
- `join()`: Gộp chuổi theo ký tự chèn vào `" "` - theo khoảng trắng 
- `strip()`: Loại bỏ khoảng trắng đầu và cuối chuổi 
- `replace()`: Thay thế chuỗi 
- `find()`: Tìm kiếm ký tự trong chuỗi
- `startswith(obj)`: Bắt đầu bằng ký tự `obj` trong chuỗi 
- `endswith(obj)`: Kết thúc bằng ký tự `obj` trong chuỗi
- f-string: `f"value is {x:.2f}"` - trong dấu `{}` là biến cần truyền để in ra màn hình
- Ngoài ra có thể sử dụng `""" """` để chèn một string dài

### Collection (Tập hợp)

Giống như các ngôn ngữ lập trình khác các tập hợp được xem là lưu trữ nhiều biến được khai báo trong một chương trình 

_**List (Danh sách)**_

- Dặc điểm của danh sách có thể lưu trữ nhiều dữ liệu cho phép trùng nhau và các dữ liệu đặt trong dấu `[]`.
- Các list có thể lồng nhau, có thể thay thế được
- Thường dược dùng cho các danh sách có thứ tự

```python
list = [1, 2, 3, 4, 5]
```

- Truy cập: `list[0]` - cho phần tử đầu, `list[-1]` - cho phần tử cuối  
- Cắt lát (Slicing): `list[1:3]` (lấy từ index 1 đến 2).
- Thêm: `append(x)` (vào cuối), `insert(index, x)` (vào vị trí bất kỳ), `extend(iterable)` (nối danh sách).
- Xóa: `pop()` (xóa cuối và trả về giá trị), `remove(x)` (xóa phần tử đầu tiên có giá trị x).
- Sắp xếp: `sort()` (thay đổi trực tiếp) hoặc `sorted(my_list)` (trả về list mới).

**Tuple (Bộ)**

- Giống với list nhưng không thể thay đổi sau khi tạo và các dữ liệu được đặt trong `()`
- Được sử dụng trong việc lưu trữ dữ liệu cố định, bảo vệ dữ liệu không bị sửa đổi

```python
tuple = (1, 2, 3, 5)
```

- Truy cập: Giống List `(my_tuple[0], my_tuple[:2])`.
- Thao tác chỉnh sửa: Không có.
- Thao tác tìm kiếm: `count(x)` (đếm số lần xuất hiện), `index(x)` (tìm vị trí của x).
- Ứng dụng đặc biệt: Dùng để unpack dữ liệu: `x, y = (10, 20)`.


**Sets (Tập hợp)**

- Dặc điểm là không cho phép trùng lặp, không quan tâm đến thứ tự và các dữ liệu được đặt trong dấu `{}`
- Có thể thay đổi được
- Dược sử dụng trong các phép toán tập hợn, tìm phần tử duy nhất

```python
set = {1, 2, 3, 5}
```

- Khởi tạo set rộng có thể dùng `set()` hoặc `{}`
- Truy cập: Chỉ có thể duyệt qua bằng vòng lặp for.
- Thêm/Xóa: `add(x)`, `remove(x)` (lỗi nếu không có x), `discard(x)` (không lỗi nếu không có x).
- Phép toán tập hợp:
  - Hợp (Union): `set1 | set2` hoặc `set1.union(set2)`
  - Giao (Intersection): `set1 & set2` hoặc `set1.intersection(set2)`
  - Hiệu (Difference): `set1` - `set2`

**Dictionary (Thư viện)**

- Tổ chức theo `{key:value}`. Key phải là duy nhất
- Key cho phép trùng lặp còn Value thì không
- Thay đổi được
- Được sử dụng như một cách để lưu trữ thông tin (user, giao dịch)

```python
my_dict = {'name': 'An', 'age': 20}
```

- Truy cập: `my_dict['name']` (lỗi nếu key không tồn tại) hoặc an toàn hơn là `my_dict.get('name', 'Không thấy')`.
- Thêm/Sửa: `my_dict['job'] = 'Dev'` (Nếu key chưa có thì thêm mới, nếu có rồi thì ghi đè).
-  Xóa: `del my_dict['age']` hoặc `my_dict.pop('age')`.
- Lấy danh sách thành phần:
  - `y_dict.keys()`: Lấy toàn bộ Key.
  - `my_dict.values()`: Lấy toàn bộ Value.
  - `my_dict.items()`: Lấy cặp (Key, Value) dưới dạng Tuple.

### Control Flow

- `if` điều kiện - `elif` điều kiện khác - `else` điều kiện cuối 
- `for` vòng lặp - `for i in list` lặp các phần tử trong list - `for i, j in dict.items()` lặp các phần tử trong dict - `for i in ranges(10)` lặp từ 0 đến 9
- `while` lặp - `break` điều kiện thỏa thì dừng lại - `continues` thỏa điều kiện thì tiếp tục 
- `range`: Chỉ vùng dữ liệu
- `enumerate(chuỗi cần đuyệt, giá trị bắt đầu=0)`: Duyệt phần tử kèm chỉ số 
- `zip()`: Ghép song song nhiều danh sách 

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
---

## 1.2. Function (Hàm)

Hầu hết các chương trình đều được viết dựa trên hàm, giống như là một tập hợp các bước thao tác của một tác vụ nào đó. 
Để định nghĩa một hàm ta dùng `def`

```python
def test():
  return True # def là từ khóa | test là tên hàm
```

**Hàm có tham số**

Tham số được định nghĩa là parameter được sử dụng để biểu hiện giá trị được truyền vào hàm 

```python
def get(name):
  print(f"{name}") # name là tham số truyền vào hàm
```

**Hàm trả về giá trị**

Là hàm kết thúc bằng `return` có thể là số, chữ, danh sách, ... hay phép tính 

```python
def test():
  return True

```

**Parameter mặc định**

Là tham số mặc định khi ta gọi hàm mà không truyền bất kỳ tham số nào khác vào hàm, và khi ta truyền một giá trị nào khác giá trị mặc
định ta sẽ xem nó là Keyword Argument

```python
def greet(name="Guest"):
    print(name)  # Khi không truyền giá trị nào khác vào name thì mặc định giá trị của name sẽ là "Guest"
```

**Scope**

- Local: là biến khi khai báo trong hàm ta sẽ sử dụng nó trong hàm và khi ra ngoài hàm thì sẽ không được sử dụng
- Global: là biến có thể sử dụng ở bất cứ đầu trong chương trình
- `*args`: Cho phép gom các tham số vị trí không cố định thành một `tuple`
- `**kwargs`: Cho phép gom các tham số vị trí không cố định thành một `dict`
  - `*args` và `**kwargs` giúp tùy nghi truyền thêm tham số mà không cần làm vỡ code, giúp mở rộng, sửa code dễ dàng

**Lambda function**

Hay gọi là hàm vô danh là hàm nhỏ, chỉ chứa một biểu thức duy nhất và không có tên khi khai báo 

```python
lambda arguments: expression

# Hàm thông thường
def add_one(x):
    return x + 1

# Viết tương đương bằng Lambda
add_one_lambda = lambda x: x + 1

print(add_one_lambda(5)) # Kết quả: 6
```

**Đệ quy**

Là kỹ thuật mà một hàm tự gọi lại chính nó, chia nhỏ một bài toán lớn thành các bài toán nhỏ
- Điều kiện dừng gọi lại chính nó
- Bước đệ quy là nơi hàm gọi lại chính nó với bài toán có quy mô nhỏ hơn

**Docstrings**

Là chuỗi văn bản dặt đầu hàm, class, hoặc module giải thích chức năng, tham số và kết quả trả về. Giúp IDE hiển thị gợi ý
- Chuẩn viết DocStrings trong Python AI DSK

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

--- 


## 1.3. OOP (Hướng đối tượng) [](#1.2)

Giống với các ngôn ngữ lập trình khác thì Python cũng có các phương thức liên quan đến Hướng đối tượng 

**Class (Lớp)**

Lớp giống như một đối tượng lớn chứa tất cả các thuộc tính chung khi muốn gọi một đối tượng mỡi phải xuất phát từ class 

```python
class Employee: # Lớp nhân viên
  pass
```

**Object (Đối tượng)**

Là một thực thể cụ thể được tạo ra từ class. Nó mang các thuộc tính và hành vi riêng 

```python
class Meo:
  pass

meo_vang = Meo() # meo_vang là một object
```


**Constructor `(__init__)`**

Hàm khởi tạo từ động chạy ngay khi một đối tượng tạo ra. Dùng để gắn giá trị ban đầu cho thuộc tính của đối tượng

```python
class Meo:
    def __init__(self, ten):
        self.ten = ten # Gán tên ngay khi tạo mèo

meo = Meo("Miu") # Lúc này meo.ten sẽ là "Miu"
```

**Instance Method**

Là hàm thông thường trong class, hoạt động trực tiếp trên từng Object cụ thể. Luôn có tham số `self` ở đầu để đại diện cho chính Object đó.

```python
class Meo:
    def __init__(self, ten):
        self.ten = ten
        
    def keu(self): # Instance Method
        return f"{self.ten} kêu meo meo"
```

**Class Method**

Là hàm quản lý các vấn đề chung của cả Class chứ không riêng gì một Object. Dùng decorator `@classmethod` và nhận tham số đầu tiên là `cls` (chính là Class đó)

```python
class Meo:
    so_luong = 0
    def __init__(self):
        Meo.so_luong += 1
        
    @classmethod
    def lay_so_luong(cls): # Class Method
        return f"Tổng số mèo: {cls.so_luong}"
```

**Static Method**

Là một hàm tiện ích nằm trong Class nhưng hoàn toàn độc lập, không dùng dữ liệu của Object (`self`) hay Class (`cls`). Dùng decorator `@staticmethod`.

```python
class CungCu:
    @staticmethod
    def doi_tieng_meo(chuoi): # Static Method
        return chuoi.replace("người", "meo")
```

**Các tính chất hướng đối tượng**
- Kế thừa (Inheritance)
  Cho phép một Class con sử dụng lại các thuộc tính và hàm của Class cha, giúp tránh lặp code thông qua từ khóa `super()`.

  ```python
  class DongVat: # Cha
    def an(self):
        return "Đang ăn..."

  class Cho(DongVat): # Con kế thừa từ Cha
      def sua(self):
          return "Gâu gâu!"
  
  milu = Cho()
  print(milu.an())  # Kế thừa từ cha: "Đang ăn..."
  print(milu.sua()) # Của riêng con: "Gâu gâu!"
  ```
- Đóng gói (Encapsulation)
  Che giấu dữ liệu bên trong Object để tránh bị bên ngoài sửa đổi lung tung. Trong Python, ta dùng dấu gạch dưới `_ (protect)` hoặc `__ (private)`.

  ```python
    class TaiKhoan:
    def __init__(self, so_du):
        self.__so_du = so_du # Private, bên ngoài không gọi trực tiếp được

    def xem_so_du(self): # Hàm gián tiếp để truy cập an toàn
        return self.__so_du
    
    tk = TaiKhoan(1000)
    # print(tk.__so_du) # LỖI! Không truy cập được.
    print(tk.xem_so_du()) # Đúng: 1000
  ```

- Đa hình (Polymorphism)
  Cùng một tên hàm nhưng các Class khác nhau sẽ thực hiện theo các cách khác nhau.

  ```python
  class Cho:
    def keu(self): return "Gâu gâu"

  class Meo:
      def keu(self): return "Meo meo"
  
  # Hàm nhận vào bất kỳ con gì và bắt nó kêu
  def lam_cho_keu(dong_vat):
      print(dong_vat.keu())
  
  lam_cho_keu(Cho()) # In ra: Gâu gâu
  ```
  
- Abstract Class (Lớp trừu tượng)
  Là một "khung thiết kế" chung. Bạn không thể tạo Object trực tiếp từ Abstract Class, mà bắt buộc các Class con phải kế thừa và định nghĩa cụ thể các hàm của nó. Dùng thư viện `abc`
  ```python
  from abc import ABC, abstractmethod

  class Hinh(ABC): # Abstract Class
      @abstractmethod
      def tinh_dien_tich(self):
          pass
  
  class HinhVuong(Hinh):
      def __init__(self, canh):
          self.canh = canh
      def tinh_dien_tich(self): # Bắt buộc phải viết hàm này, nếu không sẽ lỗi
          return self.canh * self.canh
  ```

**Các phương thức khác**

- `__str__`, `__repr__`: Chuyển đổi đối tượng thành String
  - `__str__`: Dùng để hiển thị cho Người dùng cuối (End-user). Ưu tiên tính dễ đọc, thân thiện. Gọi khi dùng print(obj) hoặc str(obj).
  - `__repr__`: Dùng cho Developer / Debugging. Yêu cầu tính chính xác, ngắn gọn, hiển thị rõ kiểu Class và thông số chính để tái tạo Object. Gọi khi dùng repr(obj) hoặc gõ tên biến trong console/Jupyter Notebook.

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

- `__len__`, `__getitem__`: Class hoạt động thành List hoặc Dict (truy cập bằng chỉ số)
  - Dataset trong PyTorch bắt buộc phải implement 2 hàm này để DataLoader có thể chia batch và iterate qua dữ liệu.
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
print(len(dataset))       # Output: 2 (thông qua __len__)
print(dataset[0])         # Output: {'text': 'Học Python AI', 'label': tensor(1)} (thông qua __getitem__)
```

**Các Decorator vói lớp tĩnh trong Python**

| Decorator | Cú pháp phương thức | Tham số bắt buộc | Bản chất & Mục đích chính | Trường hợp sử dụng tiêu biểu trong AI / MLOps |
| :--- | :--- | :--- | :--- | :--- |
| **`@property`** | `def my_attribute(self):` | `self` | Biến một Method thành Attribute (chỉ đọc hoặc có validation). Cho phép truy cập dạng `obj.my_attribute` thay vì `obj.my_attribute()`. | • Kiểm tra trạng thái GPU/RAM khả dụng.<br>• Tính toán dynamic attribute (vd: `is_v2`, `embedding_dim`).<br>• Bảo vệ biến nội bộ không bị sửa đổi trực tiếp. |
| **`@classmethod`** | `def my_factory(cls, ...):` | `cls` (đại diện cho Class) | Tác động lên cấp độ Class thay vì Instance. Thường dùng làm **Factory Methods** để khởi tạo Object từ nhiều nguồn dữ liệu khác nhau. | • `Model.from_pretrained("bert-base")`<br>• `Config.from_json(json_str)`<br>• `Dataset.from_pandas(df)` |
| **`@staticmethod`** | `def my_utility(...):` | Không nhận `self` hay `cls` | Là một hàm tiện ích (Utility function) độc lập, được nhóm vào bên trong Class cho gọn scope/namespace. Không truy cập trạng thái Class hay Instance. | • Preprocessing/Clean văn bản (`clean_text()`).<br>• Validate format đường dẫn file config.<br>• Helper math operations độc lập. |
| **`@abstractmethod`** | `def predict(self, x):` | `self` (hoặc `cls`) | Đi kèm với `ABC` (Abstract Base Class) để tạo khung thiết kế. **Bắt buộc** các Class con kế thừa phải viết code triển khai phương thức này. | • Định nghĩa Base Provider chuẩn LangChain/LlamaIndex (`BaseLLM`, `BaseVectorStore`).<br>• Interface chung cho Custom PyTorch Pipeline/Transform. |

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
print(clean_str)         # Output: "Prompt có newline  "
```

## 1.4. Pythonic Code 

Cách sử dụng code chuẩn Python 

- **Comprehension:** Giúp tạo list, dict, set bằng cú pháp 1 dòng trong khi đó

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

- **Genenerator Experssions:** Xử lý dữ liệu lớn mà không làm tràn RAM. Dùng ngoặc `()` và chỉ tính toán từng phần tử khi được yêu cầu

```python
# GIẢ SỬ: File chứa 10 triệu dòng text
# BAD: Nạp cả 10 triệu dòng vào RAM dưới dạng List
# memory_heavy = [clean_text(line) for line in open("huge_corpus.txt")]

# GOOD: Dùng Generator Expression, RAM dùng chỉ ~0 MB
memory_efficient = (line.strip().lower() for line in open("huge_corpus.txt"))

# Duyệt từng dòng một để xử lý / đưa vào pipeline
first_sample = next(memory_efficient)
```
- `map()`: Áp dụng một hàm lên toàn bộ phần tử của chuỗi dữ liệu
- `filter()`: Lọc các phần tử thỏa mãn điều kiện trả về `True`
- `reduce()`: Tích lũy hàm 2 tham số lên lần lượt các phần tử từ trái sang phải để rút gọn tập hợp thành 1 giá trị duy nhất
- Unpaking: `a, b, *rest = lst`: Trích xuất các phần tử trong danh sách thành 1 biến riêng lẻ
```python
# Lấy phần tử đầu, phần tử cuối và gom tất cả phần còn lại
batch_data = ["sample_0", "sample_1", "sample_2", "sample_3", "target"]

first_sample, *middle_samples, label = batch_data

print(first_sample)     # 'sample_0'
print(middle_samples)    # ['sample_1', 'sample_2', 'sample_3']
print(label)             # 'target'
```

- `any()`: Trả về `True` nếu có ít nhất 1 phần tử là `True`
- `all()`: Trả về `True` nếu tất cả phần tử là `True` 
- `sorted()`: Sắp xếp theo chỉ đinh trong hàm 
- `collection` :
  - `Counter`: Đếm tần suất phần tử
  - `defaultdict`: Tự động khởi tạo giá trị mặc định cho một key chưa tồn tại khi bạn truy cập hoặc gán giá trị.
  
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
  - `deque`: Hành chờ cấu trúc dữ liệu

---
 
## 1.5. File handing [](#1.4)

Các thao tác quan trọng như: 

**Đọc file**

Để có thể đọc file ta sử dụng từ khóa `with`

```python
with open("data.txt", "r") as file:
  content = file.read()
```

Thao tác chính: 
- Với data.txt chính là tên file ta muốn đọc
- as file: Tên biến lưu để thực hiện trong chương trình
- Từ khóa "r" thể hiện thao tác đọc
- content = file.read(): Biến content lưu toàn bộ nội dung của file

**Đọc từng dòng trong file**

```python
with open("data.txt") as file:
    for line in file:
        print(line.strip())
```

Thao tác chính: 
- Duyệt từng dòng trong file và xóa khoảng trắng giữa chúng

**Ghi file**

```python
with open("output.txt", "w") as file:
    file.write("Hello")
```

Thao tác chính: 
- Với output.txt chính là tên file ta muốn ghi
- as file: Tên biến lưu để thực hiện trong chương trình
- Từ khóa "w" thể hiện thao tác ghi
- file.write(): Ghi nội dung mới trong file

**Append**
-Thêm dòng mới vào trong file

```python
with open("log.txt", "a") as file:
    file.write("New log\n")
```

Thao tác chính: 
- Với log.txt chính là tên file ta muốn ghi
- as file: Tên biến lưu để thực hiện trong chương trình
- Từ khóa "a" thể hiện tha
```python
# Đọc
with open("students.csv") as file:
    reader = csv.DictReader(file) # Xác định đọc file
    for row in reader:
        print(row)

# Ghi
with open("students.csv", "w", newline="") as file:
  writer = csv.writer(file)
  writer.writerow(["id","name","score"]) # Thêm cột mới
  writer.writerow([1,"An",8.5]) # Thêm dòng mới
```

Thao tác chính: 
- Với students.csv chính là tên file ta muốn ghio tác thêm mới vào trong file
- file.write(): Ghi nội dung mới trong file lưu ý thêm có `\n` xuống dòng

**Thao tác với file csv**

- Để đọc và ghi file csv ta cần khai báo thư viện `imoport csv`
- `os`: Thao tác hệ thống cũ hơn, làm việc với đường dẫn dạng chuỗi string
- `pathlib.Path`: Thao tác đường dẫn theo hướng đối tượng (Khuyên dùng)
- `glob`: Tìm kiếm filke theo pattern (`*.jpg`)

## 1.6. Sửa lỗi & Debugging  [](#1.3)


Trong quá trình xây dựng một chương trình nào đó ta sẽ phải đối mặt với những lỗi như kiểu dữ liệu không đúng, file không tồn tại, ... Exception ra đời nhằm giám sát chất lượng dữ liệu, xây dựng một hệ thống 
có thể tự phục hồi và tìm ra lỗi tự động giảm tải bớt nhân công và thời gian 

**Try Except**

Đóng gói đoạn code có nguy cơ lỗi. Nếu lỗi xảy ra, chương trình không bị sập mà lập tức nhảy vào khối `except` để xử lý.

```python
try:
    # Khối lệnh có nguy cơ xảy ra lỗi (ví dụ: ép kiểu dữ liệu lỗi)
    age = int("không phải số")
except ValueError:
    # Khối lệnh xử lý khi bắt đúng loại lỗi ValueError
    print("Lỗi: Vui lòng nhập vào một số nguyên hợp lệ!")
```

**Multiple Except**

Bắt riêng biệt từng loại lỗi khác nhau từ trên xuống dưới. Thêm `Exception` ở cuối cùng làm lưới bảo hiểm cho các lỗi chưa lường trước.

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
    # Bắt tất cả các loại lỗi còn lại (đóng vai trò như một lưới bọc an toàn)
    print(f"Đã xảy ra lỗi hệ thống không xác định: {e}")
```

**Else**

Chỉ chạy khi khối `try` thực thi thành công hoàn toàn và không xảy ra bất kỳ lỗi nào.

```python
try:
    # Thực hiện một tác vụ đọc dữ liệu từ API hoặc File
    score = int("95")
except ValueError:
    print("Lỗi định dạng số.")
else:
    # Khối này CHỈ chạy khi khối try hoàn thành trơn tru và KHÔNG có exception nào xảy ra
    print(f"Xử lý dữ liệu thành công. Điểm số của user là: {score}")
```

**Finally**

Luôn luôn chạy cho dù khối `try` có lỗi hay không, có bị `return` thoát hàm hay không. Dùng để dọn dẹp hệ thống (đóng file, ngắt kết nối DB)

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

Chủ động ép chương trình ném ra (kích hoạt) một lỗi khi dữ liệu vi phạm logic nghiệp vụ (mặc dù code không sai cú pháp).
- **raise:** Chủ động ném lỗi khi dữ liệu vi phạm nghiệp vụ.

```python
def check_replicate_count(count):
    if count < 0:
        # Chủ động ném (trigger) ra một ngoại lệ khi dữ liệu vi phạm logic nghiệp vụ
        raise ValueError("Số lượng bản sao (replicate) không được là số âm!")
    return True

try:
    check_replicate_count(-5)
except ValueError as e:
    print(f"Hệ thống phát hiện dữ liệu bất thường: {e}")
```

**Custom Exception**

Tự tạo ra một loại lỗi mới bằng cách kế thừa lớp `Exception`, giúp định nghĩa tường minh tên lỗi theo đúng nghiệp vụ của dự án.
- Custom Exception: Tạo lớp kế thừa từ Exception để quản lý lỗi theo domain AI.


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

Thay thế hoàn toàn cho lệnh `print()` trong code nghiệp vụ. Cung cấp các mức độ `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

- `breakpoint()` / `pdb`: Dừng chwuong trình lại để xem biến trức tiếp từ Terminal

```python
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Bắt đầu huấn luyện mô hình...")

# Đặt điểm dừng Debug
x = 10
# breakpoint() # Chương trình sẽ dừng lại ở đây để tương tác CLI
```

--- 

## 1.7. Hiệu suất và Bộ nhớ

Chiến lược giúp tối ưu hiệu năng tính toán và quản lý dung lượng RAM/VRAM.

- **Generators & `yield`**: Cho phép viết hàm trả về dữ liệu từng phần một dưới dạng Stream mà không nạp toàn bộ vào RAM cùng một lúc. Rất quan trọng khi xử lý streaming token từ LLM.Pythondef stream_llm_response():

```python
tokens = ["Xin", "chào", "bạn", "tôi", "là", "AI"]
    for token in tokens:
        yield token

for chunk in stream_llm_response():
    print(chunk, end=" ")
```

- **Thư viện `itertools`**: Tạo ra các con lặp hiệu năng cao (C-speed) phục vụ cho tính toán tổ hợp.
  - `itertools.chain()`: Nối nhiều iterable.
  - `itertools.islice()`: Cắt lát generator.
  - `itertools.product()`: Tích Cartesian (dùng làm Grid Search hyperparameter đơn giản).
- **Benchmarking** (`timeit`, `cProfile`)
  - `timeit`: Đo thời gian thực thi của một đoạn code nhỏ.
  - `cProfile`: Phân tích hiệu năng (profiling) xem hàm nào tốn nhiều thời gian nhất trong ứng dụng.
- **Shallow vs Deep Copy**:
  - **Shallow Copy (`copy.copy()`):** Tạo đối tượng mới nhưng giữ nguyên tham chiếu đến các đối tượng con mutable bên trong.
  - **Deep Copy (`copy.deepcopy()`)**: Sao chép toàn bộ đối tượng và tất cả các đối tượng con một cách độc lập hoàn toàn.
- **Vectorization (Vector hóa)**: Thay thế vòng lặp Python thuần (`for` loop) bằng các thao tác vector hóa của NumPy/PyTorch. Các phép toán này được viết bằng C/CUDA mang lại tốc độ nhanh hơn từ 10x đến 100x.

---

## 1.8. NumPy (Nền tảng tính toán Mảng)Thư viện lõi cho đại số tuyến tính và xử lý mảng đa chiều trong Python.Tạo mảng & Thuộc tính cơ bảnPythonimport numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # Dimensions: (2, 3)
print(arr.ndim)   # Số chiều: 2
print(arr.dtype)  # Kiểu dữ liệu: int64

zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
eye = np.eye(3)   # Ma trận đơn vị
Thay đổi hình dạng (Reshaping) & Nối mảng (Stacking)reshape(): Thay đổi kích thước mảng mà không đổi dữ liệu.flatten(): Chuyển mảng thành 1D (tạo ra copy mới).ravel(): Chuyển mảng thành 1D (trả về view nếu có thể, tiết kiệm bộ nhớ).np.vstack(), np.hstack(), np.stack(): Ghép các mảng theo chiều dọc/ngang/chiều mới.Indexing nâng cao & BroadcastingBoolean Indexing: Lọc dữ liệu theo điều kiện logic.np.where(condition, x, y): Chọn $x$ nếu thỏa điều kiện, ngược lại chọn $y$.Broadcasting Rules: Cho phép thực hiện phép toán giữa các mảng khác kích thước nếu thỏa mãn điều kiện tương thích về shape (ví dụ: cộng ma trận với một vector).Pythondata = np.array([10, 25, 30, 5])
filtered = data[data > 15]  # [25, 30]
replaced = np.where(data > 15, 1, 0)  # [0, 1, 1, 0]
Phép toán Ma trận & AggregationsPhép nhân ma trận: Sử dụng np.dot(A, B) hoặc toán tử @.np.linalg.inv(): Tìm ma trận nghịch đảo.np.linalg.eig(): Tìm trị riêng và vector riêng.Aggregations: np.mean(), np.sum(), np.std(), np.argmax() kèm theo tham số axis (ví dụ: axis=0 cho cột, axis=1 cho dòng).Pythonmatrix = np.array([[1, 2], [3, 4]])
# Nhân ma trận
result = matrix @ matrix
# Tìm chỉ số max theo dòng
max_idx = np.argmax(matrix, axis=1)
1.9. Pandas (Xử lý và Phân tích Dữ liệu Bảng)Thư viện chuẩn để thao tác với Tabular Data (DataFrame & Series).Khởi tạo & Khảo sát Dữ liệuPythonimport pandas as pd

df = pd.read_csv("data.csv")
print(df.head(5))       # 5 dòng đầu
print(df.info())        # Thông tin kiểu dữ liệu và giá trị Null
print(df.describe())    # Thống kê mô tả (Mean, Std, Min, Max, Quantiles)
print(df.shape)         # Kích thước (rows, cols)
Truy xuất & Lọc dữ liệu (loc vs iloc)loc: Truy xuất bằng Nhãn (Label/Name) của dòng và cột.iloc: Truy xuất bằng Chỉ số nguyên (Integer Index).Python# Lọc bằng loc (Boolean Mask)
high_income = df.loc[df['income'] > 50000, ['name', 'income']]

# Lọc bằng iloc (hàng từ 0 đến 5, cột từ 0 đến 2)
subset = df.iloc[0:5, 0:2]
Xử lý Missing Dataisna() / isnull(): Kiểm tra giá trị thiếu.dropna(): Bỏ bớt các dòng/cột chứa giá trị Null.fillna(): Điền giá trị thay thế (Mean, Median, Mode hoặc gán cố định).Gom nhóm & Biến đổi (GroupBy, Reshape, Merge)GroupBy & Aggregation: df.groupby('category').agg({'price': 'mean', 'id': 'count'})Merge (Join): Kết nối các DataFrame tương tự SQL (inner, left, right, outer).Concat: Nối theo chiều dọc hoặc chiều ngang.Pivot Table & Melt: Soay và tái cấu trúc dạng bảng (Wide vs Long format).Datetime Parsing: pd.to_datetime() - chuyển cột chuỗi sang kiểu thời gian để trích xuất year, month, day.1.10. Code Quality & Cấu trúc Dự án (Software Engineering Practices)Biến các đoạn code dạng script thành một Dự án phần mềm chuẩn mực, sẵn sàng bảo trì.Môi trường ảo & Quản lý Phụ thuộcTạo môi trường cách ly dependency để tránh xung đột phiên bản:Bashpython -m venv .venv
source .venv/bin/activate  # Linux/Mac
# Hoặc dùng conda: conda create -n ai_env python=3.10
Lưu và cài đặt danh sách thư viện: pip freeze > requirements.txt và pip install -r requirements.txt.Cấu trúc Module & PackageThêm file __init__.py vào thư mục để biến thư mục đó thành một Python Package.Tách code thành các submodule rõ ràng (data/, models/, utils/).Type Hints & DataclassesType Hints: Khai báo kiểu dữ liệu cho tham số và giá trị trả về giúp IDE hiển thị gợi ý và bắt lỗi tĩnh.dataclasses: Tạo lớp chứa dữ liệu thuần túy một cách nhanh chóng mà không cần viết boilerplate __init__.Pythonfrom dataclasses import dataclass

@dataclass(frozen=True)  # frozen=True biến instance thành Immutable
class ModelConfig:
    model_name: str
    learning_rate: float = 0.001
    batch_size: int = 32

config = ModelConfig(model_name="ResNet18")
Testing, Linting & FormattingUnit Testing với pytest: Viết test tự động cho các hàm quan trọng.Linters (ruff / flake8): Kiểm tra lỗi cú pháp và chuẩn PEP8.Formatters (black): Tự động format code theo chuẩn thống nhất.1.11. Python cho AI WorkflowsCác công cụ trợ lực cho chu trình nghiên cứu, phát triển và thử nghiệm mô hình AI.Môi trường Interactive & Tiến trìnhJupyter Notebooks / Google Colab: Môi trường thử nghiệm nhanh, hỗ trợ GPU/TPU miễn phí. Sử dụng các Magic commands như %timeit, %matplotlib inline.tqdm: Tạo thanh tiến trình (progress bar) cho các vòng lặp huấn luyện dài.Pythonfrom tqdm import tqdm
import time

for epoch in tqdm(range(100), desc="Training Model"):
    time.sleep(0.01)
Quản lý Tham số CLI & Configargparse: Truyền tham số trực tiếp từ dòng lệnh khi thực thi script.YAML / Hydra Config: Quản lý cấu hình mô hình tập trung từ các file .yaml ngoại vi.python-dotenv (.env): Lưu trữ an toàn các hằng số nhạy cảm như API Keys mà không lo dính vào Git.Pythonimport os
from dotenv import load_dotenv

load_dotenv()  # Nạp các biến từ file .env
api_key = os.getenv("OPENAI_API_KEY")
Tái lập thực nghiệm (Reproducibility) & Lưu trữSeeding: Cố định seed cho tất cả các thư viện sinh số ngẫu nhiên để đảm bảo thực nghiệm có thể tái lập 100%.Pythonimport random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
Lưu trữ mô hình: joblib.dump(), torch.save().1.12. Async Python (Lập trình Bất đồng bộ cho AI APIs)Kiến thức bắt buộc để xây dựng các ứng dụng tương tác với LLM APIs (OpenAI, Claude, vLLM) nhằm tối ưu latency và throughput.Cơ chế async / await & Event LoopLập trình bất đồng bộ cho phép chương trình thực hiện các công việc khác trong thời gian chờ phản hồi I/O (ví dụ: chờ API trả về token) thay vì chặn toàn bộ hệ thống (blocking).Pythonimport asyncio

async def fetch_llm_response(prompt: str) -> str:
    print(f"Đang gửi prompt: {prompt}")
    await asyncio.sleep(2)  # Giả lập lệnh chờ I/O từ Network
    return f"Response cho: {prompt}"

async def main():
    result = await fetch_llm_response("Hello AI")
    print(result)

# Chạy Event Loop
asyncio.run(main())
Thực thi đồng thời với asyncio.gather()Gửi nhiều request API cùng một lúc thay vì gọi tuần tự từng request.Pythonasync def process_batch_prompts(prompts: list[str]):
    # Gửi song song tất cả các prompts sang API
    tasks = [fetch_llm_response(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

# Tốc độ phản hồi sẽ bằng thời gian request chậm nhất, thay vì tổng thời gian tất cả request
Async HTTP Clients (httpx, aiohttp)Sử dụng các thư viện hỗ trợ Async Native để làm việc với REST APIs hoặc Server-Sent Events (SSE) khi Streaming LLM Responses.Thư việnĐặc điểmTrường hợp sử dụngrequestsSynchronous (Blocking)Script đơn giản, không yêu cầu concurrency cao.httpxHỗ trợ cả Sync & AsyncChuẩn Production mới, được OpenAI SDK và FastAPI tích hợp sẵn.aiohttpAsync thuần túyHệ thống Async lớn, giao tiếp WebSocket hoặc Streaming pipeline nặng.Tại sao Streaming LLM lại cần Async?Các mô hình Generative AI sinh dữ liệu dưới dạng chuỗi các token theo thời gian. Sử dụng Async kết hợp với Async Generators (async for ... in ...) cho phép client nhận và render từng token lên giao diện người dùng ngay lập tức khi nó vừa được tạo ra mà không làm đứng (freeze) luồng xử lý chính của ứng dụng.


