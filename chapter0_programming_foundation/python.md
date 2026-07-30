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

Là hàm thông thường trong class, hoạt động trực tiếp trên từng Object cụ thể. Luôn có tham số self ở đầu để đại diện cho chính Object đó.

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
  Cho phép một Class con sử dụng lại các thuộc tính và hàm của Class cha, giúp tránh lặp code.

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
  
- Abstract Class (cơ bản)
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


## 1.3. Exception  [](#1.3)

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


---

## 1.4. File handing [](#1.4)

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
- Từ khóa "a" thể hiện thao tác thêm mới vào trong file
- file.write(): Ghi nội dung mới trong file lưu ý thêm có `\n` xuống dòng

**Thao tác với file csv**

- Để đọc và ghi file csv ta cần khai báo thư viện `imoport csv`

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
- Với students.csv chính là tên file ta muốn ghi

