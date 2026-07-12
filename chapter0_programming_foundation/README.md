# Phase 1. Nền tảng lập trình 
> ⭐ Mục này lưu trữ lý thuyết và bài tập liên quan đến `Python` + `Git` + `SQL`
---
# ☑️ 1. Python [](#1)

> Mục tiệu: Thành thạo thao tác câu lệnh trong `Python`

## 1.1. Biến & Kiểu dữ liệu [](#1.1)

Biến giống như là để định nghĩa hoặc nói đơn giản chính là tên gọi lưu trữ cho một giá trị (số, string, text, boolean) dữ liệu:

```python
name = "john" # Đây là biến dữ liệu là sting
age = 20 # Biến dữ liệu số
height = 1.72 # Biến dữ liệu số thập phân
is_student = True # Biến dữ liệu boolean 
```

**Bên cạnh đó ta cũng có các kiểu dữ liệu:**
- `str`: Kiểu dữ liệu string 
- `int`: Kiểu dữ liệu số nguyên
- `float`: Kiểu dữ liệu số thực
- `bool`: Kiểu dữ liệu boolean (True / False)

### 1.1.1. Các câu lệnh phổ biến sử dụng 
- `type()`: Xác định kiểu dữ liệu được lưu trữ tỏng biến
- `id()`:
- `print()`: Xuất giá trị của dữ liệu dược lưu trữ trong biến 
- `f_string`: Chuyển thành string
- `round()`: Làm tròn số thập phân 

### 1.1.2. Collection (Tập hợp)

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


**Set (Tập hợp)**

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

### 1.1.3. Function (Hàm)

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


## 1.2. OOP (Hướng đối tượng) [](#1.2)

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

---

## 1.5. Requests [](#1.5)
## 1.6. JSON [](#1.6)
## 1.7. Pandas [](#1.7)
## 1.8. Logging [](#1.8)



## 1.9. Thư viện [](#1.9)

---
# 2. SQL [](#2)
## 2.1. Lệnh `SELECT` [](#2.1)
## 2.2. Lệnh `JOIN` [](#2.2)
## 2.3. Lệnh `GROUP BY` [](#2.3)
## 2.4. Lệnh `HAVING` [](#2.4)
## 2.5. Lệnh `WINDOW FUNCTION` [](#2.5)
## 2.6. Lệnh `CTE` [](#2.6)
## 2.7. Lệnh `SUBQUERY` [](#2.7)
## 2.8. Lệnh `CASE WHEN` [](#2.7)
---

# 3. Git [](#3)
## 3.1. clone [](#3.1)
## 3.2. branch [](#3.1)
## 3.3. merge [](#3.1)
## 3.4. rebase [](#3.1)
## 3.5. pull request [](#3.1)
---
