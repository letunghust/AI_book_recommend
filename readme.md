	
books = pd.read_csv("./Books.csv")
users = pd.read_csv("./Users.csv")
ratings = pd.read_csv("./Ratings.csv")

# Popularity Based Recommend System
```python
ratings_with_name = ratings.merge(books, on='ISBN')
ratings_with_name.groupby('Book-Title').count()['Book-Rating']
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_ratings', ascending=False).head(50)
popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')
```

### Giải thích 
Đọc dữ liệu: Dữ liệu từ ba tập tin CSV đã được đọc vào DataFrame của pandas. Cụ thể, tập tin "Books.csv" được đọc vào DataFrame books, tập tin "Users.csv" được đọc vào DataFrame users, và tập tin "Ratings.csv" được đọc vào DataFrame ratings.

Kết hợp dữ liệu đánh giá với thông tin sách: Dữ liệu về đánh giá sách được kết hợp với thông tin về sách từ tập dữ liệu books bằng cách sử dụng trường chung là ISBN. Kết quả là một DataFrame mới có tên là ratings_with_name, chứa thông tin về đánh giá sách cùng với thông tin chi tiết về sách.

Tính số lượt đánh giá cho mỗi sách: Dữ liệu được nhóm theo tựa đề sách và đếm số lượt đánh giá cho mỗi tựa đề. Kết quả được lưu vào DataFrame num_rating_df.

Tính điểm trung bình của sách: Dữ liệu được nhóm theo tựa đề sách và tính điểm trung bình của sách. Kết quả được lưu vào DataFrame avg_rating_df.

Kết hợp số lượt đánh giá và điểm trung bình của sách: Hai DataFrame num_rating_df và avg_rating_df được kết hợp lại dựa trên tựa đề sách. Kết quả là DataFrame popular_df, chứa thông tin về số lượt đánh giá và điểm trung bình của mỗi sách.

Lọc sách phổ biến: Chỉ lấy các sách có ít nhất 250 lượt đánh giá và sắp xếp chúng theo điểm trung bình giảm dần. Kết quả được lưu vào DataFrame popular_df.

Kết hợp thông tin sách với danh sách sách phổ biến: DataFrame popular_df được kết hợp lại với DataFrame books dựa trên tựa đề sách, và sau đó loại bỏ các bản sao của sách. Kết quả là một DataFrame chứa thông tin về sách phổ biến đã lọc và sắp xếp.

# Collaborative Filtering Based Recommend System
- x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
- x[x].index
- filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(x[x].index)]
- y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
- famous_books = y[y].index
- final_ratings =  filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
- final_ratings.drop_duplicates()
- pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
- pt.fillna(0, inplace=True)

- from sklearn.metrics.pairwise import cosine_similarity
- cosine_similarity(pt)
- similarity_scores = cosine_similarity(pt)

```py
def recommend(book_name) :
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x:x[1], reverse=True)[1:11]

    data = []
    for i in similar_items :
        item = []
        # print(pt.index[i[0]])
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return data 

```

### Giải thích 
- Lọc dữ liệu: Đầu tiên, một bộ lọc được áp dụng để chỉ giữ lại các đánh giá từ người dùng đã đánh giá hơn 200 cuốn sách. Kết quả là một Series boolean với chỉ mục là các User-ID của người dùng đã đánh giá hơn 200 cuốn sách.

- Xác định cuốn sách nổi tiếng: Tiếp theo, một bộ lọc khác được áp dụng để chỉ giữ lại các cuốn sách đã nhận được ít nhất 50 lượt đánh giá. Kết quả là một danh sách các tựa đề sách nổi tiếng.

- Lọc lại dữ liệu dựa trên người dùng và cuốn sách nổi tiếng: Dữ liệu được lọc lại để chỉ chứa các đánh giá từ những người dùng đã đánh giá hơn 200 cuốn sách và các cuốn sách nổi tiếng đã nhận được ít nhất 50 lượt đánh giá.

- Tạo bảng dữ liệu pivot: Bảng dữ liệu pivot được tạo từ dữ liệu đã lọc, với tựa đề sách làm hàng và User-ID làm cột. Bảng này được sử dụng để tính toán độ tương tự giữa các cuốn sách.
=> Bảng dữ liệu này cung cấp 1 cái nhìn tổng quát về cách mà người dùng đánh giá các cuốn sách, dùng để tính toán các độ tương tự giữa các cuốn sách dựa trên cách mà người dùng đánh giá chúng.

- Tính toán ma trận tương tự: Sử dụng hàm cosine_similarity từ thư viện sklearn.metrics.pairwise, tính toán ma trận tương tự giữa các cuốn sách dựa trên điểm số của người dùng. 
    + Mỗi phần tử trong ma trận tương tự chứa 1 giá trị độ tương tự giữa 2 cuốn sách tương ứng, được tính dựa trên bảng dữ liệu pivot.
    + Ma trận tương tự sử dụng để tìm các cuốn sách tương ứng với 1 cuốn sách cụ thể. 

- Hàm gợi ý (recommend): Định nghĩa hàm recommend(book_name) để gợi ý các cuốn sách tương tự với một cuốn sách cụ thể được chỉ định. Hàm này sẽ trả về danh sách các cuốn sách tương tự với cuốn sách đó, dựa trên ma trận tương tự đã tính toán ở bước trước đó. 


## Để sử dụng TF-IDF để tìm kiếm sách gần đúng và gợi ý các cuốn sách tương tự, bạn có thể làm theo các bước sau:

1. Chuẩn bị dữ liệu:
   - Tạo một DataFrame mới chứa thông tin về tiêu đề sách và tác giả.
   - Kết hợp tiêu đề sách và tác giả thành một chuỗi duy nhất để sử dụng cho việc tính toán TF-IDF.

2. Tính toán ma trận TF-IDF:
   - Sử dụng `TfidfVectorizer` từ thư viện scikit-learn để tính toán ma trận TF-IDF cho các chuỗi kết hợp tiêu đề sách và tác giả.

3. Tìm kiếm sách gần đúng:
   - Khi người dùng nhập một từ khóa tìm kiếm, sử dụng `TfidfVectorizer` để chuyển đổi từ khóa thành vector TF-IDF.
   - Tính toán độ tương đồng cosine giữa vector TF-IDF của từ khóa và ma trận TF-IDF của các cuốn sách.
   - Lấy cuốn sách có độ tương đồng cao nhất làm kết quả tìm kiếm gần đúng.

4. Gợi ý các cuốn sách tương tự:
   - Sử dụng hàm `recommend` đã có từ trước để gợi ý các cuốn sách tương tự dựa trên cuốn sách được tìm kiếm gần đúng.

Dưới đây là đoạn mã Python minh họa cách thực hiện:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Tính toán ma trận TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['Book-Title'])

vectorizer = TfidfVectorizer()
vectorizer.fit(books['Book-Title'])


def search_book(query, vectorizer):
    # query = '1984'
    # processed = re.sub(r"[^a-zA-Z0-9]", "", query.lower())
    processed = query.lower()
    query_vec = tfidf.transform([processed])
    
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = np.argpartition(similarity, -10)[-10:]

     # Lọc các chỉ số hợp lệ
    # valid_indices = [idx for idx in indices if idx < len(pt.index)]
    results = books.iloc[indices]
    
    # results = results.sort_values("num_ratings", ascending=False)
    return results.head(5)

search_book_df =  search_book("The Long Road Hom", vectorizer)
search_book_df
search_book_merge = search_book_df.merge(final_ratings[['ISBN', 'Book-Rating']], on='ISBN', how='left').sort_values('Book-Rating', ascending=False).head(1)
search_book_merge
book = search_book_merge['Book-Title'].values[0]
book
recommend(book)
```

Trong đoạn mã trên:
- Chúng ta tạo một DataFrame mới `book_data` chứa tiêu đề sách và tác giả, và kết hợp chúng thành một chuỗi duy nhất trong cột 'Book-Info'.
- Sử dụng `TfidfVectorizer` để tính toán ma trận TF-IDF cho các chuỗi 'Book-Info'.
- Hàm `search_book` nhận từ khóa tìm kiếm, chuyển đổi nó thành vector TF-IDF, tính toán độ tương đồng cosine với ma trận TF-IDF của các cuốn sách, và trả về cuốn sách có độ tương đồng cao nhất.
- Hàm `recommend_books` sử dụng `search_book` để tìm kiếm sách gần đúng, sau đó sử dụng hàm `recommend` đã có từ trước để gợi ý các cuốn sách tương tự.

Với cách tiếp cận này, người dùng có thể nhập từ khóa tìm kiếm gần đúng và hệ thống sẽ tìm kiếm cuốn sách phù hợp nhất dựa trên TF-IDF, sau đó gợi ý các cuốn sách tương tự dựa trên collaborative filtering.

### Sử dụng thêm kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) để tiền xử lý từ khóa tìm kiếm và dữ liệu sách. Cụ thể, chúng ta có thể sử dụng các bước như:

Chuyển đổi tất cả các ký tự thành chữ thường (lowercase).
Loại bỏ các ký tự đặc biệt và dấu câu.
Loại bỏ các từ dừng (stop words) như "the", "a", "an", vv.
Áp dụng kỹ thuật stemming hoặc lemmatization để rút gọn các từ về dạng gốc.

``` py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tải các tài nguyên cần thiết của NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Khởi tạo lemmatizer
lemmatizer = WordNetLemmatizer()

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    # Chuyển đổi thành chữ thường
    text = text.lower()
    
    # Loại bỏ các ký tự đặc biệt và dấu câu
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Loại bỏ các từ dừng
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    # Áp dụng lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    # Kết hợp lại thành chuỗi
    processed_text = ' '.join(lemmatized_words)
    
    return processed_text

# Tiền xử lý dữ liệu sách
book_data['Processed-Book-Info'] = book_data['Book-Info'].apply(preprocess_text)

# Tính toán ma trận TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(book_data['Processed-Book-Info'])

def search_book(query):
    # Tiền xử lý từ khóa tìm kiếm
    processed_query = preprocess_text(query)
    
    # Chuyển đổi từ khóa tìm kiếm thành vector TF-IDF
    query_vector = tfidf.transform([processed_query])
    
    # Tính toán độ tương đồng cosine
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    
    # Lấy cuốn sách có độ tương đồng cao nhất
    index = similarity_scores.argmax()
    book_title = book_data.iloc[index]['Book-Title']
    
    return book_title

def recommend_books(query):
    # Tìm kiếm sách gần đúng
    book_title = search_book(query)
    
    # Gợi ý các cuốn sách tương tự
    recommended_books = recommend(book_title)
    
    return recommended_books
```