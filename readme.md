	
books = pd.read_csv("./Books.csv")
users = pd.read_csv("./Users.csv")
ratings = pd.read_csv("./Ratings.csv")

## Popularity Based Recommend System
ratings_with_name = ratings.merge(books, on='ISBN')
ratings_with_name.groupby('Book-Title').count()['Book-Rating']
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_ratings', ascending=False).head(50)
popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')

### Giải thích 
Đọc dữ liệu: Dữ liệu từ ba tập tin CSV đã được đọc vào DataFrame của pandas. Cụ thể, tập tin "Books.csv" được đọc vào DataFrame books, tập tin "Users.csv" được đọc vào DataFrame users, và tập tin "Ratings.csv" được đọc vào DataFrame ratings.

Kết hợp dữ liệu đánh giá với thông tin sách: Dữ liệu về đánh giá sách được kết hợp với thông tin về sách từ tập dữ liệu books bằng cách sử dụng trường chung là ISBN. Kết quả là một DataFrame mới có tên là ratings_with_name, chứa thông tin về đánh giá sách cùng với thông tin chi tiết về sách.

Tính số lượt đánh giá cho mỗi sách: Dữ liệu được nhóm theo tựa đề sách và đếm số lượt đánh giá cho mỗi tựa đề. Kết quả được lưu vào DataFrame num_rating_df.

Tính điểm trung bình của sách: Dữ liệu được nhóm theo tựa đề sách và tính điểm trung bình của sách. Kết quả được lưu vào DataFrame avg_rating_df.

Kết hợp số lượt đánh giá và điểm trung bình của sách: Hai DataFrame num_rating_df và avg_rating_df được kết hợp lại dựa trên tựa đề sách. Kết quả là DataFrame popular_df, chứa thông tin về số lượt đánh giá và điểm trung bình của mỗi sách.

Lọc sách phổ biến: Chỉ lấy các sách có ít nhất 250 lượt đánh giá và sắp xếp chúng theo điểm trung bình giảm dần. Kết quả được lưu vào DataFrame popular_df.

Kết hợp thông tin sách với danh sách sách phổ biến: DataFrame popular_df được kết hợp lại với DataFrame books dựa trên tựa đề sách, và sau đó loại bỏ các bản sao của sách. Kết quả là một DataFrame chứa thông tin về sách phổ biến đã lọc và sắp xếp.

## Collaborative Filtering Based Recommend System
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
x[x].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(x[x].index)]
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_ratings =  filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
final_ratings.drop_duplicates()
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(pt)
similarity_scores = cosine_similarity(pt)

```py
def recommend(book_name) :
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x:x[1], reverse=True)[1:11]

    for i in similar_items :
        print(pt.index[i[0]])
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