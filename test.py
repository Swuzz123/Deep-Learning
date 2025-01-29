# ==================== Implementing Gradient Descent from Scratch ====================
MSE = 1/n * sum((y_pred - y_true)**2)

# Derivative of MSE by weight
dw = 1/n * 2 * sum((y_pred - y_true) * x)
# Derivative of MSE by bias
db = 1/n * 2 * sum(y_pred - y_true) 

# Update the parameters using gradient
w = w - alpha * dw
b = b -alpha * db

# Gradient Descent Algorithm
import numpy as np
def gradient_descent(x, y, alpha, num_iters):
    m = x.shape[0]
    w = np.zeros(x.shape[1])
    b = 0
    for i in range(num_iters):
        y_pred = np.dot(x, w) + b
        dw = (1/m) * np.dot(x.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        w = w - alpha * w
        b = b - alpha * b
    return w, b 

# ==================== Implementing Batch Gradient Descent (BGD) from Scratch ====================
# Example:
#   area	bedrooms	price
# 0	1056	2	        39.07
# 1	2600	4	        120.00
# 2	1440	3	        62.00

X = df.drop("price",axis = 1)
y = df['price']
# Ở đây shape của X là (3, 2)

def batch_GD(X, y_true, epochs, learning_rate = 0.01):
    
    # Đầu tiên ta sẽ lấy ra số lượng các features trong X, cụ thể ở đây là biến area và bedrooms nếu có nhiều hơn thì lấy nhiều hơn
    number_of_features = X.shape[1]
    weights = np.ones(shape = number_of_features) # tạo weights là 1 ma trận với các giá trị là 1 cụ thể ở đây sẽ là [1, 1]
    bias = 0
    total_samples = X.shape[0] # Tổng sample trong dataset của X
    
    # Dùng để lưu giá trị ở mỗi 10 hoặc 100 epochs 1 lần
    cost_list = []
    epochs_list = []
    
    """Formula loss function MSE: L(w, b) = 1/n sum(y_true - y_pred)^2
    với y_pred = weight*x + bias"""
    
    for i in range(epochs):
        # start
        y_pred = np.dot(weights, X.T) + bias 
        
        # Đạo hàm từng phần theo từng biến w và b
        w_derivative = -(2/ total_samples) * np.dot(X.T * (y_true - y_pred))
        b_derivative = -(2/ total_samples) * np.sum(y_true - y_pred)
        
        # Cập nhật trong số sau mỗi epochs
        weights = weights - learning_rate * w_derivative
        bias = bias - learning_rate * b_derivative
        
        cost = np.mean(np.square(y_true - y_pred))
        # end
        
        # Chỗ này là để lưu lại sau 1 quá trình cập nhật dài để xem ví 1000 epochs thì sau 10 epochs giá trị của cost là bao nhiêu
        if i%10 == 0:
            cost_list.append(cost)
            epochs_list.append(i)
            
    return weights, bias, cost, cost_list, epochs_list
    
# ==================== Implementing Stochastic Gradient Descent (SGD) from Scratch ====================
# Example:
#   area	bedrooms	price
# 0	1056	2	        39.07
# 1	2600	4	        120.00
# 2	1440	3	        62.00

X = df.drop("price",axis = 1)
y = df['price']
# Ở đây shape của X là (3, 2)

def stochastic_GD(X, y_true, epochs, learning_rate = 0.01):
    
    # Các bước đầu tương tự như BGD
    number_of_features = X.shape[1]
    weights = np.ones(shape = number_of_features)
    bias = 0
    total_samples = X.shape[0]
    
    for i in range(epochs):
        # start
        random_ind = random.randint(0, total_samples-1) # bước này để lấy ra index ngẫu nhiên 
        sample_x = X[random_ind]
        sample_y = y_true[random_ind]
                
                
        # Dùng để lưu giá trị ở mỗi 10 hoặc 100 epochs 1 lần
        cost_list = []
        epochs_list = []
        
        """Nhớ rằng SGD thì nó cập nhật liên tục parameters ở mỗi mẫu mà nó tính gd !"""
        y_pred = np.dot(weigts, sample_x.T) + bias
        
        # Đạo hàm từng phần theo từng biến w và b
        w_derivative = -(2/ total_samples) * (sample_x * (sample_y - y_pred))
        b_derivative = -(2/ total_samples) * (sample_y - y_pred) 
        
        # Cập nhật trong số sau mỗi epochs
        weights = weights - learning_rate * w_derivative
        bias = bias - learning_rate * b_derivative
        
        cost = np.mean(np.square(sample_y - y_pred))
        
        # Chỗ này là để lưu lại sau 1 quá trình cập nhật dài để xem ví 1000 epochs thì sau 10 epochs giá trị của cost là bao nhiêu
        if i%10 == 0:
            cost_list.append(cost)
            epochs_list.append(i) 
    
    return weights, bias, cost, cost_list, epochs_list
        
# ==================== Implementing Mini-Batch Gradient Descent (Mini - BGD) from Scratch ====================
# Example:
#   area	bedrooms	price
# 0	1056	2	        39.07
# 1	2600	4	        120.00
# 2	1440	3	        62.00

X = df.drop("price",axis = 1)
y = df['price']
# Ở đây shape của X là (3, 2)

def mini_BGD(X, y_true, batch_size, epochs, learning_rate = 0.01):
    
    # Bước đầu set up và lấy mẫu tương tự nhau
    number_of_features = X.shape[1]
    weights = np.ones(shape = number_of_features)
    bias = 0
    total_samples = X.shape[0]
    
    cost_list = []
    epochs_list = []
    
    for i in range(epochs):
        # Xáo trộn dữ liệu tại mỗi epochs
        random_indices = np.arrange(total_samples) # Tạo 1 mảng random
        np.random.shuffle(random_indices)
        X = X[random_indices]
        y_true = y_true[random_indices]
        
        for j in range(0, total_samples, batch_size):
            # Định nghĩa mini-batch 
            X_batch = X[j: j + batch_size]
            y_batch = y_true[j : j + batch_size]
            
            y_pred = np.dot(weights, X_batch.T) + bias
            
            # Đạo hàm từng phần theo biến w và b 
            """lưu ý: tại sao ở đây mình lại chia cho X_batch.shape[0] là vì nó đã được phân 
            ra mỗi khung nhỏ từ đâu đến đâu như nhau thì mình sẽ tính đạo hàm riêng cho mỗi ô dữ liệu đó thôi"""
            w_derivative = -(2/ X_batch.shape[0]) * np.dot(X_batch.T *(y_batch - y_pred))
            b_derivative = -(2/ X_batch.shape[0]) * np.sum(y_batch - y_pred)
            
            # Cập nhật trong số sau mỗi epochs
            weights = weights - learning_rate * w_derivative
            bias = bias - learning_rate * b_derivative
            
        # Tính cost và cập nhật lại y_pred sau khi chạy xong mỗi epochs
        y_pred_all = np.dot(weights, X.T) + bias
        cost = np.mean(np.square(y_true.flatten() - y_pred_all))
        
        if i%10 == 0:
            cost_list.append(cost)
            epochs_list.append(i)
    
    return weights, bias, cost, cost_list, epochs_list
            