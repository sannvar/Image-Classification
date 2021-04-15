# getting the data
library(keras)
mnist <- dataset_fashion_mnist()

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# plotting the data

# The right training example index was found by looking at the y_train dataset

# trouser
digit <- x_train[17,28:1,1:28]
par(pty="s") # for keeping the aspect ratio 1:1

# bag
digit <- x_train[24,28:1,1:28]
par(pty="s") # for keeping the aspect ratio 1:1
image(t(digit), col = gray.colors(256), axes = FALSE)

# ankle boot
digit <- x_train[1,28:1,1:28]
par(pty="s") # for keeping the aspect ratio 1:1
image(t(digit), col = gray.colors(256), axes = FALSE)

#processing the dataset

# flattening the dataset
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# normalizing the dataset between 0-1
x_train <- x_train / 255
x_test <- x_test / 255

# processing to categorical data
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


# model1 uses 32 hidden neurons and relu function 
model1 <- keras_model_sequential() 
model1 %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = 'softmax')

model1 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

history1 <- model1 %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)


model1 %>% evaluate(x_test, y_test)

# model2 uses 128 hidden neurons and relu function 
model2 <- keras_model_sequential() 
model2 %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = 'softmax')

model2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

history2 <- model2 %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)

model2 %>% evaluate(x_test, y_test)

# model3 uses 256 hidden neurons and relu function 
model3 <- keras_model_sequential() 
model3 %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = 'softmax')

model3 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

history3 <- model3 %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)

model3 %>% evaluate(x_test, y_test)

# model4 uses 32 hidden neurons and sigmoid function 
model4 <- keras_model_sequential() 
model4 %>% 
  layer_dense(units = 32, activation = 'sigmoid', input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = 'softmax')

model4 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

history4 <- model4 %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)

model4 %>% evaluate(x_test, y_test)

# model5 uses 128 hidden neurons and sigmoid function 
model5 <- keras_model_sequential() 
model5 %>% 
  layer_dense(units = 128, activation = 'sigmoid', input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = 'softmax')

model5 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

history5 <- model5 %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)

model5 %>% evaluate(x_test, y_test)

# model6 uses 256 hidden neurons and sigmoid function 
model6 <- keras_model_sequential() 
model6 %>% 
  layer_dense(units = 256, activation = 'sigmoid', input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = 'softmax')

model6 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

history6 <- model6 %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)

model6 %>% evaluate(x_test, y_test)


# The neural network's accuracy was highest at 88.3% but the results are not reproducible. 
# It was found that the best accuracy was achieved by the model that used 256 neurons and the ReLu activation function..
